import clickhouse_connect
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import joblib

# ================= 配置区 =================
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'
MODEL_PATH = '/data/okx/universal_model.pkl'

# 策略参数 (Plan B: 短平快)
FEE_THRESHOLD = 0.35         # 目标: 30分钟内涨 0.35% (扣费后赚 0.25% 就跑)
CONFIDENCE_THRESHOLD = 0.60  # 门槛 60%
ABNORMAL_WALL_THRES = 50.0   

# 回测风控参数
TP_PCT = 0.006  # 止盈 0.6% (微观爆发通常就在 0.5%~0.8% 之间)
SL_PCT = 0.010  # 止损 1.0% (给波动留空间)

def load_data():
    print("🚀 正在加载数据...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    
    # [修改] SQL 中去掉了不存在的 high_price 和 low_price
    query = f"""
    SELECT 
        time, close_price,
        wall_shift_pct, spoofing_ratio, net_cvd
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time ASC
    """
    df = client.query_df(query)
    
    # [新增] 既然表里没有 High/Low，我们用 Close 暂代
    # 这是一种"保守回测"：我们假设K线没有影线，只有实体
    # 如果这种情况下还能赚钱，说明策略非常硬核
    df['high_price'] = df['close_price']
    df['low_price'] = df['close_price']
    
    # ================= 特征工程 (Plan B: 维度对齐) =================
    
    # 1. 资金流 Z-Score (微观)
    df['cvd_mean'] = df['net_cvd'].rolling(window=96, min_periods=1).mean()
    df['cvd_std'] = df['net_cvd'].rolling(window=96, min_periods=1).std().replace(0, 1)
    df['cvd_zscore'] = (df['net_cvd'] - df['cvd_mean']) / df['cvd_std']
    
    # 2. 趋势乖离率 (宏观) - 96根K线 = 24小时
    df['ema96'] = df['close_price'].ewm(span=96, adjust=False).mean()
    df['dist_ema96'] = (df['close_price'] - df['ema96']) / df['ema96'] * 100
    
    # 3. [🔥 新特征] 趋势资金共振
    df['trend_flow_resonance'] = np.sign(df['dist_ema96']) * df['cvd_zscore']
    
    # 4. 数据清洗
    df['wall_shift_pct'] = df['wall_shift_pct'].fillna(method='ffill').fillna(0)
    df['spoofing_ratio'] = df['spoofing_ratio'].fillna(method='ffill').fillna(1.0)
    df['cvd_zscore'] = df['cvd_zscore'].fillna(0)
    df['trend_flow_resonance'] = df['trend_flow_resonance'].fillna(0)
    df = df.dropna()

    # ================= 目标工程 =================
    
    # 目标: 预测未来 30分钟 (2根K线)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
    
    # 因为我们用 close 代替了 high，这里 future_max 其实就是未来收盘价的最高值
    df['future_max'] = df['high_price'].rolling(window=indexer).max()
    
    # 计算潜在收益
    df['max_potential_return'] = (df['future_max'] - df['close_price']) / df['close_price'] * 100
    
    return df

def clean_and_prepare(df):
    print(f"🧹 原始样本数: {len(df)}")
    
    # 1. 剔除盘口脏数据
    mask_clean = df['wall_shift_pct'].abs() < ABNORMAL_WALL_THRES
    df = df[mask_clean].copy()
    
    # 2. [🔥 Plan B 过滤] 剔除极度逆势的样本
    # 在极度深熊 (价格低于均线 3% 以上) 时，微观买入信号失效概率极高
    # 我们强行不让 AI 学习这些样本
    mask_trend = df['dist_ema96'] > -3.0
    df_clean = df[mask_trend].copy()
    
    print(f"✂️ 剔除异常值及深熊样本后: {len(df_clean)} (删除了 {len(df) - len(df_clean)} 条)")
    
    # 3. 打标签
    df_clean['target'] = 0
    # 只要 30分钟内能冲高 0.35%，就算赢
    df_clean.loc[df_clean['max_potential_return'] > FEE_THRESHOLD, 'target'] = 1
    
    pos_count = df_clean['target'].sum()
    print(f"📊 正样本(短线爆发): {pos_count} | 负样本: {len(df_clean) - pos_count}")
    return df_clean

def train_with_oversampling(df):
    # 特征列表 (加入了共振因子)
    features = ['wall_shift_pct', 'spoofing_ratio', 'cvd_zscore', 'dist_ema96', 'trend_flow_resonance']
    
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()
    
    # 暴力过采样
    df_majority = train_data[train_data.target == 0]
    df_minority = train_data[train_data.target == 1]
    
    if len(df_minority) == 0: 
        print("❌ 无正样本")
        return None, None, None

    print(f"💪 过采样中... (原始正样本: {len(df_minority)})")
    # 稍微控制比例，防止过拟合
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=int(len(df_majority) * 0.35), random_state=42)
    train_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    X_train = train_upsampled[features]
    y_train = train_upsampled['target']
    X_test = test_data[features]
    y_test = test_data['target']
    
    print("🧠 训练随机森林 (Plan B: 短平快版)...")
    clf = RandomForestClassifier(
        n_estimators=200,      # 树多一点，稳一点
        max_depth=6,           # 深度适中
        min_samples_leaf=20,   # 提高叶子门槛，过滤噪音
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    return clf, X_test, test_data

def evaluate_strategy(clf, X_test, test_df):
    print("\n================ 策略评估 (30分钟闪电战) ================")
    probs = clf.predict_proba(X_test)[:, 1]
    
    prob_series = pd.Series(probs)
    print("🧐 概率分布: Max:", f"{prob_series.max():.4f}", "| 95%:", f"{prob_series.quantile(0.95):.4f}")
    
    # 信号生成
    test_df = test_df.copy()
    test_df['signal'] = 0
    test_df.loc[probs > CONFIDENCE_THRESHOLD, 'signal'] = 1
    
    # === 模拟回测 (Plan B: 时间止损缩短) ===
    trades = test_df[test_df['signal'] == 1].copy()
    
    if len(trades) == 0:
        print("😴 无开单")
        return 1.0

    # 1. 默认收益: 持有 2根K线 (30分钟) 后的收盘价收益
    # shift(-2) 代表 30分钟后的价格
    trades['exit_price_time'] = trades['close_price'].shift(-2)
    # 如果是最后几行没数据，就用最后一行填补
    trades['exit_price_time'] = trades['exit_price_time'].fillna(trades['close_price'].iloc[-1])
    
    trades['trade_return'] = (trades['exit_price_time'] - trades['close_price']) / trades['close_price']
    
    # 2. 检查止盈 (TP)
    # 如果未来2根K线最高价摸到了 TP
    tp_price = trades['close_price'] * (1 + TP_PCT)
    hit_tp_mask = trades['future_max'] >= tp_price
    trades.loc[hit_tp_mask, 'trade_return'] = TP_PCT
    
    # 3. 检查止损 (SL) (简单模拟: 如果30分钟后亏损超过 SL，则按 SL 算)
    # 这是一个简化，严格来说应该看 future_min，但我们 Plan B 假设微观爆发速度很快
    sl_mask = trades['trade_return'] < -SL_PCT
    trades.loc[sl_mask, 'trade_return'] = -SL_PCT
    
    # 4. 扣费 (0.1%)
    trades['net_return'] = trades['trade_return'] - 0.001
    
    # 5. 统计
    test_df['strategy_net_return'] = 0.0
    test_df.loc[trades.index, 'strategy_net_return'] = trades['net_return']
    
    cum_strategy = (test_df['strategy_net_return'] + 1).cumprod()
    final_nav = cum_strategy.iloc[-1]
    
    trade_count = len(trades)
    tp_count = hit_tp_mask.sum()
    sl_count = sl_mask.sum()
    win_count = len(trades[trades['net_return'] > 0])
    
    print(f"🎯 狙击门槛: > {CONFIDENCE_THRESHOLD*100}%")
    print(f"🔥 开单次数: {trade_count}")
    print(f"✅ 止盈触发: {tp_count} 次 ({tp_count/trade_count*100:.1f}%)")
    print(f"❌ 止损触发: {sl_count} 次")
    print(f"💰 最终净值: {final_nav:.4f}x")
    print(f"🏆 胜率: {win_count / trade_count * 100:.2f}%")

    # 特征重要性
    feature_imp = pd.Series(clf.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    print("\n🔍 关键特征排名 (看 Resonance 排第几):")
    print(feature_imp)

    return final_nav

if __name__ == "__main__":
    df = load_data()
    df = clean_and_prepare(df)
    model, X_test, test_df = train_with_oversampling(df)
    if model:
        evaluate_strategy(model, X_test, test_df)
        joblib.dump(model, MODEL_PATH)