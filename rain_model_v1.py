import clickhouse_connect
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'

def load_and_clean_data():
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    print("🚀 加载数据中...")
    
    # 1. 读取数据
    sql = f"""
    SELECT time, close_price, wall_shift_pct, net_cvd, spoofing_ratio
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time ASC
    """
    df = client.query_df(sql)
    
    # 2. 基础清洗 (inf -> nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 3. 前向填充 + 补0
    df['wall_shift_pct'] = df['wall_shift_pct'].fillna(method='ffill').fillna(0)
    df['spoofing_ratio'] = df['spoofing_ratio'].fillna(method='ffill').fillna(1.0)
    df['net_cvd'] = df['net_cvd'].fillna(0)
    
    # 4. [调试] 打印 wall_shift 的异常值情况
    max_shift = df['wall_shift_pct'].max()
    print(f"🔍 检查数据: wall_shift_pct 最大值 = {max_shift}")
    if max_shift > 10:
        print("⚠️ 警告: 发现异常巨大的 wall_shift，正在强制截断...")
        
    # 5. 强制截断 (Clip) - 修复 200% 问题
    df['wall_shift_pct'] = df['wall_shift_pct'].clip(lower=-3.0, upper=3.0)
    
    # 6. CVD 归一化
    rolling_mean = df['net_cvd'].rolling(20, min_periods=1).mean()
    rolling_std = df['net_cvd'].rolling(20, min_periods=1).std().replace(0, 1)
    df['cvd_zscore'] = (df['net_cvd'] - rolling_mean) / rolling_std
    
    # 7. 计算收益率 (Label)
    # 这一步计算出百分比 (例如 0.5)
    df['next_return'] = (df['close_price'].shift(-1) - df['close_price']) / df['close_price'] * 100
    df = df.dropna(subset=['next_return'])
    
    # 8. [调试] 检查收益率量级
    print(f"🔍 检查收益: next_return 最大值 = {df['next_return'].max()}%")
    
    # 9. 打标签 (降低门槛到 0.1%，增加正样本)
    df['label'] = 0
    df.loc[df['next_return'] > 0.1, 'label'] = 1
    
    print(f"🧹 有效样本: {len(df)} | 正样本(买入机会): {sum(df['label']==1)}")
    return df

def force_train(df):
    X = df[['wall_shift_pct', 'cvd_zscore', 'spoofing_ratio']]
    y = df['label'].astype(int)
    
    # 不打乱顺序
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    print("\n🧠 正在进行暴力训练 (Force Training)...")
    
    # [关键修改] 极端的权重！
    # 只要有机会，就让它买！哪怕由于误报导致准确率下降，我们要先看到它开单。
    # 0 (观望): 权重 1
    # 1 (买入): 权重 10 (强迫模型重视买入信号)
    force_weights = {0: 1, 1: 5}
    
    # min_samples_leaf=20: 防止过拟合，让规则更通用
    clf = DecisionTreeClassifier(
        max_depth=4, 
        criterion='entropy', 
        random_state=42, 
        class_weight=force_weights,
        min_samples_leaf=20 
    )
    clf.fit(X_train, y_train)
    
    # --- 评估 ---
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # --- 回测 (修复 15x 回报 Bug) ---
    # 1. 获取测试集的原始回报 (百分数)
    raw_ret = df.loc[X_test.index, 'next_return']
    
    # 2. [自动修复量纲] 如果回报率均值 > 1，说明没除以100
    # 正常的 15m 回报率均值应该是 0.0x%
    scale_factor = 100.0
    if raw_ret.abs().mean() < 0.1: 
        # 如果均值已经很小(比如 0.005)，说明已经是小数了，不用除以100
        scale_factor = 1.0
        
    real_returns = raw_ret / scale_factor
    
    # 市场基准
    cum_market = (real_returns + 1).cumprod()
    
    # 策略收益
    # 增加人工过滤: 墙塌了(-0.2) 坚决不买
    signals = y_pred.copy()
    signals[X_test['wall_shift_pct'] < -0.2] = 0
    
    strategy_returns = real_returns * signals
    cum_strategy = (strategy_returns + 1).cumprod()
    
    print(f"\n💰 市场基准回报: {cum_market.iloc[-1]:.4f}x (期间涨跌: {(cum_market.iloc[-1]-1)*100:.2f}%)")
    print(f"🤖 AI 策略回报:   {cum_strategy.iloc[-1]:.4f}x")
    
    # 打印规则
    print("\n📜 激进版规则树:")
    print(export_text(clf, feature_names=list(X.columns)))

if __name__ == "__main__":
    df = load_and_clean_data()
    force_train(df)