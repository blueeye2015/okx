import clickhouse_connect
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'

def load_data():
    print("🚀 正在加载数据...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    sql = f"""
    SELECT time, close_price, wall_shift_pct, net_cvd, spoofing_ratio
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time ASC
    """
    df = client.query_df(sql)
    return df

def prepare_data(df):
    # 1. 基础清洗
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. 填充缺失值 (模拟实盘，用前值填充)
    df['wall_shift_pct'] = df['wall_shift_pct'].ffill().fillna(0).clip(-5, 5)
    df['spoofing_ratio'] = df['spoofing_ratio'].ffill().fillna(1.0)
    df['net_cvd'] = df['net_cvd'].fillna(0)
    
    # 3. CVD 归一化 (Z-Score)
    rolling_mean = df['net_cvd'].rolling(20, min_periods=1).mean()
    rolling_std = df['net_cvd'].rolling(20, min_periods=1).std().replace(0, 1)
    df['cvd_zscore'] = (df['net_cvd'] - rolling_mean) / rolling_std
    
    # 4. 计算收益 (Target)
    df['next_return'] = (df['close_price'].shift(-1) - df['close_price']) / df['close_price'] * 100
    df = df.dropna()
    
    # 5. 打标签 (门槛 0.15%)
    df['label'] = 0
    df.loc[df['next_return'] > 0.15, 'label'] = 1
    
    print(f"🧹 数据准备完成: {len(df)} 条样本")
    return df

def run_backtest(df):
    # 特征选择
    X = df[['wall_shift_pct', 'cvd_zscore', 'spoofing_ratio']]
    y = df['label'].astype(int)
    
    # [关键] 划分训练集和测试集 (不打乱时间)
    # 此时 X_train 和 X_test 都包含暴跌数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # ---------------------------------------------------------
    # 1. 训练阶段 (只给 AI 看"好"数据)
    # ---------------------------------------------------------
    # 从训练集中剔除 wall_shift < -0.2 的脏数据，防止 AI 学会接飞刀
    train_mask = X_train['wall_shift_pct'] > -0.2
    X_train_clean = X_train[train_mask]
    y_train_clean = y_train[train_mask]
    
    print(f"🧠 训练集去噪: 剔除 {len(X_train) - len(X_train_clean)} 条暴跌样本")
    print("🚀 开始训练决策树...")
    
    # 使用优化过的权重 1:5 (激进进攻)
    clf = DecisionTreeClassifier(
        max_depth=4, 
        criterion='entropy', 
        random_state=42, 
        class_weight={0: 1, 1: 5.0}, 
        min_samples_leaf=20
    )
    clf.fit(X_train_clean, y_train_clean)
    
    # ---------------------------------------------------------
    # 2. 预测阶段 (测试集包含所有真实行情，包括暴跌)
    # ---------------------------------------------------------
    y_pred = clf.predict(X_test)
    
    # ---------------------------------------------------------
    # 3. 回测与风控模拟
    # ---------------------------------------------------------
    # 获取真实收益率 (除以100修复Bug)
    real_returns = df.loc[X_test.index, 'next_return'] / 100
    
    # --- 策略逻辑 ---
    final_signals = y_pred.copy()
    
    # [风控升级] 只有当 墙塌(-0.2) 且 主力砸盘(CVD<-0.5) 时，才强制空仓
    # 这模拟了我们在 production_signal.py 里的逻辑
    # 1. 获取对应的特征列
    test_wall = X_test['wall_shift_pct']
    test_cvd = X_test['cvd_zscore']
    
    # 2. 定义"真跌"条件
    mask_real_dump = (test_wall < -0.2) & (test_cvd < -0.5)
    
    # 3. 统计风控拦截次数
    triggered_count = mask_real_dump.sum()
    print(f"🛡️ 风控系统触发: {triggered_count} 次 (成功拦截暴跌)")
    
    # 4. 执行熔断 (将信号置为 0)
    final_signals[mask_real_dump] = 0
    
    # --- 计算资金曲线 ---
    market_curve = (real_returns + 1).cumprod()
    strategy_curve = (real_returns * final_signals + 1).cumprod()
    
    # --- 打印结果 ---
    print("\n" + "="*40)
    print(f"💰 市场基准回报: {market_curve.iloc[-1]:.4f}x")
    print(f"🤖 AI 策略回报:   {strategy_curve.iloc[-1]:.4f}x")
    print("="*40)
    
    # 赢家分析
    wins = real_returns[final_signals == 1] > 0
    print(f"🎯 胜率: {wins.mean():.2%} (交易次数: {sum(final_signals)})")
    
    # 打印规则
    print("\n📜 最终规则树:")
    print(export_text(clf, feature_names=list(X.columns)))

if __name__ == "__main__":
    raw_df = load_data()
    if not raw_df.empty:
        clean_df = prepare_data(raw_df)
        run_backtest(clean_df)