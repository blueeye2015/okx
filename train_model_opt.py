import clickhouse_connect
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'

def load_and_clean_data():
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    print("🚀 加载数据中...")
    
    sql = f"""
    SELECT time, close_price, wall_shift_pct, net_cvd, spoofing_ratio
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time ASC
    """
    df = client.query_df(sql)
    
    # 1. 基础清洗
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. [修复警告] 新版 Pandas 写法
    df['wall_shift_pct'] = df['wall_shift_pct'].ffill().fillna(0)
    df['spoofing_ratio'] = df['spoofing_ratio'].ffill().fillna(1.0)
    df['net_cvd'] = df['net_cvd'].fillna(0)
    
    # 3. 截断异常值
    df['wall_shift_pct'] = df['wall_shift_pct'].clip(lower=-3.0, upper=3.0)
    
    # 4. CVD 归一化
    rolling_mean = df['net_cvd'].rolling(20, min_periods=1).mean()
    rolling_std = df['net_cvd'].rolling(20, min_periods=1).std().replace(0, 1)
    df['cvd_zscore'] = (df['net_cvd'] - rolling_mean) / rolling_std
    
    # 5. 计算目标
    df['next_return'] = (df['close_price'].shift(-1) - df['close_price']) / df['close_price'] * 100
    df = df.dropna(subset=['next_return'])
    
    # 6. 打标签 (门槛 0.1%)
    df['label'] = 0
    df.loc[df['next_return'] > 0.1, 'label'] = 1
    
    return df

def optimize_weights(df):
    X = df[['wall_shift_pct', 'cvd_zscore', 'spoofing_ratio']]
    y = df['label'].astype(int)
    
    # 不打乱顺序
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    real_returns = df.loc[X_test.index, 'next_return'] / 100
    market_return = (real_returns + 1).cumprod().iloc[-1]
    
    print(f"\n💰 市场基准回报: {market_return:.4f}x")
    print("-" * 65)
    print(f"{'买入权重':<10} | {'准确率(Precision)':<18} | {'召回率(Recall)':<15} | {'策略回报(ROI)':<15}")
    print("-" * 65)
    
    best_roi = 0
    best_weight = 0
    
    # 循环测试权重：从 1.0 到 10.0
    for w in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        # 训练
        clf = DecisionTreeClassifier(
            max_depth=4, 
            criterion='entropy', 
            random_state=42, 
            class_weight={0: 1, 1: w}, # 动态调整权重
            min_samples_leaf=20
        )
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        
        # 应用人工熔断 (墙塌不买)
        final_signals = y_pred.copy()
        final_signals[X_test['wall_shift_pct'] < -0.2] = 0
        
        # 计算指标
        prec = precision_score(y_test, final_signals, zero_division=0)
        rec = recall_score(y_test, final_signals, zero_division=0)
        
        # 计算回报
        strategy_ret = (real_returns * final_signals + 1).cumprod().iloc[-1]
        
        print(f"1 : {w:<6} | {prec:<18.2%} | {rec:<15.2%} | {strategy_ret:.4f}x {'🔥' if strategy_ret > market_return else ''}")
        
        if strategy_ret > best_roi:
            best_roi = strategy_ret
            best_weight = w

    print("-" * 65)
    print(f"🏆 最佳买入权重: 1 : {best_weight} (回报 {best_roi:.4f}x)")

if __name__ == "__main__":
    try:
        data = load_and_clean_data()
        optimize_weights(data)
    except Exception as e:
        print(f"Error: {e}")