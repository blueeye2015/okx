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
    print("🚀 加载全量数据...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    sql = f"""
    SELECT time, close_price, wall_shift_pct, net_cvd, spoofing_ratio
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time ASC
    """
    df = client.query_df(sql)
    return df

def feature_engineering(df):
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 1. 趋势特征 (EMA)
    df['ema_50'] = df['close_price'].ewm(span=50).mean()
    df['dist_ema'] = (df['close_price'] - df['ema_50']) / df['ema_50'] * 100
    
    # 2. 资金特征 (CVD Z-Score)
    rolling_mean = df['net_cvd'].rolling(window=96).mean() # 1天基准
    rolling_std = df['net_cvd'].rolling(window=96).std().replace(0, 1)
    df['cvd_z'] = (df['net_cvd'] - rolling_mean) / rolling_std
    
    # 3. 墙特征 (平滑)
    df['wall_smooth'] = df['wall_shift_pct'].rolling(3).mean()
    
    # 4. 构造目标 (三分类: 1=涨, 2=跌, 0=震荡)
    # 我们看未来 4小时 (16根K线)
    window = 16
    df['next_max'] = df['close_price'].shift(-1).rolling(window).max()
    df['next_min'] = df['close_price'].shift(-1).rolling(window).min()
    df['next_close'] = df['close_price'].shift(-window)
    
    # 门槛
    TARGET_PCT = 1.2 # 目标涨跌幅 1.2%
    
    df['label'] = 0
    # 做多机会: 最高价涨超 1.2% 且 最低价没跌破 -0.6% (盈亏比 2:1)
    long_cond = (df['next_max'] / df['close_price'] - 1 > TARGET_PCT/100) & \
                (df['next_min'] / df['close_price'] - 1 > -TARGET_PCT/2/100)
    
    # 做空机会: 最低价跌超 1.2% 且 最高价没涨破 0.6%
    short_cond = (df['next_min'] / df['close_price'] - 1 < -TARGET_PCT/100) & \
                 (df['next_max'] / df['close_price'] - 1 < TARGET_PCT/2/100)
    
    df.loc[long_cond, 'label'] = 1  # Long
    df.loc[short_cond, 'label'] = 2 # Short
    
    df = df.dropna()
    print(f"🧹 数据清洗完成: {len(df)} 条 | 多头样本: {sum(df['label']==1)} | 空头样本: {sum(df['label']==2)}")
    return df

def simulate_trade_path(entry_price, signal, future_prices, tp_pct=0.03, sl_pct=0.015):
    """
    模拟真实的持仓路径，检查止盈止损
    signal: 1 (Long), 2 (Short)
    """
    for price in future_prices:
        change = (price - entry_price) / entry_price
        
        if signal == 1: # 做多
            if change >= tp_pct: return tp_pct  # 止盈
            if change <= -sl_pct: return -sl_pct # 止损
            
        elif signal == 2: # 做空
            if change <= -tp_pct: return tp_pct # 止盈 (跌了赚钱)
            if change >= sl_pct: return -sl_pct # 止损 (涨了亏钱)
            
    # 如果时间到了既没止盈也没止损，按最后一根K线结算
    final_change = (future_prices.iloc[-1] - entry_price) / entry_price
    if signal == 2: final_change = -final_change
    return final_change

def run_universal_bot(df):
    features = ['dist_ema', 'cvd_z', 'wall_smooth', 'spoofing_ratio']
    X = df[features]
    y = df['label'].astype(int)
    
    # 划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    print("🧠 正在训练全能猎手 (Long & Short)...")
    clf = DecisionTreeClassifier(
        max_depth=4, 
        criterion='entropy', 
        random_state=42, 
        class_weight={0:1, 1:2, 2:2}, # 重点关注多空机会
        min_samples_leaf=20
    )
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    
    print("\n⚔️ 开启路径回测 (带止盈止损)...")
    balance = 1.0
    trade_count = 0
    wins = 0
    
    # 模拟交易参数
    TP = 0.015  # 止盈 1.5%
    SL = 0.010  # 止损 1.0% (盈亏比 1.5)
    HOLD_PERIOD = 16 # 持仓 4小时
    
    # 遍历测试集进行模拟
    # 为了速度，这里用向量化思想的简化版循环
    logs = []
    
    for i in range(len(X_test) - HOLD_PERIOD):
        signal = y_pred[i]
        if signal == 0: continue
        
        idx = X_test.index[i]
        entry_price = df.loc[idx, 'close_price']
        
        # 获取未来 N 根K线的价格路径
        future_prices = df.loc[idx:].iloc[1:HOLD_PERIOD+1]['close_price']
        
        # 结算一笔交易
        pnl = simulate_trade_path(entry_price, signal, future_prices, tp_pct=TP, sl_pct=SL)
        
        # 扣除手续费 (假设万5)
        fee = 0.0005 * 2
        net_pnl = pnl - fee
        
        balance *= (1 + net_pnl)
        trade_count += 1
        if net_pnl > 0: wins += 1
        
        logs.append(net_pnl)
        
    print("\n" + "="*40)
    print(f"📊 全能猎手战报")
    print("="*40)
    print(f"💰 最终净值: {balance:.4f}x (初始 1.0)")
    print(f"🔥 交易次数: {trade_count}")
    if trade_count > 0:
        print(f"🎯 胜率: {wins/trade_count:.2%}")
        print(f"📈 平均盈亏: {np.mean(logs)*100:.2f}%")
        
    print("\n📜 猎手准则 (1=Buy, 2=Short):")
    print(export_text(clf, feature_names=features))

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        df = feature_engineering(df)
        run_universal_bot(df)