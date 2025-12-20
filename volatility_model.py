import clickhouse_connect
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
# 注意：我们需要 High/Low 数据，所以还是要连 trades 表
SYMBOL = 'BTC-USDT' 
SYMBOL_F = 'BTCUSDT'

def load_data():
    print("🚀 正在构建数据 (支持跨K线持仓)...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    sql = f"""
    WITH 
    OHLC AS (
        SELECT 
            toStartOfInterval(event_time, INTERVAL 15 MINUTE) as time,
            argMin(price, event_time) as open,
            max(price) as high,
            min(price) as low,
            argMax(price, event_time) as close
        FROM marketdata.trades
        WHERE symbol = '{SYMBOL}'
        GROUP BY time
    ),
    Feat AS (
        SELECT * FROM marketdata.features_15m WHERE symbol = '{SYMBOL_F}'
    )
    SELECT 
        O.time, O.open, O.high, O.low, O.close,
        F.wall_shift_pct, F.net_cvd, F.spoofing_ratio
    FROM OHLC AS O
    INNER JOIN Feat AS F ON O.time = F.time
    ORDER BY O.time ASC
    """
    df = client.query_df(sql)
    return df

def feature_engineering(df):
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 1. 波动率特征
    df['amplitude'] = (df['high'] - df['low']) / df['open'] * 100
    df['prev_amp'] = df['amplitude'].shift(1)
    df['wall_volatility'] = df['wall_shift_pct'].rolling(4).std()
    df['cvd_abs'] = df['net_cvd'].abs()
    
    # 2. 目标：预测当前 K 线是否适合"接针" (波动大)
    df['label'] = (df['amplitude'] > 0.6).astype(int) # 只要振幅够大，就有机会接到
    
    df = df.dropna()
    return df

def run_extended_backtest(df):
    features = ['prev_amp', 'wall_volatility', 'cvd_abs', 'spoofing_ratio']
    X = df[features]
    y = df['label'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    print("🧠 训练弹性网格猎手...")
    clf = DecisionTreeClassifier(
        max_depth=3, 
        criterion='entropy', 
        random_state=42, 
        class_weight={0:1, 1:2},
        min_samples_leaf=20
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("\n⚔️ 开启跨K线狩猎 (Buy Dip, Sell Rally)...")
    
    # --- 策略参数 ---
    BUY_DROP = 0.006  # 下跌 0.3% 挂买单接针
    SELL_RISE = 0.006 # 反弹 0.6% 止盈 (相对于买入价)
    STOP_LOSS = 0.010 # 止损 1.5% (防止单边暴跌)
    MAX_HOLD = 16     # 最多拿 4小时 (16根K线)
    
    # --- 状态机变量 ---
    position = None # 存储字典: {'price': 90000, 'entry_time': idx, 'stop_loss': 88000, 'take_profit': 90600}
    
    trade_count = 0
    wins = 0
    losses = 0
    timeout_exits = 0
    
    logs = []
    
    # 遍历测试集
    # 注意：我们需要按照时间顺序逐根扫描
    indices = X_test.index
    
    for i in range(len(indices)):
        idx = indices[i]
        curr_open = df.loc[idx, 'open']
        curr_high = df.loc[idx, 'high']
        curr_low = df.loc[idx, 'low']
        curr_close = df.loc[idx, 'close']
        
        # 1. 如果有持仓，检查是否止盈/止损/超时
        if position is not None:
            entry_price = position['price']
            tp_price = position['take_profit']
            sl_price = position['stop_loss']
            bars_held = i - position['entry_idx']
            
            # A. 检查是否止损 (最高优先级)
            if curr_low <= sl_price:
                pnl = (sl_price - entry_price) / entry_price
                losses += 1
                logs.append(pnl)
                position = None # 平仓
                continue # 这一根K线处理完了
                
            # B. 检查是否止盈
            if curr_high >= tp_price:
                pnl = (tp_price - entry_price) / entry_price
                wins += 1
                logs.append(pnl)
                position = None # 平仓
                continue
            
            # C. 检查是否超时
            if bars_held >= MAX_HOLD:
                # 强平
                pnl = (curr_close - entry_price) / entry_price
                timeout_exits += 1
                if pnl > 0: wins += 1
                else: losses += 1
                logs.append(pnl)
                position = None
                continue
                
        # 2. 如果空仓，检查是否开单
        if position is None:
            # 只有 AI 预测波动大，且这一根K线确实跌下来了，才能接到
            if y_pred[i] == 1:
                limit_buy_price = curr_open * (1 - BUY_DROP)
                
                # 检查这一根 K 线是否触及买单
                if curr_low <= limit_buy_price:
                    # 成交！
                    trade_count += 1
                    position = {
                        'price': limit_buy_price,
                        'entry_idx': i,
                        'take_profit': limit_buy_price * (1 + SELL_RISE),
                        'stop_loss': limit_buy_price * (1 - STOP_LOSS)
                    }
                    # 注意：如果当根K线波动极大，可能直接止盈或止损，这里简化处理，下一根K线结算
    
    # --- 统计结果 ---
    total_pnl = sum(logs) - (trade_count * 0.0005 * 2) # 扣手续费
    
    print("\n" + "="*40)
    print(f"🕸️ 弹性网格战报")
    print("="*40)
    print(f"🔥 开仓次数: {trade_count}")
    if trade_count > 0:
        win_rate = wins / (wins + losses + timeout_exits)
        print(f"🎯 胜率: {win_rate:.2%} (止盈+超时盈利)")
        print(f"✅ 止盈次数: {wins}")
        print(f"❌ 止损次数: {losses}")
        print(f"⌛ 超时平仓: {timeout_exits}")
        print(f"💰 累计净回报: {total_pnl*100:.2f}%")
        print(f"📈 平均单笔: {np.mean(logs)*100:.2f}%")
    else:
        print("❄️ 没有开单")

    print("\n📜 猎手直觉 (Tree Rules):")
    print(export_text(clf, feature_names=features))

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        df = feature_engineering(df)
        run_extended_backtest(df)