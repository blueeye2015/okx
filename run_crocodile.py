import clickhouse_connect
import pandas as pd
import numpy as np

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'

def load_data():
    print("🚀 加载数据中...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    # 鳄鱼策略不需要盘口微观数据，只需要 OHLC
    # 我们用 1小时 级别的数据，过滤噪音
    sql = f"""
    SELECT 
        toStartOfInterval(event_time, INTERVAL 1 HOUR) as time,
        argMin(price, event_time) as open,
        max(price) as high,
        min(price) as low,
        argMax(price, event_time) as close
    FROM marketdata.trades
    WHERE symbol = 'BTC-USDT'
    GROUP BY time
    ORDER BY time ASC
    """
    df = client.query_df(sql)
    return df

def calculate_indicators(df):
    """手写计算经典指标，不依赖第三方库"""
    close = df['close']
    high = df['high']
    low = df['low']
    
    # 1. EMA 趋势线 (144) - 牛熊分界线
    df['ema_trend'] = close.ewm(span=144, adjust=False).mean()
    
    # 2. RSI (14) - 寻找回调
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. ATR (14) - 用于计算波动率止损
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    
    # 4. ADX (14) - 趋势强度 (计算稍微复杂点，简化版)
    # 这里用简单的波动率斜率代替 ADX 逻辑：如果 EMA 正在明显变陡，说明有趋势
    df['ema_slope'] = (df['ema_trend'] - df['ema_trend'].shift(5)) / df['ema_trend'].shift(5) * 100
    
    df = df.dropna()
    return df

def run_crocodile_strategy(df):
    print("🐊 鳄鱼开始潜伏 (Trend Following + Trailing Stop)...")
    
    # --- 策略参数 ---
    ATR_MULTIPLIER = 3.0   # 移动止损宽度 (3倍 ATR，很宽，不容易被洗下车)
    RSI_BUY_ZONE = 55      # 强趋势中的回调，往往跌不到 30，跌到 55 就要接了
    TREND_SLOPE = 0.05     # EMA 必须向上倾斜
    
    position = None
    trades = []
    equity = 1000 # 初始资金
    
    indices = df.index
    
    for i in range(len(indices)):
        idx = indices[i]
        curr_time = df.loc[idx, 'time']
        close = df.loc[idx, 'close']
        high = df.loc[idx, 'high']
        low = df.loc[idx, 'low']
        
        ema = df.loc[idx, 'ema_trend']
        rsi = df.loc[idx, 'rsi']
        slope = df.loc[idx, 'ema_slope']
        atr = df.loc[idx, 'atr']
        
        # --- 1. 持仓管理 (移动止损逻辑) ---
        if position:
            # 更新最高价 (用于计算移动止损)
            if high > position['highest_price']:
                position['highest_price'] = high
                # 移动止损上移：最高价 - 3*ATR
                new_sl = high - (atr * ATR_MULTIPLIER)
                # 止损只能上移，不能下移
                if new_sl > position['stop_loss']:
                    position['stop_loss'] = new_sl
            
            # 检查是否触发出场
            if low <= position['stop_loss']:
                exit_price = position['stop_loss']
                # 如果跳空低开，按开盘价止损
                if df.loc[idx, 'open'] < exit_price: exit_price = df.loc[idx, 'open']
                
                pnl = (exit_price - position['entry_price']) / position['entry_price']
                pnl_u = equity * pnl
                equity += pnl_u
                
                trades.append({
                    'Entry Time': position['entry_time'],
                    'Entry Price': round(position['entry_price'], 2),
                    'Exit Time': curr_time,
                    'Exit Price': round(exit_price, 2),
                    'Reason': '🛑 Trailing SL',
                    'PnL %': round(pnl * 100, 2),
                    'Equity': round(equity, 2)
                })
                position = None
                continue
        
        # --- 2. 开仓逻辑 (只做多) ---
        if position is None:
            # 鳄鱼法则：
            # 1. 价格在长期均线之上 (牛市)
            # 2. 均线在向上走 (趋势强)
            # 3. RSI 回调到了支撑区 (不再追高，等回调买)
            if (close > ema) and (slope > TREND_SLOPE) and (rsi < RSI_BUY_ZONE):
                # 还要加个过滤：不要在 RSI 极弱的时候买 (比如 < 30 可能崩盘)
                if rsi > 35:
                    entry_price = close
                    sl_price = close - (atr * ATR_MULTIPLIER)
                    
                    position = {
                        'entry_time': curr_time,
                        'entry_price': entry_price,
                        'stop_loss': sl_price,
                        'highest_price': close # 初始最高价
                    }

    # --- 打印战报 ---
    if len(trades) > 0:
        df_res = pd.DataFrame(trades)
        
        print("\n" + "="*60)
        print("🐊 鳄鱼捕猎战报")
        print("="*60)
        pd.set_option('display.width', 1000)
        print(df_res[['Entry Time', 'Entry Price', 'Exit Time', 'PnL %', 'Equity']])
        
        wins = df_res[df_res['PnL %'] > 0]
        losses = df_res[df_res['PnL %'] <= 0]
        
        avg_win = wins['PnL %'].mean() if len(wins) > 0 else 0
        avg_loss = losses['PnL %'].mean() if len(losses) > 0 else 0
        
        print("-" * 60)
        print(f"🔥 总交易次数: {len(trades)}")
        print(f"🎯 胜率: {len(wins)/len(trades):.2%}")
        print(f"💰 平均盈利 (吃肉): {avg_win:.2f}%")
        print(f"🩸 平均亏损 (割肉): {avg_loss:.2f}%")
        print(f"⚖️ 盈亏比: {abs(avg_win/avg_loss):.2f}")
        print(f"📈 最终资金: {equity:.2f} (初始1000)")
    else:
        print("❄️ 鳄鱼没有找到合适的机会出手 (空仓)")

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        df = calculate_indicators(df)
        run_crocodile_strategy(df)