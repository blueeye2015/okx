import clickhouse_connect
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# --- Configuration ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTC-USDT'   # For trades table
SYMBOL_F = 'BTCUSDT'  # For features table

def load_data():
    print("🚀 Building data (High/Low required for dip catching)...")
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
    
    # 1. Volatility Features
    df['amplitude'] = (df['high'] - df['low']) / df['open'] * 100
    df['prev_amp'] = df['amplitude'].shift(1)
    df['wall_volatility'] = df['wall_shift_pct'].rolling(4).std()
    df['cvd_abs'] = df['net_cvd'].abs()
    
    # 2. Target: Predict if current candle is "catchable" (high volatility)
    df['label'] = (df['amplitude'] > 0.6).astype(int) 
    
    df = df.dropna()
    return df

def run_extended_backtest(df):
    features = ['prev_amp', 'wall_volatility', 'cvd_abs', 'spoofing_ratio']
    X = df[features]
    y = df['label'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    print("🧠 Training Elastic Grid Hunter...")
    clf = DecisionTreeClassifier(
        max_depth=3, 
        criterion='entropy', 
        random_state=42, 
        class_weight={0:1, 1:2},
        min_samples_leaf=20
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("\n⚔️ Hunting Season Open (Buy Dip, Sell Rally)...")
    
    # --- Strategy Params ---
    BUY_DROP = 0.006  # Buy limit at -0.6% from Open
    SELL_RISE = 0.006 # TP at +0.6% from Entry
    STOP_LOSS = 0.015 # SL at -1.5% from Entry
    MAX_HOLD = 16     # Max hold 4 hours
    
    # --- State Machine ---
    position = None 
    trade_records = [] # Store details of every trade
    
    # Iterate through Test Set
    indices = X_test.index
    
    for i in range(len(indices)):
        idx = indices[i]
        curr_time = df.loc[idx, 'time']
        curr_open = df.loc[idx, 'open']
        curr_high = df.loc[idx, 'high']
        curr_low = df.loc[idx, 'low']
        curr_close = df.loc[idx, 'close']
        
        # 1. Manage Existing Position
        if position is not None:
            entry_price = position['price']
            entry_time = position['entry_time']
            tp_price = position['take_profit']
            sl_price = position['stop_loss']
            bars_held = i - position['entry_idx']
            
            exit_reason = None
            exit_price = 0.0
            pnl_pct = 0.0
            
            # A. Check Stop Loss (Priority 1)
            if curr_low <= sl_price:
                exit_price = sl_price
                exit_reason = "❌ Stop Loss"
            
            # B. Check Take Profit (Priority 2)
            elif curr_high >= tp_price:
                exit_price = tp_price
                exit_reason = "✅ Take Profit"
            
            # C. Check Timeout (Priority 3)
            elif bars_held >= MAX_HOLD:
                exit_price = curr_close
                if exit_price > entry_price:
                    exit_reason = "⌛ Timeout (Win)"
                else:
                    exit_reason = "⌛ Timeout (Loss)"
            
            # Process Exit
            if exit_reason:
                pnl_pct = (exit_price - entry_price) / entry_price
                
                trade_records.append({
                    'Entry Time': entry_time,
                    'Entry Price': round(entry_price, 2),
                    'Exit Time': curr_time,
                    'Exit Price': round(exit_price, 2),
                    'Type': exit_reason,
                    'Hold Bars': bars_held,
                    'Return %': round(pnl_pct * 100, 2),
                    'PnL ($1000)': round(1000 * pnl_pct, 2)
                })
                
                position = None # Close position
                continue # Done for this candle
                
        # 2. Open New Position
        if position is None:
            # AI predicts high volatility -> Place Limit Buy
            if y_pred[i] == 1:
                limit_buy_price = curr_open * (1 - BUY_DROP)
                
                # Check if price dropped enough to fill the order
                if curr_low <= limit_buy_price:
                    # Filled!
                    position = {
                        'price': limit_buy_price,
                        'entry_time': curr_time,
                        'entry_idx': i,
                        'take_profit': limit_buy_price * (1 + SELL_RISE),
                        'stop_loss': limit_buy_price * (1 - STOP_LOSS)
                    }
                    # Simplified: We assume fill happens, exit logic handled next bar
    
    # --- Reporting ---
    print("\n" + "="*80)
    print(f"🕸️ ELASTIC GRID DETAILED LEDGER")
    print("="*80)
    
    if len(trade_records) > 0:
        df_trades = pd.DataFrame(trade_records)
        df_trades['Cum PnL'] = df_trades['PnL ($1000)'].cumsum()
        
        # Display settings
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.expand_frame_repr', False)
        
        # Print the ledger
        print(df_trades[['Entry Time', 'Entry Price', 'Exit Time', 'Type', 'Return %', 'PnL ($1000)', 'Cum PnL']])
        
        # Statistics
        wins = len(df_trades[df_trades['Return %'] > 0])
        total_pnl = df_trades['PnL ($1000)'].sum()
        
        print("-" * 80)
        print(f"🔥 Total Trades: {len(df_trades)}")
        print(f"🎯 Win Rate: {wins / len(df_trades):.2%}")
        print(f"💰 Total PnL (per $1000): ${total_pnl:.2f}")
        
    else:
        print("❄️ No trades triggered.")

    print("\n📜 Hunter's Intuition (Tree Rules):")
    print(export_text(clf, feature_names=features))

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        df = feature_engineering(df)
        run_extended_backtest(df)