import pandas as pd
import numpy as np
import clickhouse_connect

# ==========================================
# 1. 策略配置
# ==========================================
# 您的策略参数
RATIO_THRESHOLD = 0.5       # CVD占比阈值
CVD_THRESHOLD = 2_000_000   # 净流入阈值
TP_PCT = 0.015              # 止盈 1.5%
SL_PCT = 0.010              # 止损 1.0%

# ⚠️ 关键：真实交易成本
TAKER_FEE = 0.0005          # 手续费 0.05% (Binance Taker)
SLIPPAGE = 0.0001           # 滑点 0.01% (假设流动性充足)
COST_PER_TRADE = TAKER_FEE + SLIPPAGE # 单边成本

CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'

def load_data():
    print("🚀 正在加载数据...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    sql = f"""
    SELECT time,symbol, close_price, net_cvd, cvd_ratio
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time ASC
    """
    df = client.query_df(sql)
    return df
# ==========================================
# 2. 回测逻辑函数
# ==========================================
def run_advanced_backtest(df):
    signals = []
    
    # --- 信号生成 ---
    for i, row in df.iterrows():
        signal_type = None
        if row['cvd_ratio'] < -RATIO_THRESHOLD and row['net_cvd'] < -CVD_THRESHOLD:
            signal_type = 'Long'
        elif row['cvd_ratio'] > RATIO_THRESHOLD and row['net_cvd'] > CVD_THRESHOLD:
            signal_type = 'Short'
            
        if signal_type:
            # 计算扣费后的止盈止损位
            # Long: Entry * (1+TP), Short: Entry * (1-TP)
            signals.append({
                'entry_index': i,
                'entry_time': row['time'],
                'symbol': row['symbol'],
                'type': signal_type,
                'entry_price': row['close_price'],
                'tp': row['close_price'] * (1 + TP_PCT) if signal_type == 'Long' else row['close_price'] * (1 - TP_PCT),
                'sl': row['close_price'] * (1 - SL_PCT) if signal_type == 'Long' else row['close_price'] * (1 + SL_PCT)
            })
    
    if not signals: return pd.DataFrame()

    # --- 逐笔回测 ---
    results = []
    equity = 1.0 # 初始净值 1.0
    
    for sig in signals:
        future_data = df[(df['symbol'] == sig['symbol']) & (df.index > sig['entry_index'])]
        outcome = "Holding"
        exit_price = future_data.iloc[-1]['close_price'] if not future_data.empty else sig['entry_price']
        
        # 模拟持仓过程
        for _, fut_row in future_data.iterrows():
            curr = fut_row['close_price']
            if sig['type'] == 'Long':
                if curr >= sig['tp']:
                    outcome = "Win"; exit_price = sig['tp']; break
                if curr <= sig['sl']:
                    outcome = "Loss"; exit_price = sig['sl']; break
            else:
                if curr <= sig['tp']:
                    outcome = "Win"; exit_price = sig['tp']; break
                if curr >= sig['sl']:
                    outcome = "Loss"; exit_price = sig['sl']; break
        
        # 计算毛利 (Raw PnL)
        if sig['type'] == 'Long':
            raw_pnl = (exit_price - sig['entry_price']) / sig['entry_price']
        else:
            raw_pnl = (sig['entry_price'] - exit_price) / sig['entry_price']
            
        # ⚠️ 计算净利 (Net PnL) = 毛利 - 开仓成本 - 平仓成本
        # 成本 = 进出各一次
        net_pnl = raw_pnl - (COST_PER_TRADE * 2)
        
        sig['outcome'] = outcome
        sig['raw_pnl_pct'] = round(raw_pnl * 100, 2)
        sig['net_pnl_pct'] = round(net_pnl * 100, 2)
        results.append(sig)

    return pd.DataFrame(results)


# ==========================================
# 2. 统计报告生成
# ==========================================
def generate_report(results):
    if results.empty: return "无交易记录"
    
    # 过滤掉 'Holding' 的单子，只统计已平仓的
    closed = results[results['outcome'] != 'Holding'].copy()
    
    total_trades = len(closed)
    wins = len(closed[closed['net_pnl_pct'] > 0]) # 注意：扣费后可能由盈转亏
    losses = len(closed[closed['net_pnl_pct'] <= 0])
    win_rate = (wins / total_trades) * 100
    
    avg_win = closed[closed['net_pnl_pct'] > 0]['net_pnl_pct'].mean()
    avg_loss = closed[closed['net_pnl_pct'] <= 0]['net_pnl_pct'].mean()
    profit_factor = abs(closed[closed['net_pnl_pct'] > 0]['net_pnl_pct'].sum() / 
                        closed[closed['net_pnl_pct'] <= 0]['net_pnl_pct'].sum())

    # 资金曲线
    closed['equity_curve'] = (1 + closed['net_pnl_pct']/100).cumprod()
    final_equity = closed['equity_curve'].iloc[-1]
    total_return = (final_equity - 1) * 100
    
    # 最大回撤
    closed['peak'] = closed['equity_curve'].cummax()
    closed['drawdown'] = (closed['equity_curve'] - closed['peak']) / closed['peak']
    max_drawdown = closed['drawdown'].min() * 100
    
    print("="*40)
    print("       🚀 策略回测报告 (扣费后)       ")
    print("="*40)
    print(f"总交易数   : {total_trades}")
    print(f"真实胜率   : {win_rate:.2f}% (扣费后)")
    print(f"盈亏比     : 1:{abs(avg_win/avg_loss):.2f} (平均赢 {avg_win:.2f}% / 平均输 {avg_loss:.2f}%)")
    print(f"获利因子   : {profit_factor:.2f}")
    print("-" * 40)
    print(f"总收益率   : {total_return:.2f}%")
    print(f"最大回撤   : {max_drawdown:.2f}%")
    print("="*40)


# ==========================================
# 3. 读取数据并运行 (这里模拟读取您的文件)
# ==========================================
# 假设您已经把数据加载到了 df 中
# 如果是读取 CSV: df = pd.read_csv('your_data.csv')
df = load_data()
# (此处使用您刚刚上传的数据演示)

# 运行回测
print(f"🚀 开始回测: 阈值 Ratio>{RATIO_THRESHOLD}, CVD>{CVD_THRESHOLD/10000}万")
result_df = run_advanced_backtest(df)
generate_report(result_df)
