import pandas as pd
import numpy as np

def market_maker_backtest():
    # 1. 数据加载
    df = pd.read_csv('eth_7.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # 2. 因子（只做波动率过滤）
    df['atr'] = df['high'].rolling(5, min_periods=3).max() - df['low'].rolling(5, min_periods=3).min()
    df['atr_pct'] = df['atr'] / df['close']
    
    # 低波动期：市场稳定，适合做市
    # 高波动期：风险大，不参与
    df['volatility_quiet'] = df['atr_pct'] < df['atr_pct'].rolling(720, min_periods=200).quantile(0.25)
    
    # 3. 事件定义（无方向，只看市场状态）
    # 事件：低波动 + 成交量平稳（避免异常波动）
    df['volume_ma'] = df['volume'].rolling(30, min_periods=15).mean()
    df['volume_stable'] = (df['volume'] < df['volume_ma'] * 2)  # 成交量不超过均值2倍
    
    events = df[df['volatility_quiet'] & df['volume_stable']].copy()
    
    # 4. 回测参数（纯做市）
    MAKER_FEE = 0.0005  # 挂单手续费（支付）
    MAKER_REBATE = 0.0006  # ✅ 做市商返还（赚取）
    NET_REBATE = MAKER_REBATE - MAKER_FEE  # 净收益0.01%
    
    PROFIT_TARGET = 0.0006  # 0.06%的微利
    STOP_LOSS = 0.0012  # 0.12%止损
    HOLD_TIME = 10  # 10分钟
    
    print('\n========== 做市策略配置 ==========')
    print(f'净手续费返还: {NET_REBATE*100:.3f}%')
    print(f'盈利目标: {PROFIT_TARGET*100:.2f}% | 止损: {STOP_LOSS*100:.2f}%')
    print(f'持仓时间: {HOLD_TIME}分钟')
    print(f'事件数: {len(events)}（低波动期）')
    
    if len(events) == 0:
        print('❌ 无低波动期')
        return
    
    # 5. 回测引擎（双向挂单，无方向预测）
    pnls = []
    trade_logs = []
    
    for idx, row in events.iterrows():
        try:
            # ✅ 核心：同时在买卖盘挂单（做市商逻辑）
            bid_price = row['close'] * 0.9998  # 买价：市价下方0.02%
            ask_price = row['close'] * 1.0002  # 卖价：市价上方0.02%
            
            # 观察未来
            future = df.loc[idx:idx + pd.Timedelta(minutes=HOLD_TIME)]
            if len(future) < HOLD_TIME:
                continue
            
            traded = False
            
            # 检查是否成交（先成交哪边就平哪边）
            for t in range(1, len(future)):
                high_price, low_price, close_price = future.iloc[t][['high', 'low', 'close']]
                
                # 卖单成交（先成为空头）
                if high_price >= ask_price:
                    # 寻找平仓机会（买回）
                    for e in range(t, min(t + HOLD_TIME, len(future))):
                        exit_price = future.iloc[e]['close']
                        pnl = (ask_price - exit_price) / ask_price - MAKER_FEE + MAKER_REBATE  # 空单盈利
                        
                        # 盈利目标或止损
                        if pnl >= PROFIT_TARGET:
                            pnls.append(pnl)
                            trade_logs.append({'time': idx, 'side': '卖挂单', 'pnl': pnl})
                            traded = True
                            break
                        elif pnl <= -STOP_LOSS:
                            pnls.append(pnl)
                            trade_logs.append({'time': idx, 'side': '卖止损', 'pnl': pnl})
                            traded = True
                            break
                    
                    if traded:
                        break
                
                # 买单成交（先成为多头）
                elif low_price <= bid_price:
                    # 寻找平仓机会（卖出）
                    for e in range(t, min(t + HOLD_TIME, len(future))):
                        exit_price = future.iloc[e]['close']
                        pnl = (exit_price - bid_price) / bid_price - MAKER_FEE + MAKER_REBATE  # 多单盈利
                        
                        if pnl >= PROFIT_TARGET:
                            pnls.append(pnl)
                            trade_logs.append({'time': idx, 'side': '买挂单', 'pnl': pnl})
                            traded = True
                            break
                        elif pnl <= -STOP_LOSS:
                            pnls.append(pnl)
                            trade_logs.append({'time': idx, 'side': '买止损', 'pnl': pnl})
                            traded = True
                            break
                    
                    if traded:
                        break
            
        except Exception as e:
            continue
    
    # 6. 绩效评估
    if pnls:
        pnls_array = np.array(pnls)
        
        # 剔除极端值
        q99, q01 = np.percentile(pnls_array, [99, 1])
        pnls_clean = pnls_array[(pnls_array >= q01) & (pnls_array <= q99)]
        
        if len(pnls_clean) < 3:
            pnls_clean = pnls_array
        
        avg_pnl = np.mean(pnls_clean)
        win_rate = np.mean(pnls_clean > 0)
        trades_per_day = len(pnls_clean) / (len(df) / 1440)
        sharpe = avg_pnl / (np.std(pnls_clean) + 1e-8) * np.sqrt(trades_per_day * 365)
        
        print(f'\n========== 做市结果 ==========')
        print(f'总信号数: {len(events)}')
        print(f'实际成交: {len(pnls_clean)}')
        print(f'日均交易: {trades_per_day:.1f} 笔')
        print(f'平均净收益: {avg_pnl*100:.4f}%')
        print(f'胜率: {win_rate*100:.1f}%')
        print(f'年化收益: {avg_pnl*100*trades_per_day*365:.2f}%')
        print(f'夏普比率: {sharpe:.2f}')
        print(f'税后达标: {"✅" if avg_pnl > 0.001 else "❌"}')
        
        # 交易分布
        if trade_logs:
            log_df = pd.DataFrame(trade_logs)
            print(f'\n========== 交易分布 ==========')
            print(log_df['side'].value_counts())
            
            print(f"\n📈 盈利交易：平均{log_df[log_df['pnl']>0]['pnl'].mean()*100:.3f}%")
            print(f"📉 亏损交易：平均{log_df[log_df['pnl']<0]['pnl'].mean()*100:.3f}%")
        
        # 最简单的建议
        if avg_pnl < 0:
            print('\n⚠️  做市策略也亏损，说明：')
            print('   1. 你的数据时间段市场波动过大（不适合做市）')
            print('   2. 或者手续费返还设置不对（检查交易所政策）')
            print('   3. 建议换标的（BTC）或换周期（5分钟K线）')
            
    else:
        print('❌ 无成交，市场波动太大，无法做市')

if __name__ == '__main__':
    market_maker_backtest()