# backtest_with_backtrader.py (最终健壮版 - 修复KeyError)
import os
import random
import numpy as np


import backtrader as bt
import pandas as pd
import logging
import clickhouse_connect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入我们的核心模块
#from momentum_scanner import scan_and_rank_momentum,_calculate_features_and_factor
from momentum_scanner_enhanced import scan_and_rank_momentum,_calculate_features_and_factor ##增强因子  
from trade_manager import TradeManager

# --- 全局回测配置 ---
CLICKHOUSE_CONFIG = {'host': '127.0.0.1', 'port': 8123, 'user': 'default', 'password': '12'}
WARMUP_START_DATE = '2023-01-01' 
BACKTEST_START_DATE = '2024-01-01'
BACKTEST_END_DATE = '2025-09-20'
INITIAL_CASH = 100000.0
REBALANCE_DAYS = 21

# --- Backtrader 策略 ---
class MomentumPortfolioStrategy(bt.Strategy):
    params = (
        ('all_historical_data', None),
        ('all_symbols', None), # <-- 补上缺失的声明
        ('trade_manager', None),
        ('stop_loss', 0.08),  # <--- 新增：移动止损的回撤比例 (8%)
        ('take_profit', 0.25),  # 止盈 25%
    )

    def __init__(self):
        self.rebalance_counter = 0
        self.tm = self.p.trade_manager
        self.d_names = {d._name: d for d in self.datas}
        self.all_historical_data = self.p.all_historical_data
        self.all_symbols = self.p.all_symbols
        self.position_peaks = {}  # <--- 新增：用于跟踪每个持仓达到的最高价
        self.sold_in_this_rebalance = set() # 用于记录在当前调仓日已卖出的币种，防止立即再买入
        self.trade_info = {} # <--- 用一个字典来跟踪每笔交易的开仓方向
        self.trades = []
   
    def notify_trade(self, trade):
        """
        在交易开始和结束时被调用，用于收集数据以生成详细报告。
        """
        # 当一笔交易刚刚开仓时
        if trade.justopened:
            direction = 'long' if trade.size > 0 else 'short'
            self.trade_info[trade.ref] = {
                'symbol': trade.data._name,
                'direction': direction,
                'open_datetime': self.datetime.datetime(0)
            }
            return

        # 当一笔交易关闭时
        if trade.isclosed:
            info = self.trade_info.pop(trade.ref, None)
            if not info: return

            # 创建一个简单的对象来存储完整的交易记录
            class FinalTrade: pass
            final_trade = FinalTrade()
            final_trade.data = trade.data
            final_trade.direction = info['direction']
            final_trade.pnlcomm = trade.pnlcomm
            final_trade.open_datetime = lambda: info['open_datetime']
            final_trade.close_datetime = lambda: self.datetime.datetime(0)
            
            self.trades.append(final_trade)

            # --- 日志记录 ---
            logging.info(f"--- 交易平仓 ---\n"
                         f"  币种: {trade.data._name}\n"
                         f"  方向: {info['direction']}\n"
                         f"  净利润: {trade.pnlcomm:.2f}\n"
                         f"-----------------")

     # --- 【新增】订单通知日志 ---
    def notify_order(self, order):
        """
        订单状态通知，用于调试。
        """
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status == order.Completed:
            order_type = 'BUY' if order.isbuy() else 'SELL'
            log_details =(
                f"--- 订单成交 ---\n"
                f"  日期: {self.datetime.date(0)}\n"
                f"  币种: {order.data._name}\n"
                f"  类型: {order_type}\n"
                f"  成交价: {order.executed.price:.4f}\n"
                f"  数量: {order.executed.size:.2f}"
            )
            # V V V 【核心修改】判断并记录平仓原因 V V V
            reason = "未知原因"
            if order.exectype == bt.Order.Stop or order.exectype == bt.Order.StopLimit:
                reason = "止损触发"
            elif order.exectype == bt.Order.Limit:
                reason = "止盈触发"
            elif order.exectype == bt.Order.Market or order.exectype == bt.Order.Close:
                reason = "调仓日主动平仓"
            log_details += f"  原因: {reason}\n"
            # ^ ^ ^ 修改结束 ^ ^ ^             
            
            log_details += f"  成交价: {order.executed.price:.4f}\n" \
                           f"  数量: {order.executed.size:.2f}"
            
            logging.info(log_details)

            if not order.isbuy():
                print(f"[DEBUG] {self.datetime.date(0)} - {order.data._name} 卖出成交，size={order.executed.size}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.warning(f"--- 订单问题 ---\n"
                            f"  币种: {order.data._name}, 状态: {order.getstatusname()}")
            
    
    def _check_daily_stops(self):

        open_positions = self.getpositions()
        
        for data, pos in list(open_positions.items()):
            symbol = data._name
            if pos.size == 0: continue
            print(f"[DEBUG] {self.datetime.date(0)} - {symbol} 持仓 size={pos.size}, price={pos.price}")

            current_price = data.close[0]
            # V V V 【核心修正】简化最高价更新逻辑 V V V
            # 直接更新最高价
            self.position_peaks[symbol] = max(self.position_peaks[symbol], current_price)
            # ^ ^ ^ 修正结束 ^ ^ ^

            peak_price = self.position_peaks[symbol]
            stop_price = peak_price * (1.0 - self.p.trail_percent)

            if current_price < stop_price:
                logging.warning(f"[{symbol}] 触发移动止损！... 准备平仓并取消所有挂单。")
                
                # V V V 【核心增强】在 close 之前，取消所有挂单 V V V
                pending_orders = [o for o in self.broker.get_orders_pending() if o.data == data]
                for o in pending_orders:
                    logging.warning(f"  -> 取消 [{symbol}] 的旧挂单, ref: {o.ref}")
                    self.cancel(o)
                # ^ ^ ^ 增强结束 ^ ^ ^

                self.close(data=data)
                self.sold_in_this_rebalance.add(symbol)
            else:
                logging.info(f"[{symbol}] 持仓正常。...")

    
    def next(self):
        # 每日开始时清空当日卖出记录
        self.sold_in_this_rebalance.clear()
        
        current_date = self.datetime.date(0)
        if current_date < pd.to_datetime(BACKTEST_START_DATE).date():
            return
        
        # 只在调仓日执行再平衡逻辑
        self.rebalance_counter += 1
        if self.rebalance_counter % REBALANCE_DAYS != 0:
            return

        # --- 调仓日逻辑开始 ---
        logging.warning(f"\n--- 调仓日: {current_date.isoformat()} ---")
        
        # ... (数据准备和因子计算部分，这部分逻辑是正确的，保持不变) ...
        historical_data_today = self.all_historical_data[self.all_historical_data['timestamp'] <= pd.to_datetime(current_date)]
        current_ranks_df = self.scan_and_rank_in_memory(historical_data_today, self.all_symbols)
        if current_ranks_df.empty:
            # ... (代码不变) ...
            return
        current_ranks = current_ranks_df.set_index('symbol').rename(columns={'RVol': 'rank'})
        
        # --- 再平衡：卖出不符合条件的持仓 ---
        open_positions = self.getpositions()
        for data, pos in list(open_positions.items()):
            symbol = data._name
            if pos.size == 0: continue
            
            should_sell = False
            # 如果在新排名中找不到，或者排名过低，则准备卖出
            if symbol not in current_ranks.index or current_ranks.loc[symbol, 'rank'] < 0.65:
                should_sell = True

            if should_sell:
                logging.info(f"[{symbol}] 动量/信号不符，准备平仓并取消所有关联挂单。")
                
                # 【核心健壮性修复】在手动close之前，取消所有由buy_bracket创建的挂单
                pending_orders = [o for o in self.broker.get_orders_open() if o.data == data]
                for o in pending_orders:
                    logging.warning(f"  -> 取消 [{symbol}] 的旧挂单 (ref: {o.ref})")
                    self.cancel(o)

                self.close(data=data)
                self.sold_in_this_rebalance.add(symbol)
            else:
                logging.info(f"[{symbol}] 动量维持，继续持有。")
        
        # --- 择优建仓：买入排名最高的币种 ---
        open_positions_symbols = {d._name for d in self.datas if self.getposition(d).size > 0}
        open_counts = {cat: 0 for cat in self.tm.max_positions_config}
        for pos_symbol in open_positions_symbols:
            cat = self.tm.get_market_cap_category(pos_symbol)
            if cat in open_counts: open_counts[cat] += 1
        
        for symbol, signal_info in current_ranks.sort_values('rank', ascending=False).iterrows():
            current_pos_count = len([p for p in self.getpositions() if self.getposition(p).size > 0])
            if current_pos_count >= self.tm.total_max_positions: break
            if symbol not in self.d_names: continue
            if symbol in open_positions_symbols: continue
            if symbol in self.sold_in_this_rebalance: continue # 如果当日刚卖出，则不再买入
            
            category = self.tm.get_market_cap_category(symbol)
            if category == 'unknown': continue
            
            if open_counts.get(category, 0) < self.tm.max_positions_config.get(category, 0):
                logging.warning(f"[{symbol}] 符合建仓条件 (rank: {signal_info['rank']:.2f}, category: {category})，准备开仓。")
                
                data = self.d_names[symbol]
                current_price = data.close[0]
                if current_price <= 0: continue

                # 计算下单数量和止盈止损价格
                target_value = self.broker.get_value() * (1 / self.tm.total_max_positions) * 0.95
                size = target_value / current_price
                stop_price = current_price * (1.0 - self.p.stop_loss)
                limit_price = current_price * (1.0 + self.p.take_profit)
                
                # 执行括号订单
                self.buy_bracket(
                    data=data,
                    size=size,
                    exectype=bt.Order.Close,
                    stopprice=stop_price,
                    limitprice=limit_price
                )
                
                open_counts[category] += 1
                open_positions_symbols.add(symbol)
        
        logging.info(f"{'='*25} 调仓日结束 {'='*25}")


    # 将 momentum_scanner 的核心逻辑直接集成到策略内部，以便在回测的每一天调用
    def scan_and_rank_in_memory(self, all_data_df, symbols_list):
        results = []
        for symbol in symbols_list:
            group = all_data_df[all_data_df['symbol'] == symbol]
            if not group.empty:
                
                # --- V V V 终极手段：在每次调用前，强制重置随机种子 V V V ---
                np.random.seed(42)
                random.seed(42)
                # --- ^ ^ ^ 修改结束 ^ ^ ^ ---
                res = _calculate_features_and_factor(group)
                if res is not None:
                    res['symbol'] = symbol
                    results.append(res)
        if not results: return pd.DataFrame()
        results_df = pd.DataFrame(results)
        signal_df = pd.DataFrame({
            'symbol': results_df['symbol'],
            'current_price': results_df['latest_price'],
            'rank': results_df['D_MOM']
        })
        return signal_df.sort_values('rank', ascending=False).reset_index(drop=True)
    


def run_backtrader():
    global all_prices_df

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # --- 1. 数据准备 ---
    ch_client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
    logging.info("正在获取回测所需的全量数据...")
    all_symbols_query = f"SELECT DISTINCT symbol FROM marketdata.okx_klines_1d WHERE symbol LIKE '%-USDT' GROUP BY symbol HAVING min(timestamp) <= toDateTime('{WARMUP_START_DATE}') order by symbol"
    symbols_list = [row[0] for row in ch_client.query(all_symbols_query).result_rows]
    
    
    # --------- BTC 作为市场收益 ---------
    all_prices_query = f"""
    WITH btc_ret AS (
        SELECT
            timestamp,
            (close - lag(close) OVER (ORDER BY timestamp)) / lag(close) OVER (ORDER BY timestamp) AS btc_ret
        FROM marketdata.okx_klines_1d
        WHERE symbol = 'BTC-USDT'
    )
    SELECT
        k.timestamp,
        k.symbol,
        avg(k.open)  AS open,
        max(k.high)  AS high,
        min(k.low)   AS low,
        avg(k.close) AS close,
        sum(k.volume) AS volume,
        b.btc_ret     AS mkt_ret
    FROM marketdata.okx_klines_1d AS k
    LEFT JOIN btc_ret AS b
        ON k.timestamp = b.timestamp
    WHERE k.symbol IN {tuple(symbols_list)}
    AND k.timestamp >= toDateTime('{WARMUP_START_DATE}')
    GROUP BY k.timestamp, k.symbol, b.btc_ret
    ORDER BY k.timestamp, k.symbol
    """

    all_prices_df = ch_client.query_df(all_prices_query)
    all_prices_df['timestamp'] = pd.to_datetime(all_prices_df['timestamp'])
    # --- 核心修正点在这里 ---
    # 策略内部用于计算的 all_historical_data 也必须与 backtrader 的“天真”时间系统兼容。
    # 在传递给策略之前，统一将它的时区信息剥离。
    if all_prices_df['timestamp'].dt.tz is not None:
        logging.info("检测到 'timestamp' 列包含时区信息，正在将其转换为时区天真(naive)类型...")
        all_prices_df['timestamp'] = all_prices_df['timestamp'].dt.tz_localize(None)


    # --- 2. 初始化Backtrader引擎 ---
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.set_coc(True) # 设置以当日收盘价成交
    cerebro.broker.set_shortcash(False) #设置禁止做空

  
    logging.info(f"正在向Backtrader引擎添加 {len(symbols_list)} 个币种的数据...")
    date_index = pd.date_range(start=WARMUP_START_DATE, end=BACKTEST_END_DATE, freq='D')
    for symbol in symbols_list:
        df = all_prices_df[all_prices_df['symbol'] == symbol].set_index('timestamp')
        if df.empty: continue
         #    这样 '2024-02-01 00:00:00+08:00' 就变成了 '2024-02-01 00:00:00'
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # 2. 标准化到午夜，确保能和 date_index 的每日零点对齐
        df.index = df.index.normalize()
        
        df = df[~df.index.duplicated(keep='last')]
        df_aligned = df.reindex(date_index)
        df_aligned[['open', 'high', 'low', 'close']] = df_aligned[['open', 'high', 'low', 'close']].ffill()
        df_aligned['open'].fillna(df_aligned['close'], inplace=True)
        df_aligned['high'].fillna(df_aligned['close'], inplace=True)
        df_aligned['low'].fillna(df_aligned['close'], inplace=True)
        df_aligned['volume'].fillna(0, inplace=True)
        df_aligned.dropna(subset=['close'], inplace=True)
        
        if not df_aligned.empty:
            # --- 新增的诊断代码 ---
            first_valid_date = df_aligned.index.min().date()
            last_valid_date = df_aligned.index.max().date()
            
            # 我们只打印几个关键币种的信息，避免刷屏
            if symbol in ['BTC-USDT', 'ETH-USDT'] or first_valid_date > pd.to_datetime('2024-01-01').date():
                 print(f"DEBUG: 为 {symbol} 添加数据 | "
                       f"数据起始: {first_valid_date} | "
                       f"数据结束: {last_valid_date} | "
                       f"数据行数: {len(df_aligned)}")
            # --- 诊断代码结束 ---
            data_feed = bt.feeds.PandasData(
                dataname=df_aligned, # 在传入时再设置索引
                datetime=None, open='open', high='high', low='low', close='close', volume='volume',
                openinterest=-1
            )
            cerebro.adddata(data_feed, name=symbol)
        else:
             # <--- 如果 df_aligned 变为空，也打印出来
             print(f"DEBUG: 警告！{symbol} 在处理后数据为空，被跳过。")

    tm_for_logic = TradeManager(ch_client, trading_client=None, dry_run=True, backtest_mode=True)
    #tm_for_logic.open_positions = {}
    cerebro.addstrategy(MomentumPortfolioStrategy, 
                        all_historical_data=all_prices_df,
                        all_symbols=symbols_list, # <-- 不再传递一次性算好的信号
                        trade_manager=tm_for_logic)

    # ... (分析器和报告部分与上一版相同，无需修改)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days, compression=1, factor=365)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    logging.warning("--- Backtrader 回测开始 ---")
    results = cerebro.run()
    logging.warning("--- Backtrader 回测结束 ---")
    if not results:
        logging.error("回测未能运行任何策略，请检查数据和时间范围。")
        return
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    print("\n" + "="*50)
    print("                BACKTRADER PERFORMANCE REPORT")
    print("="*50)
    print(f"初始资金: {INITIAL_CASH:,.2f}")
    print(f"最终资金: {final_value:,.2f}")
    print(f"总回报率: {(final_value / INITIAL_CASH - 1):.2%}")
    print("-" * 50)
    print("关键绩效指标:")
    analysis = strat.analyzers
   # ###################### 核心修正点在这里 ######################
    sharpe = analysis.sharpe_ratio.get_analysis().get('sharperatio')
    drawdown_analysis = analysis.drawdown.get_analysis()
    drawdown = drawdown_analysis.max.drawdown if drawdown_analysis and drawdown_analysis.max else 0.0
    returns_analysis = analysis.returns.get_analysis()
    rann = returns_analysis.get('rann')
    
    print(f"  - 夏普比率 (年化): {sharpe:.3f}" if isinstance(sharpe, (int, float)) else "  - 夏普比率 (年化): N/A")
    print(f"  - 最大回撤: {drawdown:.2%}") # drawdown 永远是数值
    print(f"  - 年化回报率: {rann:.2%}" if isinstance(rann, (int, float)) else "  - 年化回报率: N/A")
    
    trade_analysis = analysis.trade_analyzer.get_analysis()
    
    # 安全地获取交易统计信息
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    won_trades = trade_analysis.get('won', {}).get('total', 0)
    pnl_net_average = trade_analysis.get('pnl', {}).get('net', {}).get('average', 0)

    if total_trades > 0:
        print("-" * 50)
        print("交易统计:")
        print(f"  - 总交易次数: {total_trades}")
        print(f"  - 胜率: {won_trades / total_trades:.2%}")
        print(f"  - 平均盈利/亏损: {pnl_net_average:.2f}")
    else:
        print("-" * 50)
        print("交易统计: 回测期间没有已平仓的交易。")
    # #############################################################
    print("="*50)

    # --- V V V 在这里增加下面的代码 V V V ---
    # --- 增加图表绘制和最终持仓分析 ---
    logging.info("\n--- 最终持仓分析 ---")
    open_positions = strat.getpositions()
    if open_positions:
        # backtrader 的 positions 是一个字典，键是 data feed，值是 Position 对象
        # 我们需要通过 data feed 的 _name 属性来获取币种名称
        for data, pos in open_positions.items():
            symbol = data._name
            unrealized_pnl = (data.close[0] - pos.price) * pos.size
            logging.info(f"持仓: {symbol}, "
                         f"数量: {pos.size:.2f}, "
                         f"开仓均价: {pos.price:.4f}, "
                         f"当前价: {data.close[0]:.4f}, "
                         f"未实现盈亏: {unrealized_pnl:.2f}")
    else:
        logging.info("回测结束时无持仓。")

    logging.warning("正在生成回测图表, 请在运行目录下查找 backtest_plot.html 文件...")
    # 使用 iplot=False 和 savefig=True 来确保在任何环境下都能生成文件
    cerebro.plot(style='candlestick', iplot=False, savefig=True, figfilename='backtest_plot.html')
    # ---------------  细粒度币种分析 ---------------
    all_trades = strat.trades  # Backtrader 策略对象里已收集全部 trade
    df_sym = analyze_per_symbol(all_trades)

    print('\n============  各币种交易明细  ============')
    print(df_sym.to_string())

    # 保存 CSV 方便 Excel 打开
    df_sym.to_csv('symbol_trade_stats.csv')

    # 画图
    plot_pnl_charts(df_sym)
# ---------------------------------------------
    # --- ^ ^ ^ 增加结束 ^ ^ ^ ---

# ======================  币种级细粒度分析  ======================
# ======================  币种级细粒度分析  ======================
def analyze_per_symbol(trades):
    records = []
    sym_trades = {}
    for t in trades:
        sym = t.data._name
        sym_trades.setdefault(sym, []).append(t)

    for sym, tlist in sym_trades.items():
        # V V V 【注意】这里不再需要 closed 列表，因为传入的 trades 已经是平仓交易了 V V V
        # closed = [t for t in tlist if t.isclosed]
        # if not closed:
        #     continue

        buy_num = sell_num = 0
        # V V V 使用新的 .direction 属性来判断 V V V
        for t in tlist: # 直接遍历 tlist
            if t.direction == 'long':
                buy_num += 1
            elif t.direction == 'short':
                sell_num += 1
        # ^ ^ ^ 修正结束 ^ ^ ^
        pnls = [t.pnlcomm for t in tlist]
        total_pnl = sum(pnls)
        avg_pnl   = total_pnl / len(pnls) if pnls else 0
        max_win   = max(pnls) if pnls else 0
        max_loss  = min(pnls) if pnls else 0
        days = [(t.close_datetime() - t.open_datetime()).days for t in tlist]
        avg_days = np.mean(days) if days else 0
        max_days = max(days) if days else 0
        min_days = min(days) if days else 0

        records.append({
            'symbol': sym,
            'buy_times': buy_num,
            'sell_times': sell_num,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_win': max_win,
            'max_loss': max_loss,
            'avg_hold_days': avg_days,
            'max_hold_days': max_days,
            'min_hold_days': min_days,
        })
    return pd.DataFrame(records).set_index('symbol')


def plot_pnl_charts(df):
    """画两张图并保存"""
    plt.rcParams['figure.figsize'] = (14, 5)
    fig, ax = plt.subplots(1, 2)

    # 图1：各币种总盈亏
    colors = ['g' if v >= 0 else 'r' for v in df['total_pnl']]
    df['total_pnl'].sort_values(ascending=False).plot.bar(ax=ax[0], color=colors)
    ax[0].set_title('Total PnL by Symbol')
    ax[0].set_ylabel('PnL (USD)')

    # 图2：盈亏分布
    ax[1].hist(df['total_pnl'], bins=max(10, len(df)//2), color='skyblue', edgecolor='black')
    ax[1].set_title('PnL Distribution')
    ax[1].set_xlabel('Total PnL')
    ax[1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('symbol_pnl_analysis.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    run_backtrader()