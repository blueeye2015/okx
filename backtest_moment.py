# backtest_with_backtrader.py (最终健壮版 - 修复KeyError)
import backtrader as bt
import pandas as pd
import logging
import clickhouse_connect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入我们的核心模块
from momentum_scanner import scan_and_rank_momentum,_calculate_features_and_factor
from trade_manager import TradeManager

# --- 全局回测配置 ---
CLICKHOUSE_CONFIG = {'host': '127.0.0.1', 'port': 8123, 'user': 'default', 'password': '12'}
WARMUP_START_DATE = '2023-01-01' 
BACKTEST_START_DATE = '2024-01-01'
BACKTEST_END_DATE = '2025-09-20'
INITIAL_CASH = 100000.0
REBALANCE_DAYS = 7

# --- Backtrader 策略 ---
class MomentumPortfolioStrategy(bt.Strategy):
    params = (
        ('all_historical_data', None),
        ('all_symbols', None), # <-- 补上缺失的声明
        ('trade_manager', None),
    )

    def __init__(self):
        self.rebalance_counter = 0
        self.tm = self.p.trade_manager
        self.d_names = {d._name: d for d in self.datas}
        self.all_historical_data = self.p.all_historical_data
        self.all_symbols = self.p.all_symbols

    # --- V V V 在这里增加下面的方法 V V V ---
    def notify_trade(self, trade):
        """
        这个方法会在每次交易状态发生变化时被调用
        我们只关心交易完成（平仓）时的状态
        """
        if trade.isclosed:
            logging.info(f"--- 交易平仓 ---")
            logging.info(f"币种: {trade.data._name}")
            logging.info(f"毛利润: {trade.pnl:.2f}")
            logging.info(f"净利润: {trade.pnlcomm:.2f}") # pnlcomm = pnl - commission
            logging.info(f"-----------------")
    # --- ^ ^ ^ 增加结束 ^ ^ ^ ---

    def next(self):
        self.rebalance_counter += 1
        if self.rebalance_counter % REBALANCE_DAYS != 0:
            return

        current_date = self.datetime.date(0)
        if current_date < pd.to_datetime(BACKTEST_START_DATE).date():
            return
        logging.warning(f"\n--- 调仓日: {current_date.isoformat()} ---")
        logging.info(f"当前总资产: {self.broker.getvalue():,.2f} | 现金: {self.broker.get_cash():,.2f}")

        # 1. 截取直到“今天”为止的所有历史数据
        historical_data_today = self.all_historical_data[self.all_historical_data['timestamp'] <= pd.to_datetime(current_date)]
        
       # 2. 在“今天”这个时间点上，重新运行因子扫描
        logging.info("正在为当前调仓日动态计算因子排名...")
        current_ranks_df = self.scan_and_rank_in_memory(historical_data_today, self.all_symbols)
        if current_ranks_df.empty:
            logging.warning("当前日期无法生成有效信号，清空所有持仓。")
            for symbol in list(self.getpositions().keys()): # 获取当前持仓的拷贝
                self.close(data=self.d_names[symbol])
            return
        
        logging.info(f"成功生成 {len(current_ranks_df)} 个信号，排名前5如下:")
        top_5_signals = current_ranks_df.head(5)
        for idx, row in top_5_signals.iterrows():
            logging.info(f"  - 排名 {idx+1}: {row['symbol']} (因子: {row['RVol']:.2f}, 价格: {row['current_price']})")

        current_ranks = current_ranks_df.set_index('symbol').rename(columns={'RVol': 'rank'})
        
        # --- 再平衡：卖出不符合条件的持仓 ---
        open_symbols_before_sell = [d._name for d in self.datas if self.getpositionbyname(d._name).size]
        
        for symbol in open_symbols_before_sell:
            if symbol in current_ranks.index:
                rank_info = current_ranks.loc[symbol]
                if rank_info['rank'] < 0.5:
                    logging.info(f"[{symbol}] 动量衰退 (rank: {rank_info['rank']:.2f})，准备平仓。")
                    self.close(data=self.d_names[symbol])
                else:
                    # <--- 新增日志: 打印“决定持有”的理由 ---
                    logging.info(f"[{symbol}] 动量维持 (rank: {rank_info['rank']:.2f} >= 0.5)，继续持有。")
            else: # 如果在新排名中找不到，说明信号已消失，也应平仓
                logging.info(f"[{symbol}] 信号消失，准备平仓。")
                self.close(data=self.d_names[symbol])
        
        # --- 择优建仓：买入排名最高的币种 ---
        num_open_positions = sum(1 for pos in self.broker.positions if self.broker.getposition(pos).size)
        open_positions_symbols = {d._name for d in self.datas if self.getposition(d).size > 0}
        
        open_counts = {cat: 0 for cat in self.tm.max_positions_config}
        for pos_symbol in open_positions_symbols:
            cat = self.tm.get_market_cap_category(pos_symbol)
            if cat in open_counts: open_counts[cat] += 1
            
        if num_open_positions < self.tm.total_max_positions:
            # 按排名从高到低遍历
            for symbol, signal_info in current_ranks.sort_values('rank', ascending=False).iterrows():
                # 【关键安全检查】确保这个币种的数据真实存在于回测引擎中
                if symbol not in self.d_names:
                    continue

                current_pos_count = sum(1 for pos in self.broker.positions if self.broker.getposition(pos).size)
                if current_pos_count >= self.tm.total_max_positions:
                    break
                if symbol in open_positions_symbols: continue
                
                category = self.tm.get_market_cap_category(symbol)
                if category == 'unknown': continue

                if open_counts.get(category, 0) < self.tm.max_positions_config.get(category, 0):
                    logging.warning(f"[{symbol}] 符合建仓条件 (rank: {signal_info['rank']:.2f}, category: {category})，准备开仓。")
                    target_value = self.broker.get_value() * (1 / self.tm.total_max_positions) * 0.95
                    self.order_target_value(target=target_value, data=self.d_names[symbol])
                    open_counts[category] += 1
                    open_positions_symbols.add(symbol)
        
        logging.info(f"{'='*25} 调仓日结束 {'='*25}")

    # 将 momentum_scanner 的核心逻辑直接集成到策略内部，以便在回测的每一天调用
    def scan_and_rank_in_memory(self, all_data_df, symbols_list):
        results = []
        for symbol in symbols_list:
            group = all_data_df[all_data_df['symbol'] == symbol]
            if not group.empty:
                res = _calculate_features_and_factor(group)
                if res is not None:
                    res['symbol'] = symbol
                    results.append(res)
        if not results: return pd.DataFrame()
        results_df = pd.DataFrame(results)
        signal_df = pd.DataFrame({
            'symbol': results_df['symbol'],
            'current_price': results_df['latest_price'],
            'RVol': results_df['factor']
        })
        return signal_df.sort_values('RVol', ascending=False).reset_index(drop=True)

# 全局变量，用于在策略中访问
all_prices_df = None

def run_backtrader():
    global all_prices_df

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # --- 1. 数据准备 ---
    ch_client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
    logging.info("正在获取回测所需的全量数据...")
    all_symbols_query = f"SELECT DISTINCT symbol FROM marketdata.okx_klines_1d WHERE symbol LIKE '%-USDT' GROUP BY symbol HAVING min(timestamp) <= toDateTime('{WARMUP_START_DATE}') order by symbol"
    symbols_list = [row[0] for row in ch_client.query(all_symbols_query).result_rows]
    
    
    all_prices_query = f"SELECT timestamp, symbol, open, high, low, close, volume FROM marketdata.okx_klines_1d WHERE symbol IN {tuple(symbols_list)} AND timestamp >= toDateTime('{WARMUP_START_DATE}')"
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

    tm_for_logic = TradeManager(ch_client, trading_client=None, dry_run=True)
    tm_for_logic.open_positions = {}
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
    # --- ^ ^ ^ 增加结束 ^ ^ ^ ---


if __name__ == '__main__':
    run_backtrader()