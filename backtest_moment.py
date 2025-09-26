# backtest_with_backtrader.py (最终健壮版 - 修复KeyError)
import backtrader as bt
import pandas as pd
import logging
import clickhouse_connect

# 导入我们的核心模块
from momentum_scanner import scan_and_rank_momentum
from trade_manager import TradeManager

# --- 全局回测配置 ---
CLICKHOUSE_CONFIG = {'host': '127.0.0.1', 'port': 8123, 'user': 'default', 'password': '12'}
BACKTEST_START_DATE = '2024-01-01'
BACKTEST_END_DATE = '2025-09-20'
INITIAL_CASH = 100000.0
REBALANCE_DAYS = 7

# --- Backtrader 策略 ---
class MomentumPortfolioStrategy(bt.Strategy):
    params = (
        ('signals_df', None),
        ('trade_manager', None),
    )

    def __init__(self):
        self.rebalance_counter = 0
        self.signals = self.p.signals_df.set_index('symbol')
        self.tm = self.p.trade_manager
        # 创建一个字典，便于通过名字快速访问数据源
        self.d_names = {d._name: d for d in self.datas}

    def next(self):
        self.rebalance_counter += 1
        if self.rebalance_counter % REBALANCE_DAYS != 0:
            return

        current_date_str = self.datetime.date(0).isoformat()
        logging.warning(f"\n--- 调仓日: {current_date_str} ---")

        current_ranks = self.signals.rename(columns={'RVol': 'rank'})
        
        # --- 再平衡：卖出不符合条件的持仓 ---
        open_symbols_before_sell = [d._name for d in self.datas if self.getposition(d).size > 0]
        for symbol in open_symbols_before_sell:
            if symbol in current_ranks.index:
                rank_info = current_ranks.loc[symbol]
                if rank_info['rank'] < 0.5:
                    logging.info(f"[{symbol}] 动量衰退 (rank: {rank_info['rank']:.2f})，准备平仓。")
                    self.close(data=self.d_names[symbol])
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


def run_backtrader():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # --- 1. 数据准备 ---
    ch_client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
    logging.info("正在获取回测所需的全量数据...")
    all_symbols_query = "SELECT DISTINCT symbol FROM marketdata.okx_klines_1d WHERE symbol LIKE '%-USDT'"
    symbols_list = [row[0] for row in ch_client.query(all_symbols_query).result_rows]
    
    ranked_signals_df = scan_and_rank_momentum(ch_client, symbols_list)
    symbols_with_signal = ranked_signals_df['symbol'].unique().tolist()
    
    all_prices_query = f"SELECT timestamp, symbol, open, high, low, close, volume FROM marketdata.okx_klines_1d WHERE symbol IN {tuple(symbols_with_signal)}"
    all_prices_df = ch_client.query_df(all_prices_query)
    all_prices_df['datetime'] = pd.to_datetime(all_prices_df['timestamp'])
    all_prices_df = all_prices_df.drop(columns=['timestamp'])

    # --- 2. 初始化Backtrader引擎 ---
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)

    # --- 3. 为每个币种添加数据到引擎中 ---
    # ###################### 核心修正点在这里 ######################
    successfully_loaded_symbols = set() # 创建一个集合来记录成功加载的币种
    # #############################################################
    
    logging.info(f"正在向Backtrader引擎添加 {len(symbols_with_signal)} 个币种的数据...")
    for symbol in symbols_with_signal:
        df = all_prices_df[all_prices_df['symbol'] == symbol]
        # 增加去重逻辑，作为双重保险
        df = df.drop_duplicates(subset=['datetime'], keep='last')
        
        if not df.empty:
            data_feed = bt.feeds.PandasData(
                dataname=df.set_index('datetime'), # 在传入时再设置索引
                datetime=None, open='open', high='high', low='low', close='close', volume='volume',
                openinterest=-1
            )
            cerebro.adddata(data_feed, name=symbol)
            successfully_loaded_symbols.add(symbol) # 记录成功

    logging.info(f"成功向引擎加载了 {len(successfully_loaded_symbols)} 个币种的数据。")

    # --- 4. 添加策略 ---
    # 【关键安全检查】只把成功加载了数据的信号传入策略
    final_signals_df = ranked_signals_df[ranked_signals_df['symbol'].isin(successfully_loaded_symbols)]
    
    tm_for_logic = TradeManager(ch_client, trading_client=None, dry_run=True)
    tm_for_logic.open_positions = {}
    cerebro.addstrategy(MomentumPortfolioStrategy, 
                        signals_df=final_signals_df, 
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

if __name__ == '__main__':
    run_backtrader()