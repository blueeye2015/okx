import backtrader as bt
import pandas as pd
import os
import numpy as np
import psycopg2 
import logging
import glob   
from datetime import datetime
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns         

load_dotenv('.env')
POSTGRES_CONFIG  = os.getenv("DB_DSN1")

BENCHMARK_SYMBOL = '000300.SH'
ADJUST_TYPE = 'hfq'
CACHE_DIR = 'factor_cache'
INITIAL_CASH = 1000000.0
# --- 1. 自定义数据加载器 (不变) ---
class PandasDataWithFactor(bt.feeds.PandasData):
    lines = ('factor',)
    params = (('factor', -1),)

# --- 2. 核心交易策略 (已更新) ---
class MLFactorStrategy(bt.Strategy):
    params = dict(
        top_n_pct=0.1,
        rebalance_monthday=20,
        benchmark_symbol=BENCHMARK_SYMBOL,
        # --- 新增：止盈止损参数 ---
        stop_loss_pct=0.08,      # 8% 止损
        take_profit_pct=0.25,    # 25% 止盈
        debug=True
    )
    def __init__(self):
        self.benchmark = self.getdatabyname(self.p.benchmark_symbol)
        self.sma200 = bt.indicators.SimpleMovingAverage(self.benchmark.close, period=200)
        self.stocks = [d for d in self.datas if d._name != self.p.benchmark_symbol]
        self.add_timer(when=bt.timer.SESSION_END, monthdays=[self.p.rebalance_monthday], cheat=False)
        self.trade_history = []
        self.last_rebalance_month = -1
        self.closed_trades = []
        
        logging.info("策略初始化完成。")
        logging.info(f"止损线设置为: {self.p.stop_loss_pct:.2%}")
        logging.info(f"止盈线设置为: {self.p.take_profit_pct:.2%}")

    def notify_trade(self, trade):
        """
        当交易关闭时，记录交易详情。 (不变)
        """
        if trade.isclosed:
            log_msg = (f"交易完成: {trade.data._name}, 方向: '买入', 毛利: {trade.pnl:.2f}, 净利: {trade.pnlcomm:.2f}")
            print(log_msg)
            
            self.closed_trades.append({
                'symbol': trade.data._name,
                'open_date': bt.num2date(trade.dtopen).date(),
                'close_date': bt.num2date(trade.dtclose).date(),
                'duration_days': trade.barlen,
                'pnl': trade.pnl,
                'pnl_net': trade.pnlcomm
            })
    
    # <<<--- 核心修改区域：重写 next() 方法以实现止盈止损 ---<<<
    def next(self):
        """
        此方法在每个交易日被调用，用于执行止盈止损检查。
        月度调仓逻辑已完全由 notify_timer 处理，此处不再重复。
        """
        current_date = self.datetime.date(0)
        
        # 使用 list() 创建持仓字典的副本进行迭代，防止在迭代过程中修改字典
        for data, pos in list(self.getpositions().items()):
            # 确保有持仓 (pos.size 是持有的股数)
            if pos.size == 0:
                continue

            # backtrader 自动记录的平均买入价
            entry_price = pos.price 
            current_price = data.close[0]
            
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 1. 检查止损条件
            if pnl_pct <= -self.p.stop_loss_pct:
                print(f"--- 止损触发 ---")
                print(f"日期: {current_date}, 股票: {data._name}")
                print(f"买入价: {entry_price:.2f}, 当前价: {current_price:.2f}, 浮亏: {pnl_pct:.2%}")
                self.close(data=data) # 以市价单平仓
            
            # 2. 检查止盈条件
            elif pnl_pct >= self.p.take_profit_pct:
                print(f"--- 止盈触发 ---")
                print(f"日期: {current_date}, 股票: {data._name}")
                print(f"买入价: {entry_price:.2f}, 当前价: {current_price:.2f}, 浮盈: {pnl_pct:.2%}")
                self.close(data=data) # 以市价单平仓
    # <<<--- 核心修改区域结束 ---<<<

    def notify_timer(self, timer, when, *args, **kwargs):
        """
        此方法由定时器在每月的指定日期触发，负责执行调仓。(不变)
        """
        self.rebalance_portfolio()

    def rebalance_portfolio(self):
        """
        核心的月度调仓逻辑。(不变)
        """
        current_date = self.datetime.date(0)
        rankings = []
        
        for d in self.stocks:
            if len(d) > 0 and d.factor[0] > -1:
                rankings.append((d.factor[0], d))
        
        rankings.sort(key=lambda x: x[0], reverse=True)
        if not rankings: return

        top_n = int(len(self.stocks) * self.p.top_n_pct)
        target_stocks = [d for score, d in rankings[:top_n]]
        target_stock_names = {d._name for d in target_stocks}

        is_bull_market = self.benchmark.close[0] > self.sma200[0]
        total_portfolio_weight = 1.0 if is_bull_market else 0.5 # 修正：牛市100%，熊市50%

        print("-" * 60)
        if is_bull_market:
            print(f"调仓日: {self.datetime.date(0)} - 市场状态: 牛市 (目标总仓位: {total_portfolio_weight:.0%})")
        else:
            print(f"调仓日: {self.datetime.date(0)} - 市场状态: 熊市 (目标总仓位: {total_portfolio_weight:.0%})")
        
        target_pct_per_stock = total_portfolio_weight / len(target_stocks) if target_stocks else 0
        print(f"目标持仓股票数量: {len(target_stocks)}, 每只目标仓位: {target_pct_per_stock:.2%}")
        
        # 卖出不在目标列表中的股票
        for data, pos in self.getpositions().items():
            if pos.size != 0 and data._name not in target_stock_names and data._name != self.p.benchmark_symbol:
                print(f"调仓卖出: {data._name} (清空现有持仓)")
                self.order_target_percent(data=data, target=0.0)

        # 买入/调整目标股票的仓位
        for d in target_stocks:
            print(f"调仓买入/调整: {d._name}, 目标仓位: {target_pct_per_stock:.2%}")
            self.order_target_percent(data=d, target=target_pct_per_stock)
        print("-" * 60)

    def stop(self):
        """
        回测结束时调用。(不变)
        """
        print("\n--- 回测结束 ---")
        final_value = self.broker.getvalue()
        print(f"初始资金: {INITIAL_CASH:,.2f}")
        print(f'最终资产价值: {final_value:,.2f}')
        print(f"总回报率: {(final_value / INITIAL_CASH - 1):.2%}")


# --- 3. 主执行逻辑 (不变) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # --- 数据准备 ---
    logging.info("开始数据准备...")
    logging.info(f"正在从缓存目录 '{CACHE_DIR}' 加载所有因子分块文件...")
    chunk_files = glob.glob(os.path.join(CACHE_DIR, '*.parquet'))
    if not chunk_files:
        print(f"错误：在目录 '{CACHE_DIR}' 中未找到任何 .parquet 缓存文件。")
        print("请先运行因子计算脚本以生成因子缓存。")
        exit()

    all_chunks = [pd.read_parquet(f) for f in chunk_files]
    df_factor = pd.concat(all_chunks, ignore_index=True)
    logging.info(f"成功合并 {len(all_chunks)} 个分块文件，共计 {len(df_factor)} 行因子数据。")

    symbols_to_run = df_factor['symbol'].unique().tolist()
    start_date = df_factor['trade_date'].min().strftime('%Y-%m-%d')
    end_date = df_factor['trade_date'].max().strftime('%Y-%m-%d')
    if BENCHMARK_SYMBOL not in symbols_to_run:
        symbols_to_run.append(BENCHMARK_SYMBOL)
    stock_symbols_list = [s for s in symbols_to_run if s != BENCHMARK_SYMBOL]

    logging.info("正在连接数据库以获取价格数据...")
    try:
        conn = psycopg2.connect(POSTGRES_CONFIG)
        suspension_threshold_ratio = 0.7 
        sql_suspended = f"""
        WITH MarketDaysPerYear AS (
            SELECT EXTRACT(YEAR FROM trade_date) as yr, COUNT(*) as total_market_days
            FROM public.index_daily WHERE ts_code = '{BENCHMARK_SYMBOL}' GROUP BY yr
        ),
        StockDaysPerYear AS (
            SELECT symbol, EXTRACT(YEAR FROM trade_date) as yr, COUNT(*) as stock_trading_days
            FROM public.stock_history WHERE adjust_type = '{ADJUST_TYPE}' GROUP BY symbol, yr
        )
        SELECT DISTINCT symbol FROM StockDaysPerYear s
        JOIN MarketDaysPerYear m ON s.yr = m.yr
        WHERE s.stock_trading_days < m.total_market_days * {suspension_threshold_ratio};
        """
        suspended_symbols = pd.read_sql_query(sql_suspended, conn)['symbol'].tolist()
        logging.info(f"清洗规则: 找到 {len(suspended_symbols)} 只存在长期停牌记录的股票将被剔除。")

        pre_cleaned_list = [s for s in stock_symbols_list if s not in suspended_symbols]
        logging.info(f"初步清洗后，剩余 {len(pre_cleaned_list)} 只股票。")      

        all_stocks_query = f"SELECT trade_date, symbol, close FROM public.stock_history WHERE symbol IN {tuple(pre_cleaned_list)} AND trade_date BETWEEN '{start_date}'::date AND '{end_date}'::date AND adjust_type = '{ADJUST_TYPE}'"
        df_stocks = pd.read_sql_query(all_stocks_query, conn)
        df_stocks['trade_date'] = pd.to_datetime(df_stocks['trade_date'])
        
        index_price_query = f"SELECT trade_date, ts_code AS symbol, close FROM public.index_daily WHERE ts_code = '{BENCHMARK_SYMBOL}' AND trade_date BETWEEN '{start_date}'::date AND '{end_date}'::date"
        df_index = pd.read_sql_query(index_price_query, conn)
        df_index['trade_date'] = pd.to_datetime(df_index['trade_date'])
        conn.close()

        df_prices = pd.concat([df_stocks, df_index], ignore_index=True)
        logging.info("价格数据获取完毕。")
    except Exception as e:
        logging.error(f"从数据库获取价格数据失败: {e}")
        exit()
    
    df_all_data = pd.merge(df_prices, df_factor, on=['trade_date', 'symbol'], how='left')
    df_all_data['factor'].fillna(-1, inplace=True)
    df_all_data.set_index('trade_date', inplace=True)
    df_all_data.sort_index(inplace=True)
    
    MAX_STOCKS_TO_RUN = None 
    if MAX_STOCKS_TO_RUN:
        symbols_to_run = pre_cleaned_list[:MAX_STOCKS_TO_RUN] + [BENCHMARK_SYMBOL]
        logging.warning(f"注意：为提高速度，仅回测前 {MAX_STOCKS_TO_RUN} 只股票及基准。")
    else:
        symbols_to_run = pre_cleaned_list + [BENCHMARK_SYMBOL]

    # --- Cerebro引擎设置 ---
    cerebro = bt.Cerebro()

    logging.info(f"正在向Backtrader添加 {len(symbols_to_run)} 个数据源...")
    for symbol in symbols_to_run:
        df_sym = df_all_data[df_all_data['symbol'] == symbol][['close', 'factor']].copy()
        if df_sym.empty: continue
        
        df_sym['open'] = df_sym['high'] = df_sym['low'] = df_sym['close']
        df_sym['volume'] = 0
        
        data_feed = PandasDataWithFactor(dataname=df_sym)
        cerebro.adddata(data_feed, name=symbol)

    cerebro.addstrategy(MLFactorStrategy)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001, stocklike=True)
    cerebro.broker.set_shortcash(False)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    logging.info("一切准备就绪，开始运行Backtrader回测...")
    results = cerebro.run()
    
    # --- 打印分析结果 ---
    strat = results[0]
    print("\n" + "="*50)
    print("--- 交易分析报告 ---")

    sharpe_analysis = strat.analyzers.sharpe_ratio.get_analysis()
    sharpe_ratio = sharpe_analysis.get('sharperatio') if sharpe_analysis else None
    
    if sharpe_ratio is not None:
        print(f"夏普率: {sharpe_ratio:.2f}")
        
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    if drawdown_analysis and 'max' in drawdown_analysis and 'drawdown' in drawdown_analysis.max:
        print(f"最大回撤: {drawdown_analysis.max.drawdown:.2f}%")
    
    trade_analysis = strat.analyzers.tradeanalyzer.get_analysis()
    if hasattr(trade_analysis, 'total') and trade_analysis.total.total > 0:
        print(f"总交易次数: {trade_analysis.total.total}")
        print(f"盈利交易次数: {trade_analysis.won.total}")
        print(f"亏损交易次数: {trade_analysis.lost.total}")
        print(f"胜率: {trade_analysis.won.total / trade_analysis.total.total * 100:.2f}%")
        print(f"平均每笔盈利: {trade_analysis.won.pnl.average:.2f}")
        print(f"平均每笔亏损: {trade_analysis.lost.pnl.average:.2f}")
        if trade_analysis.lost.pnl.average != 0:
            print(f"盈亏比: {abs(trade_analysis.won.pnl.average / trade_analysis.lost.pnl.average):.2f}")
    print("="*50)
    
    logging.info("正在生成详细交易分析...")

    if strat.closed_trades:
        df_trades = pd.DataFrame(strat.closed_trades)
        df_trades.to_csv('trade_log.csv', index=False, encoding='utf-8-sig')
        logging.info("交易记录已保存至 trade_log.csv")
        print("\n--- 最近5笔交易记录 ---")
        print(df_trades.tail())

        plt.figure(figsize=(12, 7))
        sns.histplot(df_trades['pnl_net'], kde=True, bins=50)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('单笔交易净利 (PnL) 分布图', fontsize=16)
        plt.xlabel('净利 (Pnl Net)')
        plt.ylabel('交易次数 (Frequency)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('pnl_distribution.png')
        logging.info("收益分布图已保存至 pnl_distribution.png")
    else:
        logging.warning("策略中未记录任何已完成的交易。")