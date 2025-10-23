# visual_backtest.py (Refactored for Dynamic Factor Calculation)
import backtrader as bt
import pandas as pd
import os
import numpy as np
import psycopg2
import logging
from datetime import datetime
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 关键：从您的因子计算脚本中导入核心函数
from momentum_scanner_a_share import calc_monthly_ic

load_dotenv('.env')
POSTGRES_CONFIG = os.getenv("DB_DSN1")

# --- 1. 配置项 ---
BENCHMARK_SYMBOL = '000300.SH'
ADJUST_TYPE = 'hfq'
INITIAL_CASH = 1000000.0
# 回测的起始和结束日期
START_DATE_BACKTEST = '2014-01-01'
END_DATE_BACKTEST = '2018-12-31'
# 因子计算需要更早的数据，这里多取4年用于模型训练
FACTOR_DATA_START_DATE = (pd.to_datetime(START_DATE_BACKTEST) - pd.DateOffset(years=4)).strftime('%Y-%m-%d')


# --- 2. 核心交易策略 (动态因子版) ---
class DynamicMLFactorStrategy(bt.Strategy):
    params = dict(
        top_n_pct=0.1,
        rebalance_day=20, # 改为 rebalance_day，更清晰
        benchmark_symbol=BENCHMARK_SYMBOL,
        db_config=POSTGRES_CONFIG,
        adjust_type=ADJUST_TYPE,
        debug=True
    )

    def __init__(self):
        self.benchmark = self.getdatabyname(self.p.benchmark_symbol)
        self.sma200 = bt.indicators.SimpleMovingAverage(self.benchmark.close, period=200)
        self.stocks_in_universe = [d for d in self.datas if d._name != self.p.benchmark_symbol]
        
        # 移除旧的timer，使用next进行月度调仓逻辑
        self.last_rebalance_month = -1
        self.closed_trades = []
        self.db_conn = None # 初始化数据库连接为空
        
        logging.info("动态因子策略初始化完成。")

    def start(self):
        """在回测开始时建立数据库连接"""
        try:
            self.db_conn = psycopg2.connect(self.p.db_config)
            logging.info("策略内部已成功建立数据库连接。")
        except Exception as e:
            logging.error(f"策略无法连接到数据库: {e}")
            self.env.runstop() # 如果无法连接数据库，则停止回测

    def stop(self):
        """在回测结束时关闭数据库连接并打印最终结果"""
        if self.db_conn:
            self.db_conn.close()
            logging.info("策略内部数据库连接已关闭。")
            
        print("\n--- 回测结束 ---")
        final_value = self.broker.getvalue()
        print(f"初始资金: {INITIAL_CASH:,.2f}")
        print(f'最终资产价值: {final_value:,.2f}')
        print(f"总回报率: {(final_value / INITIAL_CASH - 1):.2%}")


    def notify_trade(self, trade):
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

    def next(self):
        current_date = self.datetime.date(0)
        
        # 每月只在 rebalance_day 或之后的首个交易日执行一次
        if current_date.day >= self.p.rebalance_day and current_date.month != self.last_rebalance_month:
            self.last_rebalance_month = current_date.month
            self.rebalance_portfolio()

    def rebalance_portfolio(self):
        current_date = self.datetime.date(0)
        logging.info(f"\n{'='*30} 调仓日: {current_date} {'='*30}")

        # --- 动态因子计算核心 ---
        try:
            # 1. 获取截至当前回测日的所有历史数据用于因子计算
            factor_data_end = current_date.strftime('%Y-%m-%d')
            all_symbols_in_universe = [d._name for d in self.stocks_in_universe]
            
            logging.info(f"正在为 {current_date} 动态计算因子...")
            df_data_for_factor = get_data_for_factor_calculation(
                self.db_conn, 
                all_symbols_in_universe,
                FACTOR_DATA_START_DATE, # 因子计算需要长历史数据
                factor_data_end # 截至当前回测日
            )

            # 2. 调用外部函数计算因子
            # 注意：calc_monthly_ic会计算历史所有月份的因子，我们只取最新的
            _, df_daily_factor = calc_monthly_ic(df_data_for_factor, all_symbols_in_universe)
            
            if df_daily_factor is None or df_daily_factor.empty:
                logging.warning("当前日期未能计算出任何因子值，跳过本次调仓。")
                return

            # 3. 获取最新的因子排名
            # 找到离当前日期最近的因子值
            df_daily_factor['trade_date'] = pd.to_datetime(df_daily_factor['trade_date'])
            latest_factors = df_daily_factor[df_daily_factor['trade_date'] <= pd.to_datetime(current_date)]
            if latest_factors.empty:
                logging.warning(f"截至 {current_date} 未找到有效的因子值。")
                return

            rankings = latest_factors.groupby('symbol')['factor'].last().sort_values(ascending=False)
            
        except Exception as e:
            logging.error(f"在 {current_date} 的因子计算过程中发生严重错误: {e}")
            return # 如果计算失败，则跳过本次调仓

        # --- 后续交易逻辑 (与之前类似) ---
        top_n = int(len(self.stocks_in_universe) * self.p.top_n_pct)
        target_stock_names = set(rankings.head(top_n).index)
        
        is_bull_market = self.benchmark.close[0] > self.sma200[0]
        total_portfolio_weight = 1.0 if not is_bull_market else 0.5 # 熊市满仓，牛市半仓

        print(f"市场状态: {'牛市' if is_bull_market else '熊市'} (总仓位: {total_portfolio_weight:.0%})")
        print(f"目标持仓股票: {list(target_stock_names)}")

        target_pct_per_stock = total_portfolio_weight / len(target_stock_names) if target_stock_names else 0

        # 卖出不在新目标列表中的股票
        for data, pos in self.getpositions().items():
            if pos.size != 0 and data._name not in target_stock_names:
                print(f"卖出: {data._name}")
                self.order_target_percent(data=data, target=0.0)

        # 买入/调整目标股票的仓位
        for stock_name in target_stock_names:
            d = self.getdatabyname(stock_name)
            self.order_target_percent(data=d, target=target_pct_per_stock)


# --- 3. 辅助函数：用于获取因子计算所需的全量数据 ---
def get_data_for_factor_calculation(conn, symbols_list, start_date, end_date):
    """
    一个独立的函数，用于为给定的符号和日期范围获取所有需要的数据（价格，市值，财务）。
    这个函数整合了 backtest_a_share.py 中的数据获取逻辑。
    """
    # 价格
    prices_query = f"""
    SELECT trade_date, symbol, close 
    FROM public.stock_history 
    WHERE symbol IN {tuple(symbols_list)} 
      AND trade_date BETWEEN '{start_date}'::date AND '{end_date}'::date 
      AND adjust_type = '{ADJUST_TYPE}'
    """
    df_prices = pd.read_sql_query(prices_query, conn)
    
    # 市值
    mv_query = f"""
    SELECT trade_date, security_code AS symbol, total_mv
    FROM public.daily_basic
    WHERE security_code IN {tuple(symbols_list)}
      AND trade_date BETWEEN '{start_date}'::date AND '{end_date}'::date
    """
    df_mv = pd.read_sql_query(mv_query, conn)
    
    # 财务
    profit_query = f"""
    SELECT security_code as symbol, report_date, deduct_parent_netprofit
    FROM public.profit_sheet
    WHERE security_code IN {tuple(symbols_list)}
    """
    df_profit = pd.read_sql_query(profit_query, conn)

    # --- 数据合并逻辑 (从 backtest_a_share.py 移植) ---
    df_prices['trade_date'] = pd.to_datetime(df_prices['trade_date'], errors='coerce')
    df_mv['trade_date'] = pd.to_datetime(df_mv['trade_date'], errors='coerce')
    df_profit['report_date'] = pd.to_datetime(df_profit['report_date'], errors='coerce')
    
    df_prices.dropna(subset=['trade_date'], inplace=True)
    df_mv.dropna(subset=['trade_date'], inplace=True)
    df_profit.dropna(subset=['report_date'], inplace=True)

    df_merged = pd.merge(df_prices, df_mv, on=['trade_date', 'symbol'], how='left')
    df_profit.rename(columns={'report_date': 'trade_date'}, inplace=True)
    
    df_merged = df_merged.sort_values(by=['trade_date'])
    df_profit = df_profit.sort_values(by=['trade_date'])
    df_profit.drop_duplicates(subset=['symbol', 'trade_date'], keep='last', inplace=True)

    df_final = pd.merge_asof(
        df_merged.reset_index(drop=True),
        df_profit.reset_index(drop=True),
        on='trade_date',
        by='symbol',
        direction='backward'
    )
    return df_final


# --- 4. 主执行逻辑 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    
    # --- 数据准备 ---
    logging.info("开始数据准备...")
    
    # 1. 获取回测期间的所有股票列表和价格数据
    try:
        conn = psycopg2.connect(POSTGRES_CONFIG)
        logging.info("主程序成功连接数据库以准备回测数据。")
        
        # 获取一个在整个回测期间都存在的、稳定的股票列表
        sql_symbols = f"""
        SELECT symbol FROM public.stock_history 
        WHERE adjust_type = '{ADJUST_TYPE}' AND symbol != '{BENCHMARK_SYMBOL}'
        GROUP BY symbol 
        HAVING min(trade_date) <= '{START_DATE_BACKTEST}'::date AND max(trade_date) >= '{END_DATE_BACKTEST}'::date
        ORDER BY symbol
        """
        stock_symbols_list = pd.read_sql_query(sql_symbols, conn)['symbol'].tolist()
        logging.info(f"筛选出在 {START_DATE_BACKTEST} 到 {END_DATE_BACKTEST} 期间持续存在的 {len(stock_symbols_list)} 只股票作为回测宇宙。")
        stock_symbols_list_for_test = stock_symbols_list[:1000]
        symbols_to_run = stock_symbols_list_for_test + [BENCHMARK_SYMBOL]

        # 获取回测所需的价格数据 (只需要收盘价)
        all_prices_query = f"""
        SELECT trade_date, symbol, close 
        FROM public.stock_history 
        WHERE symbol IN {tuple(symbols_to_run)} 
          AND trade_date BETWEEN '{START_DATE_BACKTEST}'::date AND '{END_DATE_BACKTEST}'::date 
          AND adjust_type = '{ADJUST_TYPE}'
        UNION ALL
        SELECT trade_date, ts_code AS symbol, close 
        FROM public.index_daily 
        WHERE ts_code = '{BENCHMARK_SYMBOL}' 
          AND trade_date BETWEEN '{START_DATE_BACKTEST}'::date AND '{END_DATE_BACKTEST}'::date
        """
        df_prices = pd.read_sql_query(all_prices_query, conn)
        df_prices['trade_date'] = pd.to_datetime(df_prices['trade_date'])
        
        conn.close()
        logging.info("主程序数据库连接已关闭。")

    except Exception as e:
        logging.error(f"从数据库获取回测价格数据失败: {e}")
        exit()

    # --- Cerebro引擎设置 ---
    cerebro = bt.Cerebro()

    logging.info(f"正在向Backtrader添加 {len(symbols_to_run)} 个数据源...")
    for symbol in symbols_to_run:
        df_sym = df_prices[df_prices['symbol'] == symbol].copy()
        if df_sym.empty: continue
        
        df_sym.set_index('trade_date', inplace=True)
        df_sym.sort_index(inplace=True)
        df_sym['open'] = df_sym['high'] = df_sym['low'] = df_sym['close']
        df_sym['volume'] = 0
        
        # 使用标准的PandasData，因为因子是动态计算的，不再需要自定义Feed
        data_feed = bt.feeds.PandasData(dataname=df_sym, name=symbol)
        cerebro.adddata(data_feed)

    cerebro.addstrategy(DynamicMLFactorStrategy)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001, stocklike=True)
    cerebro.broker.set_shortcash(False)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')

    logging.info("一切准备就绪，开始运行动态因子回测... (这可能需要很长时间)")
    results = cerebro.run()
    
    # --- 结果分析与绘图 (与您之前的代码一致) ---
    strat = results[0]
    # ... (后续打印结果部分不变) ...
    print("\n" + "="*50)
    print("--- 交易分析报告 ---")

    # 稳健地处理夏普率的返回结果
    sharpe_analysis = strat.analyzers.sharpe_ratio.get_analysis()
    sharpe_ratio = None
    if isinstance(sharpe_analysis, dict):
        sharpe_ratio = sharpe_analysis.get('sharperatio')
    elif isinstance(sharpe_analysis, float):
        sharpe_ratio = sharpe_analysis
    
    if sharpe_ratio is not None:
        print(f"夏普率: {sharpe_ratio:.2f}")
        
    # 稳健地处理最大回撤
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    if drawdown_analysis and 'max' in drawdown_analysis and 'drawdown' in drawdown_analysis.max:
        print(f"最大回撤: {drawdown_analysis.max.drawdown:.2f}%")
    
    # 交易分析
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
    # <<<--- 核心修改区域结束 ---<<<
    print("="*50)
    
    logging.info("正在生成详细交易分析...")

    # 3. 从策略实例中直接访问我们记录的 closed_trades 列表
    if strat.closed_trades:
        df_trades = pd.DataFrame(strat.closed_trades)
        df_trades.to_csv('trade_log.csv', index=False, encoding='utf-8-sig') # 增加编码以防中文乱码
        logging.info("交易记录已保存至 trade_log.csv")
        print("\n--- 最近5笔交易记录 ---")
        print(df_trades.tail())

        # 绘制收益分布图
        plt.figure(figsize=(12, 7))
        # 使用 seaborn 绘制直方图和核密度估计曲线
        sns.histplot(df_trades['pnl_net'], kde=True, bins=50)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('单笔交易净利 (PnL) 分布图', fontsize=16, fontproperties="SimHei") # 指定中文字体
        plt.xlabel('净利 (Pnl Net)', fontproperties="SimHei")
        plt.ylabel('交易次数 (Frequency)', fontproperties="SimHei")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('pnl_distribution.png')
        logging.info("收益分布图已保存至 pnl_distribution.png")
    else:
        logging.warning("策略中未记录任何已完成的交易。")