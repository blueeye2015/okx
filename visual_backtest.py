import backtrader as bt
import pandas as pd
import os
import numpy as np
import psycopg2 # <<<--- 新增
import logging
import glob   # <<<--- 新增
from datetime import datetime
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns          # <<<--- 新增
from collections import defaultdict

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

# --- 2. 核心交易策略 (不变) ---
class MLFactorStrategy(bt.Strategy):
    params = dict(
        top_n_pct=0.1,
        rebalance_monthday=20,
        benchmark_symbol=BENCHMARK_SYMBOL,
        debug=True # 新增debug开关
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

    def notify_trade(self, trade):
        """
        兼容任何版本 Backtrader 的 notify_trade：
        entry_price 用 trade.price
        exit_price  用最后一笔反向成交的成交价
        """
        if not trade.isclosed:
            return

        # 1. 基本字段
        symbol      = trade.data._name
        open_date   = bt.num2date(trade.dtopen).date()
        close_date  = bt.num2date(trade.dtclose).date()
        duration    = trade.barlen
        size        = int(trade.size)          # 正数
        entry_price = round(trade.price, 4)    # 开仓均价（官方提供）

        # ---- 关键修正：安全地获取 exit_price ----
        try:
            if len(trade.history) > 0:
                exit_price = round(trade.history[-1].price, 4)
            else:
                # history 尚为空，用当前收盘价近似
                exit_price = round(trade.data.close[0], 4)
        except Exception:
            # 兜底方案
            exit_price = round(trade.data.close[0], 4)

        # 3. 收益 & 盈亏
        pct_ret = round(exit_price / entry_price - 1, 4)
        pnl_gross = round(trade.pnl, 2)
        pnl_net   = round(trade.pnlcomm, 2)

        # 4. 落库
        self.closed_trades.append({
            'symbol'       : symbol,
            'open_date'    : open_date,
            'close_date'   : close_date,
            'duration_days': duration,
            'size'         : size,
            'entry_price'  : entry_price,
            'exit_price'   : exit_price,
            'return'       : pct_ret,
            'pnl'          : pnl_gross,
            'pnl_net'      : pnl_net
        })

        # 5. 可选调试打印
        if self.p.debug:
            print(f'{open_date} -> {close_date}  {symbol}  '
                f'{size}股  {entry_price} → {exit_price}  '
                f'收益 {pct_ret:.2%}  净利 {pnl_net:.2f}')
                   
    def next(self):
        """
        Backtrader的核心方法，每个bar（每个交易日）都会被调用一次。
        """
        current_date = self.datetime.date(0)
        current_month = current_date.month

        # 如果当前月份与上次调仓月份不同，说明进入了新的一个月，触发调仓
        if self.last_rebalance_month != current_month:
            self.last_rebalance_month = current_month
            print(f"\n进入新的月份: {current_date.year}-{current_date.month}，执行月度调仓...")
            self.rebalance_portfolio()
            
    def notify_timer(self, timer, when, *args, **kwargs):
        self.rebalance_portfolio()

    def rebalance_portfolio(self):
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
        total_portfolio_weight = 0.5 if is_bull_market else 1.0

        if is_bull_market:
            print(f"\n调仓日: {self.datetime.date(0)} - 市场状态: 牛市 (仓位: 50%)")
        else:
            print(f"\n调仓日: {self.datetime.date(0)} - 市场状态: 熊市 (仓位: 100%)")
        
        target_pct_per_stock = total_portfolio_weight / len(target_stocks) if target_stocks else 0
        print(f"目标持仓股票数量: {len(target_stocks)}, 每只目标仓位: {target_pct_per_stock:.2%}")
        
        # --- 【关键修正】使用正确的 .items() 方式遍历持仓 ---
        for data, pos in self.getpositions().items():
            if pos.size != 0 and data._name not in target_stock_names:
                print(f"卖出: {data._name} (清空现有持仓)")
                self.order_target_percent(data=data, target=0.0)

        # 买入/调整目标股票的仓位
        for d in target_stocks:
            self.order_target_percent(data=d, target=target_pct_per_stock)

    def stop(self):
        print("\n--- 回测结束 ---")
        final_value = self.broker.getvalue()
        print(f"初始资金: {INITIAL_CASH:,.2f}")
        print(f'最终资产价值: {final_value:,.2f}')
        print(f"总回报率: {(final_value / INITIAL_CASH - 1):.2%}")


# --- 3. 主执行逻辑 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # <<<--- 核心修改区域开始 ---<<<
    # --- 数据准备 ---
    logging.info("开始数据准备...")
    # 1. 从缓存目录加载所有因子分块文件
    logging.info(f"正在从缓存目录 '{CACHE_DIR}' 加载所有因子分块文件...")
    chunk_files = glob.glob(os.path.join(CACHE_DIR, '*.parquet'))
    if not chunk_files:
        print(f"错误：在目录 '{CACHE_DIR}' 中未找到任何 .parquet 缓存文件。")
        print("请先运行 backtest_a_share.py 以生成因子缓存。")
        exit()

    all_chunks = [pd.read_parquet(f) for f in chunk_files]
    df_factor = pd.concat(all_chunks, ignore_index=True)
    logging.info(f"成功合并 {len(all_chunks)} 个分块文件，共计 {len(df_factor)} 行因子数据。")


    # 从因子文件中获取回测范围
    symbols_to_run = df_factor['symbol'].unique().tolist()
    start_date = df_factor['trade_date'].min().strftime('%Y-%m-%d')
    end_date = df_factor['trade_date'].max().strftime('%Y-%m-%d')
    if BENCHMARK_SYMBOL not in symbols_to_run:
        symbols_to_run.append(BENCHMARK_SYMBOL)
    stock_symbols_list = [s for s in symbols_to_run if s != BENCHMARK_SYMBOL]

    # 连接数据库以获取价格数据
    logging.info("正在连接数据库以获取价格数据...")
    try:
        conn = psycopg2.connect(POSTGRES_CONFIG)

        # # --- 步骤 2.2: 使用交易日缺失率来识别并剔除长期停牌股 ---
        suspension_threshold_ratio = 0.7  # 交易日占比低于70%即认为长期停牌
        sql_suspended = f"""
        WITH MarketDaysPerYear AS (
            -- 1. 计算市场每年的总交易日 (以沪深300为基准)
            SELECT 
                EXTRACT(YEAR FROM trade_date) as yr, 
                COUNT(*) as total_market_days
            FROM public.index_daily
            WHERE ts_code = '{BENCHMARK_SYMBOL}'
            GROUP BY yr
        ),
        StockDaysPerYear AS (
            -- 2. 计算每只股票每年的实际交易日
            SELECT 
                symbol, 
                EXTRACT(YEAR FROM trade_date) as yr, 
                COUNT(*) as stock_trading_days
            FROM public.stock_history
            WHERE adjust_type = '{ADJUST_TYPE}'
            GROUP BY symbol, yr
        )
        -- 3. 找出在任何一年交易日占比不足的股票
        SELECT DISTINCT symbol
        FROM StockDaysPerYear s
        JOIN MarketDaysPerYear m ON s.yr = m.yr
        WHERE s.stock_trading_days < m.total_market_days * {suspension_threshold_ratio};
        """
        suspended_symbols = pd.read_sql_query(sql_suspended, conn)['symbol'].tolist()
        logging.info(f"清洗规则2: 找到 {len(suspended_symbols)} 只存在长期停牌记录的股票将被剔除 (年交易日占比<{suspension_threshold_ratio:.0%})。")

        # # --- 步骤 2.4: 组合排除列表并进行初步清洗 ---
        pre_cleaned_list = [s for s in stock_symbols_list if s not in suspended_symbols]
        logging.info(f"初步清洗后，剩余 {len(stock_symbols_list)} 只股票。")      

        all_stocks_query = f"SELECT trade_date, symbol, close FROM public.stock_history WHERE symbol IN {tuple(pre_cleaned_list)} AND trade_date BETWEEN '{start_date}'::date AND '{end_date}'::date AND adjust_type = '{ADJUST_TYPE}'"
        df_stocks = pd.read_sql_query(all_stocks_query, conn)
        df_stocks['trade_date'] = pd.to_datetime(df_stocks['trade_date']) # 确保价格表的日期也是datetime
        
        index_price_query = f"SELECT trade_date, ts_code AS symbol, close FROM public.index_daily WHERE ts_code = '{BENCHMARK_SYMBOL}' AND trade_date BETWEEN '{start_date}'::date AND '{end_date}'::date"
        df_index = pd.read_sql_query(index_price_query, conn)
        df_index['trade_date'] = pd.to_datetime(df_index['trade_date']) # 确保指数表的日期也是datetime
        conn.close()

        df_prices = pd.concat([df_stocks, df_index], ignore_index=True)
        logging.info("价格数据获取完毕。")
    except Exception as e:
        logging.error(f"从数据库获取价格数据失败: {e}")
        exit()
    
    # 合并价格和因子数据
    df_all_data = pd.merge(df_prices, df_factor, on=['trade_date', 'symbol'], how='left')
    df_all_data['factor'].fillna(-1, inplace=True)
    df_all_data.set_index('trade_date', inplace=True)
    df_all_data.sort_index(inplace=True)
    
    # <<<--- 核心修改区域结束 ---<<<

    # 为了快速演示，可以只回测部分股票。设为None则回测全部。
    MAX_STOCKS_TO_RUN = None 
    if MAX_STOCKS_TO_RUN:
        symbols_to_run = pre_cleaned_list[:MAX_STOCKS_TO_RUN] + [BENCHMARK_SYMBOL]
        logging.warning(f"注意：为提高速度，仅回测前 {MAX_STOCKS_TO_RUN} 只股票及基准。")

    # --- Cerebro引擎设置 ---
    cerebro = bt.Cerebro()

    logging.info(f"正在向Backtrader添加 {len(symbols_to_run)} 个数据源...")
    for symbol in symbols_to_run:
        df_sym = df_all_data[df_all_data['symbol'] == symbol][['close', 'factor']]
        if df_sym.empty: continue
        
        df_sym['open'] = df_sym['high'] = df_sym['low'] = df_sym['close']
        df_sym['volume'] = 0
        
        data_feed = PandasDataWithFactor(dataname=df_sym)
        cerebro.adddata(data_feed, name=symbol)

    cerebro.addstrategy(MLFactorStrategy)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001, stocklike=True)
    ##cerebro.broker.set_coc(True) # 设置以当日收盘价成交
    cerebro.broker.set_shortcash(False) #设置禁止做空

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    logging.info("一切准备就绪，开始运行Backtrader回测...")
    results = cerebro.run()
    
    # --- 打印分析结果 ---
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

    # <<<--- 核心修改区域结束 ---<<<
    