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
from collections import defaultdict
import time
import gc

# 加载配置
load_dotenv('.env')
POSTGRES_CONFIG  = os.getenv("DB_DSN1")
BENCHMARK_SYMBOL = '000300.SH'
ADJUST_TYPE = 'hfq'
CACHE_DIR = 'factor_cache'
INITIAL_CASH = 1000000.0

# --- 1. 修复数据加载器：显式映射列名 ---
class PandasDataWithFactor(bt.feeds.PandasData):
    lines = ('factor',)
    # 显式指定 'factor' 线对应 DataFrame 中的 'factor' 列
    # 只有当 DataFrame 中确实有 'volume' 列时，volume 索引才有效，否则建议显式指定
    params = (
        ('factor', 'factor'), 
        ('datetime', None), # 使用索引作为时间
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
    )

def get_price_limit(symbol: str) -> float:
    if symbol.startswith(('300', '688')): return 0.20
    elif symbol.startswith('8'): return 0.30
    else: return 0.10

# --- 2. 策略逻辑 (保持原有逻辑，增加调试信息) ---
class MLFactorStrategy(bt.Strategy):
    params = dict(
        top_n_pct=0.02,
        rebalance_monthday=1, # 月初第1个交易日
        benchmark_symbol=BENCHMARK_SYMBOL,
        debug=False,
        bull_position=1.00,
        bear_position=1.00,
        stop_loss_pct=0.08,
        take_profit_pct=0.25,
    )

    def __init__(self):
        self.benchmark = self.getdatabyname(self.p.benchmark_symbol)
        # 注意：如果 Benchmark 数据缺失，SMA 会导致 next() 等待。
        # 这里使用 safer 的方式，允许 benchmark 缺失时策略仍能运行（虽然逻辑上可能需要 benchmark）
        #self.sma200 = bt.indicators.SimpleMovingAverage(self.benchmark.close, period=200)
        
        self.stocks = [d for d in self.datas if d._name != self.p.benchmark_symbol]
        self.add_timer(when=bt.timer.SESSION_END, monthdays=[self.p.rebalance_monthday], cheat=False)
        
        self.last_rebalance_month = -1
        self.closed_trades = []
        self._rebalance_count = 0
        self.daily_holdings = []
        self.stock_entry_price = defaultdict(lambda: None)
        logging.info("策略初始化完成。")

    def notify_trade(self, trade):
        if not trade.isclosed: return
        
        symbol = trade.data._name
        entry_price = round(trade.price, 4)
        try:
            exit_price = round(trade.history[-1].price, 4) if len(trade.history) > 0 else round(trade.data.close[0], 4)
        except:
            exit_price = round(trade.data.close[0], 4)
            
        pct_ret = round(exit_price / entry_price - 1, 4)
        self.closed_trades.append({
            'symbol': symbol, 
            'open_date': bt.num2date(trade.dtopen).date(), 
            'close_date': bt.num2date(trade.dtclose).date(),
            'pnl_net': round(trade.pnlcomm, 2),
            'return': pct_ret
        })

    def next(self):
        print(f"当前回测时间: {self.datetime.date(0)}")
        # 打印进度，确保回测正在运行且时间正确
        if self.datetime.date(0).month != getattr(self, '_last_log_month', -1):
             print(f"[{self.datetime.date(0)}] 回测进行中... 持仓数: {len(self.getpositions())}")
             self._last_log_month = self.datetime.date(0).month

        current_month = self.datetime.date(0).month
        current_day = self.datetime.date(0).day
        
        # 保险调仓
        if current_day >= 28 and self.last_rebalance_month != current_month:
            self.rebalance_portfolio()
            self.last_rebalance_month = current_month
        
        # 止盈止损逻辑
        for data, pos in self.getpositions().items():
            if pos.size == 0: continue
            entry = self.stock_entry_price.get(data._name)
            if not entry: continue
            
            ret = data.close[0] / entry - 1
            if ret < -self.p.stop_loss_pct:
                self.order_target_percent(data=data, target=0.0)
                self.stock_entry_price[data._name] = None
            elif ret > self.p.take_profit_pct:
                self.order_target_percent(data=data, target=0.0)
                self.stock_entry_price[data._name] = None

    def notify_timer(self, timer, when, *args, **kwargs):
        current_month = self.datetime.date(0).month
        if self.last_rebalance_month != current_month:
            self.rebalance_portfolio()
            self.last_rebalance_month = current_month

    def _is_limit_up(self, data):
        if len(data) < 2: return False
        limit = get_price_limit(data._name)
        return data.high[0] >= round(data.close[-1] * (1 + limit), 2) - 0.01

    def _is_limit_down(self, data):
        if len(data) < 2: return False
        limit = get_price_limit(data._name)
        return data.low[0] <= round(data.close[-1] * (1 - limit), 2) + 0.01

    def rebalance_portfolio(self):
        self._rebalance_count += 1
        rankings = []
        
        # 这里是关键：只有 factor 有效且 > -1 才会被选中
        # 如果 PandasData 映射错误，d.factor[0] 可能全是 NaN 或 0
        for d in self.stocks:
            if len(d) > 0 and not np.isnan(d.factor[0]) and d.factor[0] > -1:
                rankings.append((d.factor[0], d))
        
        rankings.sort(key=lambda x: x[0], reverse=True)
        
        top_n = int(len(self.stocks) * self.p.top_n_pct)
        # 如果早期只有1700只，top_n 会自动变小，符合逻辑
        target_stocks_full = [d for score, d in rankings[:top_n]]
        
        target_stocks = [d for d in target_stocks_full if not self._is_limit_up(d)]
        
        if not target_stocks: return
        
        # 简化的调仓逻辑
        target_names = {d._name for d in target_stocks}
        weight = 1.0 / len(target_stocks) if target_stocks else 0
        
        for data, pos in self.getpositions().items():
            if pos.size != 0 and data._name not in target_names:
                if not self._is_limit_down(data):
                    self.order_target_percent(data=data, target=0.0)
        
        for d in target_stocks:
            self.order_target_percent(data=d, target=weight)
            if self.stock_entry_price[d._name] is None:
                self.stock_entry_price[d._name] = d.close[0]

# --- 交易成本 ---
class AStockCommission(bt.CommInfoBase):
    params = (('stocklike', True), ('commtype', bt.CommInfoBase.COMM_PERC), ('percabs', True),)
    def _getcommission(self, size, price, pseudoexec):
        turnover = abs(size) * price
        return turnover * 0.0003 if size > 0 else turnover * (0.0003 + 0.001)

def set_oom_sacrificial():
    try:
        # 将当前进程的 OOM 评分设为最大 (1000)
        # 这样一旦内存紧张，Linux 会优先杀掉这个脚本，保全 ClickHouse
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
        print("当前进程已设置为 OOM 优先献祭对象 (Score=1000)")
    except Exception as e:
        print(f"设置 OOM Score 失败 (可能需要 root 权限): {e}")

# --- 3. 主程序优化 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    
    #set_oom_sacrificial()
    # 1. 加载因子
    logging.info("1. 加载因子数据...")
    chunk_files = glob.glob(os.path.join(CACHE_DIR, '*.parquet'))
    if not chunk_files:
        logging.error("未找到 parquet 文件")
        exit(1)
        
    df_factor = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    df_factor['trade_date'] = pd.to_datetime(df_factor['trade_date'])
    
    # 【诊断】打印因子时间范围
    min_factor_date = df_factor['trade_date'].min()
    max_factor_date = df_factor['trade_date'].max()
    logging.info(f"因子数据范围: {min_factor_date.date()} -> {max_factor_date.date()}")
    logging.info(f"因子覆盖股票数: {df_factor['symbol'].nunique()}")

    all_symbols = df_factor['symbol'].unique().tolist()
    symbols_to_run = all_symbols[:400]
    if BENCHMARK_SYMBOL not in symbols_to_run:
        symbols_to_run.append(BENCHMARK_SYMBOL)

    # 2. 从数据库加载价格
    logging.info("2. 从数据库加载价格数据...")
    try:
        conn = psycopg2.connect(POSTGRES_CONFIG)
        
        # 仅查询因子存在的日期范围
        start_str = min_factor_date.strftime('%Y-%m-%d')
        end_str = max_factor_date.strftime('%Y-%m-%d')
        
        placeholders = ','.join(['%s'] * len(symbols_to_run))
        
        # 查询个股
        query_stock = f"""
            SELECT trade_date, symbol, open, high, low, close 
            FROM public.stock_history 
            WHERE symbol IN ({placeholders}) 
              AND trade_date BETWEEN %s AND %s 
              AND adjust_type = %s
        """
        df_stocks = pd.read_sql_query(query_stock, conn, params=[*symbols_to_run, start_str, end_str, ADJUST_TYPE])
        
        # 查询指数 (分开查询更安全，防止表结构差异)
        query_index = """
            SELECT trade_date, ts_code AS symbol, open, high, low, close 
            FROM public.index_daily 
            WHERE ts_code = %s AND trade_date BETWEEN %s AND %s
        """
        df_index = pd.read_sql_query(query_index, conn, params=[BENCHMARK_SYMBOL, start_str, end_str])
        conn.close()
        
        df_prices = pd.concat([df_stocks, df_index], ignore_index=True)
        df_prices['trade_date'] = pd.to_datetime(df_prices['trade_date'])
        
        # 【诊断】打印价格时间范围
        if df_prices.empty:
            logging.error("数据库未返回任何价格数据！请检查 adjust_type 或 日期范围。")
            exit(1)
            
        logging.info(f"价格数据范围: {df_prices['trade_date'].min().date()} -> {df_prices['trade_date'].max().date()}")
        logging.info(f"价格覆盖股票数: {df_prices['symbol'].nunique()}")
        
        # 【关键诊断】检查 Benchmark 数据开始时间
        bench_data = df_prices[df_prices['symbol'] == BENCHMARK_SYMBOL]
        if not bench_data.empty:
            logging.info(f"Benchmark ({BENCHMARK_SYMBOL}) 数据范围: {bench_data['trade_date'].min().date()} -> {bench_data['trade_date'].max().date()}")
            if bench_data['trade_date'].min().year > min_factor_date.year:
                logging.warning(f"!!! 警告: Benchmark 数据开始于 {bench_data['trade_date'].min().date()}，晚于因子数据。这会导致回测延迟开始！")
        else:
            logging.error(f"严重: 缺少 Benchmark {BENCHMARK_SYMBOL} 数据！")

    except Exception as e:
        logging.error(f"数据库错误: {e}")
        exit(1)

    # 3. 合并数据
    logging.info("3. 合并因子与价格...")
    # 使用 merge 保留价格数据（因为 Backtrader 需要连续的价格，因子可以缺失）
    df_all_data = pd.merge(df_prices, df_factor, on=['trade_date', 'symbol'], how='left')
    df_all_data['factor'].fillna(-1, inplace=True)
    
    # 设置 volume (如果 SQL 没查出来)
    if 'volume' not in df_all_data.columns:
        df_all_data['volume'] = 0
    else:
        df_all_data['volume'].fillna(0, inplace=True)

    # 4. 初始化 Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.addcommissioninfo(AStockCommission())
    
    # 5. 高效添加数据 (使用 groupby 替代循环切片)
    logging.info("4. 添加数据到 Cerebro (使用 GroupBy 优化)...")
    t0 = time.time()
    
    # 必须确保 benchmark 先添加（可选，但在某些逻辑下有助于对齐）
    if BENCHMARK_SYMBOL in symbols_to_run:
        df_bench = df_all_data[df_all_data['symbol'] == BENCHMARK_SYMBOL].copy()
        df_bench.set_index('trade_date', inplace=True)
        df_bench.sort_index(inplace=True)
        cerebro.adddata(PandasDataWithFactor(dataname=df_bench), name=BENCHMARK_SYMBOL)
    
    # 批量添加其他股票
    # 这里的 groupby 极其高效，避免了你原来 O(N^2) 的筛选
    grouped = df_all_data[df_all_data['symbol'] != BENCHMARK_SYMBOL].groupby('symbol')
    
    count = 0
    for symbol, df_sym in grouped:
        if df_sym.empty: continue
        
        # 必须设置索引并排序
        df_sym = df_sym.set_index('trade_date').sort_index()
        
        # 检查该股票的最早日期
        logging.debug(f"{symbol} start: {df_sym.index[0]}") 
        
        data_feed = PandasDataWithFactor(dataname=df_sym)
        cerebro.adddata(data_feed, name=symbol)
        count += 1
        
    logging.info(f"成功添加 {count} 只股票，耗时 {time.time()-t0:.2f}秒")

    # 6. 运行回测
    cerebro.addstrategy(MLFactorStrategy)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    logging.info("开始运行回测...")
    results = cerebro.run(preload=False, runonce=False)
    strat = results[0]

    # ... (后续打印结果部分代码保持不变) ...
    print("\n" + "="*50)
    final_value = cerebro.broker.getvalue()
    print(f"最终资金: {final_value:,.2f}")
    if strat.closed_trades:
        df_log = pd.DataFrame(strat.closed_trades)
        df_log.to_csv('trade_log_v3.csv', index=False, encoding='utf-8-sig')
        print(f"交易记录已保存，共 {len(df_log)} 笔")
    else:
        print("无交易产生，请检查是否所有因子都被过滤了 (-1)")