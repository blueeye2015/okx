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

load_dotenv('.env')
POSTGRES_CONFIG  = os.getenv("DB_DSN1")
BENCHMARK_SYMBOL = '000300.SH'
ADJUST_TYPE = 'hfq'
CACHE_DIR = 'factor_cache'
INITIAL_CASH = 1000000.0

# 1. 数据加载器 (显式指定列，防止读取错位)
class PandasDataWithFactor(bt.feeds.PandasData):
    lines = ('factor',)
    params = (
        ('factor', 'factor'), 
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
    )

# 2. 策略类 (已修复报错和无交易问题)
class MLFactorStrategy(bt.Strategy):
    params = dict(
        top_n_pct=0.02,
        rebalance_monthday=1, 
        debug=True,
        stop_loss_pct=0.08,      
        take_profit_pct=0.25,    
    )

    def __init__(self):
        print("\n" + "!"*50)
        print("【启动确认】新版策略类已加载！")
        print("!"*50 + "\n")

        # 1. 打印 Data0 的身份信息 (最关键的一步)
        data0 = self.datas[0]
        print(f"【Data0 身份核查】名称: {data0._name}")
        # 注意：在 init 阶段无法获取 data0.datetime.date(0)，要在 next 里看
        
        # 排除 Data0 (Benchmark)，只操作个股
        self.stocks = self.datas[1:] 
        
        self.add_timer(when=bt.timer.SESSION_END, monthdays=[self.p.rebalance_monthday], cheat=False)
        self.last_rebalance_month = -1
        
        # 【关键修复】初始化列表 (防止 AttributeError)
        self.closed_trades = [] 
        
        self.stock_entry_price = defaultdict(lambda: None)
        # 增加一个标记，只打印一次 Data0 时间
        self.first_bar_checked = False 

    def next(self):
        # ----------------------------------------------------------------------
        # 【核心诊断】这里打印的时间，就是 Backtrader 认定的“世界原点”
        # ----------------------------------------------------------------------
        if not self.first_bar_checked:
            current_date = self.datetime.date(0)
            data0_date = self.datas[0].datetime.date(0)
            
            print("\n" + "="*40)
            print(f"🔴 回测第一帧时间: {current_date}")
            print(f"🔵 Data0 当前时间 : {data0_date}")
            print(f"🔎 Data0 数据源名称: {self.datas[0]._name}")
            print("="*40 + "\n")
            
            self.first_bar_checked = True
        # ----------------------------------------------------------------------

        # 心跳包
        if self.datetime.date(0).month == 6 and self.datetime.date(0).day <= 3:
            print(f"[{self.datetime.date(0)}] 进度... 资金: {self.broker.getvalue():.0f}")

        current_month = self.datetime.date(0).month
        current_day = self.datetime.date(0).day
        
        # 保险调仓
        if current_day >= 28 and self.last_rebalance_month != current_month:
            self.rebalance_portfolio()
            self.last_rebalance_month = current_month
            
        # ... (后续止盈止损逻辑保持不变) ...
        # =========================================================
        # 🧪 【实验】注释掉止盈止损，让利润奔跑
        # =========================================================
        # for data, pos in self.getpositions().items():
        #     if pos.size == 0: continue
        #     entry = self.stock_entry_price.get(data._name)
        #     if not entry: continue
        #     ret = data.close[0] / entry - 1
        #     if ret < -self.p.stop_loss_pct:
        #         self.order_target_percent(data=data, target=0.0)
        #         self.stock_entry_price[data._name] = None
        #     elif ret > self.p.take_profit_pct:
        #         self.order_target_percent(data=data, target=0.0)
        #         self.stock_entry_price[data._name] = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        symbol = trade.data._name
        try:
            exit_price = trade.history[-1].price if len(trade.history) > 0 else trade.data.close[0]
        except:
            exit_price = trade.data.close[0]
        pct_ret = (exit_price / trade.price) - 1
        self.closed_trades.append({
            'symbol': symbol, 'open_date': bt.num2date(trade.dtopen).date(), 
            'close_date': bt.num2date(trade.dtclose).date(), 'pnl_net': trade.pnlcomm, 'return': pct_ret
        })


    def notify_timer(self, timer, when, *args, **kwargs):
        self.rebalance_portfolio()

    def _is_limit_up(self, data):
        if len(data) < 2: return False
        return data.high[0] >= round(data.close[-1] * 1.1, 2) - 0.01

    def _is_limit_down(self, data):
        if len(data) < 2: return False
        return data.low[0] <= round(data.close[-1] * 0.9, 2) + 0.01

    def rebalance_portfolio(self):
        # 仅在每年的 6 月 1 日附近打印一次诊断，避免刷屏
        is_debug_day = (self.datetime.date(0).month == 6 and self.datetime.date(0).day <= 5)
        
        valid_stocks = []
        reject_counts = {'nan_close': 0, 'nan_factor': 0, 'low_factor': 0, 'ok': 0}
        
        for d in self.stocks:
            # 1. 过滤幽灵数据 (Reindex 产生的空值)
            if len(d) == 0 or np.isnan(d.close[0]):
                reject_counts['nan_close'] += 1
                continue
            
            # 2. 过滤无效因子
            if np.isnan(d.factor[0]):
                reject_counts['nan_factor'] += 1
                continue
                
            # 3. 过滤低分因子
            if d.factor[0] <= -0.99:
                reject_counts['low_factor'] += 1
                continue
            
            # 通过筛选
            reject_counts['ok'] += 1
            valid_stocks.append((d.factor[0], d))
        
        # --- 打印诊断信息 ---
        if is_debug_day:
            print(f"\n[{self.datetime.date(0)}] 选股漏斗:")
            print(f"  - 幽灵数据(未上市): {reject_counts['nan_close']}")
            print(f"  - 因子缺失(NaN):   {reject_counts['nan_factor']}")
            print(f"  - 因子无效(-1):    {reject_counts['low_factor']}")
            print(f"  - ✅ 最终入选:      {reject_counts['ok']}")
            if reject_counts['ok'] == 0:
                print("  ⚠️ 警告：无任何股票入选，请检查因子数据！")

        # 1. 选股
        valid_stocks = []
        for d in self.stocks:
            # 增加检查：d.close[0] > 0.01
            # 这样就绝对不会买到我们填充的 0 元幽灵数据
            if len(d) > 0 and \
               d.close[0] > 0.01 and \
               not np.isnan(d.close[0]) and \
               not np.isnan(d.factor[0]) and \
               d.factor[0] > -0.99:
                valid_stocks.append((d.factor[0], d))
        
        valid_stocks.sort(key=lambda x: x[0], reverse=True)
        
        # 2. 【修复2】保底买入逻辑
        top_n = int(len(self.stocks) * self.p.top_n_pct)
        if top_n == 0 and len(valid_stocks) > 0: top_n = 5 # 强制最少买5只

        target_stocks = [d for score, d in valid_stocks[:top_n] if not self._is_limit_up(d)]
        if not target_stocks: return

        # 3. 调仓
        weight = 0.95 / len(target_stocks)
        target_names = {d._name for d in target_stocks}
        
        for data, pos in self.getpositions().items():
            if pos.size != 0 and data._name not in target_names:
                if not self._is_limit_down(data):
                    self.order_target_percent(data=data, target=0.0)
                    self.stock_entry_price[data._name] = None
        
        for d in target_stocks:
            self.order_target_percent(data=d, target=weight)
            if self.stock_entry_price[d._name] is None:
                self.stock_entry_price[d._name] = d.close[0]

# 3. 主程序
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    
    # --- A. 加载因子 ---
    logging.info("1. 加载因子...")
    chunk_files = glob.glob(os.path.join(CACHE_DIR, '*.parquet'))
    df_factor = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    df_factor['trade_date'] = pd.to_datetime(df_factor['trade_date'])
    # 3. 主程序
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    
    # --- A. 加载因子 ---
    logging.info("1. 加载因子...")
    chunk_files = glob.glob(os.path.join(CACHE_DIR, '*.parquet'))
    if not chunk_files:
        logging.error("未找到因子文件")
        exit(1)
        
    df_factor = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    df_factor['trade_date'] = pd.to_datetime(df_factor['trade_date'])
    
    min_date, max_date = df_factor['trade_date'].min(), df_factor['trade_date'].max()
    all_symbols = df_factor['symbol'].unique().tolist()
    
    # 扩大测试范围 (此时无论你选多少，都不会再跳时间了)
    symbols_to_run = all_symbols 
    if BENCHMARK_SYMBOL not in symbols_to_run: symbols_to_run.append(BENCHMARK_SYMBOL)
    
    # --- B. 分离加载 Benchmark ---
    logging.info("2. 加载数据 (强力对齐模式)...")
    conn = psycopg2.connect(POSTGRES_CONFIG)
    
    # 2.1 查指数
    df_bench = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close FROM index_daily WHERE ts_code=%s AND trade_date BETWEEN %s AND %s ORDER BY trade_date",
        conn, params=[BENCHMARK_SYMBOL, min_date, max_date])
    df_bench['trade_date'] = pd.to_datetime(df_bench['trade_date'])
    
    # --- Benchmark 清洗与去重 ---
    if 'volume' not in df_bench.columns: df_bench['volume'] = 0
    df_bench['factor'] = -1
    # 必须去重，否则 reindex 会报错
    df_bench.drop_duplicates(subset=['trade_date'], inplace=True)
    df_bench.set_index('trade_date', inplace=True)
    # 填补中间可能的空洞日期 (Business Day) - 可选，这里直接用现有指数日期做标尺
    df_bench.sort_index(inplace=True)
    
    # 获取“上帝时间轴”
    FULL_TIMELINE = df_bench.index
    logging.info(f"基准时间轴长度: {len(FULL_TIMELINE)} 天 ({FULL_TIMELINE[0].date()} -> {FULL_TIMELINE[-1].date()})")

    # 2.2 查个股
    stock_syms = [s for s in symbols_to_run if s != BENCHMARK_SYMBOL]
    placeholders = ','.join(['%s'] * len(stock_syms))
    df_stocks = pd.read_sql_query(
        f"SELECT trade_date, symbol, open, high, low, close FROM stock_history WHERE symbol IN ({placeholders}) AND trade_date BETWEEN %s AND %s AND adjust_type=%s",
        conn, params=[*stock_syms, min_date, max_date, ADJUST_TYPE])
    conn.close()
    df_stocks['trade_date'] = pd.to_datetime(df_stocks['trade_date'])
    
    # --- C. 合并个股因子 ---
    logging.info("3. 合并个股因子...")
    df_all = pd.merge(df_stocks, df_factor, on=['trade_date', 'symbol'], how='left')
    del df_stocks, df_factor
    gc.collect()

    # -------------------------------------------------------------
    # 🔍 【诊断插入点】检查合并质量
    # -------------------------------------------------------------
    print("\n" + "="*40)
    print("🕵️‍♂️ 因子合并质量检查")
    # 1. 检查是否有任何有效因子
    valid_factors = df_all[df_all['factor'].notna()]
    print(f"原始匹配到的因子行数: {len(valid_factors)} / {len(df_all)}")
    
    # 2. 模拟空值填充后的情况
    df_all['factor'].fillna(-1, inplace=True)
    valid_factors_final = df_all[df_all['factor'] > -0.99]
    print(f"填充后，有效因子(> -0.99)行数: {len(valid_factors_final)}")
    
    # 3. 如果有效因子很少，打印样本看看 Key 是否匹配
    if len(valid_factors_final) == 0:
        print("❌ 警告：因子合并完全失败！请检查 Symbol 格式！")
        print(f"股票表中 Symbol 样例: {df_stocks['symbol'].iloc[0]}")
        print(f"因子表中 Symbol 样例: {df_factor['symbol'].iloc[0]}")
    else:
        print("✅ 因子合并成功，存在有效数据。")
    print("="*40 + "\n")
    
    # --- D. 初始化 Cerebro ---
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.addstrategy(MLFactorStrategy) # 记得添加策略！
    
    # 【第一步】添加 Benchmark 作为 Data0
    start_dt = FULL_TIMELINE[0].to_pydatetime()
    cerebro.adddata(PandasDataWithFactor(dataname=df_bench, fromdate=start_dt), name=BENCHMARK_SYMBOL)
    logging.info(f"Data0 (Benchmark) 已加载")

    # 【第二步】个股暴力对齐 (Procrustean Alignment)
    logging.info("正在对齐个股数据 (Reindexing & Filling)...")
    grouped = df_all.groupby('symbol')
    
    add_count = 0
    price_columns = ['open', 'high', 'low', 'close'] # 需要清洗的价格列

    for symbol, df_s in grouped:
        # 1. 设置索引并排序
        df_s.set_index('trade_date', inplace=True)
        df_s = df_s[~df_s.index.duplicated(keep='first')]
        
        # 2. 暴力重置索引
        df_aligned = df_s.reindex(FULL_TIMELINE)
        
        # =================================================================
        # 🚑 【核心修复】消灭价格中的 NaN
        # =================================================================
        
        # A. 处理停牌/数据缺失 (用前一天的价格填充)
        # 注意：ffill 会把上市前的 NaN 依然留着（因为前面没有值）
        df_aligned[price_columns] = df_aligned[price_columns].fillna(method='ffill')
        
        # B. 处理上市前的数据 (填充为 0)
        # 这样 Backtrader 读到的是 0 元，我们策略里不买 0 元的票即可
        df_aligned[price_columns] = df_aligned[price_columns].fillna(0.0)
        
        # C. 填充成交量 (没数据就是 0 量)
        if 'volume' not in df_aligned.columns:
            df_aligned['volume'] = 0
        else:
            df_aligned['volume'] = df_aligned['volume'].fillna(0)
            
        # D. 填充因子
        df_aligned['factor'] = df_aligned['factor'].fillna(-1)
        
        # =================================================================
        
        # 4. 喂给 Cerebro
        cerebro.adddata(PandasDataWithFactor(dataname=df_aligned), name=symbol)
        add_count += 1

    logging.info(f"已添加 {add_count} 只完全对齐的股票")
    logging.info("开始回测 (runonce=False)...")
    results = cerebro.run(preload=False, runonce=False)
    
    # --- E. 结果 ---
    if results:
        strat = results[0]
        print(f"\n最终资金: {cerebro.broker.getvalue():,.2f}")
        
        if hasattr(strat, 'closed_trades') and strat.closed_trades:
            df_res = pd.DataFrame(strat.closed_trades)
            print(f"交易笔数: {len(df_res)}")
            print(df_res.tail())
            df_res.to_csv('trade_log_final.csv', index=False)
        else:
            print("警告：没有产生交易记录。")
    else:
        print("Cerebro 返回结果为空！")