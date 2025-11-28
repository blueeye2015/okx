#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visual_backtest.py  –  完整版：补齐 + 涨跌停过滤 + 换仓成本
"""
import os, glob, logging, multiprocessing as mp
import pandas as pd
import numpy as np
import psycopg2
import backtrader as bt
from dotenv import load_dotenv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv('.env')
POSTGRES_CONFIG = os.getenv('DB_DSN1')
BENCHMARK_SYMBOL = '000300.SH'
CACHE_DIR = 'factor_cache_per_stock'
ADJUST_TYPE = 'hfq'
INITIAL_CASH = 1_000_000

# ========== 补齐与交易开关 ==========
PAD_PRICE = True        # 是否补价格（一字板）
PAD_FACTOR = True       # 是否补因子（行业 median）
PAD_TRADE = False       # False=补齐标的只排名不下单；True=允许交易
INDUSTRY_CSV = 'industry_sw2021.csv'   # symbol,industry 两列
LIMIT_FILTER = True     # 是否启用涨跌停过滤

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# -------------------- 1. 加载因子 --------------------
def _read_single_parquet(f: str) -> pd.DataFrame | None:
    """读一个 parquet 文件，返回统一列名的 DataFrame"""
    df = pd.read_parquet(f)
    df = df.rename(columns={'ts_code': 'symbol', 'code': 'symbol',
                            'datetime': 'trade_date', 'date': 'trade_date'})
    cols = {'trade_date', 'symbol', 'factor'}
    miss = cols - set(df.columns)
    if miss:
        logging.warning(f"{os.path.basename(f)} 缺少列 {miss}，跳过")
        return None
    df = df[list(cols)].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

def load_factor_cache(start: str, end: str):
    files = glob.glob(os.path.join(CACHE_DIR, '*.parquet'))
    if not files:
        raise FileNotFoundError(f'{CACHE_DIR} 下没有 parquet 文件！')
    
    with mp.Pool(min(8, mp.cpu_count())) as pool:
        chunks = pool.map(_read_single_parquet, files)
    
    df = pd.concat([c for c in chunks if c is not None], ignore_index=True)
    mask = (df['trade_date'] >= start) & (df['trade_date'] <= end)
    df = df[mask].dropna(subset=['factor'])
    logging.info(f'因子缓存加载：{len(df)} 行，{df["symbol"].nunique()} 只股票')
    return df

# -------------------- 2. 加载价格（带补齐） --------------------
# -------------------- 2. 从数据库取真实 OHLCV（含补齐） --------------------
def load_price_data(start: str, end: str):
    conn = psycopg2.connect(POSTGRES_CONFIG)

    sql_pool = f"""
    WITH listed AS (
        SELECT symbol
        FROM public.stock_basic
        WHERE list_date <= '{start}'
          AND (delist_date IS NULL OR delist_date > '{end}')
    ),
    market_days AS (
        SELECT EXTRACT(YEAR FROM trade_date) AS yr,
               COUNT(*) AS mkt_days
        FROM public.index_daily
        WHERE ts_code = '{BENCHMARK_SYMBOL}'
          AND trade_date BETWEEN '{start}' AND '{end}'
        GROUP BY yr
    ),
    stock_days AS (
        SELECT symbol,
               EXTRACT(YEAR FROM trade_date) AS yr,
               COUNT(*) AS sd
        FROM public.stock_history
        WHERE adjust_type = 'hfq'
          AND trade_date BETWEEN '{start}' AND '{end}'
        GROUP BY symbol, yr
    )
    SELECT s.symbol
    FROM listed s
    JOIN stock_days d ON d.symbol = s.symbol
    JOIN market_days m ON m.yr = d.yr
    GROUP BY s.symbol
    HAVING COUNT(*) = EXTRACT(YEAR FROM '{end}'::DATE) - EXTRACT(YEAR FROM '{start}'::DATE) + 1         -- 每年都要有记录
       AND MIN(d.sd) >= MIN(m.mkt_days) * 0.8;  -- 每年占比≥阈值
    """
    symbols = pd.read_sql(sql_pool, conn)['symbol'].tolist()
    if BENCHMARK_SYMBOL not in symbols:
        symbols.append(BENCHMARK_SYMBOL)

    # 1) 真实 OHLCV
    sql = """
    SELECT trade_date, symbol, open, high, low, close, volume
    FROM   public.stock_history
    WHERE  symbol IN %s
      AND  trade_date BETWEEN %s AND %s
      AND  adjust_type = %s
    """
    df_stock = pd.read_sql(sql, conn, params=(tuple(symbols), start, end, ADJUST_TYPE))

    # 2) 基准
    sql_idx = """
    SELECT trade_date, ts_code AS symbol, open, high, low, close, vol as volume
    FROM   public.index_daily
    WHERE  ts_code = %s
      AND  trade_date BETWEEN %s AND %s
    """
    df_idx = pd.read_sql(sql_idx, conn, params=(BENCHMARK_SYMBOL, start, end))

    
    conn.close()

    df_price = pd.concat([df_stock, df_idx], ignore_index=True)
    df_price['trade_date'] = pd.to_datetime(df_price['trade_date'])

    # 3) 仅对「数据库里 truly 无价格」的股票补首根真实 K 线
    if PAD_PRICE:
        all_want = set(symbols) - {BENCHMARK_SYMBOL}
        have_price = set(df_price['symbol'])
        miss_price = all_want - have_price

        if miss_price:
            conn = psycopg2.connect(POSTGRES_CONFIG)
            sql_first = """
            SELECT symbol,
                   MIN(trade_date)                        AS list_day,
                   (array_agg(open  ORDER BY trade_date))[1] AS open,
                   (array_agg(high  ORDER BY trade_date))[1] AS high,
                   (array_agg(low   ORDER BY trade_date))[1] AS low,
                   (array_agg(close ORDER BY trade_date))[1] AS close,
                   0                                         AS volume
            FROM   public.stock_history
            WHERE  symbol IN %s
              AND  adjust_type = %s
            GROUP  BY symbol
            """
            df_first = pd.read_sql(sql_first, conn, params=(tuple(miss_price), ADJUST_TYPE))
            conn.close()

            # 把首根 K 线日期设为回测起点（仅用于占位，volume=0 标记为补齐）
            df_first['trade_date'] = pd.to_datetime(start)
            df_price = pd.concat([df_price, df_first], ignore_index=True)
            logging.info(f'补齐 {len(miss_price)} 只无价格股票（真实首根 K 线）')

    return df_price,symbols

# -------------------- 3. Backtrader 策略 --------------------
class PandasDataWithFactor(bt.feeds.PandasData):
    lines = ('factor', 'is_padded')
    params = (('factor', -1), ('is_padded', -1))

class MLFactorStrategy(bt.Strategy):
    params = dict(top_n_pct=0.1,
                  rebalance_monthday=20,
                  benchmark_symbol=BENCHMARK_SYMBOL,
                  debug=True)

    def __init__(self):
        self.bm = self.getdatabyname(self.p.benchmark_symbol)
        self.sma200 = bt.ind.SimpleMovingAverage(self.bm.close, period=200)
        self.stocks = [d for d in self.datas if d._name != self.p.benchmark_symbol]
        self.add_timer(when=bt.timer.SESSION_END, monthdays=[self.p.rebalance_monthday])
        self.closed_trades = []
        self.last_reb_month = -1
        logging.info(f"策略初始化完成，共加载 {len(self.stocks)} 只股票")

    def is_limit_up(self, data):
        """涨停：high==low==close 且收在最高价"""
        return LIMIT_FILTER and (data.high[0] == data.low[0] == data.close[0] == data.high[0])

    def is_limit_down(self, data):
        """跌停：high==low==close 且收在最低价"""
        return LIMIT_FILTER and (data.high[0] == data.low[0] == data.close[0] == data.low[0])
    
    # ---------- 工具：日志打印 ----------
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logging.info(f'{dt.isoformat()} {txt}')

    # ---------- 核心：next 内调仓 ----------
    def next(self):
        current_date = self.datas[0].datetime.date(0)
        current_month = current_date.month

        # 1. 每月指定日调仓（可改成第 N 个交易日逻辑）
        if current_month == self.last_reb_month or current_date.day != self.p.rebalance_monthday:
            return
        self.last_reb_month = current_month

        if self.p.debug:
            self.log(f'========== 调仓日 ==========')

        # 2. 排名 & 过滤
        rankings = []
        for d in self.stocks:
            if len(d) and d.factor[0] > -1:
                if self.is_limit_up(d):
                    if self.p.debug:
                        self.log(f'{d._name} 涨停，跳过')
                    continue
                rankings.append((d.factor[0], d, d.is_padded[0]))

        if not rankings:
            self.log('无有效因子股票！')
            return

        rankings.sort(reverse=True, key=lambda x: x[0])
        top_n = int(len(self.stocks) * self.p.top_n_pct)
        targets = {d for _, d, _ in rankings[:top_n]}

        # 3. 仓位权重
        is_bull = self.bm.close[0] > self.sma200[0]
        weight = 0.5 if is_bull else 1.0
        per_weight = weight / len(targets) if targets else 0

        if self.p.debug:
            self.log(f'目标持仓 {len(targets)} 只，每只股票仓位={per_weight:.2%}')

        # 4. 清仓非目标 + 跌停强平
        for d, pos in self.getpositions().items():
            if pos.size and (d._name not in targets or self.is_limit_down(d)):
                self.log(f'卖出: {d._name} ({pos.size}股)')
                self.order_target_percent(d, 0)

        # 5. 买入目标
        buy_cnt = 0
        for d in targets:
            if d in self.getpositions():
                continue  # 已持仓跳过
            if self.is_limit_up(d):
                self.log(f'{d._name} 涨停，跳过买入')
                continue
            if hasattr(d, 'is_padded') and d.is_padded[0] and not PAD_TRADE:
                self.log(f'{d._name} 补齐标的，跳过交易')
                continue

            self.log(f'买入: {d._name} 仓位={per_weight:.2%}')
            self.order_target_percent(d, per_weight)
            buy_cnt += 1

        if self.p.debug:
            self.log(f'本次调仓：买入 {buy_cnt} 只，卖出 {len([d for d, pos in self.getpositions().items() if pos.size and d._name not in targets])} 只')

    # ---------- 交易记录 ----------
    def notify_trade(self, trade):
        cost_basis = trade.price * abs(trade.size)
        ret = round(trade.pnlcomm / cost_basis, 4) if cost_basis else 0.0

        if trade.isclosed:
            self.closed_trades.append({
                'symbol': trade.data._name,
                'open_date': bt.num2date(trade.dtopen).date(),
                'close_date': bt.num2date(trade.dtclose).date(),
                'size': int(trade.size),
                'entry_price': round(trade.price, 4),
                'exit_price': round(trade.history[-1].price if trade.history else trade.data.close[0], 4),
                'return': ret,
                'pnl_net': round(trade.pnlcomm, 2)
            })

# -------------------- 4. 主流程 --------------------
def main():
    START_DATE, END_DATE = '2018-01-01', '2025-12-31'
    
    # 验证打印
    files = glob.glob(os.path.join(CACHE_DIR, '*.parquet'))
    print('① parquet 文件数：', len(files))
    
    # 1. 读因子
    df_factor= load_factor_cache(START_DATE, END_DATE)
    print('② 因子表股票数：', df_factor['symbol'].nunique())
    
    # 2. 读价格（自动补齐）
    df_price ,symbols = load_price_data(START_DATE, END_DATE)
    print('② 完整数据股票数：', len(symbols))

    price_symbols = set(df_price['symbol']) - {BENCHMARK_SYMBOL}
    print('③ 补齐后区间有价格：', len(price_symbols))
    
    # 3. 合并
    df_all = pd.merge(df_price, df_factor, on=['trade_date', 'symbol'], how='left')
    df_all['factor'] = df_all['factor'].fillna(-1)
    
    # 4. 补因子（行业 median）
    if PAD_FACTOR:
        try:
            ind_map = pd.read_csv(INDUSTRY_CSV)
            df_all = df_all.merge(ind_map, on='symbol', how='left')
            
            ind_median = df_all.groupby('industry')['factor'].median().to_dict()
            df_all['factor'] = df_all.groupby('industry')['factor'].transform(
                lambda x: x.fillna(ind_median.get(x.name, 0))
            )
            df_all['factor'] = df_all['factor'].fillna(0)
            
            # 标记补数据
            df_all['is_padded'] = df_all['volume'] == 0
            logging.info(f'按行业补齐因子，标记 {df_all["is_padded"].sum()} 只补数据股票')
        except FileNotFoundError:
            logging.warning(f'{INDUSTRY_CSV} 未找到，跳过因子补齐')
            df_all['is_padded'] = False
    else:
        df_all['is_padded'] = False
    
    df_all.set_index('trade_date', inplace=True)
    df_all.sort_index(inplace=True)
    
    print('④ 最终可参与股票：', len(set(df_all['symbol']) - {BENCHMARK_SYMBOL}))

    # =============  立即验证：调仓日是否有数据  =============
    cal = pd.date_range(START_DATE, END_DATE, freq='B')          # 工作日日历
    reb_days = [d for d in cal if d.day == 20]                   # 所有 20 号
    print('调仓日（20号）总数：', len(reb_days))
    print('首个股调仓日：', reb_days[0] if reb_days else '无')
    print('data 最小日期：', df_all.index.min())
    print('data 最大日期：', df_all.index.max())
    # =========================================================
    
    # 5. 喂给 Backtrader
    cerebro = bt.Cerebro()
    for sym in symbols:
        df_sym = df_all[df_all['symbol'] == sym].copy()
        if df_sym.empty:
            continue

        # 1. 确保补齐标记存在
        if 'is_padded' not in df_sym.columns:
            df_sym['is_padded'] = False

        # 2. 直接搬运真实 OHLCV（数据库已带回）
        df_sym = df_sym[['open', 'high', 'low', 'close', 'volume', 'factor', 'is_padded']]

        # 3. 仅补齐标的把 volume 标 0（真实股票保留原 volume）
        df_sym['volume'] = np.where(df_sym['is_padded'], 0, df_sym['volume'])

        data = PandasDataWithFactor(dataname=df_sym)
        cerebro.adddata(data, name=sym)
    
    cerebro.addstrategy(MLFactorStrategy)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.set_slippage_perc(0.002)  # 0.2%滑点
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='ret')
    
    logging.info('开始回测...')
    results = cerebro.run()
    strat = results[0]
    
    # 6. 结果打印（修复 None 格式化错误）
    final = cerebro.broker.getvalue()
    print('\n' + '='*50)
    print(f'回测区间：{START_DATE} ~ {END_DATE}')
    print(f'初始资金：{INITIAL_CASH:,.0f}')
    print(f'最终市值：{final:,.0f}')
    print(f'总收益：{(final / INITIAL_CASH - 1):.2%}')
    
    sharpe_val = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    if sharpe_val is not None:
        print(f'夏普率：{sharpe_val:.2f}')
    else:
        print('夏普率：N/A (无交易)')
    
    dd_val = strat.analyzers.dd.get_analysis().get("max", {}).get("drawdown")
    if dd_val is not None:
        print(f'最大回撤：{dd_val:.2f}%')
    else:
        print('最大回撤：N/A (无交易)')
    print('='*50)
    
    # 7. 收益分布图
    if hasattr(strat, 'closed_trades') and strat.closed_trades:
        df_trades = pd.DataFrame(strat.closed_trades)
        df_trades.to_csv('trade_log.csv', index=False)
        plt.figure(figsize=(8, 5))
        sns.histplot(df_trades['pnl_net'], kde=True, bins=30)
        plt.axvline(0, color='r', ls='--')
        plt.title('Trade PnL Distribution')
        plt.savefig('pnl_distribution.png')
        logging.info('trade_log.csv & pnl_distribution.png 已生成')

if __name__ == '__main__':
    main()