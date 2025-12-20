#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版回测框架
- 市场择时（熊市降仓）
- 因子分数加权
- 波动率自适应止盈止损
- 真实交易成本
"""
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
POSTGRES_CONFIG = os.getenv("DB_DSN1")
BENCHMARK_SYMBOL = '000300.SH'
ADJUST_TYPE = 'hfq'
CACHE_DIR = 'factor_cache_per_stock'
INITIAL_CASH = 1000000.0

# 自定义数据加载器
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

def get_price_limit(symbol: str) -> float:
    if symbol.startswith(('300', '688')): return 0.20
    elif symbol.startswith('8'): return 0.30
    else: return 0.10

# 增强版策略
class MLFactorStrategy(bt.Strategy):
    params = dict(
        top_n_pct=0.03,  # 改善：从2%提高到3%，更分散
        rebalance_monthday=1,
        debug=True,
        stop_loss_base=0.08,
        take_profit_base=0.25,
        volatility_lookback=20,
    )

    def __init__(self):
        print("\n" + "!"*50)
        print("【启动确认】增强版策略已加载！")
        print("!"*50 + "\n")

        # 市场择时指标
        self.market_ma120 = bt.indicators.SMA(self.datas[0].close, period=120)
        self.market_ma250 = bt.indicators.SMA(self.datas[0].close, period=250)
        
        # 动态仓位控制
        self.target_position_ratio = 1.0
            
        # 1. 打印 Data0 的身份信息 (最关键的一步)
        data0 = self.datas[0]
        print(f"【Data0 身份核查】名称: {data0._name}")
        
        # 股票列表（排除Data0）
        self.stocks = self.datas[1:]
        self.add_timer(when=bt.timer.SESSION_END, monthdays=[self.p.rebalance_monthday], cheat=False)
        self.last_rebalance_month = -1
        
        # 交易记录
        self.closed_trades = []
        self.stock_entry_price = defaultdict(lambda: None)
        self.first_bar = True

    def next(self):
        # =================================================================
        # 🎯 核心修复：显式调用调仓 + 保险兜底
        # =================================================================
        current_date = self.datetime.date(0)
        
        # 首根K线立即调仓（确保有初始持仓）
        if self.first_bar:
            print(f"[{current_date}] 首根K线，强制调仓！")
            self.rebalance_portfolio()
            self.first_bar = False
            self.last_rebalance_month = current_date.month
            return
        
        # 每月1号调仓（主逻辑）
        if current_date.day == self.p.rebalance_monthday:
            print(f"[{current_date}] 定时调仓触发")
            self.rebalance_portfolio()
            self.last_rebalance_month = current_date.month
        
        # 月底保险调仓（防止定时器失效）
        elif current_date.day >= 28 and self.last_rebalance_month != current_date.month:
            print(f"[{current_date}] 月底保险调仓触发")
            self.rebalance_portfolio()
            self.last_rebalance_month = current_date.month
        
        # =================================================================
        # 市场择时
        # =================================================================
        if len(self.datas[0]) > 250:
            market_price = self.datas[0].close[0]
            market_ma250 = self.market_ma250[0]
            
            if market_price < market_ma250 * 0.95:  # 熊市
                self.target_position_ratio = 0.3
            elif market_price < self.market_ma120[0]:  # 震荡
                self.target_position_ratio = 0.7
            else:  # 牛市
                self.target_position_ratio = 1.0
        
        # =================================================================
        # 动态止盈止损
        # =================================================================
        for data, pos in self.getpositions().items():
            if pos.size == 0 or data._name == BENCHMARK_SYMBOL:
                continue
            
            entry = self.stock_entry_price.get(data._name)
            if not entry:
                continue
            
            # ✅ 修复：多取一根K线，确保close_hist[:-1]与high/low长度一致
            lookback = self.p.volatility_lookback + 1  # 21 instead of 20
            close_hist = np.array(data.close.get(size=lookback))
            if len(close_hist) < lookback:
                continue  # 数据不足
            
            # high/low保持原长度
            high = np.array(data.high.get(size=self.p.volatility_lookback))
            low = np.array(data.low.get(size=self.p.volatility_lookback))
            
            # 现在close_hist[:-1]也是20个元素，可以正常广播
            tr = np.maximum(
                high - low,
                np.maximum(
                    abs(high - close_hist[:-1]),
                    abs(low - close_hist[:-1])
                )
            )
            atr = np.mean(tr[-10:])
            vol_ratio = atr / close_hist[-1]
            
            # 动态阈值
            dynamic_stop = max(0.05, self.p.stop_loss_base * vol_ratio * 10)
            dynamic_profit = self.p.take_profit_base * (1 + vol_ratio * 5)
            
            # 执行
            ret = data.close[0] / entry - 1
            if ret < -dynamic_stop or ret > dynamic_profit:
                print(f"[{current_date}] {data._name} 止盈止损平仓: {ret:.2%}")
                self.order_target_percent(data=data, target=0.0)
                self.stock_entry_price[data._name] = None

    def notify_trade(self, trade):
        if not trade.isclosed or trade.data._name == BENCHMARK_SYMBOL:
            return
        
        symbol = trade.data._name
            
            # ✅ 正确计算收益率
        position_cost = trade.price * abs(trade.size)
        pct_ret = trade.pnlcomm / position_cost if position_cost > 0 else 0
            
            # ✅ 增加买卖价格（后复权）
        entry_price = trade.price  # 开仓均价（后复权）
        exit_price = trade.data.close[0]  # 平仓价（后复权）
            
        self.closed_trades.append({
            'symbol': symbol,
            'open_date': bt.num2date(trade.dtopen).date(),
            'close_date': bt.num2date(trade.dtclose).date(),
            'pnl_net': trade.pnlcomm,
            'return': pct_ret,
            'price_entry': entry_price,  # 买入价
            'price_exit': exit_price     # 卖出价
        })

    def notify_timer(self, timer, when, *args, **kwargs):
        # 保留定时器作为备用，但不再依赖它
        pass

    def _is_limit_up(self, data):
        if len(data) < 2: return False
        limit = get_price_limit(data._name)
        return data.high[0] >= round(data.close[-1] * (1 + limit), 2) - 0.01

    def _is_limit_down(self, data):
        if len(data) < 2: return False
        limit = get_price_limit(data._name)
        return data.low[0] <= round(data.close[-1] * (1 - limit), 2) + 0.01

    def rebalance_portfolio(self):
        is_debug_day = (self.datetime.date(0).month == 6 and self.datetime.date(0).day <= 5)
        
        # 筛选有效股票
        valid_stocks = []
        reject_counts = {'nan_close': 0, 'nan_factor': 0, 'low_factor': 0, 'limit_up': 0, 'ok': 0}
        
        for d in self.stocks:
            # 过滤条件
            if len(d) == 0 or np.isnan(d.close[0]) or d.close[0] < 0.01:
                reject_counts['nan_close'] += 1
                continue
            
            if np.isnan(d.factor[0]):
                reject_counts['nan_factor'] += 1
                continue
                
            if d.factor[0] <= -0.99:
                reject_counts['low_factor'] += 1
                continue
            
            if self._is_limit_up(d):
                reject_counts['limit_up'] += 1
                continue
            
            reject_counts['ok'] += 1
            valid_stocks.append((d.factor[0], d))
        
        # 打印诊断
        if is_debug_day:
            print(f"\n[{self.datetime.date(0)}] 选股漏斗:")
            print(f"  - 无效数据: {reject_counts['nan_close']}")
            print(f"  - 因子缺失: {reject_counts['nan_factor']}")
            print(f"  - 因子无效: {reject_counts['low_factor']}")
            print(f"  - 涨停不可买: {reject_counts['limit_up']}")
            print(f"  - ✅ 最终入选: {reject_counts['ok']}")
        
        if not valid_stocks:
            return
        
        # 排序并选择
        valid_stocks.sort(key=lambda x: x[0], reverse=True)
        top_n = int(len(self.stocks) * self.p.top_n_pct)
        if top_n == 0 and len(valid_stocks) > 0: top_n = 5
        
        target_stocks = [d for score, d in valid_stocks[:top_n]]
        
        # 因子分数加权
        factor_scores = np.array([d.factor[0] for d in target_stocks])
        
        # 去极值和归一化
        p10, p90 = np.percentile(factor_scores, 10), np.percentile(factor_scores, 90)
        
        # ✅ 修复：使用max防止分母为0
        denom = max(p90 - p10, 1e-8)
        factor_scores = (factor_scores - p10) / denom
        factor_scores = np.clip(factor_scores, 0, 1)
        
        # 权重分配（归一化到目标仓位）
        if factor_scores.sum() > 0:
            weights = factor_scores / factor_scores.sum() * self.target_position_ratio
        else:
            weights = np.ones(len(target_stocks)) / len(target_stocks) * self.target_position_ratio
        
        # 调仓执行
        target_names = {d._name for d in target_stocks}
        
        for data, pos in self.getpositions().items():
            if pos.size != 0 and data._name not in target_names and not self._is_limit_down(data):
                self.order_target_percent(data=data, target=0.0)
                self.stock_entry_price[data._name] = None
        
        for i, d in enumerate(target_stocks):
            current_pos = self.getposition(d).size
            if current_pos == 0 and not self._is_limit_up(d):
                self.order_target_percent(data=d, target=weights[i])
                if self.stock_entry_price[d._name] is None:
                    self.stock_entry_price[d._name] = d.close[0]

# 印花税成本模型
class StampDutyCommissionScheme(bt.CommInfoBase):
    params = (
        ('stamp_duty', 0.001),
        ('commission', 0.00025),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )
    
    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买入
            return abs(size) * price * self.p.commission
        else:  # 卖出
            return abs(size) * price * (self.p.commission + self.p.stamp_duty)

# 主程序
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    
    # 加载因子
    logging.info("1. 加载因子...")
    chunk_files = glob.glob(os.path.join(CACHE_DIR, '*.parquet'))
    if not chunk_files:
        logging.error("未找到因子文件")
        exit(1)
        
    df_factor = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    df_factor['trade_date'] = pd.to_datetime(df_factor['trade_date'])
    
    min_date, max_date = df_factor['trade_date'].min(), df_factor['trade_date'].max()
    logging.info(f"因子覆盖时间: {min_date.date()} -> {max_date.date()}")
    
    # 在加载因子后检查
    print("\n" + "="*50)
    print("📊 因子数据合法性检查")
    print("="*50)

    # 因子分布统计
    factor_stats = df_factor['factor'].describe()
    print("因子统计:")
    print(factor_stats)

    # 检查因子是否在合理范围 [0,1]
    out_of_range = df_factor[(df_factor['factor'] < -1) | (df_factor['factor'] > 1)]
    if len(out_of_range) > 0:
        print(f"❌ 发现 {len(out_of_range)} 条因子超出[-1,1]范围")
    else:
        print("✅ 因子范围正常")

    # 检查因子是否全为-1（无效）
    all_invalid = (df_factor['factor'] == -1).all()
    if all_invalid:
        print("❌ 因子全为-1，无效因子！")
    else:
        valid_factor_rate = (df_factor['factor'] > -0.99).mean()
        print(f"✅ 有效因子占比: {valid_factor_rate:.2%}")

    print("="*50 + "\n")

    # 股票池
    all_symbols = df_factor['symbol'].unique().tolist()
    symbols_to_run = all_symbols
    if BENCHMARK_SYMBOL not in symbols_to_run:
        symbols_to_run.append(BENCHMARK_SYMBOL)
    
    # 连接数据库
    logging.info("2. 加载数据...")
    conn = psycopg2.connect(POSTGRES_CONFIG)
    
    # 加载基准
    df_bench = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close FROM index_daily WHERE ts_code=%s AND trade_date BETWEEN %s AND %s ORDER BY trade_date",
        conn, params=[BENCHMARK_SYMBOL, min_date, max_date]
    )
    df_bench['trade_date'] = pd.to_datetime(df_bench['trade_date'])
    df_bench['volume'] = 0
    df_bench['factor'] = -1
    
    # 加载个股数据
    stock_syms = [s for s in symbols_to_run if s != BENCHMARK_SYMBOL]
    placeholders = ','.join(['%s'] * len(stock_syms))
    df_stocks = pd.read_sql_query(
        f"SELECT trade_date, symbol, open, high, low, close, volume FROM stock_history WHERE symbol IN ({placeholders}) AND trade_date BETWEEN %s AND %s AND adjust_type=%s",
        conn, params=[*stock_syms, min_date, max_date, ADJUST_TYPE]
    )
    conn.close()
    
    df_stocks['trade_date'] = pd.to_datetime(df_stocks['trade_date'])
    
    # 合并因子
    logging.info("3. 合并因子...")
    df_all = pd.merge(df_stocks, df_factor, on=['trade_date', 'symbol'], how='left')
    df_all['factor'].fillna(-1, inplace=True)
    
    # 在 visual_backtest_v4.py 合并数据后添加诊断
    print("\n" + "="*50)
    print("🔍 价格数据合法性检查")
    print("="*50)

    # 检查后复权价格是否异常
    price_stats = df_all.groupby('symbol')['close'].agg(['min', 'max', 'mean'])
    print("价格极值统计:")
    print(f"  最低价格: {price_stats['min'].min():.4f}")
    print(f"  最高价格: {price_stats['max'].max():.4f}")
    print(f"  平均价格: {price_stats['mean'].mean():.4f}")

    # 检查是否有价格<=0的幽灵数据
    invalid_prices = df_all[df_all['close'] <= 0]
    if len(invalid_prices) > 0:
        print(f"❌ 发现 {len(invalid_prices)} 条价格<=0的异常数据！")
        print(invalid_prices[['trade_date', 'symbol', 'close']].head())
    else:
        print("✅ 价格数据无负值或零值")

    # 检查是否有价格日涨幅超过20%（非新股）
    df_all['return'] = df_all.groupby('symbol')['close'].pct_change()
    extreme_moves = df_all[(df_all['return'].abs() > 0.2) & (df_all['close'] > 10)]
    if len(extreme_moves) > 0:
        print(f"⚠️ 发现 {len(extreme_moves)} 条涨幅超20%的异常波动")
        print(extreme_moves[['trade_date', 'symbol', 'return', 'close']].head())
    else:
        print("✅ 价格波动正常")
    print("="*50 + "\n")
    # 基准时间轴
    FULL_TIMELINE = pd.to_datetime(df_bench['trade_date']).sort_values()
    logging.info(f"基准时间轴: {len(FULL_TIMELINE)} 天")
    
    # 初始化Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.addstrategy(MLFactorStrategy)
    
    # 添加Benchmark
    start_dt = FULL_TIMELINE[0].to_pydatetime()
    cerebro.adddata(PandasDataWithFactor(dataname=df_bench.set_index('trade_date'), fromdate=start_dt), name=BENCHMARK_SYMBOL)
    
    # 添加成本
    cerebro.broker.setcommission(commission=0.00025, stocklike=True, commtype=bt.CommInfoBase.COMM_PERC)
    cerebro.broker.addcommissioninfo(StampDutyCommissionScheme())
    cerebro.broker.set_slippage_perc(perc=0.0005, slip_open=True, slip_match=True, slip_out=False)
    
    # 个股对齐
    logging.info("4. 对齐个股数据...")
    grouped = df_all.groupby('symbol')
    add_count = 0
    price_columns = ['open', 'high', 'low', 'close']
    
    for symbol, df_s in grouped:
        df_s.set_index('trade_date', inplace=True)
        df_s = df_s[~df_s.index.duplicated(keep='first')]
        df_aligned = df_s.reindex(FULL_TIMELINE)
        
        # 填充价格（停牌ffill，上市前0）
        df_aligned[price_columns] = df_aligned[price_columns].fillna(method='ffill').fillna(0.0)
        df_aligned['volume'] = df_aligned['volume'].fillna(0)
        df_aligned['factor'] = df_aligned['factor'].fillna(-1)
        
        cerebro.adddata(PandasDataWithFactor(dataname=df_aligned), name=symbol)
        add_count += 1
    
    logging.info(f"已添加 {add_count} 只股票")
    
    # 运行回测
    logging.info("5. 开始回测...")
    results = cerebro.run(preload=False, runonce=False)
    
    # 结果分析
    if results:
        strat = results[0]
        final_value = cerebro.broker.getvalue()
        print(f"\n{'='*50}")
        print(f"最终资金: {final_value:,.2f}")
        print(f"收益率: {(final_value/INITIAL_CASH-1)*100:.2f}%")
        
        if hasattr(strat, 'closed_trades') and strat.closed_trades:
            df_res = pd.DataFrame(strat.closed_trades)
            print(f"交易笔数: {len(df_res)}")
            print(f"胜率: {(df_res['pnl_net'] > 0).mean():.2%}")
            print(f"平均每笔收益: {df_res['pnl_net'].mean():.2f}")
            df_res.to_csv('trade_log_enhanced.csv', index=False)
            print(f"\n交易记录已保存至 trade_log_enhanced.csv")
        else:
            print("⚠️ 无交易记录")
    else:
        logging.error("回测返回空结果")