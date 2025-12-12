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
REBALANCE_DAY_RANGE = (1, 5) ##1-5开仓
BENCHMARK_SYMBOL = '000300.SH'
ADJUST_TYPE = 'hfq'
CACHE_DIR = 'factor_cache'
INITIAL_CASH = 1000000.0

# --- 1. 自定义数据加载器 (不变) ---
class PandasDataWithFactor(bt.feeds.PandasData):
    lines = ('factor',)
    params = (('factor', -1),)

# ========================
# 辅助函数：根据股票代码判断涨跌幅限制
# ========================
def get_price_limit(symbol: str) -> float:
    """返回对应股票的涨跌幅限制比例"""
    if symbol.startswith(('300', '688')):  # 创业板、科创板
        return 0.20
    elif symbol.startswith('8'):           # 北交所（如需）
        return 0.30
    else:                                  # 主板
        return 0.10
    
# --- 2. 核心交易策略 (重大修改) ---
class MLFactorStrategy(bt.Strategy):
    params = dict(
        top_n_pct=0.02,          # 优化：更集中 0.1→0.05
        rebalance_monthday=2,    # 优化：月初调仓 20→1
        benchmark_symbol=BENCHMARK_SYMBOL,
        debug=False,             # 关闭调试日志提升速度
        bull_position=1.00,      # 新增：牛市仓位50%→80%
        bear_position=1.00,      # 熊市满仓
        # 新增：止损止盈
        stop_loss_pct=0.08,      # 15%强制止损
        take_profit_pct=0.25,    # 25%止盈
        min_daily_amount=50000000,  # 新增：最小日均成交额5000万
    )

    def __init__(self):
        self.benchmark = self.getdatabyname(self.p.benchmark_symbol)
        self.sma200 = bt.indicators.SimpleMovingAverage(self.benchmark.close, period=200)
        self.stocks = [d for d in self.datas if d._name != self.p.benchmark_symbol]
        # 移除next()中的调仓，仅保留timer机制
        self.add_timer(when=bt.timer.SESSION_END, monthdays=[self.p.rebalance_monthday], 
                       cheat=False, # 设为True可在当前bar成交，更接近实盘
                       )
        self.last_rebalance_month = -1
        self.closed_trades = []
        self._rebalance_count = 0  # 新增：调仓计数
        self.daily_holdings = []
        # 新增：记录入场价
        self.stock_entry_price = defaultdict(lambda: None)

        logging.info("策略初始化完成。")

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        symbol      = trade.data._name
        open_date   = bt.num2date(trade.dtopen).date()
        close_date  = bt.num2date(trade.dtclose).date()
        duration    = trade.barlen
        size        = int(trade.size)
        entry_price = round(trade.price, 4)

        # 安全获取exit_price
        try:
            exit_price = round(trade.history[-1].price, 4) if len(trade.history) > 0 else round(trade.data.close[0], 4)
        except:
            exit_price = round(trade.data.close[0], 4)

        pct_ret = round(exit_price / entry_price - 1, 4)
        pnl_gross = round(trade.pnl, 2)
        pnl_net   = round(trade.pnlcomm, 2)

        self.closed_trades.append({
            'symbol': symbol, 'open_date': open_date, 'close_date': close_date,
            'duration_days': duration, 'size': size, 'entry_price': entry_price,
            'exit_price': exit_price, 'return': pct_ret, 'pnl': pnl_gross, 'pnl_net': pnl_net
        })

        if self.p.debug:
            print(f'{open_date} -> {close_date}  {symbol}  '
                  f'{size}股  {entry_price} → {exit_price}  '
                  f'收益 {pct_ret:.2%}  净利 {pnl_net:.2f}')
                   
    # --- 删除next()中的调仓逻辑，避免重复触发 ---
    # --- 【关键修复】保留 next() 保持策略活跃 ---
    def next(self):
        """
        必须保留此方法以维持策略的正常bar处理循环。
        timer的触发依赖于next()的执行链。
        此处不直接调仓，仅做月度保险或日内风控。
        """
        # 保险机制：如果 timer 因节假日未触发，在月末强制调仓
        current_month = self.datetime.date(0).month
        current_day = self.datetime.date(0).day
        
        # 如果是月末最后交易日（保险逻辑），且本月未调仓
        if current_day >= 28 and self.last_rebalance_month != current_month:
            logging.warning(f"Timer未触发，月末保险调仓: {self.datetime.date(0)}")
            self.rebalance_portfolio()
            self.last_rebalance_month = current_month
        # 记录每日持仓用于事后分析
        holdings = {d._name: pos.size for d, pos in self.getpositions().items() if pos.size != 0}
        self.daily_holdings.append({'date': self.datetime.date(0), 'holdings': holdings})

        # --- 【严格执行】止损止盈 ---
        for data, pos in self.getpositions().items():
            if pos.size == 0: continue
            
            entry_price = self.stock_entry_price.get(data._name)
            if entry_price is None: continue
            
            current_price = data.close[0]
            ret = current_price / entry_price - 1
            
            # 止损（8%严格止损）
            if ret < -self.p.stop_loss_pct:
                logging.info(f"止损: {data._name} 亏损 {ret:.2%}")
                self.order_target_percent(data=data, target=0.0)
                self.stock_entry_price[data._name] = None  # 清除记录
                continue
            
            # 止盈（20%锁定利润）
            if ret > self.p.take_profit_pct:
                logging.info(f"止盈: {data._name} 盈利 {ret:.2%}")
                self.order_target_percent(data=data, target=0.0)
                self.stock_entry_price[data._name] = None  # 清除记录

    def notify_timer(self, timer, when, *args, **kwargs):
        """主调仓逻辑，由timer精确控制"""
        current_month = self.datetime.date(0).month
        if self.last_rebalance_month != current_month:
            logging.info(f"\nTimer触发调仓: {self.datetime.date(0)}")
            self.rebalance_portfolio()
            self.last_rebalance_month = current_month

    def _is_limit_up(self, data):
																 
        if len(data) < 2:
            return False
        symbol = data._name
        limit_pct = get_price_limit(symbol)
        prev_close = data.close[-1]
        limit_up = round(prev_close * (1 + limit_pct), 2)
											 
        return data.high[0] >= limit_up - 1e-5

    def _is_limit_down(self, data):
									  
        if len(data) < 2:
            return False
        symbol = data._name
        limit_pct = get_price_limit(symbol)
        prev_close = data.close[-1]
        limit_down = round(prev_close * (1 - limit_pct), 2)
        return data.low[0] <= limit_down + 1e-5

    def rebalance_portfolio(self):
        self._rebalance_count += 1
        current_date = self.datetime.date(0)
        rankings = []
       
        for d in self.stocks:
            if len(d) > 0 and not np.isnan(d.factor[0]) and d.factor[0] > -1:
                rankings.append((d.factor[0], d))
        
        rankings.sort(key=lambda x: x[0], reverse=True)
        if not rankings: 
            logging.warning(f"{current_date} 无有效因子数据，跳过调仓")
            return

        top_n = int(len(self.stocks) * self.p.top_n_pct)
        target_stocks_full = [d for score, d in rankings[:top_n]]
        
        # --- 【核心新增】涨停不买逻辑 ---
        target_stocks = []
        for d in target_stocks_full:
            if self._is_limit_up(d):
                logging.info(f"{current_date} {d._name} 涨停({d.close[0]:.2f})，剔除买入名单")
                continue
            target_stocks.append(d)
        
        if not target_stocks:
            logging.warning(f"{current_date} 所有目标股票均涨停，无买入标的")
            target_stocks = []  # 将持有现金

        target_stock_names = {d._name for d in target_stocks}

        is_bull_market = self.benchmark.close[0] > self.sma200[0]
        total_portfolio_weight = 1.0 if is_bull_market else 1.0

        logging.info(f"\n调仓日: {current_date} - 市场状态: {'牛市(50%仓位)' if is_bull_market else '熊市(100%仓位)'}")
        logging.info(f"目标持仓股票数量: {len(target_stocks)}/共{len(target_stocks_full)}只(涨停剔除{len(target_stocks_full)-len(target_stocks)}只)")
        
        target_pct_per_stock = total_portfolio_weight / len(target_stocks) if target_stocks else 0
        
        # --- 【核心新增】跌停不卖逻辑 ---
        for data, pos in self.getpositions().items():
            if pos.size == 0: 
                continue
            
            # 如果股票在目标列表外，或跌停，则清仓
            if data._name not in target_stock_names:
                if self._is_limit_down(data):
                    logging.info(f"{current_date} {data._name} 跌停({data.close[0]:.2f})，暂停卖出")
                    continue  # 跌停不卖出，保留持仓至下周期
                logging.info(f"卖出: {data._name} (清空现有持仓)")
                self.order_target_percent(data=data, target=0.0)

        # 买入/调整目标股票的仓位
        for d in target_stocks:
            self.order_target_percent(data=d, target=target_pct_per_stock)

    def stop(self):
        # --- 删除有bug的IC分析，用简单统计 ---
        logging.info("\n--- 回测结束 ---")
        final_value = self.broker.getvalue()
        logging.info(f"初始: {INITIAL_CASH:,.0f}, 最终: {final_value:,.0f}, 收益: {(final_value/INITIAL_CASH-1):.2%}")
        
        print("\n" + "="*50)
        print("--- 收益归因诊断 ---")
        print(f"总调仓次数: {self._rebalance_count}")
        
        # 持仓分析
        if self.daily_holdings:
            avg_holdings = np.mean([len(h['holdings']) for h in self.daily_holdings])
            print(f"平均每日持仓: {avg_holdings:.1f}只股票")
        
        # 交易分析
        if self.closed_trades:
            df_trades = pd.DataFrame(self.closed_trades)
            print(f"总交易笔数: {len(df_trades)}")
            print(f"平均每笔收益: {df_trades['pnl_net'].mean():.2f}")
            print(f"胜率: {(df_trades['pnl_net']>0).mean():.2%}")
        
        # 检查股票覆盖
        traded_symbols = set(t['symbol'] for t in self.closed_trades)
        print(f"股票池覆盖: {len(traded_symbols)}/{len(self.stocks)}只曾被持有")
        
        # 关键：平均持仓天数判断因子周期
        if self.closed_trades:
            df_trades['duration_days'] = (pd.to_datetime(df_trades['close_date']) - 
                                          pd.to_datetime(df_trades['open_date'])).dt.days
            avg_duration = df_trades['duration_days'].mean()
            print(f"\n平均持仓天数: {avg_duration:.1f}天")
            
            if avg_duration >= 40:
                print("✓ 因子适合拉长周期（双月/季度调仓）")
            elif avg_duration >= 20:
                print("○ 因子适合月度调仓")
            else:
                print("✗ 因子适合短周期（双周/周度调仓）")
        

# --- A股精细交易成本模型 ---
class AStockCommission(bt.CommInfoBase):
    params = (
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),
    )
    
    def _getcommission(self, size, price, pseudoexec):
        turnover = abs(size) * price
        if size > 0:  # 买入：佣金万3（不足5元按5元，此处简化）
            return turnover * 0.0003
        else:         # 卖出：佣金万3 + 印花税千1
            return turnover * (0.0003 + 0.001)
        
# --- 3. 主执行逻辑 ---  
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    
    logging.info("开始数据准备...")
    chunk_files = glob.glob(os.path.join(CACHE_DIR, '*.parquet'))
    if not chunk_files:
        logging.error(f"在目录 '{CACHE_DIR}' 中未找到任何缓存文件。")
        exit(1)

    # 方案B：读取时立即转换日期类型
    all_chunks = []
    for f in chunk_files:
        chunk = pd.read_parquet(f, columns=['trade_date', 'symbol', 'factor'])
        # 确保日期类型正确
        chunk['trade_date'] = pd.to_datetime(chunk['trade_date'], errors='coerce')
        if chunk['trade_date'].isna().any():
            logging.warning(f"文件 {f} 中包含无效日期，已被转换为NaT")
        all_chunks.append(chunk)
    
    df_factor = pd.concat(all_chunks, ignore_index=True)
    logging.info(f"成功合并 {len(all_chunks)} 个分块文件，共计 {len(df_factor)} 行因子数据。")

    symbols_to_run = df_factor['symbol'].unique().tolist()
    start_date = df_factor['trade_date'].min().strftime('%Y-%m-%d')
    end_date = df_factor['trade_date'].max().strftime('%Y-%m-%d')
    if BENCHMARK_SYMBOL not in symbols_to_run:
        symbols_to_run.append(BENCHMARK_SYMBOL)

    # 提前过滤股票列表（修复MAX_STOCKS_TO_RUN无效问题）
    MAX_STOCKS_TO_RUN = None
    if MAX_STOCKS_TO_RUN:
        stock_symbols_list = [s for s in symbols_to_run if s != BENCHMARK_SYMBOL][:MAX_STOCKS_TO_RUN]
        logging.warning(f"快速测试模式：仅回测前 {MAX_STOCKS_TO_RUN} 只股票")
    else:
        stock_symbols_list = [s for s in symbols_to_run if s != BENCHMARK_SYMBOL]

    try:
        conn = psycopg2.connect(POSTGRES_CONFIG)
        
        # 修复SQL注入风险
        if not stock_symbols_list:
            logging.error("股票列表为空，无法查询价格数据")
            exit(1)
        
        placeholders = ','.join(['%s'] * len(stock_symbols_list))
        all_stocks_query = f"""
            SELECT trade_date, symbol, open, high, low, close 
            FROM public.stock_history 
            WHERE symbol IN ({placeholders}) 
              AND trade_date BETWEEN %s AND %s 
              AND adjust_type = %s
        """
        df_stocks = pd.read_sql_query(all_stocks_query, conn, 
                                      params=[*stock_symbols_list, start_date, end_date, ADJUST_TYPE])
        
        index_price_query = """
            SELECT trade_date, ts_code AS symbol, open, high, low, close 
            FROM public.index_daily 
            WHERE ts_code = %s AND trade_date BETWEEN %s AND %s
        """
        df_index = pd.read_sql_query(index_price_query, conn, 
                                     params=[BENCHMARK_SYMBOL, start_date, end_date])
        
        conn.close()
        df_prices = pd.concat([df_stocks, df_index], ignore_index=True)
        
        # 方案A：合并前统一转换（双重保险）
        df_prices['trade_date'] = pd.to_datetime(df_prices['trade_date'], errors='coerce')
        df_factor['trade_date'] = pd.to_datetime(df_factor['trade_date'], errors='coerce')
        
        logging.info(f"获取价格数据完成，共 {len(df_prices)} 行。")
    except Exception as e:
        logging.error(f"数据库查询失败: {e}", exc_info=True)
        exit(1)
    
    # 合并数据（此时类型已一致，不会报错）
    df_all_data = pd.merge(df_prices, df_factor, on=['trade_date', 'symbol'], how='left')
    df_all_data['factor'].fillna(-1, inplace=True)
    df_all_data.set_index('trade_date', inplace=True)
    df_all_data.sort_index(inplace=True)
    
    del df_prices,df_factor,df_stocks,df_index
    gc.collect()
    # --- Cerebro引擎设置 ---
    cerebro = bt.Cerebro()

    # 添加精细交易成本
    cerebro.broker.addcommissioninfo(AStockCommission())

    # 添加滑点模型
    cerebro.broker.set_slippage_perc(
        perc=0.001,  # 0.1%滑点
        slip_open=True,
        slip_limit=True,
        slip_match=True,
        slip_out=False
    )
    
    logging.info(f"添加 {len(symbols_to_run)} 个数据源...")
    t0 = time.time()
    # 为每个symbol预处理数据
    symbol_data_map = {}
    for symbol in symbols_to_run:
        df_sym = df_all_data[df_all_data['symbol'] == symbol][['open', 'high', 'low', 'close', 'factor']]
        if df_sym.empty:
            logging.warning(f"{symbol} 无数据，跳过")
            continue
        
        # 预处理数据
        df_sym['volume'] = 0
        df_sym.sort_index(inplace=True)  # 确保数据按时间排序
        
        symbol_data_map[symbol] = df_sym
    t1 = time.time()
    logging.info(f"预处理所有 symbol 数据: {t1 - t0:.2f}s")
    # 批量添加数据
    for symbol, df_sym in symbol_data_map.items():
        
        data_feed = PandasDataWithFactor(dataname=df_sym)
        cerebro.adddata(data_feed, name=symbol)
    t2 = time.time()
    logging.info(f"【耗时】查询价格数据: {t2 - t1:.2f}s")
    cerebro.addstrategy(MLFactorStrategy)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.001, stocklike=True)
    cerebro.broker.set_shortcash(False)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    logging.info("一切准备就绪，开始回测...")
    results = cerebro.run()
    
    # --- 打印分析结果 ---
    strat = results[0]

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
    