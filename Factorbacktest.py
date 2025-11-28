import backtrader as bt
import pandas as pd
import datetime

# ----------------------------------------------------------------------
# 1. 自定义数据源 (Data Feed)
# ----------------------------------------------------------------------
# 我们需要告诉 backtrader 我们的 CSV 中有哪些额外的列
class FactorData(bt.feeds.PandasData):
    """
    一个包含我们所有自定义因子的数据源
    """
    # 告诉 backtrader 'lines' (数据线) 上有哪些新因子
    lines = (
        'volume_delta',
        'volume_delta_ma5',
        'vwap_ask10',
        'vwap_bid10',
        'spread_10',
        'prev_vwap_bid',
        'avg_taker_buy_size',
        'avg_taker_sell_size',
        'avg_size_delta',
        'eaten_high_qty',
        'eaten_prev_high_qty',
        'new_high_qty',
        'prev_vwap',
        'fill_ratio',
        'net_removed',
    )

    # 将 'lines' 映射到 CSV 中的列名
    # (-1 表示自动按列名匹配)
    params = (
        ('volume_delta', -1),
        ('volume_delta_ma5', -1),
        ('vwap_ask10', -1),
        ('vwap_bid10', -1),
        ('spread_10', -1),
        ('prev_vwap_bid', -1),
        ('avg_taker_buy_size', -1),
        ('avg_taker_sell_size', -1),
        ('avg_size_delta', -1),
        ('eaten_high_qty', -1),
        ('eaten_prev_high_qty', -1),
        ('new_high_qty', -1),
        ('prev_vwap', -1),
        ('fill_ratio', -1),
        ('net_removed', -1),
    )

# ----------------------------------------------------------------------
# 2. 定义 "慢涨" 策略
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 2. 定义 "慢涨" 策略 (版本 2 - 优化)
# ----------------------------------------------------------------------
class SlowGrindStrategy(bt.Strategy):
    """
    一个专门捕捉 "慢涨" 行情的策略
    """
    params = (
        # <== 修改: 
        # 我们需要 delta_ma5 超过一个 "有意义" 的正阈值才入场
        # 这里的 1 代表 5分钟平均净买入量 > 1 个ETH (你需要根据标的调整)
        ('entry_delta_threshold', 1.0), 
        
        # <== 修改: 
        # 只要 delta_ma5 失去势头 (变为负数) 我们就离场
        # 这里的 -0.5 给了一个小的缓冲，防止频繁进出
        ('exit_delta_threshold', -0.5),
        
        ('print_log', True),
    )

    def __init__(self):
        # 方便地引用我们关心的数据线
        self.dataclose = self.datas[0].close
        
        # 引用我们的自定义因子
        self.delta_ma5 = self.datas[0].volume_delta_ma5
        self.vwap_bid10 = self.datas[0].vwap_bid10
        self.prev_vwap_bid = self.datas[0].prev_vwap_bid
        
        self.order = None # 用于跟踪订单状态

    def log(self, txt, dt=None):
            """ 打印日志的辅助函数 """
            if self.params.print_log:
                # <== 修正: 正确的 backtrader 获取 datetime 对象的方法
                dt = dt or self.datas[0].datetime.datetime(0) 
                print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """ 订单通知 """
        if order.status in [order.Submitted, order.Accepted]:
            return # 订单已提交/接受，无需操作

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        # 检查是否有待处理订单
        if self.order:
            return
            
        # (可选) 打印每一根K线的因子值，用于调试
        # self.log(f'Close: {self.dataclose[0]:.2f}, Delta_MA5: {self.delta_ma5[0]:.2f}, Vwap_Bid: {self.vwap_bid10[0]:.2f}, Prev_Vwap_Bid: {self.prev_vwap_bid[0]:.2f}')

        # 检查是否持仓
        if not self.position:
            # === 入场条件 (更严格) ===
            # 1. 5分钟净主动买入为正 (买方持续施压)
            condition1 = self.delta_ma5[0] > self.params.entry_delta_threshold
            # 2. 买10 VWAP 抬高 (被动支撑跟进)
            condition2 = self.vwap_bid10[0] > self.prev_vwap_bid[0]
            # 3. 确保 vwap_bid10 是有效值 (非0)
            condition3 = self.vwap_bid10[0] > 0 and self.prev_vwap_bid[0] > 0

            if condition1 and condition2 and condition3:
                self.log(f'BUY CREATE, Close: {self.dataclose[0]:.2f}, Delta_MA5: {self.delta_ma5[0]:.2f}')
                self.order = self.order_target_percent(target=0.95) 

        else:
            # === 离场条件 (更灵敏) ===
            # 1. 5分钟净主动买入势头消失 (转为负数)
            condition_exit = self.delta_ma5[0] < self.params.exit_delta_threshold

            if condition_exit:
                self.log(f'SELL CREATE (Exit), Close: {self.dataclose[0]:.2f}, Delta_MA5: {self.delta_ma5[0]:.2f}')
                self.order = self.order_target_percent(target=0)
# ----------------------------------------------------------------------
# 3. 运行回测
# ----------------------------------------------------------------------
if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # 添加策略
    cerebro.addstrategy(SlowGrindStrategy)

    # 加载数据
    print("正在加载数据...")
    df = pd.read_csv(
        'eth_factors_1min.csv', 
        index_col='datetime', 
        parse_dates=True
    )
    
    # 将数据馈送到 backtrader
    # 注意：我们使用自定义的 FactorData
    data = FactorData(
        dataname=df,
        fromdate=datetime.datetime(2025, 11, 12, 16, 0, 0), # 从你指定的时间开始
        # todate=... # 可以指定结束时间
    )
    
    cerebro.adddata(data)

    # 设置初始资金
    cerebro.broker.setcash(100_000.0)

    # 设置佣金 (根据你的用户摘要，你做加密货币，设置一个合理的费率)
    cerebro.broker.setcommission(commission=0.001) # 0.1%

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    

    # 运行回测
    print('正在运行回测...')
    results = cerebro.run()
    strat = results[0]
    returns_analysis = strat.analyzers.returns.get_analysis()
    
    # 打印最终结果
    final_value = cerebro.broker.getvalue()
    print(f'\n--- 回测结果 ---')
    print(f'初始资金: 100000.00')
    print(f'最终资金: {final_value:.2f}')
    print(f'总回报率: {(final_value - 100_000) / 100_000 * 100:.2f}%')
    
    # 打印分析器结果
    strat = results[0]
    print(f'夏普比率 (Sharpe): {strat.analyzers.sharpe_ratio.get_analysis().get("sharpe_ratio", "N/A")}')
    print(f'最大回撤 (DrawDown): {strat.analyzers.drawdown.get_analysis().max.drawdown:.2f}%')
    # <== 修正部分
    if 'rnorm100' in returns_analysis:
        # 如果有年化回报率，则打印
        print(f'年化回报率 (APR): {returns_analysis.rnorm100:.2f}%')
    else:
        # 否则，打印总回报率
        # rtot 是一个乘数 (例如 0.732), 转换为百分比
        total_return_pct = (returns_analysis.get('rtot', 1.0) - 1.0) * 100
        print(f'总回报率 (Total): {total_return_pct:.2f}% (周期太短，无法计算年化)')
    # <== 修正结束

    # 绘制图表 (需要安装 matplotlib: pip install matplotlib)
    print("正在绘制图表...")
    cerebro.plot(style='candlestick')