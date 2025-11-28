import matplotlib
matplotlib.use('Agg')   # 必须放第一行
import matplotlib.pyplot as plt
import backtrader as bt
import pandas as pd
import datetime as dt
import time
import clickhouse_connect
import os


# 1. 扩展 Lines
class PandasDataExtend(bt.feeds.PandasData):
    lines = ('net_removed', 'vwap_ask10', 'fill_ratio', 
             'sell_avg_size', 'funding_rate')
    params = tuple((col, -1) for col in lines)

CLICKHOUSE = {
    'host': 'localhost', 'port': 8123,
    'username': 'default', 'password': '12',
    'database': 'marketdata'
}
client = clickhouse_connect.get_client(**CLICKHOUSE)

# ------------------------------------------------------------------
# 实时取数函数（与回测版完全一致，只改时间范围）
# ------------------------------------------------------------------
def fetch_trade_depth(start: dt.datetime, end: dt.datetime):
    sql = f"""
    WITH
    ohlcv AS (
        SELECT
            toStartOfMinute(trade_time)               AS datetime,
            argMin(price, trade_time)                 AS open,
            max(price)                                AS high,
            min(price)                                AS low,
            argMax(price, trade_time)                 AS close,
            sum(quantity)                             AS volume,
            sumIf(quantity, buyer_order_maker=0)      AS taker_sell_qty,
            sumIf(quantity, buyer_order_maker=1)      AS taker_buy_qty,
            countIf(buyer_order_maker=0)              AS taker_sell_cnt
        FROM marketdata.trades
        WHERE symbol = 'BTC-USDT'
          AND trade_time >= '{start:%Y-%m-%d %H:%M:%S}'
          AND trade_time <  '{end:%Y-%m-%d %H:%M:%S}'
        GROUP BY toStartOfMinute(trade_time)
    ),
    ask1m AS (
        SELECT
            toStartOfMinute(event_time) AS datetime,
            sumIf(quantity, side='ask')  AS ask_add,
            sumIf(quantity, side='bid')  AS bid_add
        FROM marketdata.depth
        WHERE symbol = 'BTCUSDT'
          AND event_time >= '{start:%Y-%m-%d %H:%M:%S}'
          AND event_time <  '{end:%Y-%m-%d %H:%M:%S}'
        GROUP BY datetime
    ),
    ask10 AS (
        SELECT
            toStartOfMinute(event_time) AS datetime,
            sum(price * quantity) / sum(quantity) AS vwap_ask10
        FROM (
            SELECT event_time, price, quantity,
                   row_number() OVER (PARTITION BY toStartOfMinute(event_time) ORDER BY price ASC) AS rn
            FROM marketdata.depth
            WHERE symbol = 'BTCUSDT' AND side = 'ask'
              AND event_time >= '{start:%Y-%m-%d %H:%M:%S}'
              AND event_time <  '{end:%Y-%m-%d %H:%M:%S}'
        ) WHERE rn <= 10
        GROUP BY datetime
    )
    SELECT
        o.datetime as datetime,
        o.open, o.high, o.low, o.close, o.volume,
        o.taker_sell_qty, o.taker_buy_qty, o.taker_sell_cnt,
        COALESCE(a.ask_add,0) - COALESCE(a.bid_add,0) AS net_removed,
        s.vwap_ask10,
        0.0 AS fill_ratio,
        o.taker_sell_qty / NULLIF(o.taker_sell_cnt,0) AS sell_avg_size,
        0.0 AS funding_rate
    FROM ohlcv o
    LEFT JOIN ask1m a USING (datetime)
    LEFT JOIN ask10 s USING (datetime)
    ORDER BY datetime
    """
    df = client.query_df(sql)
    df.rename(columns={'minute_bucket': 'datetime'}, inplace=True)  # ← 新增
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df


# ------------------------------------------------------------------
# 策略类（与回测完全一致）
# ------------------------------------------------------------------
class RemoveFltStrategy(bt.Strategy):
    params = (
        ('remove_thr', -2000),
        ('fill_thr', 0.2),
        ('fund_thr', -0.0001),
        ('size_thr', 2.0),
        ('stop_ticks', 100),
        ('pos_max', 0.5),
    )

    def __init__(self):
        self.net_removed = self.data.net_removed
        self.vwap = self.data.vwap_ask10
        self.trade_price = self.data.close
        self.fill_ratio = self.data.fill_ratio
        self.sell_avg = self.data.sell_avg_size
        self.funding = self.data.funding_rate
        self.sell_avg_mean = bt.indicators.SimpleMovingAverage(self.sell_avg, period=20)
        self.sell_avg_std = bt.indicators.StdDev(self.sell_avg, period=20)

    def next(self):
        if len(self.sell_avg_std) == 0:
            return
        raw = (self.net_removed[0] < self.p.remove_thr and
               self.fill_ratio[0] > self.p.fill_thr and
               self.funding[0] > self.p.fund_thr)
        upper = self.sell_avg_mean[0] + self.p.size_thr * self.sell_avg_std[0]
        if self.sell_avg[0] > upper:
            raw = False
        if self.position.size > 0 and self.trade_price[0] < self.vwap[0] - self.p.stop_ticks:
            self.sell(size=self.position.size)
            return
        if raw and self.position.size == 0:
            size = min(self.p.pos_max, self.net_removed[0] / 5000)
            self.buy(size=size)
        if self.position.size > 0 and abs(self.vwap[0] - self.trade_price[0]) < 50:
            self.sell(size=self.position.size)


# ------------------------------------------------------------------
# 实时主循环
# ------------------------------------------------------------------
def run():
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=50)

    df = fetch_trade_depth(dt.datetime(2025, 11, 6, 23, 30),
                           dt.datetime(2025, 11, 7, 0, 30))
    data = PandasDataExtend(dataname=df,
                               open='open', high='high', low='low', close='close',
                               volume='volume', net_removed='net_removed',
                               vwap_ask10='vwap_ask10', fill_ratio='fill_ratio',
                               sell_avg_size='sell_avg_size', funding_rate='funding_rate')
    cerebro.adddata(data)
    cerebro.addstrategy(RemoveFltStrategy)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    results = cerebro.run()
    strat = results[0]
    # 防 None 处理
    sharpe_val = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    if sharpe_val is None:
        sharpe_val = 0.0

    print('=== 纯 trade+depth 回测 ===')
    print('夏普  : %.2f' % sharpe_val)
    print('最大回撤 : %.2f %%' % strat.analyzers.dd.get_analysis()['max']['drawdown'])

    # 安全取成交次数
    trades_an = strat.analyzers.trades.get_analysis()
    closed_cnt = trades_an.get('total', {}).get('closed', 0)   # 无成交时返回 0
    print('总成交 :', closed_cnt)

    # 其余不变
    cerebro.plot(style='line', iplot=False, savefig=True, path='bt_trade_depth.png')
    trade_list = [{'datetime': t.data.datetime.datetime(), 'pnl': t.pnl, 'pnlcomm': t.pnlcomm,
                'size': t.size, 'price': t.price} for t in strat._trades if t.isclosed]
    pd.DataFrame(trade_list).to_csv('bt_trade_depth_trades.csv', index=False)
    print('成交明细 → bt_trade_depth_trades.csv')

if __name__ == '__main__':
    run()