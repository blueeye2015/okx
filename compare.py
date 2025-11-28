import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# 1. 读你已有的两张表 ----------------------------------------------------------
price_df = pd.read_csv('price.csv', parse_dates=['minute_bucket'])
cancel_df = pd.read_csv('removed.csv', parse_dates=['minute_bucket'])

# 2. 拉取 Yahoo 1-min K 线（UTC 对齐）------------------------------------------
ticker = yf.Ticker("BTC-USD")
start = datetime(2025, 11, 5, 22, 0)
end   = datetime(2025, 11, 6, 15, 30)
kline = ticker.history(start=start, end=end, interval='1m', prepost=False)
kline.index = kline.index.tz_convert(None).rename('minute_bucket')

# 3. 合并三张表 ----------------------------------------------------------------
combo = (price_df
         .merge(cancel_df, on='minute_bucket', how='outer')
         .merge(kline[['Open', 'High', 'Low', 'Close']].reset_index(),
                on='minute_bucket', how='outer')
         .sort_values('minute_bucket')
         .interpolate())          # 少量缺失用线性插值补齐

# 4. 画图 ----------------------------------------------------------------------
plt.figure(figsize=(16,8))
ax1 = plt.subplot(2,1,1)
ax1.plot(combo.minute_bucket, combo.avg_price, label='Local avg_price', color='royalblue')
ax1.plot(combo.minute_bucket, combo.Close,     label='Yahoo Close',   color='black', alpha=.6)
ax1.set_ylabel('Price (USD)')
ax1.grid(alpha=.3)
ax1.legend(loc='upper left')
ax1.set_title('BTCUSDT 挂单均价 vs 市场成交价  (2025-11-05 22:00 ~ 11-06 15:00)')

ax2 = plt.subplot(2,1,2, sharex=ax1)
ax2.step(combo.minute_bucket, combo.ask_orders_added,   label='ask_orders_added',   where='post', color='g')
ax2.step(combo.minute_bucket, combo.ask_orders_removed, label='ask_orders_removed', where='post', color='r')
ax2.fill_between(combo.minute_bucket,
                 combo.ask_orders_removed,
                 combo.ask_orders_added,
                 where=(combo.ask_orders_removed > combo.ask_orders_added),
                 color='red', alpha=.2, label='net removed')
ax2.set_ylabel('Ask orders count')
ax2.grid(alpha=.3)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()