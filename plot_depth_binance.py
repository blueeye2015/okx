import pandas as pd, matplotlib.pyplot as plt

price  = pd.read_csv('price1.csv', parse_dates=['minute_bucket'])
cancel = pd.read_csv('removed1.csv', parse_dates=['minute_bucket'])
kline  = pd.read_csv('binance_btc_1m_2215.csv', parse_dates=['minute_bucket'])

df = (price.merge(cancel, on='minute_bucket')
        .merge(kline, on='minute_bucket', how='outer')
        .sort_values('minute_bucket').interpolate())
df['net_removed'] = df['ask_orders_removed'] - df['ask_orders_added']

fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(df.minute_bucket, df.avg_price, label='depth avg_price', color='royalblue')
ax.plot(df.minute_bucket, df.close,     label='Binance close',   color='k', alpha=.6)
ax.set_ylabel('Price (USDT)'); ax.grid(alpha=.3)

ax2 = ax.twinx()
ax2.step(df.minute_bucket, df.net_removed, where='post', color='r', alpha=.7, label='net removed')
ax2.fill_between(df.minute_bucket, 0, df.net_removed,
                 where=(df.net_removed > 0), color='r', alpha=.2)
ax2.set_ylabel('Net removed ask orders')
ax.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.title('BTCUSDT 挂单净撤单 vs 币安成交价（代理拉取）')
plt.tight_layout(); 
plt.savefig('depth_vs_price.png', dpi=300, bbox_inches='tight')
print('已保存 → depth_vs_price.png')
