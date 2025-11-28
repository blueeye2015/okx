import pandas as pd
import clickhouse_connect

CLICKHOUSE = dict(
    host='localhost',
    port=8123,
    database='marketdata',
    username='default',
    password='12'
)
client = clickhouse_connect.get_client(**CLICKHOUSE)

# 1. 基准快照（<= id_start 的最新一张）
snap = client.query(f'''
    SELECT * FROM depth_snapshot
    WHERE symbol = 'BTCUSDT' AND update_id <= {79721209118}
    ORDER BY update_id DESC LIMIT 1
''').first_item

# 2. 顺序拉 diff
df = client.query_df(f'''
    SELECT event_time, side, price, quantity, final_update_id
    FROM depth
    WHERE symbol = 'BTCUSDT'
      AND final_update_id > {79721209118}
      AND final_update_id <= {79738761011}
    ORDER BY final_update_id
''')

# 3. 内存镜像 + 回放
mirror_bid = {}   # price -> qty
mirror_ask = {}
bid1, ask1 = [], []

for row in df.itertuples():
    px, qty = row.price, row.quantity
    if row.side == 'bid':
        mirror_bid[px] = qty
        if qty == 0:
            mirror_bid.pop(px, None)
    else:
        mirror_ask[px] = qty
        if qty == 0:
            mirror_ask.pop(px, None)

    # 记录当前买一 / 卖一
    bid_top = max(mirror_bid) if mirror_bid else None
    ask_top = min(mirror_ask) if mirror_ask else None
    bid1.append((row.event_time, bid_top, mirror_bid.get(bid_top)))
    ask1.append((row.event_time, ask_top, mirror_ask.get(ask_top)))

# 生成 DataFrame
bid_df = pd.DataFrame(bid1, columns=['t', 'bid_px', 'bid_qty'])
ask_df = pd.DataFrame(ask1, columns=['t', 'ask_px', 'ask_qty'])

import matplotlib.pyplot as plt
plt.plot(bid_df.t, bid_df.bid_px, label='Bid-1')
plt.plot(ask_df.t, ask_df.ask_px, label='Ask-1')
plt.axvline(pd.Timestamp('2025-11-05 00:15:00'), color='green', linestyle='--', label='绿柱时刻')
plt.savefig('/data/okx/bid_ask_1min.png', dpi=150, bbox_inches='tight')
print('图已保存为 /data/okx/bid_ask_1min.png')
import matplotlib.dates as mdates
tz = bid_df.t.dt.tz
# 1. 只取 10 分钟
center = pd.Timestamp('2025-11-05 00:15:00').tz_localize(tz)
start  = center - pd.Timedelta(minutes=5)
end    = center + pd.Timedelta(minutes=5)

bid_z = bid_df[(bid_df.t >= start) & (bid_df.t <= end)]
ask_z = ask_df[(ask_df.t >= start) & (ask_df.t <= end)]

# 2. 画图
plt.figure(figsize=(12, 4))
plt.plot(bid_z.t, bid_z.bid_px, label='Bid-1')
plt.plot(ask_z.t, ask_z.ask_px, label='Ask-1')
plt.axvline(center, color='green', linestyle='--')

# 3. 稀疏刻度
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('/data/okx/clear_10min.png', dpi=200)