# save as binance_kline_http_proxy.py
import requests, pandas as pd, time
from datetime import datetime

# 1. 这里换成你的 HTTP 代理地址
PROXY = {
    'http':  'http://127.0.0.1:7890',  # 按需改
    'https': 'http://127.0.0.1:7890',  # 没有用户/密码就留空
}

URL = 'https://api.binance.com/api/v3/klines'
SYMBOL, INTERVAL = 'BTCUSDT', '1m'
START_STR = '2025-11-05 22:00:00'
END_STR   = '2025-11-07 13:30:00'

def ts2ms(s): return int(pd.Timestamp(s).timestamp()*1000)
out, start = [], ts2ms(START_STR)
end = ts2ms(END_STR)

while start < end:
    params = dict(symbol=SYMBOL, interval=INTERVAL,
                  startTime=start, limit=1000)
    r = requests.get(URL, params=params, proxies=PROXY, timeout=10)
    if r.status_code != 200:
        print('err', r.status_code, r.text[:200]); break
    data = r.json()
    if not data: break
    out.extend([[pd.Timestamp(int(row[0]), unit='ms'), float(row[4])] for row in data])
    start = int(data[-1][6]) + 1
    time.sleep(.15)

df = (pd.DataFrame(out, columns=['minute_bucket','close'])
        .drop_duplicates()
        .sort_values('minute_bucket'))
df.to_csv('binance_btc_1m_2215.csv', index=False)
print('HTTP 代理拉取完成，共', len(df), '条记录 → binance_btc_1m_2215.csv')