from Historic_Crypto import HistoricalData, Cryptocurrencies, LiveCryptoData

# 1. 获取可用的加密货币对列表
available_pairs = Cryptocurrencies().all_currencies()

# 2. 获取历史数据
# 参数说明：
# - 交易对（例如：BTC-USD）
# - 时间间隔（例如：86400 表示日数据）
# - 开始时间
# - 结束时间
btc_historical = HistoricalData(
    'BTC-USD',
    86400,
    '2020-01-01-00-00',
    '2020-02-01-00-00'
).retrieve_data()

# 3. 获取实时数据
live_price = LiveCryptoData('BTC-USD').return_data()


