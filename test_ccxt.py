import ccxt


class DeltaNeutralStrategy():
    def __init__(self):
        # 代理设置
        proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据您的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据您的实际代理地址修改
        }
        # 初始化交易所API（这里以binance为例）
        self.exchange = ccxt.okx({
            'apiKey': 'ba7f444f-e83e-4dd1-8507-bf8dd9033cbc',
            'secret': 'D5474EF76B0A7397BFD26B9656006480',
            'password': 'TgTB+pJoM!d20F',
            'enableRateLimit': True,
            'proxies': proxies,  # 添加代理设置
            'timeout': 30000,    # 设置超时时间为30秒
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })

        # 验证连接
        try:
            self.exchange.load_markets()
            print("交易所连接成功")
        except Exception as e:
            print(f"交易所连接失败: {str(e)}")
            raise e

if __name__ == "__main__":
    strategy = DeltaNeutralStrategy()
    balance = strategy.exchange.fetchFundingRateHistory (symbol='BTC/USDT:USDT',limit = 100)
    
    #postition = strategy.exchange.fetch_position('')
    print(balance)
    