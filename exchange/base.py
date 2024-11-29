import ccxt
import logging

class ExchangeBase:
    _instance = None
    _exchange = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExchangeBase, cls).__new__(cls)
        return cls._instance
    
    @property
    def exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            self._exchange = self._create_exchange()
        return self._exchange
    
    def _create_exchange(self):
        proxies = {
            'http': 'http://127.0.0.1:7890',
            'https': 'http://127.0.0.1:7890'
        }
        return ccxt.okx({
            'apiKey': 'your-api-key',
            'secret': 'your-secret',
            'password': 'your-password',
            'enableRateLimit': True,
            'proxies': proxies,
            'timeout': 30000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })
