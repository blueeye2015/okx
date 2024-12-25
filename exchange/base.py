import ccxt
import logging
import os
from dotenv import load_dotenv

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
        load_dotenv('D:\OKex-API\.env')
        # 检查必要的环境变量是否存在
        required_env_vars = ['API_KEY', 'SECRET_KEY', 'PASSPHRASE']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        proxies = {
            'http': 'http://127.0.0.1:7890',
            'https': 'http://127.0.0.1:7890'
        }
        return ccxt.okx({
            'apiKey': os.getenv('API_KEY'),
            'secret': os.getenv('SECRET_KEY'),
            'password': os.getenv('PASSPHRASE'),
            'enableRateLimit': True,
            'proxies': proxies,
            'timeout': 30000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })
