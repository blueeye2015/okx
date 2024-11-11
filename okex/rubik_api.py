from .client import Client
from .consts import *

class RubikApi(Client):
    def __init__(self, api_key, api_secret_key, passphrase, use_server_time=False, flag='1', proxies=None):
        super().__init__(api_key, api_secret_key, passphrase, use_server_time, flag, proxies)
        
    # Get Candlesticks
    def take_volume(self, ccy, instType, begin='', end='', period=''):
        params = {'ccy': ccy, 'instType' : instType, 'begin': begin, 'end': end, 'period': period}
        return self._request_with_params(GET, TAKE_VOLUME, params)
    
    