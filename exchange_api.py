import time
import requests
import logging
import base64
from urllib.parse import urlencode, quote
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import Config

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.session.proxies.update(Config.PROXIES)
        self.time_offset = 0
        self.sync_server_time()

    def sync_server_time(self):
        try:
            res = requests.get(f"{Config.BASE_URL}/fapi/v1/time", proxies=Config.PROXIES, timeout=5)
            server_time = res.json()['serverTime']
            self.time_offset = server_time - int(time.time() * 1000)
            logger.info(f"⏳ 时间校准完成，偏移: {self.time_offset}ms")
        except Exception as e:
            logger.error(f"❌ 时间校准失败: {e}")

    def _get_signature(self, payload):
        private_key_str = Config.SECRET_KEY
        if not private_key_str.startswith("-----BEGIN"):
            private_key_str = f"-----BEGIN PRIVATE KEY-----\n{private_key_str}\n-----END PRIVATE KEY-----"
        private_key = load_pem_private_key(private_key_str.encode('utf-8'), password=None)
        signature = private_key.sign(payload.encode('utf-8'))
        return base64.b64encode(signature).decode('utf-8')

    def send_request(self, method, endpoint, params=None):
        if params is None: params = {}
        params['timestamp'] = int(time.time() * 1000) + self.time_offset
        params['recvWindow'] = 60000
        query_string = urlencode(params)
        signature = self._get_signature(query_string)
        url = f"{Config.BASE_URL}{endpoint}?{query_string}&signature={quote(signature)}"
        headers = {'X-MBX-APIKEY': Config.API_KEY, 'Content-Type': 'application/json'}
        
        try:
            response = self.session.request(method, url, headers=headers, timeout=10)
            if response.status_code >= 400:
                logger.error(f"API Error ({response.status_code}): {response.text}")
                return None
            return response.json()
        except Exception as e:
            logger.warning(f"⚠️ 网络请求异常: {e}")
            return None

    def get_market_price(self, symbol):
        res = self.send_request('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
        return float(res['price']) if res else None

    def place_order(self, symbol, side, quantity, order_type='MARKET', price=None, reduce_only=False, stop_price=None):
        params = {
            'symbol': symbol, 
            'side': side, 
            'type': order_type, 
            'quantity': str(quantity) # 转换成字符串防止科学计数法
        }
        if reduce_only: params['reduceOnly'] = 'true'
        if price: params['price'] = str(price)
        if order_type == 'LIMIT': params['timeInForce'] = 'GTC'
            
        return self.send_request('POST', '/fapi/v1/order', params)

    def cancel_all_orders(self, symbol):
        return self.send_request('DELETE', '/fapi/v1/allOpenOrders', {'symbol': symbol})
    
    # exchange_api.py 新增方法

    def get_position_risk(self, symbol):
        """获取指定交易对的实时持仓、均价和风险情况"""
        # 注意使用 v2 接口获取更详细的数据
        res = self.send_request('GET', '/fapi/v2/positionRisk', {'symbol': symbol})
        if res and isinstance(res, list):
            for pos in res:
                if pos['symbol'] == symbol:
                    return {
                        'amount': abs(float(pos['positionAmt'])),
                        'entry_price': float(pos['entryPrice']),
                        'liquidation_price': float(pos['liquidationPrice']),
                        'unrealized_pnl': float(pos['unRealizedProfit']),
                        'margin_type': pos['marginType'], # 确认是 cross 还是 isolated
                        'leverage': pos['leverage']
                    }
        return None