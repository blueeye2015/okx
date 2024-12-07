import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

from exchange.base import ExchangeBase

class MarketAnalyzer(ExchangeBase):
    """
    市场分析器
    负责分析市场数据，获取符合条件的交易对
    """
    
    def __init__(self, proxy_host: str = "127.0.0.1", proxy_port: str = "7890"):
        """
        初始化市场分析器
        
        Args:
            proxy_host (str): 代理服务器主机地址
            proxy_port (str): 代理服务器端口
        """
        super().__init__()
        self.cache: Dict[str, Dict] = {}
        self.cache_timeout = 3600  # 缓存1小时
        self.last_cache_time: Optional[float] = None
        self.proxies = {
            'http': f'http://{proxy_host}:{proxy_port}',
            'https': f'http://{proxy_host}:{proxy_port}'
        }
        
    def _is_cache_valid(self) -> bool:
        """
        检查缓存是否有效
        
        Returns:
            bool: 如果缓存有效返回True，否则返回False
        """
        if not self.last_cache_time:
            return False
        return time.time() - self.last_cache_time < self.cache_timeout
    
    def _update_cache(self, data: Dict[str, Dict]) -> None:
        """
        更新缓存数据
        
        Args:
            data (Dict[str, Dict]): 新的缓存数据
        """
        self.cache = data
        self.last_cache_time = time.time()
                    
    def get_market_cap_data(self) -> Dict[str, Dict]:
        """
        从 CoinGecko 获取市值数据
        
        Returns:
            Dict[str, Dict]: 市值数据，格式如：
            {
                'BTC': {
                    'market_cap': 800000000000,
                    'first_listed': '2009-01-03'
                },
                ...
            }
        
        Raises:
            requests.RequestException: 当API请求失败时
        """
        # 如果缓存有效，直接返回缓存数据
        if self._is_cache_valid():
            return self.cache
            
        try:
            # 1. 首先获取 OKX 的最小下单数量数据
            okx_url = "https://www.okx.com/api/v5/public/instruments"
            okx_params = {
                'instType': 'SPOT'
            }
            
            okx_response = requests.get(
                okx_url,
                proxies=self.proxies,
                params=okx_params,
                timeout=10
            )
            okx_response.raise_for_status()
            okx_data = okx_response.json()
            
            # 创建最小下单量字典
            min_size_dict = {
                item['baseCcy']: float(item['minSz'])
                for item in okx_data.get('data', [])
                if item['quoteCcy'] == 'USDT'  # 只考虑 USDT 交易对
            }
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 250,  # 获取前250个币种
                'page': 1,
                'sparkline': False
            }
            
            response = requests.get(
                url,
                proxies=self.proxies,
                params=params,
                timeout=10  # 设置超时时间
            )
            response.raise_for_status()  # 检查响应状态
            data = response.json()
            
            # 3. 处理数据并更新缓存
            market_data = {}
            excluded_coins = []
            
            for item in data:
                if item['market_cap'] is None:
                    continue
                    
                symbol = item['symbol'].upper()
                current_price = item.get('current_price', 0)
                
                # 计算最小购买金额
                min_size = min_size_dict.get(symbol, 0)
                min_purchase_amount = current_price * min_size if min_size > 0 else float('inf')
                
                # 如果最小购买金额大于3USDT，排除该币种
                if min_purchase_amount > 3:
                    excluded_coins.append(f"{symbol}(${min_purchase_amount:.2f})")
                    continue
                    
                market_data[symbol] = {
                    'market_cap': item['market_cap'],
                    'first_listed': item.get('genesis_date')
                }
            
            # 记录被排除的币种
            if excluded_coins:
                logging.info(f"因最小购买金额超过3USDT被排除的币种: {', '.join(excluded_coins)}")
            
            self._update_cache(market_data)
            return market_data
            
        except requests.RequestException as e:
            logging.error(f"获取市值数据失败: {str(e)}")
            # 如果请求失败但有缓存，返回缓存数据
            if self.cache:
                logging.warning("使用缓存的市值数据")
                return self.cache
            return {}

    def get_valid_symbols(self, min_market_cap: float = 5000000, min_age_months: int = 1) -> List[str]:
        """
        获取符合条件的交易对
        
        Args:
            min_market_cap (float): 最小市值（美元）
            min_age_months (int): 最小上市月数
            
        Returns:
            List[str]: 符合条件的交易对列表
        """
        try:
            # 获取交易所支持的所有交易对
            markets = self.exchange.load_markets()
            
            # 获取市值数据
            market_cap_data = self.get_market_cap_data()
            
            # 当前时间
            current_time = datetime.now()
            min_list_date = current_time - timedelta(days=90 * min_age_months)
            
            valid_symbols = []
            
            for symbol, market in markets.items():
                try:
                    # 只考虑USDT交易对
                    if not symbol.endswith('/USDT'):
                        continue
                        
                    base_currency = market['base']  # 基础货币 (例如 BTC, ETH)
                    
                    # 检查是否有市值数据
                    if base_currency not in market_cap_data:
                        continue
                        
                    market_info = market_cap_data[base_currency]
                    
                    # 检查市值
                    if market_info['market_cap'] < min_market_cap:
                        continue
                        
                    # 检查上市时间
                    if market_info['first_listed']:
                        list_date = datetime.strptime(market_info['first_listed'], '%Y-%m-%d')
                        if list_date > min_list_date:
                            continue
                            
                    # 将交易所格式转换为我们的格式 (BTC/USDT -> BTC-USDT)
                    formatted_symbol = symbol.replace('/', '-')
                    valid_symbols.append(formatted_symbol)
                    
                except Exception as e:
                    logging.warning(f"处理交易对 {symbol} 时出错: {str(e)}")
                    continue
            
            logging.info(f"找到 {len(valid_symbols)} 个符合条件的交易对")
            return valid_symbols
            
        except Exception as e:
            logging.error(f"获取有效交易对时出错: {str(e)}")
            return []
            
    def analyze_market_trend(self, symbol: str, days: int = 7) -> Dict:
        """
        分析市场趋势（示例方法）
        
        Args:
            symbol (str): 交易对符号
            days (int): 分析的天数
            
        Returns:
            Dict: 分析结果
        """
        # 这里可以添加更多的市场分析逻辑
        pass
