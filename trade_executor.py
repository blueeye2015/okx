import logging
import time
import os
from okex.Trade_api import TradeAPI # 确保您的库路径正确
from okex.Market_api import MarketAPI
from dotenv import load_dotenv

load_dotenv('.env')
# --- 您的API凭证 ---

API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
PASSPHRASSE = os.getenv('PASSPHRASE')
IS_DEMO = 0 # 1 for demo, 0 for real

# 代理配置 (如果不需要，请设为 None)
proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据你的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据你的实际代理地址修改
        } 

class TradingClient:
    def __init__(self):
        try:
            self.trade_api = TradeAPI(API_KEY, SECRET_KEY, PASSPHRASSE, False, str(IS_DEMO),proxies)
            self.market_api = MarketAPI(API_KEY, SECRET_KEY, PASSPHRASSE, False, str(IS_DEMO),proxies)
            logging.info("交易客户端初始化成功。")
        except Exception as e:
            logging.error(f"交易客户端初始化失败: {e}")
            self.trade_api = None
            self.market_api = None

    def place_limit_order(self, symbol, px, side, size, client_order_id):
        if not self.trade_api:
            logging.error("交易API未初始化，无法下单。")
            return None
        try:
            logging.warning(f"准备下单: {side.upper()} {size:.6f} {symbol} (Client ID: {client_order_id})")
            result = self.trade_api.place_order(
                instId=symbol,
                tdMode='cash',
                side=side,
                px=px,
                ordType='limit',
                sz=f"{size:.8f}", # 使用高精度字符串
                clOrdId=client_order_id
            )
            logging.info(f"下单API返回: {result}")
            if result and result.get('code') == '0':
                order_id = result['data'][0]['ordId']
                logging.warning(f"市价单 {side.upper()} {symbol} 提交成功! 交易所订单ID: {order_id}")
                return result
            else:
                error_msg = result.get('msg', '未知错误')
                logging.error(f"下单失败: {error_msg}")
                return None
        except Exception as e:
            logging.error(f"执行下单时发生异常: {e}")
            return None
        
    def place_market_order(self, symbol, side, size, client_order_id):
        if not self.trade_api:
            logging.error("交易API未初始化，无法下单。")
            return None
        try:
            logging.warning(f"准备下单: {side.upper()} {size:.6f} {symbol} (Client ID: {client_order_id})")
            result = self.trade_api.place_order(
                instId=symbol,
                tdMode='cash',
                side=side,
                ordType='market',
                sz=f"{size:.8f}", # 使用高精度字符串
                clOrdId=client_order_id
            )
            logging.info(f"下单API返回: {result}")
            if result and result.get('code') == '0':
                order_id = result['data'][0]['ordId']
                logging.warning(f"市价单 {side.upper()} {symbol} 提交成功! 交易所订单ID: {order_id}")
                return result
            else:
                error_msg = result.get('msg', '未知错误')
                logging.error(f"下单失败: {error_msg}")
                return None
        except Exception as e:
            logging.error(f"执行下单时发生异常: {e}")
            return None

    def check_order_status(self, symbol, client_order_id):
        if not self.trade_api:
            logging.error("交易API未初始化，无法查询订单。")
            return None
        try:
            result = self.trade_api.get_orders(instId=symbol, clOrdId=client_order_id)
            logging.info(f"查询订单 ({client_order_id}) API返回: {result}")
            if result and result.get('code') == '0' and result['data']:
                return result['data'][0] # 返回订单的详细信息
            else:
                logging.error(f"查询订单失败: {result.get('msg', '订单不存在或查询失败')}")
                return None
        except Exception as e:
            logging.error(f"查询订单时发生异常: {e}")
            return None
        
    def get_latest_prices(self, symbols_list):
        """
        【新!】获取一个或多个币种的最新价格。
        :param symbols_list: 一个包含交易对字符串的列表, e.g., ['BTC-USDT-SWAP', 'ETH-USDT-SWAP']
        :return: 一个字典, key是交易对, value是最新价格, e.g., {'BTC-USDT-SWAP': 70000.1, ...}
        """
        if not self.market_api:
            logging.error("行情API未初始化，无法获取价格。")
            return {}
        
        if not symbols_list:
            return {}

        try:
            # OKX API V5 不再支持在get_tickers的instId中传多个币种
            # 我们需要循环查询，但这个库可能封装了批量查询
            # 查阅您使用的库的文档，get_tickers的'instType'参数可以获取一类产品的行情
            # 最直接的方式是循环获取单个ticker
            
            prices_dict = {}
            logging.info(f"准备查询 {len(symbols_list)} 个币种的最新价格: {symbols_list}")

            # OKX V5 Get Tickers (获取所有行情) 的instType参数更适合批量获取
            # 但为了精确，我们循环获取单个币种的ticker
            for symbol in symbols_list:
                result = self.market_api.get_ticker(symbol)
                if result and result.get('code') == '0' and result['data']:
                    last_price = result['data'][0]['last']
                    prices_dict[symbol] = float(last_price)
                else:
                    logging.warning(f"未能获取到 [{symbol}] 的价格。API返回: {result.get('msg', '未知错误')}")
                
                time.sleep(0.1) # 每次查询之间短暂休眠，防止触发限频
            
            logging.info(f"价格获取完成: {prices_dict}")
            return prices_dict

        except Exception as e:
            logging.error(f"获取最新价格时发生异常: {e}")
            return {}

 