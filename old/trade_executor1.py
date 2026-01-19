# trade_executor.py
import logging
import time
import os
from okex.Trade_api import TradeAPI 
from okex.Market_api import MarketAPI
from okex.Account_api import AccountAPI
from dotenv import load_dotenv

load_dotenv('.env')


class TradingClient:
    def __init__(self):
        try:
            # --- API凭证 ---
            API_KEY = os.getenv('API_KEY')
            SECRET_KEY = os.getenv('SECRET_KEY')
            PASSPHRASE = os.getenv('PASSPHRASE')
            IS_DEMO = '0' # 默认为实盘
            proxies = { 'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890' } 

            self.trade_api = TradeAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, IS_DEMO , proxies=proxies)
            self.market_api = MarketAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, IS_DEMO, proxies=proxies)
            self.account_api = AccountAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, IS_DEMO, proxies=proxies)
            logging.info(f"交易客户端初始化成功。模式: {'模拟' if IS_DEMO == '1' else '实盘'}")
        except Exception as e:
            logging.error(f"交易客户端初始化失败: {e}")
            raise # 初始化失败时直接抛出异常，让主程序停止

    def get_usdt_balance(self):
        """查询交易账户中可用的USDT余额。"""
        try:
            res = self.account_api.get_account()
            if res and res.get('code') == '0':
                for detail in res['data'][0]['details']:
                    if detail['ccy'] == 'USDT':
                        avail_balance = float(detail['availBal'])
                        logging.info(f"查询到可用USDT余额: {avail_balance}")
                        return avail_balance
            logging.error(f"查询USDT余额失败: {res.get('msg', '未知错误')}")
            return 0.0
        except Exception as e:
            logging.error(f"查询余额时发生异常: {e}", exc_info=True)
            return 0.0
        
    def place_market_order_by_amount(self, symbol, side, amount_usdt, client_order_id):
        """按金额（USDT）下市价单"""
        try:
            logging.warning(f"准备按金额下单: {side.upper()} {amount_usdt} USDT worth of {symbol} (Client ID: {client_order_id})")
            result = self.trade_api.place_order(
                instId=symbol,
                tdMode='cash', # 使用全仓模式
                side=side,
                ordType='market',
                sz=str(amount_usdt), 
                tgtCcy='quote_ccy' if side=='buy' else 'base_ccy' , # 指定sz的单位是计价货币(USDT)
                clOrdId=client_order_id
            )
            logging.info(f"下单API返回: {result}")
            return result
        except Exception as e:
            logging.error(f"执行按金额下单时发生异常: {e}")
            return None

    def place_market_order_by_size(self, symbol, side, size, client_order_id):
        """按数量（币）下市价单"""
        try:
            logging.warning(f"准备按数量下单: {side.upper()} {size} of {symbol} (Client ID: {client_order_id})")
            result = self.trade_api.place_order(
                instId=symbol,
                tdMode='cash',
                side=side,
                ordType='market',
                sz=f"{size:.8f}", # 确保币的数量有足够精度
                tgtCcy='quote_ccy' if side=='buy' else 'base_ccy' , # 指定sz的单位是计价货币(USDT)
                clOrdId=client_order_id
            )
            logging.info(f"下单API返回: {result}")
            return result
        except Exception as e:
            logging.error(f"执行按数量下单时发生异常: {e}")
            return None
        
    # ###################### 【核心修正】统一的限价单函数 ######################
    def place_limit_order(self, symbol, side, size, price, client_order_id):
        """
        按指定的数量（币）和价格下限价单。
        """
        try:
            logging.warning(f"准备下限价单: {side.upper()} {size:.8f} of {symbol} at price {price} (Client ID: {client_order_id})")
            result = self.trade_api.place_order(
                instId=symbol,
                tdMode='cash',
                side=side,
                ordType='limit',
                sz=f"{size:.8f}", # sz 永远是币的数量
                px=str(price),    # 价格
                clOrdId=client_order_id
            )
            logging.info(f"下单API返回: {result}")
            return result
        except Exception as e:
            logging.error(f"执行限价单时发生异常: {e}", exc_info=True)
            return None
    # #####################################################################

    def place_limit_order_by_size(self, symbol, side, size, client_order_id):
        """按数量（币）下市价单"""
        try:
            logging.warning(f"准备按数量下单: {side.upper()} {size} of {symbol} (Client ID: {client_order_id})")
            result = self.trade_api.place_order(
                instId=symbol,
                tdMode='cash',
                side=side,
                ordType='limit',
                sz=f"{size:.8f}", # 确保币的数量有足够精度
                clOrdId=client_order_id
            )
            logging.info(f"下单API返回: {result}")
            return result
        except Exception as e:
            logging.error(f"执行按数量下单时发生异常: {e}")
            return None
            
    # ###################### 【核心修正】获取单个币种价格的函数 ######################
    def get_latest_price(self, symbol: str):
        """
        获取单个币种的最新成交价。
        """
        if not symbol: 
            return None
        try:
            result = self.market_api.get_ticker(symbol)
            if result and result.get('code') == '0' and result['data']:
                return float(result['data'][0]['last'])
            else:
                logging.error(f"获取 [{symbol}] 价格失败: {result.get('msg', '未知错误')}")
                return None
        except Exception as e:
            logging.error(f"获取 [{symbol}] 价格时发生异常: {e}")
            return None
    # #####################################################################
        # # 为提高效率，一次性获取所有SWAP产品的ticker
        # try:
        #     all_tickers_result = self.market_api.get_tickers('SPOT')
        #     if all_tickers_result.get('code') == '0':
        #         tickers_data = {ticker['instId']: float(ticker['last']) for ticker in all_tickers_result['data']}
        #         for symbol in symbols:
        #             if symbol in tickers_data:
        #                 prices_dict[symbol] = tickers_data[symbol]
        #             else:
        #                 logging.warning(f"在批量获取的行情中未找到 [{symbol}] 的价格。")
        #     else:
        #         logging.error(f"批量获取行情失败: {all_tickers_result.get('msg')}")
        #         # 降级为单个查询
        #         for symbol in symbols_list:
        #             result = self.market_api.get_ticker(symbol)
        #             if result and result.get('code') == '0' and result['data']:
        #                 prices_dict[symbol] = float(result['data'][0]['last'])
        #             time.sleep(0.1)

        # except Exception as e:
        #     logging.error(f"获取最新价格时发生异常: {e}")
        
        # logging.info(f"价格获取完成: {prices_dict}")
        # return prices_dict