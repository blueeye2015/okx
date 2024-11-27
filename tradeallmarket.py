import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

class MarketScanner:
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.min_market_cap = 20000  # 最小市值2000w美元
        self.min_listing_days = 30   # 最小上市天数
        self.valid_symbols = []      # 符合条件的交易对
        self.symbol_data = {}        # 存储每个交易对的相关数据
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def get_valid_symbols(self) -> List[str]:
        """获取所有符合条件的交易对"""
        try:
            markets = self.exchange.load_markets()
            valid_pairs = []
            
            for symbol, market in markets.items():
                if market['quote'] == 'USDT' and market['spot']:
                    # 检查是否是现货交易对
                    listing_info = self.check_listing_time(symbol)
                    if not listing_info['is_valid']:
                        continue
                        
                    # 检查市值
                    market_cap = self.calculate_market_cap(symbol)
                    if market_cap < self.min_market_cap:
                        continue
                        
                    valid_pairs.append({
                        'symbol': symbol,
                        'listing_date': listing_info['listing_date'],
                        'market_cap': market_cap
                    })
                    
            self.valid_symbols = valid_pairs
            logging.info(f"找到 {len(valid_pairs)} 个符合条件的交易对")
            return valid_pairs
            
        except Exception as e:
            logging.error(f"获取有效交易对时出错: {str(e)}")
            return []
            
    def check_listing_time(self, symbol: str) -> Dict:
        """检查交易对的上市时间"""
        try:
            # 获取K线数据来确定上市时间
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1d', limit=1000)
            if not ohlcv:
                return {'is_valid': False}
                
            first_candle_time = datetime.fromtimestamp(ohlcv[0][0]/1000)
            days_listed = (datetime.now() - first_candle_time).days
            
            return {
                'is_valid': days_listed >= self.min_listing_days,
                'listing_date': first_candle_time,
                'days_listed': days_listed
            }
            
        except Exception as e:
            logging.error(f"检查{symbol}上市时间时出错: {str(e)}")
            return {'is_valid': False}
            
    def calculate_market_cap(self, symbol: str) -> float:
        """计算交易对的市值（以美元计）"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            # 获取币种信息
            currency = symbol.split('/')[0]
            
            # 这里需要通过某种方式获取流通量，可以通过API或其他数据源
            # 这里使用一个模拟值作为示例
            circulation = self.get_circulation(currency)
            
            market_cap = ticker['last'] * circulation
            return market_cap
            
        except Exception as e:
            logging.error(f"计算{symbol}市值时出错: {str(e)}")
            return 0
            
    def get_circulation(self, currency: str) -> float:
        """获取币种的流通量"""
        # 这里需要实现具体的获取流通量的逻辑
        # 可以通过API或其他数据源获取
        pass

class BreakoutDetector:
    def __init__(self):
        self.breakout_threshold = 0.05  # 5%的突破阈值
        self.volume_threshold = 2.0     # 成交量是平均值的2倍
        
    def detect_breakout(self, symbol: str, klines: pd.DataFrame) -> bool:
        """检测价格突破"""
        try:
            # 计算价格变化
            price_change = (klines['close'].iloc[-1] - klines['close'].iloc[-2]) / klines['close'].iloc[-2]
            
            # 计算成交量比较
            avg_volume = klines['volume'].rolling(20).mean().iloc[-1]
            current_volume = klines['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            # 检查是否满足突破条件
            is_breakout = price_change > self.breakout_threshold and volume_ratio > self.volume_threshold
            
            if is_breakout:
                logging.info(f"{symbol} 发生突破，价格变化: {price_change:.2%}, 成交量比: {volume_ratio:.2f}")
                
            return is_breakout
            
        except Exception as e:
            logging.error(f"检测{symbol}突破时出错: {str(e)}")
            return False

class SwingTrader:
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.positions = {}  # 当前持仓
        self.stop_loss = 0.05  # 5%止损
        self.take_profit = 0.1  # 10%获利
        self.swing_threshold = 0.03  # 3%的高抛低吸阈值
        
    def execute_trade(self, symbol: str, side: str, amount: float) -> bool:
        """执行交易"""
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            
            if order['status'] == 'closed':
                logging.info(f"{symbol} {side}单成功执行，数量: {amount}")
                return True
            return False
            
        except Exception as e:
            logging.error(f"{symbol}执行{side}单时出错: {str(e)}")
            return False
            
    def manage_position(self, symbol: str, current_price: float):
        """管理持仓，执行高抛低吸"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        entry_price = position['entry_price']
        
        # 检查止损
        if current_price < entry_price * (1 - self.stop_loss):
            self.execute_trade(symbol, 'sell', position['amount'])
            del self.positions[symbol]
            logging.info(f"{symbol} 触发止损，平仓")
            return
            
        # 检查获利
        if current_price > entry_price * (1 + self.take_profit):
            self.execute_trade(symbol, 'sell', position['amount'])
            del self.positions[symbol]
            logging.info(f"{symbol} 达到目标利润，平仓")
            return
            
        # 高抛低吸逻辑
        price_change = (current_price - entry_price) / entry_price
        if abs(price_change) >= self.swing_threshold:
            if price_change > 0:  # 高抛
                sell_amount = position['amount'] * 0.3  # 卖出30%的持仓
                if self.execute_trade(symbol, 'sell', sell_amount):
                    position['amount'] -= sell_amount
                    position['partial_sells'].append({
                        'price': current_price,
                        'amount': sell_amount
                    })
            else:  # 低吸
                if position['partial_sells']:  # 如果之前有高抛
                    buy_amount = min(
                        position['partial_sells'][-1]['amount'],
                        self.calculate_affordable_amount(symbol, current_price)
                    )
                    if self.execute_trade(symbol, 'buy', buy_amount):
                        position['amount'] += buy_amount
                        position['partial_sells'].pop()

    def calculate_affordable_amount(self, symbol: str, price: float) -> float:
        """计算当前余额可以购买的数量"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            return (usdt_balance * 0.95) / price  # 留5%作为手续费缓冲
        except Exception as e:
            logging.error(f"计算可购买数量时出错: {str(e)}")
            return 0

class MarketMonitor:
    def __init__(self, exchange_id: str = 'okx'):
        self.exchange = ccxt.okx({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        self.scanner = MarketScanner(self.exchange)
        self.detector = BreakoutDetector()
        self.trader = SwingTrader(self.exchange)
        self.monitoring_interval = 60  # 60秒检查一次
        
    def run(self):
        """运行市场监控"""
        while True:
            try:
                # 获取符合条件的交易对
                valid_symbols = self.scanner.get_valid_symbols()
                
                for symbol_info in valid_symbols:
                    symbol = symbol_info['symbol']
                    
                    # 获取K线数据
                    klines = self.get_klines(symbol)
                    if klines is None:
                        continue
                        
                    current_price = klines['close'].iloc[-1]
                    
                    # 管理现有持仓
                    self.trader.manage_position(symbol, current_price)
                    
                    # 检测新的突破机会
                    if self.detector.detect_breakout(symbol, klines):
                        # 计算购买数量
                        amount = self.trader.calculate_affordable_amount(symbol, current_price)
                        if amount > 0:
                            if self.trader.execute_trade(symbol, 'buy', amount):
                                self.trader.positions[symbol] = {
                                    'entry_price': current_price,
                                    'amount': amount,
                                    'entry_time': datetime.now(),
                                    'partial_sells': []
                                }
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"市场监控运行时出错: {str(e)}")
                time.sleep(self.monitoring_interval)
                
    def get_klines(self, symbol: str) -> pd.DataFrame:
        """获取K线数据"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(f"获取{symbol}K线数据时出错: {str(e)}")
            return None

if __name__ == "__main__":
    monitor = MarketMonitor()
    monitor.run()
