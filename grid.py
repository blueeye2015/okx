import psycopg2
import numpy as np
from datetime import datetime
import time
from typing import List, Dict
import logging
from decimal import Decimal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitcoinGridStrategy:
    def __init__(
        self, 
        upper_price: float, 
        lower_price: float, 
        grid_num: int, 
        investment: float,
        db_params: Dict[str, str]
    ):
        """
        初始化网格策略
        :param upper_price: 网格上限价格
        :param lower_price: 网格下限价格
        :param grid_num: 网格数量
        :param investment: 总投资额
        :param db_params: 数据库连接参数
        """
        self.upper_price = upper_price
        self.lower_price = lower_price
        self.grid_num = grid_num
        self.investment = investment
        self.db_params = db_params
        
        # 计算网格价格点
        self.grid_prices = np.linspace(lower_price, upper_price, grid_num)
        # 计算每个网格的投资额
        self.per_grid_investment = investment / (grid_num - 1)
        
        # 初始化持仓状态字典，记录每个网格是否持有
        self.grid_positions = {i: False for i in range(grid_num)}
        
        # 记录当前持有的BTC数量和USDT余额
        self.btc_balance = Decimal('0')
        self.usdt_balance = Decimal(str(investment))
        
        # 初始化数据库
        self.init_database()
        
        # 初始化网格
        self.initialize_grid_positions()

    def init_database(self):
        """初始化数据库表结构"""
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            
            # 创建订单表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS grid_orders (
                    id SERIAL PRIMARY KEY,
                    order_type VARCHAR(4),  -- BUY/SELL
                    price DECIMAL(20, 8),
                    amount DECIMAL(20, 8),
                    total_value DECIMAL(20, 8),
                    grid_index INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(10),
                    btc_balance DECIMAL(20, 8),
                    usdt_balance DECIMAL(20, 8)
                )
            """)
            
            # 创建网格状态表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS grid_positions (
                    grid_index INTEGER PRIMARY KEY,
                    price DECIMAL(20, 8),
                    is_holding BOOLEAN,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}")
            raise e
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def initialize_grid_positions(self):
        """初始化网格持仓"""
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            
            # 清空旧的网格状态
            cur.execute("DELETE FROM grid_positions")
            
            # 插入新的网格状态
            for i, price in enumerate(self.grid_prices):
                cur.execute("""
                    INSERT INTO grid_positions (grid_index, price, is_holding)
                    VALUES (%s, %s, %s)
                """, (i, float(price), False))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"网格初始化失败: {str(e)}")
            raise e
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def record_order(self, order_type: str, price: float, amount: float, grid_index: int):
        """记录订单到数据库"""
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            
            total_value = Decimal(str(price)) * Decimal(str(amount))
            
            # 更新余额
            if order_type == 'BUY':
                self.btc_balance += Decimal(str(amount))
                self.usdt_balance -= total_value
            else:  # SELL
                self.btc_balance -= Decimal(str(amount))
                self.usdt_balance += total_value
            
            # 记录订单
            cur.execute("""
                INSERT INTO grid_orders 
                (order_type, price, amount, total_value, grid_index, status, btc_balance, usdt_balance)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                order_type, price, amount, float(total_value), grid_index, 'FILLED',
                float(self.btc_balance), float(self.usdt_balance)
            ))
            
            # 更新网格状态
            cur.execute("""
                UPDATE grid_positions
                SET is_holding = %s, last_updated = CURRENT_TIMESTAMP
                WHERE grid_index = %s
            """, (order_type == 'BUY', grid_index))
            
            self.grid_positions[grid_index] = (order_type == 'BUY')
            
            conn.commit()
            logger.info(f"订单记录成功: {order_type} {amount} BTC @ {price}")
            logger.info(f"当前余额: BTC={float(self.btc_balance)}, USDT={float(self.usdt_balance)}")
            
        except Exception as e:
            logger.error(f"订单记录失败: {str(e)}")
            raise e
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def execute_grid_trading(self, current_price: float):
        """
        执行网格交易
        :param current_price: 当前价格
        """
        # 找到当前价格所在的网格
        grid_index = np.searchsorted(self.grid_prices, current_price)
        
        # 检查每个网格的状态并执行交易
        for i in range(len(self.grid_prices)):
            grid_price = self.grid_prices[i]
            
            # 如果价格低于网格价格且该网格未持仓，执行买入
            if current_price <= grid_price and not self.grid_positions[i]:
                if self.usdt_balance >= self.per_grid_investment:
                    amount = self.per_grid_investment / current_price
                    self.record_order('BUY', current_price, amount, i)
                    logger.info(f"网格 {i} 买入: 价格={current_price}")
            
            # 如果价格高于网格价格且该网格已持仓，执行卖出
            elif current_price >= grid_price and self.grid_positions[i]:
                if self.btc_balance > 0:
                    amount = self.per_grid_investment / current_price
                    if amount <= float(self.btc_balance):
                        self.record_order('SELL', current_price, amount, i)
                        logger.info(f"网格 {i} 卖出: 价格={current_price}")

def main():
    # 数据库连接参数
    db_params = {
        "dbname": "your_db_name",
        "user": "your_username",
        "password": "your_password",
        "host": "localhost",
        "port": "5432"
    }
    
    # 策略参数
    strategy = BitcoinGridStrategy(
        upper_price=50000,  # 上限价格
        lower_price=40000,  # 下限价格
        grid_num=10,        # 网格数量
        investment=100000,  # 总投资额
        db_params=db_params
    )
    
    # 模拟价格波动场景
    test_prices = [
        42000,  # 初始价格
        45000,  # 上涨
        48000,  # 继续上涨
        44000,  # 回落
        41000,  # 继续下跌
        46000   # 再次上涨
    ]
    
    # 模拟运行
    for price in test_prices:
        logger.info(f"\n当前价格: {price}")
        strategy.execute_grid_trading(price)
        time.sleep(1)  # 演示用，实际运行时可以调整间隔

if __name__ == "__main__":
    main()
