import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import talib
import logging
import math
import argparse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum, Boolean
from sqlalchemy.orm import declarative_base  # 新的导入方式
import requests

class DeltaNeutralStrategy():
    def __init__(self):
        # 代理设置
        self.proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据您的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据您的实际代理地址修改
        }
        # 初始化交易所API（这里以binance为例）
        self.exchange = ccxt.okx({
            'apiKey': 'ba7f444f-e83e-4dd1-8507-bf8dd9033cbc',
            'secret': 'D5474EF76B0A7397BFD26B9656006480',
            'password': 'TgTB+pJoM!d20F',
            'enableRateLimit': True,
            'proxies': self.proxies,  # 添加代理设置
            'timeout': 30000,    # 设置超时时间为30秒
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })



        # 验证连接
        try:
            self.exchange.load_markets()
            logging.info("交易所连接成功")
        except Exception as e:
            logging.info(f"交易所连接失败: {str(e)}")
            raise e

                # 策略参数
        self.symbol = None  # 交易对
        self.base_position_size = 1.0  # 基础仓位大小
        self.profit_target = 0.015  # 止盈目标 10%
        self.stop_loss = 0.008  # 止损线 8%
        self.max_value = 20 #最大持仓金额
        self.step_value = 2 #每次开仓金额       
        # 仓位记录
        self.position_perc = 0.1 #总仓位占比
        self.engine = create_engine('postgresql://postgres:12@localhost:5432/market_data')
        Session = sessionmaker(bind=self.engine)
        self.session = Session()            

    def generate_trade_signals(self):
        """获取交易对"""
        try:         
            query = text("""
                SELECT distinct symbol FROM trend_records_5m WHERE consecutive_count>20 AND timestamp >= NOW() - INTERVAL '30 minutes' 
                and symbol not in ('CITY-USDT','SWEAT-USDT','CHZ-USDT','GALFT-USDT','TRA-USDT','ARG-USDT','POR-USDT','MENGO-USDT','SPURS-USDT')
                and ma60_r2>0.6 and ma60_slope>0.0001;
            """)

            df = pd.read_sql(
                query,
                self.engine
            )
            # 返回纯列表，不包含表头
            return df['symbol'].tolist() 
            
        except Exception as e:
            logging.info(f"生成交易信号时出错: {str(e)}")
            return None
       
    def check_exit_conditions(self, current_price ,entry_price):
        """检查出场条件"""
        #current_price = indicators['current_price']
        profit_pct = (current_price - entry_price) / entry_price
        
        # 止盈条件
        if profit_pct >= self.profit_target:
            return True, "TAKE_PROFIT"
        
        # 止损条件
        if profit_pct <= -self.stop_loss:
            return True, "STOP_LOSS"    
            
        return False, None
 
    def analyze_cost_distribution(self):
        """分析成本分布"""
        try:
            # 获取最近的成交记录
            trades = self.exchange.fetch_trades(self.symbol, limit=1000)
            df = pd.DataFrame([{
                'price': trade['price'],
                'amount': trade['amount']
            } for trade in trades])
            
            # 计算成交量加权平均价格（VWAP）
            vwap = (df['price'] * df['amount']).sum() / df['amount'].sum()
            
            # 如果当前价格显著高于VWAP，可能不是好的买入点
            current_price = self.exchange.fetch_ticker(self.symbol)['last']

            logging.info(f"vwap:{vwap},current_price:{current_price}")

            if current_price > vwap * 1.02:  # 价格超过VWAP 2%
                return False, "PRICE_ABOVE_VWAP"
                
            return True, None
            
        except Exception as e:
            logging.error(f"分析成本分布时出错: {str(e)}")
            return False, "ERROR"

    def get_min_amount(self):
        """
        从 OKX 获取币种数据
        
        
        """            
        try:
            # 获取 OKX 的币种数据
            okx_url = "https://www.okx.com/api/v5/public/instruments"
            okx_params = {
                'instType': 'SPOT',
                'instId': self.symbol
            }
            
            okx_response = requests.get(
                okx_url,
                proxies=self.proxies,
                params=okx_params,
                timeout=10
            )
            okx_response.raise_for_status()
            okx_data = okx_response.json()
                        
            min_size = float(okx_data['data'][0]['minSz'])

            # 计算最小购买金额
            min_purchase_amount =  min_size if min_size > 0 else float('inf')

            return min_purchase_amount
                        
        except requests.RequestException as e:
            logging.error(f"获取OKX数据失败: {str(e)}")
            return None

    def calculate_position_size(self,symbol):
        """计算合适的开仓数量"""
        try:
            # 获取账户余额
            balance = self.exchange.fetch_balance()
            
            # 获取现货账户USDT余额
            spot_usdt = float(balance['USDT']['free'])

            # 计算账户总资产（包括USDT和所有持仓）
            positions = balance['info']['data'][0]['details']
            total_asset_value = spot_usdt  # 先加入USDT余额
            
            # 计算所有币种持仓的总价值
            total_position_value = 0
            for pos in positions:
                position_value = float(pos['eqUsd'])
                total_position_value += position_value
                total_asset_value += position_value
        
            # 计算当前持仓占总资产的比例
            position_ratio = (total_position_value / total_asset_value * 100) if total_asset_value > 0 else 0
            
            # 获取最新订单簿价格
            orderbook_prices = self.get_orderbook_mid_price()
            if not orderbook_prices:
                return 0, 0
          
            current_price = orderbook_prices['best_ask']  # 使用卖一价格计算，这样更保守

            # 添加安全边际，预留一些余额防止价格波动
            safety_margin = 1.02  # 预留2%的安全边际
            estimated_price = current_price * safety_margin
            
            logging.info(f"账户USDT余额: {spot_usdt}")
            # 计算最大可买数量，考虑手续费
            fee_rate = 0.0008  # 挂单手续费0.08%
            total_cost_ratio = 1 + fee_rate

            # 计算现货可以开的数量
            max_position = (self.step_value / estimated_price) / total_cost_ratio
            
            ##获取最小下单数量
            min_amount = self.get_min_amount()
            position = 1.5*min_amount if min_amount > max_position else max_position # 如果计算数量比最小小单数还小，为了以后能顺利卖出，乘以1.5
            #获取现货持仓总价值
            positions = balance['info']['data'][0]['details']
            symbol_value = sum(float(pos['eqUsd']) for pos in positions
                               if pos['ccy'] == str(self.symbol).replace('-USDT',''))


          
            logging.info(f"当前{self.symbol}价格: {current_price} USDT")
            logging.info(f"现货下单数量: {position}")
            logging.info(f"现货已持仓金额: {symbol_value} ")  # 除以10显示实际张数
            logging.info(f"预计使用保证金: {position * current_price} USDT")
            
            return position,spot_usdt,symbol_value,current_price,position_ratio
            
        except Exception as e:
            logging.info(f"计算仓位大小时出错: {str(e)}")
            return 0

    def get_orderbook_mid_price(self):
        """获取订单簿买一卖一中间价"""
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol)
            best_bid = orderbook['bids'][0][0] if len(orderbook['bids']) > 0 else None
            best_ask = orderbook['asks'][0][0] if len(orderbook['asks']) > 0 else None
            
            if best_bid is None or best_ask is None:
                return None
                
            mid_price = (best_bid + best_ask) / 2
            return {
                'mid_price': mid_price,
                'best_bid': best_bid,
                'best_ask': best_ask
            }
        except Exception as e:
            logging.error(f"获取订单簿价格时出错: {str(e)}")
            return None
        
    def get_position(self, symbol):
        """获取当前持仓数量"""
        try:
            balance = self.exchange.fetch_balance()
            # 获取币种名称（例如从 'BTC/USDT' 中获取 'BTC'）
            currency = symbol.split('-')[0]
            available = float(balance[currency]['free'])
            
            return available
        except Exception as e:
            logging.error(f"获取持仓数量时出错: {str(e)}")
            return 0
    
    def place_limit_order(self, side, amount, price, buy_order_id=None):
        """
        下限价单并等待成交
        side: 'buy' 或 'sell'
        amount: 交易数量
        price: 限价价格
        """
        try:
            if side == 'sell':
                available_position = self.get_position(self.symbol)
                if available_position < amount:
                    logging.warning(f"可用仓位不足，需要 {amount}，实际 {available_position}") 
                    amount = available_position            
                    
                
            # 下限价单
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price,
                params={                    
                    'tdMode': 'cash',
                    'timeInForce': 'GTC'  # Good Till Cancel
                }
            )
            
            order_id = order['id'] if side == 'buy' else buy_order_id #如果是sell则用buyorderid
            start_time = time.time()

            # 记录订单到数据库
            self.record_order(self.symbol, side, price, amount, order_id)
            
            # 等待订单成交或超时
            while time.time() - start_time < 1:  # 1秒超时
                # 检查订单状态
                order_status = self.exchange.fetch_order(order_id, self.symbol)
                
                if order_status['status'] == 'closed':
                    # 记录订单到数据库
                    self.update_order_status(order_id, 'CLOSED', side)
                    logging.info(f"订单已成交: {order_id}")
                    return True, order_id
                    
                time.sleep(0.1)  # 短暂休眠，避免频繁查询
                
            # 超时取消订单
            try:
                self.exchange.cancel_order(order_id, self.symbol)
                self.update_order_status(order_id, 'CANCEL', side)
                logging.info(f"订单超时已取消: {order_id}")
            except Exception as e:
                logging.error(f"取消订单时出错: {str(e)}")
            return False, None
            
        except Exception as e:
            logging.error(f"下限价单时出错: {str(e)}")
            return False, None

    def record_order(self, symbol, order_type, price, quantity, order_id):
        """记录订单到数据库"""
        try:
            self.session.execute(
                text("""
                    INSERT INTO trade_orders 
                    (id, symbol, order_type, price, quantity, timestamp, strategy_name, status) 
                    VALUES (:order_id, :symbol, :order_type, :price, :quantity, :timestamp, :strategy, :status)
                """),
                {
                    'order_id': order_id,
                    'symbol': symbol,
                    'order_type': order_type.upper(),
                    'price': price,
                    'quantity': quantity,
                    'timestamp': datetime.now(),
                    'strategy': 'ma60',
                    'status': 'OPEN'
                }
            )
            self.session.commit()
        except Exception as e:
            logging.error(f"记录订单时出错: {str(e)}")
            self.session.rollback()

    def update_order_status(self, order_id, status, side):
        """更新订单状态"""
        try:
            self.session.execute(
                text("""
                    UPDATE trade_orders 
                    SET status = :status 
                    WHERE id = :order_id
                    AND side = :side
                """),
                {'status': status, 'order_id': order_id, 'side': side}
            )
            self.session.commit()
        except Exception as e:
            logging.error(f"更新订单状态时出错: {str(e)}")
            self.session.rollback()

    def get_active_positions(self):
        """从数据库获取特定交易对的活跃仓位"""
        try:
            
            query = text("""
                SELECT price, quantity, timestamp, symbol 
                FROM trade_orders 
                WHERE order_type = 'BUY' 
                AND status = 'CLOSED'
                AND id NOT IN (
                    SELECT id 
                    FROM trade_orders 
                    WHERE order_type = 'SELL' 
                    AND status = 'CLOSED'
                )
            """)
                
            
            # 使用 pandas 读取查询结果
            df = pd.read_sql(query, self.engine)
            
            # 转换为字典列表
            positions = df.to_dict('records')
            return positions
        except Exception as e:
            logging.error(f"获取活跃仓位时出错: {str(e)}")
            return []   

    def record_trade_pair(self, order_id, profit, sell_price):
        """
        记录交易对的盈亏情况
        
        参数:
        buy_order_id: 买入订单ID
        sell_order_id: 卖出订单ID
        symbol: 交易对符号
        """
        try:     
            query = text("""
                SELECT price, quantity, timestamp
                FROM trade_orders
                WHERE id = :order_id
            """)
            
            # 执行查询        
            df = pd.read_sql(query, self.engine, params={'order_id': order_id})
            
            if not df.empty:
                logging.error(f"未找到订单信息:  sell_order_id={order_id}")
                return False
            

            buy_total = sell_price * df['price']
            # 计算盈利百分比
            profit_percentage = (profit / buy_total) * 100
            
            # 计算持仓时长
            hold_duration = datetime.now() - df['timestamp']
            
            # 插入交易对记录
            insert_query = text("""
                INSERT INTO trade_pairs (
                    order_id,
                    symbol,
                    profit_amount,
                    profit_percentage,
                    hold_duration
                ) VALUES (
                    :order_id,
                    :symbol,
                    :profit_amount,
                    :profit_percentage,
                    :hold_duration
                )
            """)
            
            self.session.execute(insert_query, {
                'order_id': order_id,                
                'symbol': self.symbol,
                'profit_amount': profit,
                'profit_percentage': profit_percentage,
                'hold_duration': hold_duration
            })
            
            self.session.commit()
            
            # 记录日志
            logging.info(f"交易对记录成功 - 交易对: {self.symbol}")
            logging.info(f"盈利金额: {profit:.2f} USDT")
            logging.info(f"盈利百分比: {profit_percentage:.2f}%")
            logging.info(f"持仓时长: {hold_duration}")
            
            return True
            
        except Exception as e:
            self.session.rollback()
            logging.error(f"记录交易对时出错: {str(e)}")
            return False


    def execute_swing_trade(self):
        """执行高抛低吸交易"""
        while True:  # 添加无限循环
            try:
                # 生成交易信号
                symbol = self.generate_trade_signals()
                if not symbol:
                    continue
                
                # 遍历每个交易对
                for self.symbol in symbol:
                    logging.info(f"处理交易对: {self.symbol}")
                    
                    # 计算当前可用的交易数量
                    positions, balance, symbol_value, current_price,position_ratio = self.calculate_position_size(self.symbol)
                    
                    if positions <= 0:
                        logging.info(f"{self.symbol} 可用资金不足，跳过")
                        continue

                    # 如果仓位占比超过阈值 则跳过
                    if position_ratio >  self.position_perc:
                        logging.info(f"{position_ratio} 仓位占比超过阈值，跳过")
                        continue

                    # 检查是否可以新开仓位
                    if balance >= (positions * current_price) and symbol_value <= self.max_value:
                        # 获取最新订单簿价格
                        fresh_prices = self.get_orderbook_mid_price()
                        if not fresh_prices:
                            continue

                        # 执行买入
                        max_attempts = 3
                        attempt = 0
                        while attempt < max_attempts:
                            success, order_id = self.place_limit_order(                                
                                side='buy',
                                amount=positions,
                                price=fresh_prices['mid_price']
                            )
                            
                            if success:
                                logging.info(f"{self.symbol} 买入成功: 价格={fresh_prices['mid_price']}, 数量={positions}")
                                break

                            attempt += 1
                            if attempt < max_attempts:
                                logging.info(f"{self.symbol} 重试买入，第{attempt}次")
                                time.sleep(0.5)
                
                # 获取当前活跃仓位
                active_positions = self.get_active_positions()
                if not active_positions:
                    continue
                
                # 检查现有仓位是否需要平仓
                for pos in active_positions:
                    self.symbol = pos['symbol']
                    current_price = self.exchange.fetch_ticker(self.symbol)['last'] #获取最新价格
                    should_exit, reason = self.check_exit_conditions(current_price, pos['price'])
                    
                    if should_exit:
                        fresh_prices = self.get_orderbook_mid_price()
                        if not fresh_prices:
                            continue

                        max_attempts = 3
                        attempt = 0
                        while attempt < max_attempts:
                            success, order_id = self.place_limit_order(
                                side='sell',
                                amount=pos['quantity'] ,
                                price=fresh_prices['mid_price']
                            )

                            if success:
                                profit = (fresh_prices['mid_price'] - pos['price']) * pos['quantity']
                                logging.info(f"{self.symbol} 平仓成功: 原因={reason}, 盈亏={profit}")
                                self.record_trade_pair(order_id, profit, fresh_prices['mid_price'])
                                break

                            attempt += 1
                            if attempt < max_attempts:
                                logging.info(f"{self.symbol} 重试卖出，第{attempt}次")
                                time.sleep(0.5)

                time.sleep(60)  # 整体循环间隔


               
            except Exception as e:
                logging.info(f"交易执行错误: {str(e)}")
                time.sleep(30)  # 出错后等待30秒
    
    def check_risk(self):
        """风险检查"""
        # 检查总持仓是否符合预期
        total_spot = self.spot_position + sum(pos['size'] for pos in self.trading_positions)
        if abs(total_spot + self.futures_position) > 0.01:  # 允许0.01的误差
            logging.info("警告: 持仓不平衡，需要调整")
        
        # 检查账户余额
        balance = self.exchange.fetch_balance()
        if balance['USDT']['free'] < 1000:  # 假设最低保证金要求
            logging.info("警告: 保证金不足")
    
    def run(self):
        """运行策略"""
              
        # 开始执行高抛低吸
        self.execute_swing_trade()

if __name__ == "__main__":
    

    # 设置日志配置
    logging.basicConfig(filename=f'ma60_strategy.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('脚本开始执行')

    try:
        strategy = DeltaNeutralStrategy()      
        strategy.run()
        
    except Exception as e:
        logging.info(f"程序启动失败: {str(e)}")