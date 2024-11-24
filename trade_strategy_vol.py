import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import talib
import logging
import math

class DeltaNeutralStrategy():
    def __init__(self):
        # 代理设置
        proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据您的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据您的实际代理地址修改
        }
        # 初始化交易所API（这里以binance为例）
        self.exchange = ccxt.okx({
            'apiKey': 'ba7f444f-e83e-4dd1-8507-bf8dd9033cbc',
            'secret': 'D5474EF76B0A7397BFD26B9656006480',
            'password': 'TgTB+pJoM!d20F',
            'enableRateLimit': True,
            'proxies': proxies,  # 添加代理设置
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
        
        # 设置为测试网（如果需要的话）
        # self.exchange.set_sandbox_mode(True)
        
        # 策略参数
        self.symbol = None  # 交易对
        self.inst_id_spot = None # OKX现货交易对格式
        self.inst_id_swap = None  # OKX永续合约交易对格式
        self.base_position_size = 1.0  # 基础仓位大小
        self.profit_target = 0.015  # 止盈目标 1.5%
        self.stop_loss = 0.08  # 止损线 8%
        self.max_value = 100 #最大持仓金额
        self.step_value = 4 #每次开仓金额
        #self.max_position = 100  # 最大持仓
        #self.min_position = 1  # 最小持仓倍数
        #self.contract_size_multiplier =None
        # 技术指标参数
        # self.rsi_period = 14
        # self.rsi_overbought = 70
        # self.rsi_oversold = 30
        # self.ma_fast = 20
        # self.ma_slow = 50
        
        # 仓位记录
        self.spot_position = 0
        self.futures_position = 0
        self.trading_positions = []  # 用于记录高抛低吸的仓位
        
        # 测试API连接
        self.test_connection()

    def test_connection(self):
        """测试API连接"""
        try:
            # 测试市场数据获取
            ticker = self.exchange.fetch_ticker(self.symbol)
            logging.info(f"当前 {self.symbol} 价格: {ticker['last']}")
            
            # 测试账户数据获取
            balance = self.exchange.fetch_balance()
            logging.info("账户连接正常")
            
            return True
        except Exception as e:
            logging.info(f"连接测试失败: {str(e)}")
            return False
            
    def analyze_trade_flow(self):
        """分析主动成交流向"""
        try:
            # 获取最近的成交记录
            trades = self.exchange.fetch_trades(self.symbol, limit=1000)
            df = pd.DataFrame([{
                'timestamp': trade['timestamp'],
                'price': trade['price'],
                'amount': trade['amount'],
                'side': trade['side'],
                'cost': trade['cost']  # price * amount
            } for trade in trades])
            
            # 计算买卖压力
            buy_volume = df[df['side'] == 'buy']['cost'].sum()
            sell_volume = df[df['side'] == 'sell']['cost'].sum()
            
            # 计算买卖比率
            buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
            
            # 计算净流入
            net_flow = buy_volume - sell_volume

            logging.info(f"buy_volume:{buy_volume},sell_volume:{sell_volume},buy_sell_ratio:{buy_sell_ratio},net_flow:{net_flow}")

            return {
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_sell_ratio': buy_sell_ratio,
                'net_flow': net_flow
            }
        except Exception as e:
            logging.info(f"分析交易流向时出错: {str(e)}")
            return None

    def detect_large_orders(self, threshold_multiplier=3):
        """检测大单交易"""
        try:
            # 获取最近的成交
            trades = self.exchange.fetch_trades(self.symbol, limit=500)
            df = pd.DataFrame([{
                'timestamp': trade['timestamp'],
                'amount': trade['amount'],
                'side': trade['side']
            } for trade in trades])
            
            # 计算平均交易量
            avg_trade_size = df['amount'].mean()
            
            # 识别大单（超过平均值3倍）
            large_trades = df[df['amount'] > avg_trade_size * threshold_multiplier]
            
            # 分析大单方向
            large_buys = large_trades[large_trades['side'] == 'buy']
            large_sells = large_trades[large_trades['side'] == 'sell']
            
            # 计算大单比率
            large_buy_ratio = len(large_buys) / len(large_trades) if len(large_trades) > 0 else 0
            
            logging.info(f"avg_trade_size:{avg_trade_size},large_buys:{len(large_buys)},large_trades:{len(large_trades)},large_buy_ratio:{large_buy_ratio}")
            
            return {
                'large_buy_ratio': large_buy_ratio,
                'large_buys_count': len(large_buys),
                'large_sells_count': len(large_sells),
                'avg_large_buy_size': large_buys['amount'].mean() if not large_buys.empty else 0,
                'avg_large_sell_size': large_sells['amount'].mean() if not large_sells.empty else 0
            }
        except Exception as e:
            logging.info(f"检测大单时出错: {str(e)}")
            return None

    def calculate_cvd(self):
        """计算累积成交量差"""
        try:
            # 获取最近的成交
            trades = self.exchange.fetch_trades(self.symbol, limit=1000)
            df = pd.DataFrame([{
                'timestamp': trade['timestamp'],
                'price': trade['price'],
                'amount': trade['amount'],
                'side': trade['side']
            } for trade in trades])
            
            # 计算每笔交易的delta
            df['delta'] = df.apply(lambda x: x['amount'] if x['side'] == 'buy' else -x['amount'], axis=1)
            
            # 计算累积delta
            df['cvd'] = df['delta'].cumsum()
            
            # 计算CVD指标
            recent_cvd = df['cvd'].iloc[-1]
            cvd_ma = df['cvd'].rolling(window=20).mean().iloc[-1]
            
            logging.info(f"recent_cvd:{recent_cvd},cvd_ma:{cvd_ma}")

            return {
                'current_cvd': recent_cvd,
                'cvd_ma': cvd_ma,
                'trend': 'bullish' if recent_cvd > cvd_ma else 'bearish'
            }
        except Exception as e:
            logging.info(f"计算CVD时出错: {str(e)}")
            return None

    def generate_trade_signals(self):
        """生成交易信号"""
        try:
            # 获取各项指标
            trade_flow = self.analyze_trade_flow()
            large_orders = self.detect_large_orders()
            cvd_data = self.calculate_cvd()
            
            if not all([trade_flow, large_orders, cvd_data]):
                return None
            
            # 计算综合得分
            score = 0
            
            # 买卖压力评分
            if trade_flow['buy_sell_ratio'] > 1.2:
                score += 1
            elif trade_flow['buy_sell_ratio'] < 0.8:
                score -= 1
                
            # 大单评分
            if large_orders['large_buy_ratio'] > 0.6:
                score += 1
            elif large_orders['large_buy_ratio'] < 0.4:
                score -= 1
                
            # CVD趋势评分
            if cvd_data['trend'] == 'bullish':
                score += 1
            else:
                score -= 1

            logging.info(f"买卖压力评分: {trade_flow['buy_sell_ratio']}，大单评分：{large_orders['large_buy_ratio']}，CVD趋势评分：{cvd_data['trend']}")

            return {
                'score': score,
                'signal': 'buy' if score > 1 else 'sell' if score < -1 else 'neutral',
                'strength': abs(score)
            }
        except Exception as e:
            logging.info(f"生成交易信号时出错: {str(e)}")
            return None
   
    def check_price_pullback(self, current_price):
        """检测价格回撤"""
        try:
            # 获取历史数据
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '15m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 计算最近的高点
            recent_high = df['high'].rolling(window=20).max().iloc[-1]
            
            # 计算从高点的回撤百分比
            pullback = (recent_high - current_price) / recent_high
            
            logging.info(f"pullback:{pullback},recent_high:{recent_high},current_price:{current_price}")

            # 如果回撤小于2%，可能不是好的买入点
            if pullback < 0.02:
                return False, "INSUFFICIENT_PULLBACK"
                
            return True, None
            
        except Exception as e:
            logging.error(f"检测价格回撤时出错: {str(e)}")
            return False, "ERROR"

    
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


    def calculate_position_size(self):
        """计算合适的开仓数量"""
        try:
            # 获取账户余额
            balance = self.exchange.fetch_balance()
            
            # 获取现货账户USDT余额
            spot_usdt = float(balance['USDT']['free'])
            
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
            
            # 向下取整到适当的小数位
            position = math.floor(max_position * 10000) / 10000  # 保留4位小数
            #获取现货持仓总价值
            positions = balance['info']['data'][0]['details']
            symbol_value = sum(float(pos['eqUsd']) for pos in positions
                               if pos['ccy'] == str(self.symbol).replace('/USDT',''))


          
            logging.info(f"当前{self.symbol}价格: {current_price} USDT")
            logging.info(f"现货下单数量: {position}")
            logging.info(f"现货已持仓金额: {symbol_value} ")  # 除以10显示实际张数
            logging.info(f"预计使用保证金: {position * current_price} USDT")
            
            return position,spot_usdt,symbol_value,current_price
            
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
            currency = symbol.split('/')[0]
            available = float(balance[currency]['free'])
            return available
        except Exception as e:
            logging.error(f"获取持仓数量时出错: {str(e)}")
            return 0
    
    def place_limit_order(self, side, amount, price):
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
                    # 清空trading_positions列表，因为当前仓位已经不足以执行任何卖出指令
                    self.trading_positions.clear()
                    return False, None
                
            # 下限价单
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price,
                params={
                    'instId': self.inst_id_spot,
                    'tdMode': 'cash',
                    'timeInForce': 'GTC'  # Good Till Cancel
                }
            )
            
            order_id = order['id']
            start_time = time.time()
            
            # 等待订单成交或超时
            while time.time() - start_time < 1:  # 1秒超时
                # 检查订单状态
                order_status = self.exchange.fetch_order(order_id, self.symbol)
                
                if order_status['status'] == 'closed':
                    logging.info(f"订单已成交: {order_id}")
                    if side == 'sell':
                        # 卖出成功后清空trading_positions
                        self.trading_positions.clear()
                    return True, order
                    
                time.sleep(0.1)  # 短暂休眠，避免频繁查询
                
            # 超时取消订单
            try:
                self.exchange.cancel_order(order_id, self.symbol)
                logging.info(f"订单超时已取消: {order_id}")
            except Exception as e:
                logging.error(f"取消订单时出错: {str(e)}")
                
            return False, None
            
        except Exception as e:
            logging.error(f"下限价单时出错: {str(e)}")
            return False, None


    def init_base_position(self):
        """初始化基础对冲仓位"""
        try:
            # 计算合适的开仓数量
            #position_size, balance,symbol_value = self.calculate_position_size()
            
            # if position_size <= 0:
            #     logging.info("没有足够的资金开仓")
            #     return False
                
            
            #现货买入
            # spot_order = self.exchange.create_order(
            #     symbol=self.symbol,
            #     type='market',
            #     side='buy',
            #     amount=position_size,  # 使用调整后的数量
            #     params={
            #         'instId': self.inst_id_spot,
            #         'tdMode': 'cash'
            #     }
            # )
            
            # 合约做空
            # 先设置杠杆
            # self.exchange.set_leverage(1, self.inst_id_swap, params={
            #     'mgnMode': 'isolated',
            #     'posSide': 'net'  # 添加 posSide 参数，设置为 'net'
            # })
            
            # futures_order = self.exchange.create_order(
            #     symbol=self.symbol,
            #     type='market',
            #     side='sell',
            #     amount=position_size/100,   # 使用调整后的数量
            #     params={
            #         'instId': self.inst_id_swap,
            #         'tdMode': 'isolated'
            #     }
            # )
            
            #self.spot_position = float(position_size)
            #self.futures_position = -float(position_size)
            
            # logging.info(f"基础对冲仓位建立完成:")
            #logging.info(f"现货订单: {spot_order}")
            # print(f"合约订单: {futures_order}")
            #logging.info(f"现货持仓: {self.spot_position}")
            
            return True
            
        except Exception as e:
            logging.info(f"建立基础对冲仓位失败: {str(e)}")
            if hasattr(e, 'response'):
                logging.info(f"错误详情: {e.response}")
            return False

    def execute_swing_trade(self):
        """执行高抛低吸交易"""
        while True:  # 添加无限循环
            try:
                # 生成交易信号
                signals = self.generate_trade_signals()
                if not signals:
                    continue
                
                # 计算当前可用的交易数量
                # 获取当前持仓
                positions,balance,symbol_value,current_price = self.calculate_position_size()
                if positions <= 0:
                    logging.info("可用资金不足，暂停交易")
                    return
                
                # 检查是否可以增加新的交易仓位
                # 根据信号执行交易
                logging.info(f"买入信号执行:{signals['signal']},数量={positions}, 强度={signals['strength']}")
                if signals['signal'] == 'buy' and signals['strength'] >= 2:
                    
                    # 检查价格回撤
                    pullback_ok, pullback_reason = self.check_price_pullback(current_price)
                    if not pullback_ok:
                        logging.info(f"价格回撤不足: {pullback_reason}")
                        continue
                    
                    # 检查成本分布
                    cost_ok, cost_reason = self.analyze_cost_distribution()
                    if not cost_ok:
                        logging.info(f"成本分布不适合: {cost_reason}")
                        continue
                    # 检查是否有足够的USDT
                    ticker = self.exchange.fetch_ticker(self.symbol)
                    required_usdt = positions * ticker['last']
                    
                    if balance >= required_usdt and symbol_value<=self.max_value:
                        max_attempts = 3  # 最大重试次数
                        attempt = 0

                        while attempt < max_attempts:
                            # 重新获取订单簿价格
                            fresh_prices = self.get_orderbook_mid_price()
                            if not fresh_prices:
                                break
                                
                            # 下限价买单
                            success, order = self.place_limit_order(
                                'buy',
                                positions,
                                fresh_prices['mid_price']
                            )
                            
                            if success:
                                self.trading_positions.append({
                                    'price': fresh_prices['mid_price'],
                                    'size': float(positions),
                                    'timestamp': datetime.now()
                                })
                                break
                        
                            attempt += 1
                            if attempt < max_attempts:
                                logging.info(f"重试下单，第{attempt}次")
                                time.sleep(0.5)  # 等待半秒后重试

                            logging.info(f"现货订单: {order}，success:{success}")
                            logging.info(f"新增交易仓位: 价格{fresh_prices['mid_price']}, 数量{positions}")

                            self.trading_positions.append({
                            'price': fresh_prices['mid_price'],
                            'size': float(positions),
                            'timestamp': datetime.now()
                            })


                        # order = self.exchange.create_order(
                        #     symbol=self.symbol,
                        #     type='market',
                        #     side='buy',
                        #     amount=positions,
                        #     params={
                        #         'instId': self.inst_id_spot,
                        #         'tdMode': 'cash'
                        #     }
                        # )
                    else:
                        logging.info(f"新增交易仓位: 价格{ticker['last']}, 数量{positions},剩余金额：{balance}，现货金额：{symbol_value}")    
                                           
                    # 风险检查
                    #self.check_risk()
                # # 检查现有仓位是否需要平仓
                for pos in self.trading_positions[:]:  # 使用切片创建副本进行遍历
                    should_exit, reason = self.check_exit_conditions(ticker['last'],pos['price'])
                    if should_exit:
                        max_attempts = 3
                        attempt = 0
                        while attempt < max_attempts:
                            fresh_prices = self.get_orderbook_mid_price()
                            if not fresh_prices:
                                break

                            success, order = self.place_limit_order(
                            'sell',
                            pos['size'],
                            fresh_prices['mid_price']
                            )

                            if success:
                                profit = (fresh_prices['mid_price'] - pos['price']) * pos['size']
                                self.trading_positions.remove(pos)
                                logging.info(f"平仓: 原因{reason}, 盈亏{profit}")
                                break
                            
                            if success:
                                logging.info(f"成功卖出 {pos['size']} 个币种，价格 {order_price}")
                                break
                            elif not self.trading_positions:  # 如果trading_positions已被清空，说明仓位不足
                                logging.info("仓位不足，终止卖出尝试")
                                break
                            
                            # 如果失败但仍有仓位，重新获取价格并重试
                            fresh_prices = self.get_orderbook_mid_price()
                            if fresh_prices:
                                order_price = self.calculate_order_price('sell', fresh_prices['mid_price'])

                        attempt += 1
                        if attempt < max_attempts:
                            logging.info(f"重试卖出，第{attempt}次")
                            time.sleep(0.5)
                        # 执行卖出
                        # order = self.exchange.create_market_sell_order(
                        #     self.symbol,
                        #     pos['size']
                        # )
                        # profit = (ticker['last'] - pos['price']) * pos['size']
                        # self.trading_positions.remove(pos)
                        # logging.info(f"平仓: 原因{reason}, 盈亏{profit}")    
                    # 休眠一段时间
                logging.info(f"trading_positions:{self.trading_positions}")
                time.sleep(60)  # 10秒检查一次
                    
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
        # 初始化基础对冲仓位
        if not self.init_base_position():
            return
        
        # 开始执行高抛低吸
        self.execute_swing_trade()

if __name__ == "__main__":
    # 设置日志配置
    logging.basicConfig(filename='trade_strategy.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('脚本开始执行')

    try:
        strategy = DeltaNeutralStrategy()
        strategy.symbol = 'MOVR-USDT'
        strategy.inst_id_spot = 'MOVR-USDT'
        strategy.inst_id_swap = 'MOVE-USDT-SWAP'
        if strategy.test_connection():
            strategy.run()
        else:
            logging.info("连接测试失败，请检查网络和API配置")
    except Exception as e:
        logging.info(f"程序启动失败: {str(e)}")
