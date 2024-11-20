import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import talib
import logging

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
        self.step_value = 10 #每次开仓金额
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
            
            return {
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_sell_ratio': buy_sell_ratio,
                'net_flow': net_flow
            }
        except Exception as e:
            print(f"分析交易流向时出错: {str(e)}")
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
            
            return {
                'large_buy_ratio': large_buy_ratio,
                'large_buys_count': len(large_buys),
                'large_sells_count': len(large_sells),
                'avg_large_buy_size': large_buys['amount'].mean() if not large_buys.empty else 0,
                'avg_large_sell_size': large_sells['amount'].mean() if not large_sells.empty else 0
            }
        except Exception as e:
            print(f"检测大单时出错: {str(e)}")
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
            
            return {
                'current_cvd': recent_cvd,
                'cvd_ma': cvd_ma,
                'trend': 'bullish' if recent_cvd > cvd_ma else 'bearish'
            }
        except Exception as e:
            print(f"计算CVD时出错: {str(e)}")
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
                
            return {
                'score': score,
                'signal': 'buy' if score > 1 else 'sell' if score < -1 else 'neutral',
                'strength': abs(score)
            }
        except Exception as e:
            print(f"生成交易信号时出错: {str(e)}")
            return None
   
    # def check_entry_conditions(self, indicators):
    #     """检查入场条件"""
    #     # RSI超卖
    #     rsi_buy = indicators['rsi'] < self.rsi_oversold
    #     # 快线在慢线上方
    #     ma_trend_up = indicators['ma_fast'] > indicators['ma_slow']
    #     # MACD金叉
    #     macd_cross = indicators['macd'] > indicators['signal']
    #     # 价格接近布林带下轨
    #     price_near_lower = indicators['current_price'] < indicators['bb_lower'] * 1.01
    #     logging.info(f"RSI超卖,rsi：{indicators['rsi']},ris_oversold:{self.rsi_oversold}")
    #     logging.info(f"快线在慢线上方,ma_fast：{indicators['ma_fast']},ma_slow:{indicators['ma_slow']}")
    #     logging.info(f"MACD金叉：{indicators['macd']}，signal:{indicators['signal']}")
    #     logging.info(f"价格接近布林带下轨,bb_lower：{indicators['bb_lower']},price:{indicators['current_price']}")
    #     return rsi_buy and ma_trend_up and macd_cross and price_near_lower
    
    def check_exit_conditions(self, indicators, entry_price):
        """检查出场条件"""
        current_price = indicators['current_price']
        profit_pct = (current_price - entry_price) / entry_price
        
        # 止盈条件
        if profit_pct >= self.profit_target:
            return True, "TAKE_PROFIT"
        
        # 止损条件
        if profit_pct <= -self.stop_loss:
            return True, "STOP_LOSS"
        
        # RSI超买
        if indicators['rsi'] > self.rsi_overbought:
            return True, "RSI_OVERBOUGHT"
            
        return False, None
 
    # def get_contract_multiplier(self):
    #     """获取合约最小下单单位"""
    #     try:
    #         # 尝试从API获取
    #         markets = self.exchange.load_markets()
    #         contract_market = markets[self.symbol]
            
    #         # 尝试从市场信息中获取合约乘数
    #         multiplier = None
    #         if 'contractSize' in contract_market:
    #             multiplier = contract_market['contractSize']
    #         elif 'lot' in contract_market:
    #             multiplier = contract_market['lot']
            
    #         if multiplier is not None:
    #             logging.info(f"从API获取到{self.symbol}的合约乘数: {multiplier}")
    #             return multiplier
                
    #         # 如果API没有提供，使用手动设置的值
    #         if self.contract_size_multiplier is not None:
    #             logging.info(f"使用手动设置的{self.symbol}合约乘数: {self.contract_size_multiplier}")
    #             return self.contract_size_multiplier
    #             # 如果需要手动设置合约乘数
    #         strategy.contract_size_multiplier = 100  # 对于HBAR设置为100
    #         # 如果都没有，使用默认值1
    #         logging.info(f"未能获取{self.symbol}合约乘数，使用默认值: 1")
    #         return 1
        
    #     except Exception as e:
    #         logging.info(f"获取合约乘数失败: {str(e)}")
    #         # 如果出错，使用手动设置的值或默认值
    #         return self.contract_size_multiplier or 1

    def calculate_position_size(self):
        """计算合适的开仓数量"""
        try:
            # 获取账户余额
            balance = self.exchange.fetch_balance()
            
            # 获取现货账户USDT余额
            spot_usdt = float(balance['USDT']['free'])
            
            # 获取当前BTC价格
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            #获取当前现货持仓
            positions = self.exchange.fetch_positions([self.inst_id_spot])
            
            #计算最大允许开仓金额（账户总值的10%）
            
            logging.info(f"账户USDT余额: {spot_usdt}")
            #logging.info(f"最大允许开仓金额: {max_position_value} USDT")     

            # 计算现货可以开的数量
            spot_size = self.step_value / current_price
            
            # 计算合约可以开的张数（向下取整到10的倍数）
            # 由于合约一张=10个单位，所以需要除以10
            #contract_size = (spot_size // 10) * 10
            
            # 确保合约张数至少是10（欧易HBAR最小10张）
            #if contract_size < 10:
            #    print("可用资金不足以开立最小合约仓位（10张）")
            #    return 0
                 
            # 重新计算实际的现货数量，使其与合约数量匹配
            #spot_size = contract_size  # 现货数量等于合约张数对应的数量
          
            logging.info(f"当前{self.symbol}价格: {current_price} USDT")
            logging.info(f"现货下单数量: {spot_size}")
            #logging.info(f"合约下单张数: {contract_size/10} 张")  # 除以10显示实际张数
            logging.info(f"预计使用保证金: {spot_size * current_price} USDT")
            
            return float(spot_size),spot_usdt
            
        except Exception as e:
            logging.info(f"计算仓位大小时出错: {str(e)}")
            return 0

    def init_base_position(self):
        """初始化基础对冲仓位"""
        try:
            # 计算合适的开仓数量
            position_size, balance = self.calculate_position_size()
            
            if position_size <= 0:
                logging.info("没有足够的资金开仓")
                return False
                
            
            #现货买入
            spot_order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side='buy',
                amount=position_size,  # 使用调整后的数量
                params={
                    'instId': self.inst_id_spot,
                    'tdMode': 'cash'
                }
            )
            
            # 合约做空
            # 先设置杠杆
            self.exchange.set_leverage(1, self.inst_id_swap, params={
                'mgnMode': 'isolated',
                'posSide': 'net'  # 添加 posSide 参数，设置为 'net'
            })
            
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
            
            self.spot_position = float(position_size)
            self.futures_position = -float(position_size)
            
            # logging.info(f"基础对冲仓位建立完成:")
            # print(f"现货订单: {spot_order}")
            # print(f"合约订单: {futures_order}")
            logging.info(f"现货持仓: {self.spot_position}")
            
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
                positions,balance = self.calculate_position_size()
                if positions <= 0:
                    logging.info("可用资金不足，暂停交易")
                    return
                
                # 检查是否可以增加新的交易仓位
                # 根据信号执行交易
                if signals['signal'] == 'buy' and signals['strength'] >= 2:
                    # 检查是否有足够的USDT
                    ticker = self.exchange.fetch_ticker(self.symbol)
                    required_usdt = positions * ticker['last']
                    
                    if balance >= required_usdt:
                        order = self.exchange.create_order(
                            symbol=self.symbol,
                            type='market',
                            side='buy',
                            amount=positions,
                            params={
                                'instId': self.inst_id_spot,
                                'tdMode': 'cash'
                            }
                        )
                        print(f"买入信号执行: 数量={positions}, 强度={signals['strength']}")
                        logging.info(f"现货订单: {order}")
                        self.trading_positions.append({
                            'price': ticker['last'],
                            'size': float(positions),
                            'timestamp': datetime.now()
                        })
                        logging.info(f"新增交易仓位: 价格{ticker['last']}, 数量{positions}")
                    
                    # # 检查现有仓位是否需要平仓
                    # for pos in self.trading_positions[:]:  # 使用切片创建副本进行遍历
                    #     should_exit, reason = self.check_exit_conditions(indicators, pos['price'])
                    #     if should_exit:
                    #         # 执行卖出
                    #         order = self.exchange.create_market_sell_order(
                    #             self.symbol,
                    #             pos['size']
                    #         )
                    #         profit = (indicators['current_price'] - pos['price']) * pos['size']
                    #         self.trading_positions.remove(pos)
                    #         logging.info(f"平仓: 原因{reason}, 盈亏{profit}")
                    
                    # 风险检查
                    self.check_risk()
                    
                    # 休眠一段时间
                    time.sleep(10)  # 10秒检查一次
                    
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
        strategy.symbol = 'ETH/USDT'
        strategy.inst_id_spot = 'ETH-USDT'
        strategy.inst_id_swap = 'ETH-USDT-SWAP'
        if strategy.test_connection():
            strategy.run()
        else:
            logging.info("连接测试失败，请检查网络和API配置")
    except Exception as e:
        logging.info(f"程序启动失败: {str(e)}")
