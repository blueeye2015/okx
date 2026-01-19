# spike_trader.py (V2.1 - Database Driven)

import time
import datetime
import logging
import uuid
import signal
import sys
import clickhouse_connect
import pandas as pd
import pytz # <--- 1. 导入新库
from old.trade_executor1 import TradingClient
from trade_manager import TradeManager # 导入交易大脑

# --- 配置区 ---
# (配置参数保持不变)
CH_HOST, CH_PORT, CH_DATABASE, CH_USERNAME, CH_PASSWORD = 'localhost', 8123, 'marketdata', 'default', '12'
SPIKE_THRESHOLD, POSITION_USDT_SIZE, TAKE_PROFIT_PCT, STOP_LOSS_PCT = 0.10, 50.0, 0.25, 0.08
SCAN_AT_SECOND = 2 
SHANGHAI_TZ = pytz.timezone('Asia/Shanghai') # <--- 2. 定义时区对象
now_aware = datetime.datetime.now(SHANGHAI_TZ) # <--- 3. 使用带时区的 now()

# 【新!】定义一个固定的列顺序，与orders表的CREATE TABLE语句完全一致
ORDER_TABLE_COLUMNS = [
    'client_order_id', 'exchange_order_id', 'symbol', 'status', 'side', 'order_type',
    'usdt_amount', 'entry_price', 'exit_price', 'size', 'created_at', 'updated_at',
    'pnl_usdt', 'pnl_pct', 'notes'
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_clickhouse_client():
    """创建并返回一个ClickHouse客户端连接"""
    try:
        client = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT, database=CH_DATABASE,
            username=CH_USERNAME, password=CH_PASSWORD)
        return client
    except Exception as e:
        logging.error(f"连接ClickHouse失败: {e}")
        return None

class SpikeTrader:
    def __init__(self, trading_client=None):
        self.ch_client = get_clickhouse_client()
        self.trading_client = TradingClient()
        # 如果没有提供外部的 trading_client，就使用真实的
        #self.trading_client = trading_client if trading_client is not None else TradingClient() 
        self.pending_orders = {}  # 待确认订单: { 'client_id': {'symbol': 'BTC-USDT', 'entry_time': ...} }
        self.open_positions = {}  # 已成交持仓: { 'symbol': {'entry_price': ..., 'size': ...} }
        self.is_shutting_down = False

        # 关键：在启动时从数据库恢复状态
        self._load_state_from_db()

    def _load_state_from_db(self):
        logging.info("--- 正在从数据库恢复程序状态 ---")
        try:
            # 查询所有未关闭的订单
            query = f"SELECT * FROM {CH_DATABASE}.orders WHERE status IN ('PENDING', 'FILLED')"
            unclosed_orders_df = self.ch_client.query_df(query)

            if unclosed_orders_df.empty:
                logging.info("没有需要恢复的活动订单。")
                return

            for _, order in unclosed_orders_df.iterrows():
                if order['status'] == 'PENDING':
                    self.pending_orders[order['client_order_id']] = {
                        'symbol': order['symbol'],
                        'entry_time': order['created_at']
                    }
                elif order['status'] == 'FILLED':
                    self.open_positions[order['symbol']] = {
                        'client_order_id': order['client_order_id'],
                        'entry_price': order['entry_price'],
                        'size': order['size']
                    }
            logging.warning(f"状态恢复成功: {len(self.pending_orders)}个待处理订单, {len(self.open_positions)}个持仓。")
        except Exception as e:
            logging.error(f"从数据库恢复状态失败: {e}")
            # 这是一个严重错误，应该终止程序
            sys.exit(1)    

    def _write_order_to_db(self, order_data):
        """【修正】向数据库插入或更新一条订单记录，使用固定的列顺序"""
        try:
            # 将字典数据按照固定顺序转换为列表
            data_to_insert = [order_data.get(col) for col in ORDER_TABLE_COLUMNS]
            self.ch_client.insert('marketdata.orders', [data_to_insert], column_names=ORDER_TABLE_COLUMNS)
        except Exception as e:
            logging.error(f"写入/更新订单到数据库失败: {e}")
    
    def scan_for_spikes(self):
        logging.info("开始扫描分钟暴涨信号...")
        query = f"""
        SELECT
            symbol,
            (close - open) / open AS spike_pct,
            close as current_price
        FROM {CH_DATABASE}.okx_klines_1m
        WHERE (timestamp, symbol) IN (
            SELECT max(timestamp), symbol FROM {CH_DATABASE}.okx_klines_1m GROUP BY symbol
        ) AND spike_pct >= {SPIKE_THRESHOLD}
        """
        try:
            signals_df = self.ch_client.query_df(query)
            if not signals_df.empty:
                logging.warning(f"🔥🔥🔥 发现 {len(signals_df)} 个暴涨信号! 🔥🔥🔥")
                for _, signal in signals_df.iterrows():
                    self.handle_new_signal(signal)
            else:
                logging.info("未发现暴涨信号。")
        except Exception as e:
            logging.error(f"执行信号扫描时出错: {e}")

    def handle_new_signal(self, signal):
        symbol = signal['symbol']
        price = signal['current_price']
        
        if symbol in self.open_positions or any(p['symbol'] == symbol for p in self.pending_orders.values()):
            logging.info(f"[{symbol}] 已有待处理订单或持仓，忽略新信号。")
            return

        client_id = f"spike{uuid.uuid4().hex[:10]}"
        size_to_buy = POSITION_USDT_SIZE / price

        
        # 步骤1: 先在数据库中创建一条PENDING记录
        order_record = {
            'client_order_id': client_id, 'exchange_order_id': '', 'symbol': symbol,
            'status': 'PENDING', 'side': 'buy', 'order_type': 'market',
            'usdt_amount': POSITION_USDT_SIZE, 'entry_price': 0, 'exit_price': 0,
            'size': 0, 'created_at': now_aware, 'updated_at': now_aware,
            'pnl_usdt': 0, 'pnl_pct': 0, 'notes': 'Order submitted'
        }
        self._write_order_to_db(order_record)

        order_result = self.trading_client.place_market_order(symbol, 'buy', size_to_buy, client_id)
        if order_result:
            self.pending_orders[client_id] = {'symbol': symbol, 'entry_time': now_aware}
        else:
            # 如果提交失败，更新数据库记录
            order_record['status'] = 'FAILED'
            order_record['notes'] = 'Failed to submit order to exchange.'
            order_record['updated_at'] = now_aware
            self._write_order_to_db(order_record)

    
    def check_pending_orders(self):
        if not self.pending_orders:
            return
        
        logging.info(f"开始检查 {len(self.pending_orders)} 个待确认订单...")
        for client_id, order_info in list(self.pending_orders.items()):
            # 规则2: 1分钟后check
            if (now_aware - order_info['entry_time']).total_seconds() < 60:
                continue # 时间未到，跳过

            symbol = order_info['symbol']
            status_result = self.trading_client.check_order_status(symbol, client_id)
            
            note = ""
            if status_result and status_result['state'] == 'filled':
                entry_price = float(status_result['avgPx'])
                size = float(status_result['accFillSz'])
                self.open_positions[symbol] = {'client_order_id': client_id, 'entry_price': entry_price, 'size': size}
                status = 'FILLED'
                note = f"Filled at {entry_price}"
            else:
                entry_price, size = 0, 0
                status = 'CANCELLED'
                note = "Order not filled within 1 minute."
            
            # 更新数据库记录
            update_record = {
                'client_order_id': client_id, 'exchange_order_id': status_result.get('ordId', '') if status_result else '',
                'symbol': symbol, 'status': status, 'side': 'buy', 'order_type': 'market',
                'usdt_amount': POSITION_USDT_SIZE, 'entry_price': entry_price, 'exit_price': 0,
                'size': size, 'created_at': order_info['entry_time'], 'updated_at': now_aware,
                'pnl_usdt': 0, 'pnl_pct': 0, 'notes': note
            }
            self._write_order_to_db(update_record)
            
            del self.pending_orders[client_id] # 无论成功与否，都从待处理中移除

    def manage_positions(self):
        """
        监控所有活跃持仓，获取最新价格，并根据规则执行止盈或止损。
        """
        # 如果没有任何持仓，则直接返回，节省API调用
        if not self.open_positions:
            return

        # 1. 准备需要查询价格的币种列表
        symbols_to_check = list(self.open_positions.keys())
        logging.info(f"开始监控 {len(symbols_to_check)} 个持仓: {symbols_to_check}")

        # 2. 一次性获取所有持仓的最新价格
        #    get_latest_prices 函数应该返回一个 {'symbol': price} 格式的字典
        current_prices = self.trading_client.get_latest_prices(symbols_to_check)

        if not current_prices:
            logging.warning("未能获取到任何持仓的最新价格，本次监控跳过。")
            return

        # 3. 逐一遍历每个持仓，独立进行判断和操作
        #    使用 list(self.open_positions.items()) 是为了安全地在循环中修改字典
        for symbol, position in list(self.open_positions.items()):
            
            current_price = current_prices.get(symbol)
            
            # 如果某个币的价格未能获取到，则跳过对它的处理
            if current_price is None:
                logging.warning(f"未能获取到 [{symbol}] 的价格，跳过本次监控。")
                continue

            entry_price = position['entry_price']
            pnl_pct = (current_price / entry_price) - 1
            
            should_close = False
            close_reason = ""

            # 检查止盈条件
            if pnl_pct >= TAKE_PROFIT_PCT:
                should_close = True
                close_reason = f"止盈(Take Profit at {pnl_pct:.2%})"

            # 检查止损条件
            if pnl_pct <= -STOP_LOSS_PCT:
                should_close = True
                close_reason = f"止损(Stop Loss at {pnl_pct:.2%})"

            # 如果触发了任一平仓条件
            if should_close:
                logging.warning(f"[{symbol}] 触发平仓条件: {close_reason}")
                                
                # 调用您自己的平仓处理函数 (例如，更新持仓字典，记录日志等)
                self.execute_sell(symbol, position, current_price, close_reason)



    def execute_sell(self, symbol, position, exit_price, reason):
        """执行卖出平仓操作并记录到数据库"""
        sell_client_id = f"close{uuid.uuid4().hex[:10]}"
        # 调用真实的下单接口
        order_result = self.trading_client.place_market_order(symbol, 'sell', position['size'], sell_client_id)
        
        if order_result:
            pnl_pct = (exit_price / position['entry_price']) - 1
            # 更新数据库记录
            final_record = {
                'client_order_id': position['client_order_id'], 'exchange_order_id': order_result['data'][0]['ordId'], 
                'symbol': symbol, 'status': reason.split(' ')[0], 'side': 'buy', 'order_type': 'market',
                'usdt_amount': POSITION_USDT_SIZE, 'entry_price': position['entry_price'], 'exit_price': exit_price,
                'size': position['size'], 'created_at': now_aware, # 应该从数据库读取创建时间
                'updated_at': now_aware,
                'pnl_usdt': (exit_price - position['entry_price']) * position['size'],
                'pnl_pct': pnl_pct, 'notes': reason
            }
            self._write_order_to_db(final_record)
            
            # 从持仓字典中移除
            if symbol in self.open_positions:
                del self.open_positions[symbol]
            logging.warning(f"成功平仓: {symbol}")

    def shutdown(self, signum, frame):
        """处理程序退出信号"""
        logging.warning("接收到关闭信号，正在优雅地退出...")
        self.is_shutting_down = True
        # 这里可以加入更复杂的逻辑，比如取消所有挂单
        logging.warning("程序已退出。")
        sys.exit(0)
    
    def run(self):

        # 注册信号处理器
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        """主循环"""
        while not self.is_shutting_down:
            now = datetime.datetime.now()
            seconds_to_wait = (60 - now.second - 1) + (1 - now.microsecond / 1_000_000) + SCAN_AT_SECOND
            if seconds_to_wait < 0: seconds_to_wait += 60

            logging.info(f"等待 {seconds_to_wait:.2f} 秒... 当前持仓: {len(self.open_positions)}, 待处理: {len(self.pending_orders)}")
            time.sleep(seconds_to_wait)
            
            try:
                self.check_pending_orders()
                self.manage_positions()
                self.scan_for_spikes()
            except Exception as e:
                logging.error(f"主循环发生严重错误: {e}")

    def main_loop(self,dry_run_mode=True):
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        if not self.ch_client: return

        # 只初始化一次交易管理器
        trade_manager = TradeManager(self.ch_client, dry_run=dry_run_mode)
        
        while True:
            now = datetime.datetime.now()
            seconds_to_wait = (60 - now.second - 1) + (1 - now.microsecond / 1_000_000) + SCAN_AT_SECOND
            if seconds_to_wait < 0: seconds_to_wait += 60

            logging.info(f"等待 {seconds_to_wait:.2f} 秒... 当前持仓: {len(self.open_positions)}, 待处理: {len(self.pending_orders)}")
            time.sleep(seconds_to_wait)

            if self.ch_client:
                try:
                    # 1. 让交易大脑去管理持仓 (止盈止损)
                    trade_manager.manage_open_positions()

                    # 2. 扫描新信号
                    signals_df = self.scan_for_spikes()
                    if signals_df is not None and not signals_df.empty:
                        # 3. 将新信号交给交易大脑处理 (建仓)
                        trade_manager.process_new_signals(signals_df)
                finally:
                    self.ch_client.close()
                    logging.info("本轮扫描/监控完成。")
           

if __name__ == "__main__":
    trader = SpikeTrader()
    #trader.run()
    trader.main_loop(dry_run_mode=False)