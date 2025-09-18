# test_spike_trader.py

import logging
import datetime
import uuid
import pandas as pd # 需要pandas来插入测试数据
from unittest.mock import MagicMock
from spike_trader import SpikeTrader, get_clickhouse_client # 导入主类和数据库连接

# 从您的主策略文件中导入SpikeTrader类
from spike_trader import SpikeTrader

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 模拟组件 ---

class MockTradingClient:
    """一个模拟的交易客户端，用于测试，不会发送真实订单。"""
    def __init__(self):
        self.submitted_orders = {}
        logging.info("【测试模式】模拟交易客户端已初始化。")

    def place_market_order(self, symbol, side, size, client_order_id):
        logging.warning(f"【测试模式】模拟下单: {side.upper()} {size:.6f} {symbol} (ID: {client_order_id})")
        # 模拟交易所接受订单
        self.submitted_orders[client_order_id] = {
            'symbol': symbol,
            'side': side,
            'size': size,
            'status': 'submitted'
        }
        # 模拟API成功返回
        return {'code': '0', 'data': [{'ordId': f'sim_{uuid.uuid4().hex[:8]}', 'clOrdId': client_order_id}]}

    def check_order_status(self, symbol, client_order_id):
        logging.info(f"【测试模式】模拟查询订单: {client_order_id}")
        if client_order_id in self.submitted_orders:
            # 模拟订单已成交
            order = self.submitted_orders[client_order_id]
            return {
                'state': 'filled',
                'avgPx': str(110.0), # 假设成交价为 110
                'accFillSz': str(order['size']),
                'ordId': f'sim_{uuid.uuid4().hex[:8]}'
            }
        return None

def setup_test_data(client, symbol, spike_price):
    """向数据库插入一条用于触发信号的测试K线"""
    logging.info(f"正在向数据库插入测试数据 for {symbol}...")
    try:
        # 清理可能存在的旧测试数据
        client.command(f"ALTER TABLE marketdata.okx_klines_1m DELETE WHERE symbol = '{symbol}' SETTINGS mutations_sync=2")
        
        # 构造一条暴涨超过10%的K线 (100 -> 110.1)
        test_kline = [{
            'timestamp': datetime.datetime.now() - datetime.timedelta(minutes=1),
            'symbol': symbol,
            'open': spike_price * 0.9, # 100
            'high': spike_price,       # 110.1
            'low': spike_price * 0.9,  # 100
            'close': spike_price,      # 110.1
            'volume': 50000
        }]
        client.insert_df('marketdata.okx_klines_1m', pd.DataFrame(test_kline))
        logging.info("测试数据插入成功！")
    except Exception as e:
        logging.error(f"插入测试数据失败: {e}")
        raise

# --- 主测试流程 ---
def run_full_test():
    print("\n" + "="*50)
    print("--- 开始全流程自动化测试 ---")
    print("="*50 + "\n")

    # 1. 初始化
    test_symbol = 'TEST-USDT-SWAP'
    mock_trader = MockTradingClient()
    strategy_bot = SpikeTrader(trading_client=mock_trader)
    
    # 2. 准备数据
    setup_test_data(strategy_bot.ch_client, test_symbol, 110.1)

    # --- 3. 模拟一分钟的完整流程 ---
    
    # a. 扫描信号并下单
    print("\n--- [测试步骤1/3] 扫描信号并执行买入 ---")
    strategy_bot.scan_for_spikes()
    assert len(strategy_bot.pending_orders) == 1, "测试失败：未将订单加入待处理列表！"
    print("✅ 成功：策略已发现信号并提交模拟订单。")

    # b. 确认订单成交
    print("\n--- [测试步骤2/3] 确认订单成交状态 ---")
    pending_id = list(strategy_bot.pending_orders.keys())[0]
    strategy_bot.pending_orders[pending_id]['entry_time'] -= datetime.timedelta(seconds=61)
    strategy_bot.check_pending_orders()
    assert len(strategy_bot.pending_orders) == 0, "测试失败：未从待处理列表移除订单！"
    assert len(strategy_bot.open_positions) == 1, "测试失败：未将订单移入持仓列表！"
    print("✅ 成功：订单已确认为成交，并加入持仓监控。")
    
    # c. 监控持仓并触发止损
    print("\n--- [测试步骤3/3] 监控持仓并触发止损 ---")
    # 模拟价格下跌超过8%
    position = strategy_bot.open_positions[test_symbol]
    stop_loss_price = position['entry_price'] * (1 - 0.09)
    mock_prices = {test_symbol: stop_loss_price}
    
    # 【核心修正】: 将模拟价格传入监控函数
    strategy_bot.manage_positions(current_prices=mock_prices)
    
    assert len(strategy_bot.open_positions) == 0, "测试失败：未能成功触发止损平仓！"
    print("✅ 成功：策略已监控到价格下跌并执行模拟止损平仓。")

    print("\n" + "="*50)
    print("--- 所有测试步骤成功通过！ ---")
    print("="*50 + "\n")


if __name__ == '__main__':
    # 为了让测试能运行，需要一个 ClickHouse 的 DataFrame 模块
    import pandas as pd
    run_full_test()