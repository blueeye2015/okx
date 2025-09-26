# test_spike_trader.py

import logging
import pandas as pd
import clickhouse_connect
from trade_manager import TradeManager # 导入我们的大脑

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_clickhouse_client():
    """连接到ClickHouse"""
    try:
        # 确保这里的连接信息是正确的
        client = clickhouse_connect.get_client(
            host='localhost', port=8123, database='marketdata',
            username='default', password='12')
        return client
    except Exception as e:
        logging.error(f"连接ClickHouse失败: {e}")
        return None

def create_mock_signals():
    """创建一个包含各种币种的模拟信号 DataFrame"""
    data = {
        'symbol': [
            'BTC-USDT-SWAP', # Large Cap
            'ETH-USDT-SWAP', # Large Cap
            'SOL-USDT-SWAP', # Mid Cap
            'DOGE-USDT-SWAP',# Mid Cap
            'PEPE-USDT',     # Small Cap
            'ORDI-USDT',     # Small Cap
            'MERL-USDT',     # Small Cap
            'WIF-USDT-SWAP', # Small Cap
            'NOT-USDT',      # Mid Cap
            'TON-USDT'       # Large Cap
        ],
        'current_price': [
            68000, 3500, 160, 0.15, 0.000012, 40, 0.3, 2.5, 0.015, 7.5
        ],
        'RVol': [ # 我们故意让一些小市值的RVol更高，测试排序
            3.1, 3.5, 4.0, 3.8, 5.5, 4.8, 6.0, 5.2, 3.2, 3.3
        ]
    }
    df = pd.DataFrame(data)
    logging.info("已创建模拟信号 DataFrame:")
    print(df.to_string())
    return df

def run_simulation():
    """运行完整的交易逻辑模拟演习"""
    ch_client = get_clickhouse_client()
    if not ch_client:
        logging.error("无法连接到数据库，测试终止。")
        return

    # 1. 初始化交易管理器，强制使用“空跑模式”
    trade_manager = TradeManager(ch_client, dry_run=True)
    
    # 2. 创建模拟信号
    mock_signals_df = create_mock_signals()

    # --- 场景一: 初始没有任何持仓 ---
    print("\n" + "="*50)
    logging.warning("场景一: 初始没有任何持仓，处理10个新信号")
    print("="*50)
    
    trade_manager.process_new_signals(mock_signals_df)
    
    print("\n--- 场景一结束后，当前模拟持仓 ---")
    print(trade_manager.open_positions)
    expected_count = trade_manager.total_max_positions
    print(f"预期持仓数量: {expected_count} | 实际持仓数量: {len(trade_manager.open_positions)}")
    if len(trade_manager.open_positions) == expected_count:
        logging.info("场景一测试通过: 持仓数量符合预期。")
    else:
        logging.error("场景一测试失败: 持仓数量不符合预期！")


    # --- 场景二: 已有部分持仓 ---
    print("\n" + "="*50)
    logging.warning("场景二: 已持有2个小市值和1个大市值币，再次处理10个新信号")
    print("="*50)

    # 手动设置一些已有的持仓
    trade_manager.open_positions = {
        'PEPE-USDT': {'entry_price': 0.000011, 'size': 9090909},
        'WIF-USDT-SWAP': {'entry_price': 2.4, 'size': 41.67},
        'BTC-USDT-SWAP': {'entry_price': 67000, 'size': 0.00149}
    }
    logging.info("已手动设置初始持仓:")
    print(trade_manager.open_positions)
    
    trade_manager.process_new_signals(mock_signals_df)

    print("\n--- 场景二结束后，当前模拟持仓 ---")
    print(trade_manager.open_positions)
    print(f"预期持仓数量: {expected_count} | 实际持仓数量: {len(trade_manager.open_positions)}")
    if len(trade_manager.open_positions) == expected_count:
        logging.info("场景二测试通过: 成功填补了剩余仓位。")
    else:
        logging.error("场景二测试失败: 持仓数量不符合预期！")

    ch_client.close()

if __name__ == "__main__":
    run_simulation()