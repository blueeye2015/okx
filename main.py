# main.py
import logging
import time
import clickhouse_connect
from trade_manager import TradeManager
from trade_executor import TradingClient # 需要导入TradingClient
from momentum_scanner import scan_and_rank_momentum

# --- 全局配置 ---
CLICKHOUSE_CONFIG = {'host': '127.0.0.1', 'port': 8123, 'user': 'default', 'password': '12'}
SCAN_INTERVAL_SECONDS = 3600*24*7
MANAGE_INTERVAL_SECONDS = 3600*24
DRY_RUN = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_symbols_to_scan(client):
    """【已修正】从数据库获取所有需要扫描的现货币种列表"""
    try:
        # 假设您的现货交易对都以 -USDT 结尾
        df = client.query_df("SELECT DISTINCT symbol FROM marketdata.okx_klines_1d WHERE symbol LIKE '%-USDT'")
        symbols = df['symbol'].tolist()
        logging.info(f"从数据库获取到 {len(symbols)} 个待扫描币种。")
        return symbols
    except Exception as e:
        logging.error(f"获取币种列表失败: {e}")
        return []

def main():
    """程序主入口"""
    # ... (代码与之前版本完全相同，无需改动)
    logging.info("启动量化交易程序...")
    try:
        ch_client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
        ch_client.ping()
        logging.info("ClickHouse连接成功。")
    except Exception as e:
        logging.error(f"无法连接到ClickHouse，程序退出: {e}")
        return
    # --- 核心改动：先创建，再传入 ---
    trading_client = TradingClient() # 在外部创建实盘交易客户端
    manager = TradeManager(ch_client, trading_client, dry_run=DRY_RUN)
    symbols_to_scan = get_symbols_to_scan(ch_client)
    if not symbols_to_scan:
        logging.error("没有要扫描的币种，程序退出。")
        return

    last_scan_time = 0
    while True:
        try:
            current_time = time.time()
            if current_time - last_scan_time >= SCAN_INTERVAL_SECONDS:
                logging.warning("="*50)
                logging.warning("开始新一轮信号扫描与排名...")
                ranked_signals_df = scan_and_rank_momentum(ch_client, symbols_to_scan)
                manager.process_new_signals(ranked_signals_df)
                last_scan_time = current_time
                logging.warning("本轮信号处理完毕。")
                logging.warning("="*50)

            manager.manage_open_positions()
            time.sleep(MANAGE_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logging.warning("接收到手动中断信号，程序退出。")
            break
        except Exception as e:
            logging.error(f"主循环发生未捕获异常: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()