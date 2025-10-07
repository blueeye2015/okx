import clickhouse_connect
from clickhouse_connect.driver.exceptions import ClickHouseError
import requests
from tqdm import tqdm
from loguru import logger
import time
from pathlib import Path
import datetime

# --- 配置区 ---
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 8123,
    'user': 'default',
    'password': '12',
    'database': 'marketdata'
}
SYMBOL_FILE = 'symbol.txt'
TABLE_NAME = 'okx_klines_1d'
BAR_INTERVAL = '1D'
# 如果数据库为空，从这个日期开始获取历史数据
DEFAULT_START_DATE = "2017-01-01"
proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据你的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据你的实际代理地址修改
        } 
# --- 函数区 ---
def load_symbols_from_file(filepath: str) -> list[str]:
    if not Path(filepath).is_file():
        logger.error(f"交易对文件 '{filepath}' 未找到。请在脚本同目录下创建该文件。")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("BTC-USDT\n")
            f.write("ETH-USDT\n")
        logger.info(f"已自动创建示例文件 '{filepath}'，请根据需要修改。")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            symbols = [line.strip() for line in f if line.strip()]
        if not symbols:
            logger.warning(f"文件 '{filepath}' 为空或不包含有效交易对。")
            return []
        logger.info(f"从 '{filepath}' 成功加载 {len(symbols)} 个交易对。")
        return symbols
    except Exception as e:
        logger.error(f"读取文件 '{filepath}' 时出错: {e}")
        return []

# --- 核心实现类 (按旧逻辑重写) ---
class OkxHistoryFetcher:
    BASE_URL = 'https://www.okx.com/api/v5/market/history-candles'
    
    def __init__(self, db_config: dict, table_name: str, bar: str = '1d'):
        self.table_name = f"{db_config.get('database', 'default')}.{table_name}"
        self.bar = bar
        try:
            self.client = clickhouse_connect.get_client(**db_config)
            logger.info("成功连接到ClickHouse数据库.")
        except Exception as e:
            logger.error(f"连接ClickHouse数据库失败: {repr(e)}")
            raise
            
        if not self.client.command(f'EXISTS TABLE {self.table_name}'):
            logger.error(f"数据表 '{self.table_name}' 不存在，请先创建。")
            raise ValueError(f"Table '{self.table_name}' does not exist.")

    def _get_latest_timestamp_ms(self, symbol: str) -> int:
        """从数据库获取指定symbol的最新时间戳(毫秒)"""
        query = f"SELECT min(timestamp) FROM {self.table_name} WHERE symbol = '{symbol}'"
        try:
            result = self.client.query(query)
            if result.result_rows and result.result_rows[0][0]:
                latest_dt = result.result_rows[0][0]
                latest_ts_ms = int(latest_dt.timestamp() * 1000)
                logger.info(f"[{symbol}] 数据库中最新时间为: {latest_dt} (ts: {latest_ts_ms})")
                return latest_ts_ms
        except Exception as e:
            logger.error(f"[{symbol}] 查询数据库最新时间失败: {repr(e)}")

        # 如果没有数据，返回默认开始日期的毫秒时间戳
        start_ts = int(datetime.datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d").timestamp() * 1000)
        logger.info(f"[{symbol}] 数据库中无数据，将从默认日期 {DEFAULT_START_DATE} 开始获取。")
        return start_ts

    def fetch_and_store_symbol(self, symbol: str):
        """为单个交易对拉取并存储数据"""
        after_ts = self._get_latest_timestamp_ms(symbol)
        all_data_to_insert = []
        
        logger.info(f"[{symbol}] 开始从时间戳 {after_ts} 之后获取数据...")

        while True:
            params = {'instId': symbol, 'bar': self.bar, 'after': str(after_ts), 'limit': '100'}
            
            try:
                response = requests.get(self.BASE_URL, proxies=proxies, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data.get('code') != '0' or not data.get('data'):
                    logger.info(f"[{symbol}] 没有更多新数据或API返回错误。")
                    break

                kline_list = data['data']
                
                # *** 核心逻辑：逐行处理 ***
                for kline in kline_list:
                    try:
                        # 尝试转换每一行的数据，失败则中断
                        formatted_kline = [
                            datetime.datetime.fromtimestamp(int(kline[0]) / 1000), # timestamp
                            symbol,
                            float(kline[1]), # open
                            float(kline[2]), # high
                            float(kline[3]), # low
                            float(kline[4]), # close
                            float(kline[5]), # volume
                        ]
                        all_data_to_insert.append(formatted_kline)
                    except (ValueError, TypeError) as e:
                        logger.error(f"[{symbol}] K线数据格式错误，停止处理。错误: {repr(e)}。问题数据行: {kline}")
                        # 使用 goto 风格的标志来跳出外层循环
                        raise StopIteration # 抛出一个特定的异常来中断

                # 更新下一轮请求的起始时间戳
                after_ts = int(kline_list[-1][0])
                time.sleep(0.2)

            except StopIteration:
                # 捕获我们自己抛出的异常，以正常中断循环
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"[{symbol}] 网络请求失败: {repr(e)}")
                break # 网络失败，中断
            except Exception as e:
                logger.error(f"[{symbol}] 发生未知错误: {repr(e)}")
                break

        # 循环结束后，统一插入数据
        if all_data_to_insert:
            logger.info(f"[{symbol}] 准备将 {len(all_data_to_insert)} 条新数据插入数据库...")
            try:
                column_names = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                self.client.insert(self.table_name, all_data_to_insert, column_names=column_names)
                logger.success(f"[{symbol}] 成功插入 {len(all_data_to_insert)} 条数据。")
            except Exception as e:
                logger.error(f"[{symbol}] 插入数据到ClickHouse时发生错误: {repr(e)}")

# --- 主程序入口 ---
if __name__ == "__main__":
    logger.add("history_fetcher_legacy_logic.log", rotation="10 MB")
    
    SYMBOLS_TO_FETCH = load_symbols_from_file(SYMBOL_FILE)

    if not SYMBOLS_TO_FETCH:
        logger.warning("交易对列表为空，程序退出。请检查 symbol.txt 文件。")
    else:
        logger.info("开始执行历史数据获取任务 (旧逻辑版)...")
        
        # 创建一个Fetcher实例
        fetcher = OkxHistoryFetcher(
            db_config=DB_CONFIG,
            table_name=TABLE_NAME,
            bar=BAR_INTERVAL
        )
        
        for symbol in tqdm(SYMBOLS_TO_FETCH, desc="总体进度"):
            try:
                fetcher.fetch_and_store_symbol(symbol)
            except Exception as e:
                logger.error(f"处理交易对 {symbol} 时发生严重错误: {repr(e)}")
                
        logger.info("所有任务执行完毕。")