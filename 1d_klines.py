import requests
import time
import datetime
import logging
import clickhouse_connect

# --- 配置区 ---
START_DATE = "2022-09-14" # 开始日期 (获取大约一年的数据)
SYMBOL_FILE = "/data/okx/symbol.txt" # 存放币种列表的文件名

# ClickHouse数据库配置
CH_HOST = 'localhost'
CH_PORT = 8123
CH_DATABASE = 'marketdata'
CH_USERNAME = 'default'
CH_PASSWORD = '12'

# 代理配置 (如果不需要，请设为 None)
proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据你的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据你的实际代理地址修改
        } 
# PROXY = None

# OKX API 配置
BASE_URL = "https://www.okx.com"
API_ENDPOINT = "/api/v5/market/candles"
URL = BASE_URL + API_ENDPOINT

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_symbols(filename):
    """从文件中加载交易对列表"""
    try:
        with open(filename, 'r') as f:
            # 读取每一行，去除空白，转换为大写，并过滤掉空行
            symbols = [line.strip().upper() for line in f if line.strip()]
            if not symbols:
                logging.warning(f"'{filename}' 文件为空, 未加载任何交易对。")
                return []
            logging.info(f"成功从 '{filename}' 加载 {len(symbols)} 个交易对。")
            return symbols
    except FileNotFoundError:
        logging.error(f"错误: '{filename}' 文件未找到。请创建该文件并填入交易对。")
        return []

def get_clickhouse_client():
    """创建并返回一个ClickHouse客户端连接"""
    try:
        client = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT, database=CH_DATABASE,
            username=CH_USERNAME, password=CH_PASSWORD
        )
        logging.info(f"成功连接到ClickHouse数据库 (host: {CH_HOST})")
        return client
    except Exception as e:
        logging.error(f"连接ClickHouse失败: {e}")
        raise

def get_historical_daily_data(client, symbol, start_date, proxies=None):
    """获取指定币种的日线数据并直接存入ClickHouse"""
    all_data_to_insert = []
    start_ts = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = ''

    while True:
        params = {"instId": symbol, "bar": "1D", "limit": "100", "after": end_ts}
        try:
            # print(f"[{symbol}] 正在获取时间戳 {end_ts} 之前的数据...") # 可以取消注释以获得更详细的日志
            response = requests.get(URL,proxies=proxies,params=params)
            response.raise_for_status()
            data = response.json()

            if data['code'] != '0' or not data['data']:
                logging.info(f"[{symbol}] 没有更多数据或API返回错误。")
                break

            kline_list = data['data']
            
            for kline in kline_list:
                formatted_kline = [
                    datetime.datetime.fromtimestamp(int(kline[0]) / 1000),
                    symbol,
                    float(kline[1]), float(kline[2]), float(kline[3]),
                    float(kline[4]), float(kline[5])
                ]
                all_data_to_insert.append(formatted_kline)
            
            oldest_ts_in_batch = int(kline_list[-1][0])
            end_ts = str(oldest_ts_in_batch)

            if oldest_ts_in_batch < start_ts:
                logging.info(f"[{symbol}] 已获取所有指定日期范围内的数据。")
                break
            
            time.sleep(0.2)

        except requests.exceptions.RequestException as e:
            logging.error(f"[{symbol}] 请求失败: {e}")
            break
        except Exception as e:
            logging.error(f"[{symbol}] 发生错误: {e}")
            break
            
    if all_data_to_insert:
        try:
            column_names = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            client.insert('okx_klines_1d', all_data_to_insert, column_names=column_names)
            logging.info(f"[{symbol}] 成功将 {len(all_data_to_insert)} 条日线数据插入ClickHouse。")
        except Exception as e:
            logging.error(f"[{symbol}] 插入数据到ClickHouse时发生错误: {e}")

if __name__ == "__main__":
    # --- 核心改动在这里 ---
    # 1. 从 symbol.txt 加载币种列表
    symbols_to_process = load_symbols(SYMBOL_FILE)

    if not symbols_to_process:
        logging.info("没有需要处理的币种，程序退出。")
    else:
        # 2. 连接数据库
        ch_client = get_clickhouse_client()
        if ch_client:
            total_symbols = len(symbols_to_process)
            # 3. 循环处理文件中的每一个币种
            for i, symbol in enumerate(symbols_to_process, 1):
                logging.info(f"--- 正在处理第 {i}/{total_symbols} 个币种: {symbol} ---")
                get_historical_daily_data(ch_client, symbol, START_DATE, proxies)
                time.sleep(0.5) # 每个币种之间也稍微停顿一下，更加稳妥
            
            ch_client.close()
            logging.info("所有币种处理完毕，数据库连接已关闭。")