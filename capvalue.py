import requests
import time
import datetime
import logging
import clickhouse_connect
import pandas as pd

# --- 配置区 ---
SYMBOL_FILE = "symbol.txt" # 我们关心的币种列表文件

# 代理配置 (如果不需要，请设为 None)
proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据你的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据你的实际代理地址修改
        } 

# ClickHouse数据库配置
CH_HOST = 'localhost'
CH_PORT = 8123
CH_DATABASE = 'marketdata'
CH_USERNAME = 'default'
CH_PASSWORD = '12'

# CoinGecko API 配置
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
PAGES_TO_FETCH = 6 # 获取4页，约1000个币种

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_symbols(filename):
    """从文件中加载我们关心的交易对的 'base currency'"""
    try:
        with open(filename, 'r') as f:
            # 例如 "BTC-USDT-SWAP" -> "BTC"
            symbols = [line.strip().upper().split('-')[0] for line in f if line.strip()]
            # 去重
            unique_symbols = sorted(list(set(symbols)))
            if not unique_symbols:
                logging.warning(f"'{filename}' 文件为空或格式不正确。")
                return []
            logging.info(f"成功从 '{filename}' 加载 {len(unique_symbols)} 个目标币种。")
            return unique_symbols
    except FileNotFoundError:
        logging.error(f"错误: '{filename}' 文件未找到。")
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

def get_and_store_market_caps(client, target_symbols,proxies):
    """从CoinGecko获取市值，在内存中去重和筛选后，再存入ClickHouse"""
    all_coins_data = []
    
    # 1. 从CoinGecko获取原始数据
    for page_num in range(1, PAGES_TO_FETCH + 1): 
        params = {
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": 250, "page": page_num, "sparkline": "false"
        }
        try:
            logging.info(f"正在获取第 {page_num} 页的市值数据...")
            response = requests.get(COINGECKO_URL,proxies=proxies, params=params)
            response.raise_for_status()
            data = response.json()
            if not data: break
            all_coins_data.extend(data)
            time.sleep(1.5)
        except Exception as e:
            logging.error(f"从CoinGecko获取数据时发生错误: {e}")
            return
            
    if not all_coins_data:
        logging.warning("未能从CoinGecko获取到任何数据。")
        return

    # 2. 使用Pandas在内存中进行智能去重
    logging.info("开始在内存中处理和去重...")
    df = pd.DataFrame(all_coins_data)
    df['symbol'] = df['symbol'].str.upper()
    
    # --- 核心去重逻辑 ---
    # 1. 按市值降序排序
    # 2. 对'symbol'列进行去重，只保留第一个出现的（也就是市值最高的那个）
    df_cleaned = df.sort_values('market_cap', ascending=False).drop_duplicates(subset=['symbol'], keep='first')
    
    # 3. 只保留我们symbol.txt中关心的币种
    df_final = df_cleaned[df_cleaned['symbol'].isin(target_symbols)]
    logging.info(f"去重和筛选后，得到 {len(df_final)} 个币种的准确市值数据。")

    # 4. 准备写入ClickHouse的数据
    all_coins_to_insert = []
    today = datetime.date.today()
    for index, row in df_final.iterrows():
        formatted_coin = [
            row.get('symbol', ''),
            row.get('id', ''),
            row.get('name', ''),
            float(row.get('current_price', 0.0) or 0.0),
            float(row.get('market_cap', 0.0) or 0.0),
            float(row.get('total_volume', 0.0) or 0.0),
            today
        ]
        all_coins_to_insert.append(formatted_coin)
            
    # 5. 写入ClickHouse
    if all_coins_to_insert:
        try:
            column_names = ['symbol', 'id', 'name', 'current_price', 'market_cap', 'total_volume', 'last_updated']
            # 先清空表，再插入干净的数据。因为ReplacingMergeTree的替换不是实时的。
            client.command(f'TRUNCATE TABLE {CH_DATABASE}.coin_info')
            logging.info("已清空 coin_info 表，准备写入新数据。")
            client.insert('coin_info', all_coins_to_insert, column_names=column_names)
            logging.info(f"成功将 {len(all_coins_to_insert)} 条准确的市值数据写入ClickHouse。")
        except Exception as e:
            logging.error(f"插入数据到ClickHouse时发生错误: {e}")

if __name__ == "__main__":
    # 从 symbol.txt 加载目标币种
    symbols_we_care_about = load_symbols(SYMBOL_FILE)
    
    if symbols_we_care_about:
        ch_client = get_clickhouse_client()
        if ch_client:
            get_and_store_market_caps(ch_client, symbols_we_care_about,proxies)
            ch_client.close()
            logging.info("市值数据处理完毕，数据库连接已关闭。")