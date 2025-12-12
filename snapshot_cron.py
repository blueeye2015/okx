#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[程序 B] 定期快照抓取器
功能：每隔 60 秒访问 REST API，获取完整 OrderBook 并存入 depth_snapshot 表。
特点：频率可控，避免 418 限流。
"""
import time
import requests
import logging
import clickhouse_connect
from datetime import datetime

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
REST_URL = "https://api.binance.com/api/v3/depth"
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
LIMIT = 1000  # 获取深度档位，最大 5000
INTERVAL = 60 # 抓取间隔 (秒)
PROXY = {"https": "http://127.0.0.1:7890"} # requests 代理格式

# --- 日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SnapshotCron")

def get_snapshot(symbol):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    }
    params = {'symbol': symbol.upper(), 'limit': LIMIT}
    
    try:
        resp = requests.get(REST_URL, params=params, headers=headers, proxies=PROXY, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"❌ 获取 {symbol} 快照失败: {e}")
        return None

def save_to_db(client, symbol, data):
    if not data: return
    
    snapshot_time = datetime.now()
    update_id = data['lastUpdateId']
    rows = []
    
    # 解析 Bids
    for px, qty in data['bids']:
        rows.append((snapshot_time, symbol, 'bid', float(px), float(qty), update_id))
    # 解析 Asks
    for px, qty in data['asks']:
        rows.append((snapshot_time, symbol, 'ask', float(px), float(qty), update_id))
        
    try:
        client.insert(
            'depth_snapshot',
            rows,
            column_names=['snapshot_time', 'symbol', 'side', 'price', 'quantity', 'update_id']
        )
        logger.info(f"📸 {symbol} 快照已保存 (id={update_id})")
    except Exception as e:
        logger.error(f"❌ 数据库写入失败: {e}")

def main():
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    logger.info("🚀 快照抓取任务已启动...")
    
    while True:
        for symbol in SYMBOLS:
            data = get_snapshot(symbol)
            save_to_db(client, symbol, data)
            time.sleep(1) # 币种之间稍微停顿一下，避免并发太高
            
        logger.info(f"😴 休眠 {INTERVAL} 秒...")
        time.sleep(INTERVAL)

if __name__ == '__main__':
    main()