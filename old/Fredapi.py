#!/usr/bin/env python3
import os, requests, psycopg2
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('.env')
API_KEY = os.getenv("FRED_API_KEY") 
DB_DSN  = os.getenv("DB_DSN")

SERIES_LIST = ["PAYEMS","CPIAUCSL","CBBTCUSD"]   # 想拉就加
LIMIT = 1000

def fetch_all_vintage(series_id, offset=0):
    """不限制 realtime，拉全部 vintage"""
    url = (f"https://api.stlouisfed.org/fred/series/observations"
       f"?series_id={series_id}&api_key={API_KEY}&file_type=json"
       f"&limit={LIMIT}&offset={offset}&sort_order=asc")
    return requests.get(url, timeout=30).json()

def write_vintage(rows):
    sql = """
    INSERT INTO fred_vintage (series_id, value_date, vintage_date, value)
    VALUES %s
    ON CONFLICT (series_id, value_date, vintage_date) DO NOTHING;
    """
    with psycopg2.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            from psycopg2.extras import execute_values
            execute_values(cur, sql, rows, page_size=1000)
    print(f"写入 {len(rows)} 条 vintage")

def sync_vintage(series_id):
    offset, total = 0, 0
    while True:
        data = fetch_all_vintage(series_id, offset)
        obs = data.get("observations", [])
        if not obs:
            break
        rows = [(series_id,
                 datetime.strptime(o["date"], "%Y-%m-%d").date(),
                 datetime.strptime(o["realtime_start"], "%Y-%m-%d").date(),
                 float(o["value"]) if o["value"] != "." else None)
                for o in obs]
        write_vintage(rows)
        total += len(rows)
        offset += LIMIT
    print(f"{series_id} vintage 完成，共 {total} 条")

if __name__ == "__main__":
    for sid in SERIES_LIST:
        sync_vintage(sid)