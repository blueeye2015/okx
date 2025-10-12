import requests, psycopg2, os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('.env')
API_KEY = os.getenv("FRED_API_KEY")
DB_DSN  = os.getenv("DB_DSN")

vintage_date = "2025-09-30"
series_id    = "PAYEMS"

url = (f"https://api.stlouisfed.org/fred/series/observations"   # ← 这里去掉空格
       f"?series_id={series_id}&api_key={API_KEY}&file_type=json"
       f"&output_type=2&vintage_date={vintage_date}")

data = requests.get(url, timeout=30).json()
rows = [(series_id,
         datetime.strptime(o["date"], "%Y-%m-%d").date(),
         vintage_date,
         float(o["value"]) if o["value"] != "." else None)
        for o in data["observations"]]

with psycopg2.connect(DB_DSN) as conn:
    with conn.cursor() as cur:
        from psycopg2.extras import execute_values
        execute_values(cur, """
            INSERT INTO fred_vintage (series_id, value_date, vintage_date, value)
            VALUES %s
            ON CONFLICT (series_id, value_date, vintage_date) DO NOTHING;
        """, rows)
print(f"已插入 {len(rows)} 条 {vintage_date} 修正快照")