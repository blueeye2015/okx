import clickhouse_connect
import pandas as pd
from datetime import datetime, timedelta
import time

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL_DEPTH = 'BTCUSDT'
SYMBOL_TRADE = 'BTC-USDT'
BATCH_DAYS = 1  # 每次计算 1 天的数据 (防止内存溢出)

def get_date_range(client):
    """获取 depth 表中数据的最早和最晚时间"""
    sql = f"SELECT min(event_time), max(event_time) FROM marketdata.depth WHERE symbol = '{SYMBOL_DEPTH}'"
    result = client.query(sql).result_rows
    start_date = result[0][0]
    end_date = result[0][1]
    
    # 如果数据里有 1970 年的脏数据，强行修正起始时间
    if start_date.year < 2020:
        print(f"⚠️ 发现早期脏数据 ({start_date})，修正起始时间为 2024-01-01")
        start_date = datetime(2024, 1, 1)
        
    return start_date, end_date

def backfill_features():
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    
    start_date, end_date = get_date_range(client)
    print(f"📅 数据范围: {start_date} -> {end_date}")
    
    current_date = start_date
    
    while current_date < end_date:
        next_date = current_date + timedelta(days=BATCH_DAYS)
        
        # 格式化时间字符串用于 SQL
        t_start = current_date.strftime('%Y-%m-%d %H:%M:%S')
        t_end = next_date.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"🔄 正在处理: {t_start} -> {t_end} ...")
        
        # ----------------------------------------------------
        # 核心 SQL: 插入 features_15m (逻辑与训练时一致)
        # ----------------------------------------------------
        sql = f"""
        INSERT INTO marketdata.features_15m
        WITH
            -- 1. 资金流向 (CVD)
            Trades AS (
                SELECT
                    toStartOfInterval(event_time, INTERVAL 15 MINUTE) as time,
                    argMax(price, event_time) as close_price,
                    sum(if(buyer_order_maker = 0, quantity, 0)) as buy_vol,
                    sum(if(buyer_order_maker = 1, quantity, 0)) as sell_vol,
                    buy_vol - sell_vol as net_cvd
                FROM marketdata.trades
                WHERE symbol = '{SYMBOL_TRADE}' 
                  AND event_time >= '{t_start}' AND event_time < '{t_end}'
                GROUP BY time
            ),
            
            -- 2. 欺诈撤单 (Spoofing) - 针对大数据的优化写法
            Spoofing AS (
                SELECT
                    toStartOfInterval(event_time, INTERVAL 15 MINUTE) as time,
                    sum(if(side='ask' AND delta < 0, abs(delta), 0)) as ask_withdraw_vol,
                    sum(if(side='bid' AND delta < 0, abs(delta), 0)) as bid_withdraw_vol,
                    if(bid_withdraw_vol > 0, ask_withdraw_vol / bid_withdraw_vol, 1.0) as spoofing_ratio
                FROM (
                    SELECT 
                        event_time, side, price,
                        quantity - lagInFrame(quantity) OVER (ORDER BY side, price, event_time) as delta,
                        price - lagInFrame(price) OVER (ORDER BY side, price, event_time) as price_diff
                    FROM marketdata.depth
                    WHERE symbol = '{SYMBOL_DEPTH}' 
                      AND event_time >= '{t_start}' AND event_time < '{t_end}'
                )
                WHERE delta < -1.0 AND price_diff = 0
                GROUP BY time
            ),
            
            -- 3. 墙的移动 (Snapshot)
            -- 注意：如果你以前没有跑 snapshot 脚本，这部分可能是空的，我们用 LEFT JOIN 兼容
            Snapshots AS (
                SELECT
                    toStartOfInterval(snapshot_time, INTERVAL 15 MINUTE) as time,
                    argMax(price, snapshot_time) as close_bid_price
                FROM marketdata.depth_snapshot
                WHERE symbol = '{SYMBOL_DEPTH}' AND side = 'bid'
                  AND snapshot_time >= '{t_start}' AND snapshot_time < '{t_end}'
                GROUP BY time
            )

        SELECT
            T.time,
            '{SYMBOL_DEPTH}' as symbol,
            T.close_price,
            
            -- 计算墙位移 % (如果没有快照数据，默认为 0)
            if(isNotNull(S.close_bid_price), (S.close_bid_price - lagInFrame(S.close_bid_price) OVER (ORDER BY T.time)) / S.close_bid_price * 100, 0) as wall_shift_pct,
            
            T.net_cvd,
            ifNull(SP.spoofing_ratio, 1.0) as spoofing_ratio,
            ifNull(SP.ask_withdraw_vol, 0) as ask_withdraw_vol,
            ifNull(SP.bid_withdraw_vol, 0) as bid_withdraw_vol
            
        FROM Trades AS T
        LEFT JOIN Spoofing AS SP ON T.time = SP.time
        LEFT JOIN Snapshots AS S ON T.time = S.time
        ORDER BY T.time
        """
        
        try:
            client.command(sql)
            print(f"✅ 完成. (已写入数据库)")
        except Exception as e:
            print(f"❌ 失败: {e}")
            # 可以在这里加 retry 逻辑
        
        current_date = next_date
        # 稍微休息一下，防止 ClickHouse 负载过高
        time.sleep(0.5)

if __name__ == "__main__":
    backfill_features()