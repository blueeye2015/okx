import clickhouse_connect
import pandas as pd
from datetime import datetime, timedelta
import time

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL_DEPTH = 'BTCUSDT'
SYMBOL_TRADE = 'BTC-USDT'
BATCH_DAYS = 1  # 每次处理1天

def get_incremental_range(client):
    """
    计算增量更新的时间范围
    """
    # 1. 找起点：查 features_15m 表里最后一条数据是几点
    try:
        sql_last = f"SELECT max(time) FROM marketdata.features_15m WHERE symbol = '{SYMBOL_DEPTH}'"
        res = client.query(sql_last).result_rows
        last_processed_time = res[0][0] if res and res[0][0] else None
    except Exception:
        last_processed_time = None

    # 2. 找终点：查 depth 表里最新数据是几点
    sql_max = f"SELECT max(event_time) FROM marketdata.depth WHERE symbol = '{SYMBOL_DEPTH}'"
    res = client.query(sql_max).result_rows
    # 如果没数据，默认现在
    max_depth_time = res[0][0] if res and res[0][0] else datetime.now()

    # 3. 判定逻辑
    if last_processed_time is None:
        print("⚠️ 特征表为空，执行全量初始化...")
        sql_min = f"SELECT min(event_time) FROM marketdata.depth WHERE symbol = '{SYMBOL_DEPTH}'"
        res = client.query(sql_min).result_rows
        start_date = res[0][0] if res and res[0][0] else datetime(2024, 1, 1)
        if start_date.year < 2020: start_date = datetime(2024, 1, 1)
        
        # [Fix] 初始时间向下取整到 15分钟
        start_date = start_date.replace(minute=start_date.minute // 15 * 15, second=0, microsecond=0)
    else:
        print(f"✅ 上次更新到: {last_processed_time}")
        start_date = last_processed_time + timedelta(minutes=15)

    # [Fix] 终点时间向下取整到 15分钟
    # 这一步至关重要！它确保我们永远不处理“当前正在进行中”的 K 线。
    # 比如现在是 10:48，我们只处理到 10:45 (不包含 10:45~11:00 这段未完成的数据)
    end_date = max_depth_time.replace(minute=max_depth_time.minute // 15 * 15, second=0, microsecond=0)

    return start_date, end_date

def update_features_incremental():
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    
    start_date, end_date = get_incremental_range(client)
    
    if start_date >= end_date:
        print("✨ 数据已是最新，无需更新。")
        return

    print(f"📅 增量任务范围: {start_date} -> {end_date}")
    
    current_date = start_date
    
    while current_date < end_date:
        # 确定这一批次的结束时间
        next_date = min(current_date + timedelta(days=BATCH_DAYS), end_date)
        
        # -----------------------------------------------------------
        # [关键逻辑] 时间窗口重叠 (Overlapping Windows)
        # -----------------------------------------------------------
        # 计算窗口 (Calc Window): 往前推 15 分钟，为了算 lag()
        calc_start = current_date - timedelta(minutes=15)
        calc_end = next_date 
        
        # 写入窗口 (Insert Window): 严格限制只写入新数据
        insert_start = current_date
        insert_end = next_date
        
        t_calc_start = calc_start.strftime('%Y-%m-%d %H:%M:%S')
        t_calc_end = calc_end.strftime('%Y-%m-%d %H:%M:%S')
        t_insert_start = insert_start.strftime('%Y-%m-%d %H:%M:%S')
        t_insert_end = insert_end.strftime('%Y-%m-%d %H:%M:%S')

        print(f"🔄 处理批次: {t_insert_start} -> {t_insert_end} (计算回溯至 {t_calc_start})")
        
        sql = f"""
        INSERT INTO marketdata.features_15m
        WITH
            -- Layer 1: 基础数据聚合 (Group By Full Expression)
            Trades AS (
                SELECT
                    toStartOfInterval(event_time, INTERVAL 15 MINUTE) as time,
                    argMax(price, event_time) as close_price,
                    sum(if(buyer_order_maker = 0, quantity, 0)) as buy_vol,
                    sum(if(buyer_order_maker = 1, quantity, 0)) as sell_vol,
                    buy_vol - sell_vol as net_cvd
                FROM marketdata.trades
                WHERE symbol = '{SYMBOL_TRADE}' 
                  AND event_time >= '{t_calc_start}' AND event_time < '{t_calc_end}'
                GROUP BY toStartOfInterval(event_time, INTERVAL 15 MINUTE) -- [修复] 不使用别名，直接重复表达式
            ),
            
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
                      AND event_time >= '{t_calc_start}' AND event_time < '{t_calc_end}'
                )
                WHERE delta < -1.0 AND price_diff = 0
                GROUP BY toStartOfInterval(event_time, INTERVAL 15 MINUTE) -- [修复]
            ),
            
            Snapshots AS (
                SELECT
                    toStartOfInterval(snapshot_time, INTERVAL 15 MINUTE) as time,
                    argMax(price, snapshot_time) as close_bid_price
                FROM marketdata.depth_snapshot
                WHERE symbol = '{SYMBOL_DEPTH}' AND side = 'bid'
                  AND snapshot_time >= '{t_calc_start}' AND snapshot_time < '{t_calc_end}' 
                GROUP BY toStartOfInterval(snapshot_time, INTERVAL 15 MINUTE) -- [修复]
            ),

            -- Layer 2: 原始计算 (包含临时字段 raw_wall_shift)
            RawFeatures AS (
                SELECT
                    T.time as time,
                    '{SYMBOL_DEPTH}' as symbol,
                    T.close_price as close_price,
                    
                    -- 计算 Raw Shift
                    if(isNotNull(S.close_bid_price) AND S.close_bid_price > 0, 
                       (S.close_bid_price - lagInFrame(S.close_bid_price) OVER (ORDER BY T.time)) / S.close_bid_price * 100, 
                       0) as raw_wall_shift,
                       
                    T.net_cvd as net_cvd,
                    ifNull(SP.spoofing_ratio, 1.0) as spoofing_ratio,
                    ifNull(SP.ask_withdraw_vol, 0) as ask_withdraw_vol,
                    ifNull(SP.bid_withdraw_vol, 0) as bid_withdraw_vol
                    
                FROM Trades AS T
                LEFT JOIN Spoofing AS SP ON T.time = SP.time
                LEFT JOIN Snapshots AS S ON T.time = S.time
                ORDER BY T.time
            ),
            
            -- Layer 3: 最终投影 (只保留目标表需要的列)
            FinalProjection AS (
                SELECT 
                    time,
                    symbol,
                    close_price,
                    -- 在这里把 raw_wall_shift 消化掉，变成 wall_shift_pct
                    if(isFinite(raw_wall_shift) AND abs(raw_wall_shift) < 50, raw_wall_shift, 0) as wall_shift_pct,
                    net_cvd,
                    spoofing_ratio,
                    ask_withdraw_vol,
                    bid_withdraw_vol
                FROM RawFeatures
                WHERE time >= '{t_insert_start}' AND time < '{t_insert_end}'
            )

        -- 最终插入：因为 FinalProjection 的结构已经和表对齐，所以 SELECT * 安全了
        SELECT * FROM FinalProjection
        """
        
        try:
            client.command(sql)
            print(f"✅ 写入成功.")
        except Exception as e:
            print(f"❌ 写入失败: {e}")
            time.sleep(5) # 出错等一下再试
        
        current_date = next_date

if __name__ == "__main__":
    update_features_incremental()