import clickhouse_connect
import pandas as pd
from datetime import datetime, timedelta
import time

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL_DEPTH = 'BTCUSDT'
SYMBOL_TRADE = 'BTC-USDT'
BATCH_DAYS = 1  # 每次处理1天 (如果落后太多天，会分批补；如果只落后几分钟，会一次跑完)

def get_incremental_range(client):
    """
    计算增量更新的时间范围
    起点 = 数据库里已有的最大时间 + 15分钟
    终点 = depth表里最新的时间
    """
    # 1. 找起点：查 features_15m 表里最后一条数据是几点
    try:
        sql_last = f"SELECT max(time) FROM marketdata.features_15m WHERE symbol = '{SYMBOL_DEPTH}'"
        last_processed_time = client.query(sql_last).result_rows[0][0]
    except Exception:
        last_processed_time = None

    # 2. 找终点：查 depth 表里最新数据是几点
    sql_max = f"SELECT max(event_time) FROM marketdata.depth WHERE symbol = '{SYMBOL_DEPTH}'"
    max_depth_time = client.query(sql_max).result_rows[0][0]

    # 3. 判定逻辑
    if last_processed_time is None:
        # 如果特征表是空的（第一次跑），就从 depth 的最早时间开始
        print("⚠️ 特征表为空，执行全量初始化...")
        sql_min = f"SELECT min(event_time) FROM marketdata.depth WHERE symbol = '{SYMBOL_DEPTH}'"
        start_date = client.query(sql_min).result_rows[0][0]
        # 修正脏数据
        if start_date.year < 2020: start_date = datetime(2024, 1, 1)
    else:
        # 如果有数据，起点就是 "上次最后时间 + 15分钟"
        print(f"✅ 上次更新到: {last_processed_time}")
        start_date = last_processed_time + timedelta(minutes=15)

    return start_date, max_depth_time

def update_features_incremental():
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    
    # 获取任务范围
    start_date, end_date = get_incremental_range(client)
    
    # 如果起点已经超过终点，说明是最新的，不用跑
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
        # 为了让第一条数据的 lag() 能算出结果，计算窗口必须往前推 15 分钟
        # 这样 SQL 引擎能看到"上一条"数据，从而算出正确的 wall_shift
        
        # 1. 计算用的窗口 (Lookback): 多取 15 分钟
        calc_start = current_date - timedelta(minutes=15)
        calc_end = next_date
        
        # 2. 写入用的窗口 (Target): 只写我们需要补的那段
        insert_start = current_date
        insert_end = next_date
        
        # 格式化时间
        t_calc_start = calc_start.strftime('%Y-%m-%d %H:%M:%S')
        t_calc_end = calc_end.strftime('%Y-%m-%d %H:%M:%S')
        t_insert_start = insert_start.strftime('%Y-%m-%d %H:%M:%S')
        t_insert_end = insert_end.strftime('%Y-%m-%d %H:%M:%S')

        print(f"🔄 处理批次: {t_insert_start} -> {t_insert_end} (计算回溯至 {t_calc_start})")
        
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
                  AND event_time >= '{t_calc_start}' AND event_time < '{t_calc_end}' -- [注意] 这里用计算窗口
                GROUP BY time
            ),
            
            -- 2. 欺诈撤单 (Spoofing)
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
                      AND event_time >= '{t_calc_start}' AND event_time < '{t_calc_end}' -- [注意] 这里用计算窗口
                )
                WHERE delta < -1.0 AND price_diff = 0
                GROUP BY time
            ),
            
            -- 3. 墙的移动 (Snapshot)
            Snapshots AS (
                SELECT
                    toStartOfInterval(snapshot_time, INTERVAL 15 MINUTE) as time,
                    argMax(price, snapshot_time) as close_bid_price
                FROM marketdata.depth_snapshot
                WHERE symbol = '{SYMBOL_DEPTH}' AND side = 'bid'
                  AND snapshot_time >= '{t_calc_start}' AND snapshot_time < '{t_calc_end}' -- [注意] 这里用计算窗口
                GROUP BY time
            ),

            -- 4. 计算逻辑 (包含 lag)
            CalculatedFeatures AS (
                SELECT
                    T.time,
                    '{SYMBOL_DEPTH}' as symbol,
                    T.close_price,
                    
                    -- 计算墙位移 (这里因为有了 Lookback 数据，第一行也能算出 lag)
                    if(isNotNull(S.close_bid_price), (S.close_bid_price - lagInFrame(S.close_bid_price) OVER (ORDER BY T.time)) / S.close_bid_price * 100, 0) as wall_shift_pct,
                    
                    T.net_cvd,
                    ifNull(SP.spoofing_ratio, 1.0) as spoofing_ratio,
                    ifNull(SP.ask_withdraw_vol, 0) as ask_withdraw_vol,
                    ifNull(SP.bid_withdraw_vol, 0) as bid_withdraw_vol
                    
                FROM Trades AS T
                LEFT JOIN Spoofing AS SP ON T.time = SP.time
                LEFT JOIN Snapshots AS S ON T.time = S.time
                ORDER BY T.time
            )

        -- 5. 最终筛选 (只写入真正属于本次增量时间段的数据)
        SELECT * FROM CalculatedFeatures
       
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