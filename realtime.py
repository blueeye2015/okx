import time
import datetime
import logging
import clickhouse_connect

# --- 配置区 ---
# ClickHouse数据库配置
CH_HOST = 'localhost'
CH_PORT = 8123
CH_DATABASE = 'marketdata'
CH_USERNAME = 'default'
CH_PASSWORD = '12'

# 扫描执行时间控制
# 在每分钟的第几秒执行扫描？(例如: 2秒，留给交易所生成和推送K线数据的时间)
SCAN_AT_SECOND = 2

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_clickhouse_client():
    """创建并返回一个ClickHouse客户端连接"""
    try:
        client = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT, database=CH_DATABASE,
            username=CH_USERNAME, password=CH_PASSWORD)
        return client
    except Exception as e:
        logging.error(f"连接ClickHouse失败: {e}")
        return None

def scan_for_signals(client):
    """
    执行核心SQL查询，扫描符合入场条件的币种。
    """
    logging.info("开始扫描入场信号...")
    
    signals_df = None

    try:
        # 【核心修正】: 使用更精确的RVol计算公式
        query = f"""
        WITH
            date_range AS (
                SELECT
                    today() AS current_day,
                    today() - INTERVAL 11 DAY AS start_11,
                    today() - INTERVAL 1 DAY AS start_1
            ),
            avg_10d_vol AS (
                SELECT
                    symbol,
                    sum(volume) / 10 AS avg_vol
                FROM {CH_DATABASE}.okx_klines_1d
                WHERE timestamp >= (SELECT start_11 FROM date_range) AND timestamp < (SELECT start_1 FROM date_range)
                GROUP BY symbol
            ),
            today_vol AS (
                SELECT
                    symbol,
                    sum(volume) AS vol_so_far
                FROM {CH_DATABASE}.okx_klines_1m
                WHERE timestamp >= toStartOfDay(now('UTC'))
                GROUP BY symbol
            ),
            minute_data_with_ma60 AS (
                SELECT
                    symbol,
                    timestamp,
                    close,
                    avg(close) OVER w60 AS ma60
                FROM {CH_DATABASE}.okx_klines_1m
                WHERE
                    symbol IN (SELECT symbol FROM {CH_DATABASE}.watchlist)
                    AND timestamp > now('UTC') - INTERVAL 65 MINUTE
                WINDOW w60 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
            ),
            minute_momentum AS (
                SELECT
                    symbol,
                    close as current_price
                FROM minute_data_with_ma60
                QUALIFY
                    row_number() OVER (PARTITION BY symbol ORDER BY timestamp DESC) = 1
                    AND
                    min(if(close > ma60, 1, 0)) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) = 1
            )
        SELECT
            m.symbol as symbol,
            m.current_price as current_price,
            -- ###################### 核心修正点在这里 ######################
            (tv.vol_so_far / (greatest(1, toHour(now('UTC')) * 60 + toMinute(now('UTC'))) / 1440.0)) / av.avg_vol AS RVol
            -- ###########################################################
        FROM minute_momentum m
        JOIN avg_10d_vol av ON m.symbol = av.symbol
        JOIN today_vol tv ON m.symbol = tv.symbol
        WHERE
            RVol > 3.0
        """
        signals_df = client.query_df(query)
        
    except Exception as e:
        logging.error(f"执行SQL扫描查询时出错: {e}")
        return

    # ... (后续处理逻辑不变)
    try:
        if signals_df is not None and not signals_df.empty:
            logging.warning("🔥🔥🔥 发现入场信号! 🔥🔥🔥")
            for _, row in signals_df.iterrows():
                logging.warning(f"  币种: {row['symbol']}, "
                                f"当前价格: {row['current_price']:.4f}, "
                                f"相对成交量(RVol): {row['RVol']:.2f}")
        else:
            logging.info("未发现信号，继续监控...")

    except Exception as e:
        logging.error(f"处理信号结果时出错: {e}")

def main_loop():
    """主循环，确保每分钟准时执行一次扫描"""
    while True:
        now = datetime.datetime.now()
        
        # 计算距离下一个周期的第 SCAN_AT_SECOND 秒还有多久
        seconds_to_wait = (60 - now.second - 1) + (1 - now.microsecond / 1_000_000) + SCAN_AT_SECOND
        if seconds_to_wait < 0: # 避免负数
            seconds_to_wait += 60

        logging.info(f"当前时间: {now.strftime('%H:%M:%S')}, 等待 {seconds_to_wait:.2f} 秒后执行下一次扫描...")
        time.sleep(seconds_to_wait)
        
        # 执行扫描
        client = get_clickhouse_client()
        if client:
            try:
                scan_for_signals(client)
            finally:
                client.close()
                logging.info("扫描完成，数据库连接已关闭。")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("程序被手动中断。")