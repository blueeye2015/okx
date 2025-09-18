import datetime
import logging
import clickhouse_connect
import pandas as pd

# --- 配置区 ---
# ClickHouse数据库配置
CH_HOST = 'localhost'
CH_PORT = 8123
CH_DATABASE = 'marketdata'
CH_USERNAME = 'default'
CH_PASSWORD = '12'

# --- 策略参数配置 ---
# Watchlist Entry Rules (必须全部满足)
MAX_MARKET_CAP = 5_000_000_000         # 最大市值: 5亿美金
MIN_PULLBACK_FROM_HIGH = 0.7          # 至少从365日高点回调70%
MAX_DISTANCE_FROM_LOW = 1.2           # 价格不高于200日低点的20%
MAX_VOLUME_RATIO = 0.5                # 10日均量需小于60日均量的50%
BBW_PERCENTILE_THRESHOLD = 0.20       # 【新!】BBW需处于过去100日的20%分位数以下
MIN_AVG_VOLUME_USDT = 50000  # 【新!】10日均成交额必须大于 5万 USDT
MIN_ACTIVITY_RATIO = 0.30    # 【新!】过去24小时，有成交的分钟数占比必须大于 10%

# Watchlist Pop-out Rules (满足任一即可)
MAX_RECOVERY_FROM_LOW = 1.5           # 从200日低点反弹超过50%
MIN_PULLBACK_RECOVERY = 0.6         # 回调幅度已经小于60%

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_clickhouse_client():
    """创建并返回一个ClickHouse客户端连接"""
    try:
        client = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT, database=CH_DATABASE,
            username=CH_USERNAME, password=CH_PASSWORD)
        logging.info("成功连接到ClickHouse数据库。")
        return client
    except Exception as e:
        logging.error(f"连接ClickHouse失败: {e}")
        raise

def get_all_metrics(client):
    """
    【核心升级】一个更强大的SQL查询，用于计算所有币种的日线级别策略指标，包含BBW。
    """
    logging.info("正在查询并计算所有币种的日线策略指标（包含BBW）...")
    
    # 这个查询通过CTE（WITH子句）分步计算，逻辑清晰且高效
    query = f"""
    WITH
        date_range AS (
            SELECT
                today() AS current_day,
                today() - INTERVAL 385 DAY AS start_385, -- 需要额外历史数据用于计算窗口函数
                today() - INTERVAL 365 DAY AS start_365,
                today() - INTERVAL 200 DAY AS start_200,
                today() - INTERVAL 100 DAY AS start_100,
                today() - INTERVAL 60 DAY AS start_60,
                today() - INTERVAL 10 DAY AS start_10
        ),
        -- 【新!】从分钟线计算交易活跃度
        activity_metrics AS (
            SELECT
                symbol,
                countIf(volume > 0) / count() AS activity_ratio
            FROM {CH_DATABASE}.okx_klines_1m
            WHERE timestamp >= (SELECT current_day FROM date_range) - INTERVAL 24 HOUR
            GROUP BY symbol
        ),
        daily_data_with_bbw AS (
            SELECT
                symbol,
                timestamp,
                high,
                low,
                close,
                volume,
                -- 计算每一天的BBW(20)，需要20天数据，所以窗口是19 PRECEDING
                if(
                    avg(close) OVER w20 > 0,
                    (4 * stddevPop(close) OVER w20) / (avg(close) OVER w20),
                    0
                ) AS bbw
            FROM {CH_DATABASE}.okx_klines_1d
            -- 预加载足够多的历史数据以进行窗口计算
            WHERE timestamp >= (SELECT start_385 FROM date_range)
            WINDOW w20 AS (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
        ),
        metrics_final AS (
            SELECT
                symbol,
                max(high) AS high_365d,
                minIf(low, timestamp >= (SELECT start_200 FROM date_range)) AS low_200d,
                avgIf(volume, timestamp >= (SELECT start_10 FROM date_range)) AS avg_vol_10d,
                avgIf(volume, timestamp >= (SELECT start_60 FROM date_range)) AS avg_vol_60d,
                argMax(close, timestamp) AS current_price,
                -- 获取最新一天的BBW值
                argMax(bbw, timestamp) AS current_bbw,
                -- 计算过去100天BBW的20%分位数
                quantileIf(0.20)(bbw, timestamp >= (SELECT start_100 FROM date_range)) AS bbw_100d_p20
            FROM daily_data_with_bbw
            WHERE timestamp >= (SELECT start_365 FROM date_range)
            GROUP BY symbol
        )
    SELECT
        m.symbol as symbol,
        ci.market_cap,
        m.current_price as current_price,
        m.high_365d,
        m.low_200d,
        m.avg_vol_10d,
        m.avg_vol_60d,
        act.activity_ratio, -- 新增活跃度字段
        m.current_bbw,
        m.bbw_100d_p20
    FROM metrics_final AS m
    JOIN {CH_DATABASE}.coin_info AS ci ON m.symbol = Concat(ci.symbol,'-USDT')
    JOIN activity_metrics AS act ON m.symbol = act.symbol -- 关联活跃度数据
    WHERE
        ci.market_cap > 0 AND m.low_200d > 0 AND m.avg_vol_60d > 0
    """
    #print(query)
    try:
        metrics_df  = client.query_df(query)
        logging.info(f"成功计算了 {len(metrics_df )} 个币种的指标。")

        # 2. 在DataFrame中计算衍生指标和 is_in_watchlist 标记
        metrics_df['snapshot_time'] = datetime.datetime.now()
        metrics_df['pullback_from_ath'] = 1 - (metrics_df['current_price'] / metrics_df['high_365d'])
        metrics_df['distance_from_low'] = (metrics_df['current_price'] / metrics_df['low_200d']) - 1
        metrics_df['volume_ratio_10_60'] = metrics_df['avg_vol_10d'] / metrics_df['avg_vol_60d']

        # 3. 准备最终要插入的数据
        columns_to_insert = [
            'snapshot_time', 'symbol', 'market_cap', 'current_price', 'high_365d', 'low_200d',
            'pullback_from_ath', 'distance_from_low', 'volume_ratio_10_60', 
            'avg_vol_10d', 'activity_ratio', 'current_bbw', 'bbw_100d_p20'
        ]
        final_insert_df = metrics_df[columns_to_insert]

        # 4. 将所有结果一次性插入到历史表中
        ch_client.insert_df(f'{CH_DATABASE}.watchlist_history', final_insert_df)
        
        logging.info(f"任务完成！共记录了 {len(final_insert_df)} 个币种的快照。")

        return metrics_df

        
    except Exception as e:
        logging.error(f"计算指标时发生SQL错误: {e}")
        return pd.DataFrame()
    

def update_watchlist(client, metrics_df):
    """根据计算出的指标和规则，更新watchlist表"""
    try:
        # 1. 获取当前watchlist中的币种
        current_watchlist_df = client.query_df(f'SELECT symbol FROM {CH_DATABASE}.watchlist')
        current_watchlist_symbols = set(current_watchlist_df['symbol'])
        logging.info(f"当前观察名单中有 {len(current_watchlist_symbols)} 个币种。")
        
        symbols_to_add = []
        symbols_to_remove = []

        # 2. 遍历所有币种，应用规则
        for _, row in metrics_df.iterrows():
            symbol = row['symbol']
            
            # 检查是否在当前watchlist中
            is_in_watchlist = symbol in current_watchlist_symbols

           # 【核心改动】: 使用新的、更严格的成交量条件
            meets_entry_conditions = (
                row['market_cap'] < MAX_MARKET_CAP and
                row['current_price'] < row['high_365d'] * (1 - MIN_PULLBACK_FROM_HIGH) and
                row['current_price'] < row['low_200d'] * MAX_DISTANCE_FROM_LOW and
                # --- 新的成交量过滤三部曲 ---
                row['avg_vol_10d'] < row['avg_vol_60d'] * MAX_VOLUME_RATIO and    # 1. 趋势萎缩
                row['avg_vol_10d'] > MIN_AVG_VOLUME_USDT and                 # 2. 绝对流动性底线
                row['activity_ratio'] > MIN_ACTIVITY_RATIO and                    # 3. 交易活跃度达标
                # --- 波动率条件不变 ---
                row['current_bbw'] < row['bbw_100d_p20'] and row['current_bbw'] > 0
            )

            # 判断是否满足移出条件
            meets_popup_conditions = (
                row['current_price'] > row['low_200d'] * MAX_RECOVERY_FROM_LOW or
                row['current_price'] > row['high_365d'] * (1 - MIN_PULLBACK_RECOVERY)
            )

            if is_in_watchlist:
                # 如果在名单中，检查是否需要移出
                if meets_popup_conditions:
                    symbols_to_remove.append(symbol)
            else:
                # 如果不在名单中，检查是否需要加入
                if meets_entry_conditions:
                    # 准备待插入的数据行
                    new_entry = {
                        'symbol': symbol,
                        'added_at': datetime.datetime.now(),
                        'market_cap_at_add': row['market_cap'],
                        'price_at_add': row['current_price'],
                        'pullback_from_ath': 1 - (row['current_price'] / row['high_365d']),
                        'strategy_version': 'v1.0'
                    }
                    symbols_to_add.append(new_entry)

        # 3. 执行数据库操作
        if symbols_to_remove:
            logging.info(f"准备从观察名单中移除 {len(symbols_to_remove)} 个币种: {symbols_to_remove}")
            # Clickhouse-connect不支持 'DELETE ... WHERE symbol IN ?' 这种参数化
            # 我们需要构建一个元组
            symbols_tuple = tuple(symbols_to_remove)
            # 处理只有一个元素时元组的语法问题
            if len(symbols_tuple) == 1:
                delete_query = f"DELETE FROM {CH_DATABASE}.watchlist WHERE symbol = '{symbols_tuple[0]}'"
            else:
                delete_query = f"DELETE FROM {CH_DATABASE}.watchlist WHERE symbol IN {symbols_tuple}"

            # Clickhouse的DELETE是异步操作，需要用ALTER TABLE ... DELETE
            alter_delete_query = delete_query.replace("DELETE FROM", "ALTER TABLE", 1) + " SETTINGS mutations_sync = 2"
            client.command(alter_delete_query)
            logging.info("移除操作已提交。")

        if symbols_to_add:
            logging.info(f"准备向观察名单中添加 {len(symbols_to_add)} 个币种: {[entry['symbol'] for entry in symbols_to_add]}")
            add_df = pd.DataFrame(symbols_to_add)
            client.insert_df(f'{CH_DATABASE}.watchlist', add_df)
            logging.info("添加操作已完成。")
            
        if not symbols_to_remove and not symbols_to_add:
            logging.info("本次运行没有币种需要更新到观察名单。")

    except Exception as e:
        logging.error(f"更新watchlist时出错: {e}")


if __name__ == "__main__":
    ch_client = get_clickhouse_client()
    if ch_client:
        all_metrics_df = get_all_metrics(ch_client)
        if not all_metrics_df.empty:
            update_watchlist(ch_client, all_metrics_df)
        ch_client.close()
        logging.info("后台任务执行完毕，数据库连接已关闭。")