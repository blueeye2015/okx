# data_handler.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import clickhouse_connect # 导入clickhouse客户端

# --- ClickHouse数据库配置 ---
# 您可以从您的 config.py 或其他配置文件中获取这些信息
# 这里我们作为示例直接写出
CLICKHOUSE_CONFIG = {
    'host': '127.0.0.1', # 或者您的数据库服务器IP
    'port': 8123,        # 通常是8123
    'user': 'default',   # 您的用户名
    'password': ''       # 您的密码
}

def _get_db_client():
    """创建一个数据库连接客户端"""
    try:
        client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
        client.ping() # 测试连接
        print("ClickHouse数据库连接成功。")
        return client
    except Exception as e:
        print(f"数据库连接失败: {e}")
        raise

def _fetch_data_from_clickhouse(table_name: str, instId: str = 'BTC-USDT-SWAP'):
    """从ClickHouse指定的表中获取K线数据"""
    client = _get_db_client()
    query = f"""
    SELECT
        ts, -- 时间戳列，假设列名为ts
        open,
        high,
        low,
        close,
        vol as volume -- 将数据库的vol列重命名为我们需要的volume
    FROM
        {table_name}
    WHERE
        instId = '{instId}' -- 根据产品ID筛选
    ORDER BY
        ts ASC
    """
    print(f"正在从表 {table_name} 查询数据...")
    # 使用 client.query_df 直接将查询结果转换为pandas DataFrame
    df = client.query_df(query)
    
    # --- 数据预处理 ---
    # 1. 将时间戳列转换为datetime对象，并设为索引
    # ClickHouse返回的时间通常是UTC时区，我们处理为本地时区或保持UTC
    df['date'] = pd.to_datetime(df['ts'])
    df = df.set_index('date')
    
    # 2. 确保数据类型正确
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
        
    return df.drop(columns=['ts'])


def _calculate_features(df):
    """(内部函数) 计算模型所需的自变量 (X) - 无需改动"""
    grouped = df.groupby(level='asset')
    df['return'] = grouped['close'].pct_change()
    df['momentum_30d'] = grouped['return'].transform(lambda x: x.rolling(30, min_periods=20).sum().shift(2))
    df['momentum_60d'] = grouped['return'].transform(lambda x: x.rolling(60, min_periods=40).sum().shift(2))
    df['momentum_90d'] = grouped['return'].transform(lambda x: x.rolling(90, min_periods=60).sum().shift(2))
    df['volatility_30d'] = grouped['return'].transform(lambda x: x.rolling(30, min_periods=20).std().shift(1))
    df['return_sign'] = np.sign(df['return'])
    streaks = grouped['return_sign'].transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
    df['streak_days'] = streaks.shift(1)
    df['volume_ma_20d'] = grouped['volume'].transform(lambda x: x.rolling(20, min_periods=15).mean().shift(1))
    df['log_volume_ma_20d'] = np.log1p(df['volume_ma_20d'])
    return df

def _calculate_directional_momentum(df, window_size=120):
    """(内部函数) 在滚动窗口上训练模型并计算方向动量因子 - 无需改动"""
    print("正在基于日线数据计算方向动量因子...")
    df_featured = _calculate_features(df)
    df_featured['target'] = (df_featured.groupby(level='asset')['return'].shift(-1) > 0).astype(int)
    df_clean = df_featured.dropna().copy()
    features = ['momentum_30d', 'momentum_60d', 'momentum_90d', 'volatility_30d', 'streak_days', 'log_volume_ma_20d']
    X = sm.add_constant(df_clean[features])
    y = df_clean['target']
    predictions = pd.Series(index=X.index, dtype=float)
    for i in tqdm(range(window_size, len(df_clean)), desc="滚动窗口计算因子"):
        X_train, y_train = X.iloc[i-window_size:i], y.iloc[i-window_size:i]
        X_pred = X.iloc[i:i+1]
        try:
            model = sm.Logit(y_train, X_train).fit(disp=0)
            predictions.iloc[i] = model.predict(X_pred)[0]
        except Exception:
            predictions.iloc[i] = np.nan
    print("方向动量因子计算完成。")
    return predictions

def get_historical_data_and_factor(daily_table, minutely_table, instId, resample_freq='1H'):
    """
    从ClickHouse加载1天和1分钟K线历史数据，计算因子，并合并成一个用于交易的DataFrame。
    """
    # 1. 从数据库加载数据
    df_daily = _fetch_data_from_clickhouse(daily_table, instId)
    df_daily['asset'] = instId
    df_daily = df_daily.set_index('asset', append=True).reorder_levels(['date', 'asset'])
    
    df_minutely = _fetch_data_from_clickhouse(minutely_table, instId)

    # 2. 计算因子
    factor_series = _calculate_directional_momentum(df_daily)

    # 3. 重采样并合并
    print(f"正在将分钟线重采样为 {resample_freq} K线...")
    logic_df = df_minutely['close'].resample(resample_freq).last().to_frame()
    logic_df['open'] = df_minutely['open'].resample(resample_freq).first()
    logic_df['high'] = df_minutely['high'].resample(resample_freq).max()
    logic_df['low'] = df_minutely['low'].resample(resample_freq).min()
    
    factor_series.index = factor_series.index.droplevel('asset')
    logic_df['d_mom_factor'] = factor_series.reindex(logic_df.index, method='ffill')
    
    print("数据准备完毕。")
    return logic_df.dropna()