import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import warnings
import clickhouse_connect

# 导入新的、更强大的模型
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 全局配置 ---
CLICKHOUSE_CONFIG = {'host': '127.0.0.1', 'port': 8123, 'user': 'default', 'password': '12'}
OUTPUT_FILE = 'ranked_signals.csv' # 定义输出文件名

def get_symbols_to_scan(client):
    """从数据库获取所有需要扫描的现货币种列表"""
    try:
        # 假设您的现货交易对都以 -USDT 结尾
        df = client.query_df("SELECT DISTINCT symbol FROM marketdata.okx_klines_1d WHERE symbol LIKE '%-USDT'")
        symbols = df['symbol'].tolist()
        logging.info(f"从数据库获取到 {len(symbols)} 个待扫描币种。")
        return symbols
    except Exception as e:
        logging.error(f"获取币种列表失败: {e}")
        return []

def _fetch_all_daily_data(ch_client, symbols_list):
    """从ClickHouse批量获取所有币种的日线数据。"""
    logging.info(f"准备从ClickHouse一次性查询 {len(symbols_list)} 个币种的日线数据...")
    symbols_tuple_str = tuple(symbols_list)
    query = f"SELECT timestamp, symbol, open, high, low, close, volume FROM marketdata.okx_klines_1d WHERE symbol IN {symbols_tuple_str} ORDER BY symbol, timestamp ASC"
    try:
        df = ch_client.query_df(query)
        df['date'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col])
        logging.info(f"成功获取并处理了 {len(df)} 行K线数据。")
        return df.drop(columns=['timestamp'])
    except Exception as e:
        logging.error(f"从ClickHouse批量获取数据失败: {e}")
        return pd.DataFrame()


def _calculate_features_and_factor(group_df, window_size=120):
    """
    【最终生产版】使用LightGBM梯度提升决策树模型计算因子。
    """
    df = group_df.copy()
    df['return'] = df['close'].pct_change()
    df['momentum_30d'] = df['return'].rolling(30, min_periods=20).sum().shift(2)
    df['momentum_60d'] = df['return'].rolling(60, min_periods=40).sum().shift(2)
    df['momentum_90d'] = df['return'].rolling(90, min_periods=60).sum().shift(2)
    df['volatility_30d'] = df['return'].rolling(30, min_periods=20).std().shift(1)
    df['return_sign'] = np.sign(df['return'])
    streaks = df['return_sign'] * (df['return_sign'].groupby((df['return_sign'] != df['return_sign'].shift()).cumsum()).cumcount() + 1)
    df['streak_days'] = streaks.shift(1)
    df['volume_ma_20d'] = df['volume'].rolling(20, min_periods=15).mean().shift(1)
    df['log_volume_ma_20d'] = np.log1p(df['volume_ma_20d'])
    df['target'] = (df['return'].shift(-1) > 0).astype(int)

    df_clean = df.dropna()
    if len(df_clean) <= window_size:
        return None

    features = ['momentum_30d', 'momentum_60d', 'momentum_90d', 'volatility_30d', 'streak_days', 'log_volume_ma_20d']
    X_data = df_clean[features]
    y_data = df_clean['target']
    
    i = len(df_clean) - 1
    X_train, y_train = X_data.iloc[i-window_size:i], y_data.iloc[i-window_size:i]
    X_pred = X_data.iloc[i:i+1]
    
    if len(y_train.unique()) < 2:
        return None

    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pred_scaled = scaler.transform(X_pred)
        
        lgb_params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 100,
            'learning_rate': 0.05, 'num_leaves': 10, 'max_depth': 3,
            'seed': 42, 'n_jobs': -1, 'verbose': -1
        }
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train_scaled, y_train)
        
        pred_proba = model.predict_proba(X_pred_scaled)[:, 1][0]
        
        return pd.Series({'factor': pred_proba, 'latest_price': df_clean['close'].iloc[-1]})
    except Exception:
        return None


def scan_and_rank_momentum(ch_client, symbols_list):
    """【最终生产版】扫描所有币种，计算因子并返回一个完整的排名列表。"""
    if not symbols_list: return pd.DataFrame()
    all_data_df = _fetch_all_daily_data(ch_client, symbols_list)
    if all_data_df.empty: return pd.DataFrame()

    logging.info("数据获取完毕，开始在内存中分组计算所有币种的因子...")
    
    results = []
    for symbol, group in tqdm(all_data_df.groupby('symbol'), desc="计算因子进度"):
        res = _calculate_features_and_factor(group)
        if res is not None:
            res['symbol'] = symbol
            results.append(res)

    successful_count = len(results)
    failed_count = len(symbols_list) - successful_count
    logging.info(f"因子计算完成：成功 {successful_count}个, 失败 {failed_count}个。")

    if not results: return pd.DataFrame()

    results_df = pd.DataFrame(results)
    signal_df = pd.DataFrame({
        'symbol': results_df['symbol'],
        'current_price': results_df['latest_price'],
        'RVol': results_df['factor']
    })
    
    signal_df = signal_df.sort_values('RVol', ascending=False).reset_index(drop=True)
    logging.info(f"扫描排名完成，共为 {len(signal_df)} 个币种生成了有效因子。")
    return signal_df

def run_scanner():
    """
    执行扫描的主函数
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("启动动量因子扫描程序...")

    try:
        ch_client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
        ch_client.ping()
        logging.info("ClickHouse连接成功。")
    except Exception as e:
        logging.error(f"无法连接到ClickHouse，程序退出: {e}")
        return

    # 1. 获取币种列表
    symbols_to_scan = get_symbols_to_scan(ch_client)
    if not symbols_to_scan:
        logging.error("没有要扫描的币种，程序退出。")
        return

    # 2. 执行扫描和排名
    ranked_signals_df = scan_and_rank_momentum(ch_client, symbols_to_scan)

    # 3. 将结果保存到文件
    if not ranked_signals_df.empty:
        try:
            ranked_signals_df.to_csv(OUTPUT_FILE, index=False)
            logging.warning(f"成功将 {len(ranked_signals_df)} 条排名信号保存到 {OUTPUT_FILE}")
        except Exception as e:
            logging.error(f"保存信号文件失败: {e}")
    else:
        logging.warning("没有生成任何有效信号，未创建信号文件。")
    
    logging.info("扫描程序执行完毕。")


if __name__ == "__main__":
    run_scanner()