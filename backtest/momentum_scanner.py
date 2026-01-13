# momentum_scanner.py (最终生产版 - LightGBM)
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import warnings

# 导入新的、更强大的模型
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)


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
            'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'deterministic': True 
        }
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train_scaled, y_train)
        
        pred_proba = model.predict_proba(X_pred_scaled)[:, 1][0]
        
        return pd.Series({'factor': pred_proba, 'latest_price': df_clean['close'].iloc[-1]})
    except Exception:
        return None
# def _calculate_features_and_factor(group_df):
#     """
#     【终极诊断版】一个100%确定性的简单动量因子，用于隔离问题。
#     这个版本完全移除了 LightGBM 模型和所有特征工程。
#     """
#     # 确保有足够的数据来计算90天的滚动值
#     if len(group_df) < 92: # (90天窗口 + 2天shift)
#         return None

#     try:
#         # 创建一个简单的、完全确定性的动量因子
#         df = group_df.copy()
        
#         # 1. 计算日收益率
#         df['return'] = df['close'].pct_change()
        
#         # 2. 将过去90天的收益率累加作为因子值
#         #    使用 .shift(2) 是为了和你之前的模型逻辑保持一致
#         df['factor'] = df['return'].rolling(90, min_periods=60).sum().shift(2)

#         # 3. 删除因子为空的行
#         df.dropna(subset=['factor'], inplace=True)

#         if df.empty:
#             return None

#         # 4. 返回最后一天的因子值和价格
#         last_row = df.iloc[-1]
        
#         # 我们只返回一个包含两个键的简单Series对象
#         return pd.Series({'factor': last_row['factor'], 'latest_price': last_row['close']})

#     except Exception as e:
#         # 如果有任何计算错误，返回None
#         # print(f"Error calculating simple factor for {group_df['symbol'].iloc[0]}: {e}")
#         return None

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