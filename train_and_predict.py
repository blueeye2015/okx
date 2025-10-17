import pandas as pd
import numpy as np
import psycopg2
import logging
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import numba
from dotenv import load_dotenv
load_dotenv('.env')
POSTGRES_CONFIG  = os.getenv("DB_DSN1")
# 导入我们修改过的因子计算脚本中的特征工程函数
# 我们需要从中“借用”特征计算的逻辑
from momentum_scanner_a_share import calc_iv 

# ... (calc_iv 和 fast_ols_std 函数与上一版相同) ...
@numba.jit(nopython=True)
def fast_ols_std(y, x):
    if len(y) < 2: return np.nan
    X = np.ones((len(x), 2)); X[:, 1] = x
    XTX = X.T @ X
    det = np.linalg.det(XTX)
    if np.abs(det) < 1e-8: return np.nan
    try:
        beta = np.linalg.inv(XTX) @ X.T @ y
        residuals = y - X @ beta
        return np.std(residuals)
    except:
        return np.nan

def calc_iv(ret_1d: pd.Series, mkt_1d: pd.Series) -> float:
    df = pd.DataFrame({'ret': ret_1d, 'mkt': mkt_1d}).dropna()
    if len(df) < 60: return np.nan
    y = df['ret'].values; x = df['mkt'].values
    return fast_ols_std(y, x)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# <<<--- 核心修改：调整实验的时间周期 ---<<<
# 数据加载范围：从2017开始，为训练集提供足够的回望数据
DATA_START_DATE = '2017-01-01'
DATA_END_DATE = '2024-12-31'
TRAIN_START_DATE = '2022-01-01'
TRAIN_END_DATE = '2024-12-31'
BACKTEST_START_DATE = '2017-01-01'
BACKTEST_END_DATE = '2021-12-31'

ADJUST_TYPE = 'hfq'
LGB_PARAMS = dict(objective='binary', n_estimators=100,
                  learning_rate=0.05, num_leaves=10,
                  max_depth=3, random_state=42, verbose=-1)

def create_features(df_full):
    """
    一个集中的函数，为整个DataFrame计算所有需要的特征。
    这是从 momentum_scanner_a_share.py 中提取并改造的。
    """
    logging.info("正在为全量数据计算特征...")
    df = df_full.copy()
    
    # 按symbol分组计算，避免数据混淆
    df['return'] = df.groupby('symbol')['close'].pct_change()
    df['mkt_ret'] = df.groupby('trade_date')['return'].transform('mean')

    # 为了加速，我们将循环改为groupby().apply()
    tqdm.pandas(desc="Calculating D-MOM")
    def d_mom_features(group):
        max_dur = 12
        group['sign'] = np.where(group['return'] > 0, 1, -1)
        group['streak_start'] = (group['sign'] != group['sign'].shift(1)).cumsum()
        group['raw_streak'] = group.groupby('streak_start').cumcount() + 1
        group['raw_streak'] = np.minimum(group['raw_streak'], max_dur)
        group['P_d'] = np.where(group['sign'] == 1, group['raw_streak'], 0)
        group['N_d'] = np.where(group['sign'] == -1, group['raw_streak'], 0)
        return group
    df = df.groupby('symbol', group_keys=False).progress_apply(d_mom_features)

    tqdm.pandas(desc="Calculating IV")
    def iv_features(group):
        group['IV'] = group['return'].expanding(756).apply(
            lambda r: calc_iv(r, group.loc[r.index, 'mkt_ret']), raw=False)
        return group
    df = df.groupby('symbol', group_keys=False).progress_apply(iv_features)
    # <<<--- 核心修改：在取对数前，强制转换数据类型 ---<<<
    df['total_mv'] = pd.to_numeric(df['total_mv'], errors='coerce')

    df['log_mv'] = np.log(df['total_mv'].clip(lower=1))
    
    rolling_max_252 = df.groupby('symbol')['close'].rolling(252, min_periods=60).max().reset_index(0, drop=True)
    rolling_min_252 = df.groupby('symbol')['close'].rolling(252, min_periods=60).min().reset_index(0, drop=True)
    df['price_pos_1y'] = (df['close'] - rolling_min_252) / (rolling_max_252 - rolling_min_252)
    df['volatility_3m'] = df.groupby('symbol')['return'].rolling(63, min_periods=20).std().reset_index(0, drop=True)
    df['profit_yoy'] = df.groupby('symbol')['deduct_parent_netprofit'].pct_change(periods=252).fillna(0)
    
    # 统一进行滞后处理
    features_to_lag = ['P_d', 'N_d', 'mkt_ret', 'IV', 'log_mv', 'price_pos_1y', 'volatility_3m', 'profit_yoy']
    for f in features_to_lag:
        df[f'{f}_t1'] = df.groupby('symbol')[f].shift(1)

    df['target'] = (df.groupby('symbol')['return'].rolling(21).sum().shift(-21) > 0).astype(int).reset_index(0, drop=True)
    
    logging.info("特征计算完毕。")
    return df

if __name__ == '__main__':
    # --- 1. 数据获取与合并 ---
    # ... (这部分代码与 backtest_a_share.py 中的数据合并逻辑几乎完全相同)
    logging.info("正在获取 2017-2023 年的全量数据...")
    conn = psycopg2.connect(POSTGRES_CONFIG)
    sql_symbols = f"SELECT symbol FROM public.stock_history WHERE adjust_type = '{ADJUST_TYPE}' GROUP BY symbol"
    all_symbols = pd.read_sql_query(sql_symbols, conn)['symbol'].tolist()
    
    prices_query = f"SELECT trade_date, symbol, close FROM public.stock_history WHERE symbol IN {tuple(all_symbols)} AND trade_date BETWEEN '{DATA_START_DATE}'::date AND '{DATA_END_DATE}'::date AND adjust_type = '{ADJUST_TYPE}'"
    df_prices = pd.read_sql_query(prices_query, conn)
    
    mv_query = f"SELECT trade_date, ts_code AS symbol, total_mv FROM public.daily_basic WHERE ts_code IN {tuple(all_symbols)} AND trade_date BETWEEN '{DATA_START_DATE}'::date AND '{DATA_END_DATE}'::date"
    df_mv = pd.read_sql_query(mv_query, conn)
    
    profit_query = f"SELECT symbol, report_date, deduct_parent_netprofit FROM public.profit_sheet WHERE symbol IN {tuple(all_symbols)}"
    df_profit = pd.read_sql_query(profit_query, conn)
    conn.close()

    df_prices['trade_date'] = pd.to_datetime(df_prices['trade_date'])
    df_mv['trade_date'] = pd.to_datetime(df_mv['trade_date'])
    df_profit['report_date'] = pd.to_datetime(df_profit['report_date'])
    df_profit.rename(columns={'report_date': 'trade_date'}, inplace=True)

    df_merged = pd.merge(df_prices, df_mv, on=['trade_date', 'symbol'], how='left')
    df_merged.sort_values(by=['trade_date'], inplace=True)
    df_profit.sort_values(by=['trade_date'], inplace=True)
    df_profit.drop_duplicates(subset=['symbol', 'trade_date'], keep='last', inplace=True)

    df_full = pd.merge_asof(df_merged, df_profit, on='trade_date', by='symbol', direction='backward')
    logging.info("全量数据合并完毕。")

    # --- 2. 特征工程 ---
    df_featured = create_features(df_full)

    # --- 3. 划分训练集和预测集 ---
    features = [
        'P_d_t1', 'N_d_t1', 'mkt_ret_t1', 'IV_t1', 'log_mv_t1', 
        'price_pos_1y_t1', 'volatility_3m_t1', 'profit_yoy_t1'
    ]
    
    df_final = df_featured.dropna(subset=features + ['target'])
    
    train_mask = (df_final['trade_date'] >= TRAIN_START_DATE) & (df_final['trade_date'] <= TRAIN_END_DATE)
    backtest_mask = (df_final['trade_date'] >= BACKTEST_START_DATE) & (df_final['trade_date'] <= BACKTEST_END_DATE)

    X_train = df_final.loc[train_mask, features]
    y_train = df_final.loc[train_mask, 'target']
    
    X_pred = df_final.loc[backtest_mask, features]
    pred_info = df_final.loc[backtest_mask, ['trade_date', 'symbol']]

    logging.info(f"训练集大小: {len(X_train)}, 预测集大小: {len(X_pred)}")

    # --- 4. 训练模型 ---
    logging.info("开始训练模型...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_train_scaled, y_train)
    logging.info("模型训练完毕。")

    # --- 5. 生成因子并保存 ---
    logging.info("正在为 2017-2019 年的数据生成因子值...")
    X_pred_scaled = scaler.transform(X_pred)
    pred_probas = model.predict_proba(X_pred_scaled)[:, 1]
    
    daily_factor = pred_info.copy()
    daily_factor['factor'] = pred_probas
    
    # 保存为 visual_backtest.py 需要的文件名
    daily_factor.to_parquet('daily_factor.parquet')
    logging.info("因子文件 'daily_factor.parquet' (2017-2019) 已生成。")
    logging.info("下一步，请运行 visual_backtest.py 来进行可视化回测。")