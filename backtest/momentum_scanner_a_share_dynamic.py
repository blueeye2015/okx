import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS, add_constant
import lightgbm as lgb
import multiprocessing
from tqdm import tqdm
import logging
import numba

# <<<--- 核心修改：重写 fast_ols_std 函数以兼容 Numba ---<<<
@numba.jit(nopython=True)
def fast_ols_std(y, x):
    """
    Numba-accelerated OLS residual standard deviation calculation.
    Uses determinant check instead of try-except for compatibility.
    """
    if len(y) < 2: # 增加一个长度检查
        return np.nan

    X = np.ones((len(x), 2))
    X[:, 1] = x
    
    XTX = X.T @ X
    
    # 检查矩阵是否为奇异矩阵 (行列式接近0)
    det = np.linalg.det(XTX)
    if np.abs(det) < 1e-8:
        return np.nan # 矩阵奇异，无法计算逆矩阵
    
    # 矩阵可逆，继续计算
    try:
        beta = np.linalg.inv(XTX) @ X.T @ y
        residuals = y - X @ beta
        return np.std(residuals)
    except: # 保留一个通用的except以防万一出现其他线性代数错误
        return np.nan

def calc_iv(ret_1d: pd.Series, mkt_1d: pd.Series) -> float:
    """
    Wrapper function to handle pandas Series and NaNs for Numba.
    """
    df = pd.DataFrame({'ret': ret_1d, 'mkt': mkt_1d}).dropna()
    if len(df) < 60:
        return np.nan
    
    y = df['ret'].values
    x = df['mkt'].values
    
    return fast_ols_std(y, x)

# ... (文件其余部分与上一版完全相同，为简洁此处省略) ...
# 您只需将上面的 fast_ols_std 和 calc_iv 函数，替换掉您文件中的对应部分即可。
# 为保证万无一失，下面是完整的代码。

# 全局变量
worker_price_df = None
worker_mkt_ret_series = None

MIN_TRAIN_DAYS = 756
LGB_PARAMS = dict(objective='binary', n_estimators=100,
                  learning_rate=0.05, num_leaves=10,
                  max_depth=3, random_state=42, verbose=-1)

def init_worker(price_data, mkt_ret_data):
    global worker_price_df, worker_mkt_ret_series
    worker_price_df = price_data
    worker_mkt_ret_series = mkt_ret_data

def calculate_factors_for_single_symbol(symbol, price_df, mkt_ret_series):
    g = price_df[price_df['symbol'] == symbol][['trade_date', 'close', 'total_mv','deduct_parent_netprofit']]
    if len(g) < MIN_TRAIN_DAYS:
        return None
    
    fac = calc_daily_factor_monthly_train(g, mkt_ret_series)
    
    if fac is not None and not fac.empty:
        fac['symbol'] = symbol
        return fac
    return None

def calc_daily_factor_monthly_train(df_sym: pd.DataFrame, mkt_ret_series: pd.Series) -> pd.DataFrame:
    df = df_sym.copy().set_index('trade_date')
    df['return'] = df['close'].pct_change()
    df = df.merge(mkt_ret_series.rename('mkt_ret'), left_index=True, right_index=True, how='left')
     # --- 1. “底部”形态特征 ---
    rolling_max_252 = df['close'].rolling(252, min_periods=60).max()
    rolling_min_252 = df['close'].rolling(252, min_periods=60).min()
    # 特征1: 当前价格在过去一年区间的位置 (0-1之间, 0代表最低点)
    df['price_pos_1y'] = (df['close'] - rolling_min_252) / (rolling_max_252 - rolling_min_252)
    df['price_pos_1y_t1'] = df['price_pos_1y'].shift(1)
    # 特征2: 近3个月的波动率
    df['volatility_3m'] = df['return'].rolling(63, min_periods=20).std()
    df['volatility_3m_t1'] = df['volatility_3m'].shift(1)
    # --- 2. “业绩困境反转”特征 ---
    # 扣非净利润(TTM)的同比增长率。pct_change(252)近似计算YoY
    # 用 fillna(0) 处理期初的NaN，以及业绩不变的情况
    df['profit_yoy'] = df['deduct_parent_netprofit'].pct_change(periods=252).fillna(0)
    df['profit_yoy_t1'] = df['profit_yoy'].shift(1)

    # <<<--- 新增：计算市值因子特征 ---<<<
    # 对市值取对数，处理极端值，使其更稳定
    # .clip(lower=1) 防止市值为0或负数时取对数出错
    df['log_mv'] = np.log(df['total_mv'].clip(lower=1))
    # 滞后一期以防未来函数
    df['log_mv_t1'] = df['log_mv'].shift(1)
    max_dur = 12
    df['sign'] = np.where(df['return'] > 0, 1, -1)
    df['streak_start'] = (df['sign'] != df['sign'].shift(1)).cumsum()
    df['raw_streak'] = df.groupby(['streak_start']).cumcount() + 1
    df['raw_streak'] = np.minimum(df['raw_streak'], max_dur)
    df['P_d'] = np.where(df['sign'] == 1, df['raw_streak'], 0)
    df['N_d'] = np.where(df['sign'] == -1, df['raw_streak'], 0)
    df['P_d_t1'] = df['P_d'].shift(1)
    df['N_d_t1'] = df['N_d'].shift(1)
    df['rm_t1'] = df['mkt_ret'].shift(1)
    df['IV'] = df['return'].expanding(MIN_TRAIN_DAYS).apply(
        lambda r: calc_iv(r, df.loc[r.index, 'mkt_ret']), raw=False)
    df['IV_t1'] = df['IV'].shift(1)
    # <<<--- 修改：将所有新特征加入列表 ---<<<
    features = [
        'P_d_t1', 'N_d_t1',         # 动量特征
        'rm_t1', 'IV_t1',           # 市场与波动特征
        'log_mv_t1',                # 市值特征
        'price_pos_1y_t1',          # 底部形态-价格位置
        'volatility_3m_t1',       # 底部形态-波动率
        'profit_yoy_t1'             # 业绩困境反转
    ]
    df_clean = df.dropna(subset=features).copy()
    df_clean['target'] = (df_clean['return'].rolling(21).sum().shift(-21) > 0).astype(int)
    if len(df_clean) < MIN_TRAIN_DAYS + 21: return pd.DataFrame()
    all_factors = []
    scaler = StandardScaler()
    training_dates = df_clean.index.to_period('M').unique().to_timestamp(how='end')
    for train_end_date in training_dates:
        valid_dates = df_clean.index[df_clean.index <= train_end_date]
        if valid_dates.empty: continue
        actual_train_end_date = valid_dates[-1]
        # 获取截止到当前训练日的所有交易日
        valid_dates_up_to_end = df_clean.index[df_clean.index <= actual_train_end_date]
        
        # === 修复：用 continue 跳过当前月份，而非 return 终止函数 ===
        if len(valid_dates_up_to_end) < MIN_TRAIN_DAYS:
            logging.debug(  # 改为debug避免日志刷屏
                f"股票在 {actual_train_end_date.date()} 累计仅{len(valid_dates_up_to_end)}天，"
                f"不足{MIN_TRAIN_DAYS}天，跳过该月"
            )
            continue  # ✅ 关键：继续下一个training_dates
        
        # 往前推 MIN_TRAIN_DAYS 个交易日
        window_start_date = valid_dates_up_to_end[-MIN_TRAIN_DAYS]
        train_df = df_clean.loc[window_start_date:actual_train_end_date]
        X_train = train_df[features].dropna()
        y_train = train_df.loc[X_train.index, 'target'].dropna()
        if y_train.nunique() < 2: continue
        pred_start_date = actual_train_end_date + pd.Timedelta(days=1)
        pred_end_date = pred_start_date + pd.offsets.MonthEnd(1)
        pred_df = df_clean.loc[pred_start_date:pred_end_date]
        if pred_df.empty: continue
        X_pred = pred_df[features]
        X_train_scaled = scaler.fit_transform(X_train)
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_train_scaled, y_train)
        pred_probas = model.predict_proba(scaler.transform(X_pred))[:, 1]
        month_factors = pd.DataFrame(pred_probas, index=X_pred.index, columns=['factor'])
        all_factors.append(month_factors)
    if not all_factors: return pd.DataFrame()
    final_factors = pd.concat(all_factors)
    return final_factors.reset_index()


def calc_monthly_ic(all_data_df: pd.DataFrame, symbols_list: list) -> tuple:
    price_df = all_data_df.copy()
    price_df['return'] = price_df.groupby('symbol')['close'].pct_change()
    mkt_ret_series = price_df.groupby('trade_date')['return'].mean()
    
    factor_list = []
    try:
        from tqdm import tqdm
        iterator = tqdm(symbols_list, desc="Calculating Factors for Chunk")
    except ImportError:
        iterator = symbols_list

    for symbol in iterator:
        fac = calculate_factors_for_single_symbol(symbol, price_df, mkt_ret_series)
        if fac is not None:
            factor_list.append(fac)

    if not factor_list:
        return pd.Series(dtype=float), pd.DataFrame()
        
    daily_factor = pd.concat(factor_list, ignore_index=True)
    daily_factor.rename(columns={'index': 'trade_date'}, inplace=True)

    daily_factor['month_end'] = pd.to_datetime(daily_factor['trade_date']) + pd.offsets.MonthEnd(0)
    factor_eom = daily_factor.groupby(['symbol', 'month_end'])['factor'].last().reset_index()
    price_df['month'] = price_df['trade_date'] + pd.offsets.MonthEnd(0)
    next_ret = price_df.groupby(['symbol', 'month'])['return'].sum().shift(-1).reset_index(name='next_ret')
    df_merge = factor_eom.merge(next_ret, left_on=['symbol', 'month_end'], right_on=['symbol', 'month'], how='inner')
    ic_list = []
    for month, g in df_merge.groupby('month_end'):
        if len(g) > 10:
            ic = g['factor'].corr(g['next_ret'])
            ic_list.append((month, ic))
    ic_series = pd.Series(dict(ic_list))
    daily_factor = daily_factor[['trade_date', 'symbol', 'factor']]
    return ic_series, daily_factor

# =================  逐日动态版  =================
def calc_factor_one_day(trade_date: pd.Timestamp,
                       price_df: pd.DataFrame,
                       mkt_ret_series: pd.Series):
    """
    只计算 **trade_date** 这一天的因子，返回
    DataFrame[symbol, factor]
    """
    # 1. 当日可交易股票（有收盘、有成交量）
    today_px = price_df.loc[trade_date]
    univ = today_px.index.get_level_values('symbol').unique().tolist()

    # 2. 取每只股票 **截止到 T 日** 的历史
    hist = (price_df.loc[pd.IndexSlice[:trade_date, :], :]
            .reset_index('trade_date'))

    factor_list = []
    for sym in univ:
        g = hist.loc[sym]
        if len(g) < 252:          # 上市不足一年直接跳过
            continue
        # 3. 计算因子（复用你原来月度训练逻辑，但只取最后一天的预测值）
        fac = _calc_one_symbol_last(g, mkt_ret_series)
        if fac is not None:
            factor_list.append({'symbol': sym, 'factor': fac})

    return pd.DataFrame(factor_list)


def _calc_one_symbol_last(df_sym: pd.DataFrame,
                         mkt_ret_series: pd.Series) -> float:
    """
    对单只股票截止到 T 日的历史，训练模型并返回 **T 日** 的因子值
    """
    df = df_sym.copy().set_index('trade_date')
    df['return'] = df['close'].pct_change()
    df = df.merge(mkt_ret_series.rename('mkt_ret'),
                 left_index=True, right_index=True, how='left')

    # —— 特征工程（与你原来完全一致） ——
    rolling_max_252 = df['close'].rolling(252, min_periods=60).max()
    rolling_min_252 = df['close'].rolling(252, min_periods=60).min()
    df['price_pos_1y'] = (df['close'] - rolling_min_252) / (rolling_max_252 - rolling_min_252)
    df['price_pos_1y_t1'] = df['price_pos_1y'].shift(1)
    df['volatility_3m'] = df['return'].rolling(63, min_periods=20).std()
    df['volatility_3m_t1'] = df['volatility_3m'].shift(1)
    df['profit_yoy'] = df['deduct_parent_netprofit'].pct_change(periods=252).fillna(0)
    df['profit_yoy_t1'] = df['profit_yoy'].shift(1)
    df['log_mv'] = np.log(df['total_mv'].clip(lower=1))
    df['log_mv_t1'] = df['log_mv'].shift(1)
    # 动量连涨连跌
    max_dur = 12
    df['sign'] = np.where(df['return'] > 0, 1, -1)
    df['streak_start'] = (df['sign'] != df['sign'].shift(1)).cumsum()
    df['raw_streak'] = df.groupby('streak_start').cumcount() + 1
    df['raw_streak'] = np.minimum(df['raw_streak'], max_dur)
    df['P_d'] = np.where(df['sign'] == 1, df['raw_streak'], 0)
    df['N_d'] = np.where(df['sign'] == -1, df['raw_streak'], 0)
    df['P_d_t1'] = df['P_d'].shift(1)
    df['N_d_t1'] = df['N_d'].shift(1)
    df['rm_t1'] = df['mkt_ret'].shift(1)
    df['IV'] = df['return'].expanding(MIN_TRAIN_DAYS).apply(
        lambda r: calc_iv(r, df.loc[r.index, 'mkt_ret']), raw=False)
    df['IV_t1'] = df['IV'].shift(1)

    features = ['P_d_t1', 'N_d_t1', 'rm_t1', 'IV_t1',
               'log_mv_t1', 'price_pos_1y_t1',
               'volatility_3m_t1', 'profit_yoy_t1']
    df_clean = df.dropna(subset=features)
    if len(df_clean) < MIN_TRAIN_DAYS:
        return None

    # 用 **最近 36 个月** 做训练
    train_df = df_clean.last('36M')
    X_train = train_df[features].values
    y_train = (train_df['return'].rolling(21).sum().shift(-21) > 0).astype(int).values
    # 去掉尾部 nan
    mask = ~np.isnan(y_train)
    X_train, y_train = X_train[mask], y_train[mask]
    if len(np.unique(y_train)) < 2:
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_train_scaled, y_train)

    # 只预测 **最后一天**
    x_last = scaler.transform(df_clean[features].iloc[-1:].values)[0]
    factor = float(model.predict_proba([x_last])[0, 1])
    return factor