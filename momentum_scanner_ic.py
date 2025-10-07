import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS, add_constant
import lightgbm as lgb

MIN_TRAIN_DAYS = 756 
LGB_PARAMS = dict(objective='binary', n_estimators=100,
                  learning_rate=0.05, num_leaves=10,
                  max_depth=3, random_state=42, verbose=-1)
# ---------- 1. 核心函数 ----------
def calc_iv(ret_1d: np.ndarray, mkt_1d: np.ndarray) -> float:
    # 1. 拼成 DataFrame 方便一起清洗
    df = pd.DataFrame({'ret': ret_1d, 'mkt': mkt_1d})
    # 2. 去掉 NaN / inf
    df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    # 3. 长度不足
    if len(df) < 60:
        return np.nan
    X = add_constant(df['mkt'])
    resid = OLS(df['ret'], X).fit().resid
    return float(resid.std())

def calc_daily_factor(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    输入：单币种对齐后的日K  [timestamp, close]
    输出：DataFrame [timestamp, factor]  每天一个因子
    """
    df = df_sym.copy()
    df['return'] = df['close'].pct_change()
    df['mkt_ret'] = df.groupby('timestamp')['return'].transform('mean')

    # D-MOM 特征
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

    # IV：expanding 窗口
    df['IV'] = df['return'].expanding(MIN_TRAIN_DAYS).apply(
        lambda r: calc_iv(r, df.loc[r.index, 'mkt_ret']), raw=False)
    df['IV_t1'] = df['IV'].shift(1)

    features = ['IV_t1', 'rm_t1', 'P_d_t1', 'N_d_t1']
    df_clean = df.dropna(subset=features)
    if len(df_clean) < MIN_TRAIN_DAYS + 1:
        return pd.DataFrame(columns=['timestamp', 'factor'])

    # 逐日 expanding 训练 & 预测
    factors = []
    scaler = StandardScaler()
    for i in range(MIN_TRAIN_DAYS, len(df_clean)):
        X_train = df_clean[features].iloc[:i]
        y_train = (df_clean['return'].shift(-21) > 0).astype(int).iloc[:i]
        if y_train.nunique() < 2:
            continue
        X_pred = df_clean[features].iloc[i:i + 1]

        scaler.fit(X_train)
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(scaler.transform(X_train), y_train)
        factor = model.predict_proba(scaler.transform(X_pred))[0, 1]
        factors.append({'timestamp': df_clean.index[i], 'factor': factor})

    return pd.DataFrame(factors)

def _calculate_features_and_factor(group_df, min_train=756):
    """
    expanding窗口训练，返回Series：{'factor':float,'latest_price':float}
    min_train=756≈36个月*21日
    """
    df = group_df.copy()
    df['return'] = df['close'].pct_change()
    df['mkt_ret'] = df.groupby('timestamp')['return'].transform('mean')   # 当日全市场等权
    # ---------- D-MOM特征 ----------
    max_dur = 12
    df['sign'] = np.where(df['return'] > 0, 1, -1)
    df['streak_start'] = (df['sign'] != df['sign'].shift(1)).cumsum()
    df['raw_streak'] = df.groupby(['symbol', 'streak_start']).cumcount() + 1
    df['raw_streak'] = np.minimum(df['raw_streak'], max_dur)
    df['P_d'] = np.where(df['sign'] == 1, df['raw_streak'], 0)
    df['N_d'] = np.where(df['sign'] == -1, df['raw_streak'], 0)
    df['P_d_t1'] = df['P_d'].shift(1)
    df['N_d_t1'] = df['N_d'].shift(1)
    df['rm_t1'] = df['mkt_ret'].shift(1)

    # IV：expanding窗口
    df['IV'] = (df['return'].expanding(min_train)
                .apply(lambda r: calc_iv(r, df.loc[r.index, 'mkt_ret']), raw=False))
    df['IV_t1'] = df['IV'].shift(1)

    # 标签：下月收益方向
    df['target'] = (df['return'].shift(-21) > 0).astype(int)

    # 特征
    features = ['IV_t1', 'rm_t1', 'P_d_t1', 'N_d_t1']
    df_clean = df.dropna(subset=features + ['target'])
    if len(df_clean) < min_train + 21:
        return None

    # 取最后一天做预测
    i = len(df_clean) - 1
    X_train, y_train = df_clean[features].iloc[:i], df_clean['target'].iloc[:i]
    X_pred = df_clean[features].iloc[i:i + 1]

    if len(y_train.unique()) < 2:
        return None

    # LightGBM训练
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    model = lgb.LGBMClassifier(objective='binary', n_estimators=100,
                               learning_rate=0.05, num_leaves=10,
                               max_depth=3, random_state=42, verbose=-1)
    model.fit(X_train_scaled, y_train)
    pred_proba = model.predict_proba(X_pred_scaled)[:, 1][0]

    return pd.Series({'factor': pred_proba, 'latest_price': df_clean['close'].iloc[-1]})

# ---------- 2. 月度IC计算 ----------
# def calc_monthly_ic(all_data_df, symbols_list):
#     """返回每月IC Series"""
#     ic_list = []
#     for month, sub in all_data_df.groupby(pd.Grouper(key='timestamp', freq='ME')):
#         ranks = []
#         rets = []
#         for symbol in symbols_list:
#             g = sub[sub['symbol'] == symbol]
#             if len(g) < MIN_TRAIN_DAYS + 21:
#                 continue
#             res = _calculate_features_and_factor(g)
#             if res is None:
#                 continue
#             ranks.append(res['factor'])
#             # 下月收益（shift -21）
#             next_ret = g['close'].pct_change().shift(-21).dropna().iloc[0] if len(g) > 21 else np.nan
#             rets.append(next_ret)
#         if len(ranks) > 10:
#             ic = pd.Series(ranks).corr(pd.Series(rets))
#             ic_list.append((month, ic))
#     return pd.Series(dict(ic_list))

def calc_monthly_ic(all_data_df: pd.DataFrame, symbols_list: list) -> pd.Series:
    """
    1. 先为每币生成每日因子
    2. 取每月最后一天的因子
    3. 与下月收益求 corr
    """
    # 1) 生成每日因子
    factor_list = []
    for symbol in symbols_list:
        g = all_data_df[all_data_df['symbol'] == symbol].set_index('timestamp')[['close']]
        if len(g) < MIN_TRAIN_DAYS :
            continue
        fac = calc_daily_factor(g)
        if not fac.empty:
            fac['symbol'] = symbol
            factor_list.append(fac)
    if not factor_list:
        return pd.Series(dtype=float)
    daily_factor = pd.concat(factor_list, ignore_index=True)

    # 2) 月底因子
    daily_factor['month_end'] = daily_factor['timestamp'] + pd.offsets.MonthEnd(0)
    factor_eom = (daily_factor.groupby(['symbol', 'month_end'])['factor']
                  .last().reset_index())

    # 3) 下月收益
    price_df = all_data_df.copy()
    price_df['month'] = price_df['timestamp'] + pd.offsets.MonthEnd(0)
    price_df['return'] = price_df.groupby('symbol')['close'].pct_change()
    next_ret = (price_df.groupby(['symbol', 'month'])['return']
                .sum().shift(-1).reset_index(name='next_ret'))

    # 4) 合并并计算 IC
    df_merge = factor_eom.merge(next_ret,
                                left_on=['symbol', 'month_end'],
                                right_on=['symbol', 'month'],
                                how='inner')
    ic_list = []
    for month, g in df_merge.groupby('month_end'):
        if len(g) > 10:
            ic = g['factor'].corr(g['next_ret'])
            ic_list.append((month, ic))
    #return pd.Series(dict(ic_list))
    #现在多返回一份日频因子表
    ic_series = pd.Series(dict(ic_list))
    daily_factor = daily_factor[['timestamp', 'symbol', 'factor']]   # 只留三列
    return ic_series, daily_factor