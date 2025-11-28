import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import logging

BENCHMARK_SYMBOL = '000300.SH'
MIN_TRAIN_DAYS   = 756
LGB_PARAMS       = dict(objective='binary', n_estimators=100,
                        learning_rate=0.05, num_leaves=10,
                        max_depth=3, random_state=42, verbose=-1)

def calc_monthly_ic(feat_df: pd.DataFrame, symbols_list: list, mkt_ret_series: pd.Series) -> tuple:
    """
    只负责：读特征 → 训练 → 预测 → 输出因子
    feat_df 必须含：trade_date,symbol,profit_yoy,log_mv,price_pos_1y,volatility_3m
    """
    factor_list = []
    for symbol in symbols_list:
        g = feat_df[feat_df['symbol'] == symbol].copy().set_index('trade_date')
        if len(g) < MIN_TRAIN_DAYS: continue

        # 无法提前的动量/IV 在这里算
        g['return'] = g['close'].pct_change()
        g['mkt_ret'] = mkt_ret_series
        g['sign'] = np.where(g['return'] > 0, 1, -1)
        streak = (g['sign'] != g['sign'].shift(1)).cumsum()
        g['raw_streak'] = g.groupby(streak).cumcount().clip(1, 12)
        g['P_d'] = np.where(g['sign'] == 1, g['raw_streak'], 0)
        g['N_d'] = np.where(g['sign'] == -1, g['raw_streak'], 0)
        g['IV'] = g['return'].expanding(MIN_TRAIN_DAYS).apply(
            lambda r: r.std(), raw=False)  # 简化 IV，可替换你的 calc_iv
        # 特征列表
        features = ['P_d', 'N_d', 'IV', 'profit_yoy', 'log_mv', 'price_pos_1y', 'volatility_3m']
        df_clean = g[features].dropna()
        if len(df_clean) < MIN_TRAIN_DAYS: continue
        df_clean['target'] = (g['return'].rolling(21).sum().shift(-21) > 0).astype(int)
        df_clean = df_clean.dropna()
        if df_clean['target'].nunique() < 2: continue

        # 滚动训练
        scaler = StandardScaler()
        train_end = df_clean.index[MIN_TRAIN_DAYS]
        train_df = df_clean.loc[:train_end]
        pred_df  = df_clean.loc[train_end+pd.Timedelta(days=1):]
        if pred_df.empty: continue
        X_train = scaler.fit_transform(train_df[features])
        y_train = train_df['target']
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_train, y_train)
        factor = model.predict_proba(scaler.transform(pred_df[features]))[:, 1]
        factor_df = pd.DataFrame({'factor': factor}, index=pred_df.index)
        factor_df['symbol'] = symbol
        factor_list.append(factor_df)

    if not factor_list:
        return pd.Series(dtype=float), pd.DataFrame()
    daily_factor = pd.concat(factor_list).reset_index()
    # IC 计算（你原有逻辑）
    daily_factor['month_end'] = pd.to_datetime(daily_factor['trade_date']) + pd.offsets.MonthEnd(0)
    price_df['month'] = price_df['trade_date'] + pd.offsets.MonthEnd(0)
    next_ret = price_df.groupby(['symbol', 'month'])['return'].sum().shift(-1).reset_index(name='next_ret')
    df_merge = daily_factor.groupby(['symbol', 'month_end']).last().reset_index().merge(
        next_ret, left_on=['symbol', 'month_end'], right_on=['symbol', 'month'], how='inner')
    ic = df_merge.groupby('month_end').apply(lambda g: g['factor'].corr(g['next_ret']))
    return ic, daily_factor[['trade_date', 'symbol', 'factor']]