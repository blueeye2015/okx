#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版因子计算引擎
- 多维度特征工程
- 滚动训练窗口+样本权重
- 过拟合控制
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import logging
import numba

# <<<--- Numba加速OLS计算 ---<<<
@numba.jit(nopython=True)
def fast_ols_std(y, x):
    if len(y) < 2:
        return np.nan
    X = np.ones((len(x), 2))
    X[:, 1] = x
    XTX = X.T @ X
    det = np.linalg.det(XTX)
    if np.abs(det) < 1e-8:
        return np.nan
    try:
        beta = np.linalg.inv(XTX) @ X.T @ y
        residuals = y - X @ beta
        return np.std(residuals)
    except:
        return np.nan

def calc_iv(ret_1d: pd.Series, mkt_1d: pd.Series) -> float:
    """计算特质波动率"""
    df = pd.DataFrame({'ret': ret_1d, 'mkt': mkt_1d}).dropna()
    if len(df) < 60:
        return np.nan
    return fast_ols_std(df['ret'].values, df['mkt'].values)

# 全局参数
MIN_TRAIN_DAYS = 500
LGB_PARAMS = {
    'objective': 'binary',
    'n_estimators': 200,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 50,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': 1
}

def calculate_factors_for_single_symbol(symbol: str, price_df: pd.DataFrame, mkt_ret_series: pd.Series):
    """
    核心增强版：为单只股票计算因子
    """
    g = price_df[price_df['symbol'] == symbol][['trade_date', 'close', 'total_mv', 'deduct_parent_netprofit', 'volume', 'symbol']]
    if len(g) < MIN_TRAIN_DAYS:
        logging.debug(f'{symbol} 数据不足{MIN_TRAIN_DAYS}天')
        return None
    
    fac = calc_daily_factor_monthly_train(g, mkt_ret_series)
    if fac is not None and not fac.empty:
        fac['symbol'] = symbol
        return fac
    return None

def calc_daily_factor_monthly_train(df_sym: pd.DataFrame, mkt_ret_series: pd.Series) -> pd.DataFrame:
    """增强版：多维度特征 + 滚动训练"""
    df = df_sym.copy().set_index('trade_date')
    df['return'] = df['close'].pct_change()
    df = df.merge(mkt_ret_series.rename('mkt_ret'), left_index=True, right_index=True, how='left')
    
    # <<<--- 特征工程增强 ---<<<
    
    # 1. 动量特征（多周期）
    df['return_1m'] = df['return'].rolling(21).sum()
    df['return_3m'] = df['return'].rolling(63).sum()
    df['return_6m'] = df['return'].rolling(126).sum()
    df['return_12m'] = df['return'].rolling(252).sum()
    
    # 2. 波动率特征
    df['vol_1m'] = df['return'].rolling(21).std()
    df['vol_3m'] = df['return'].rolling(63).std()
    df['vol_12m'] = df['return'].rolling(252).std()
    df['vol_ratio'] = df['vol_1m'] / df['vol_12m'].clip(lower=0.001)
    
    # 3. 价格位置特征（多周期）
    for window in [63, 126, 252]:
        rolling_max = df['close'].rolling(window, min_periods=window//2).max()
        rolling_min = df['close'].rolling(window, min_periods=window//2).min()
        df[f'price_pos_{window}d'] = (df['close'] - rolling_min) / (rolling_max - rolling_min).clip(lower=1e-8)
    
    # 4. 流动性特征
    df['turnover'] = (df['volume'] * df['close']) / df['total_mv']
    df['turnover_1m_avg'] = df['turnover'].rolling(21).mean()
    
    # 5. 趋势强度特征（0-3分）
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_60'] = df['close'].rolling(60).mean()
    df['ma_120'] = df['close'].rolling(120).mean()
    df['trend_score'] = (df['ma_20'] > df['ma_60']).astype(int) + \
                       (df['ma_60'] > df['ma_120']).astype(int) + \
                       (df['close'] > df['ma_20']).astype(int)
    
    # 6. RSI特征
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.clip(lower=0.001)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 7. MACD特征
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 8. 业绩特征
    df['profit_yoy'] = df['deduct_parent_netprofit'].pct_change(periods=252).fillna(0)
    df['profit_2yoy'] = df['deduct_parent_netprofit'].pct_change(periods=504).fillna(0)
    df['profit_ttm'] = df['deduct_parent_netprofit'] / df['total_mv']
    df['log_mv'] = np.log(df['total_mv'].clip(lower=1))
    
    # 9. 动量连涨连跌特征
    max_dur = 12
    df['sign'] = np.where(df['return'] > 0, 1, -1)
    df['streak_start'] = (df['sign'] != df['sign'].shift(1)).cumsum()
    df['raw_streak'] = df.groupby(['streak_start']).cumcount() + 1
    df['raw_streak'] = np.minimum(df['raw_streak'], max_dur)
    df['P_d'] = np.where(df['sign'] == 1, df['raw_streak'], 0)
    df['N_d'] = np.where(df['sign'] == -1, df['raw_streak'], 0)
    
    # <<<--- 滞后所有特征（防未来函数） ---<<<
    feature_base = [
        'return_1m', 'return_6m',
        'vol_ratio',  # 波动率变化（量缩价稳）
        'price_pos_126d',  # 半年价格位置（低位反转）
        'turnover_1m_avg', # 换手率（越低越好，或极高时的反转）
        'profit_ttm', # 估值/盈利能力（防雷） 
        'log_mv' # 市值
    ]
    
    for feat in feature_base:
        df[f'{feat}_lag'] = df[feat].shift(1)
    
    # 滞后其他特征
    df['P_d_t1'] = df['P_d'].shift(1)
    df['N_d_t1'] = df['N_d'].shift(1)
    df['rm_t1'] = df['mkt_ret'].shift(1)
    
    # IV计算
    df['IV'] = df['return'].expanding(MIN_TRAIN_DAYS).apply(
        lambda r: calc_iv(r, df.loc[r.index, 'mkt_ret']), raw=False)
    df['IV_t1'] = df['IV'].shift(1)
    
    # 最终特征列表
    features = [f'{feat}_lag' for feat in feature_base] + \
               ['IV_t1']
    
    # 数据清洗
    df_clean = df.dropna(subset=features).copy()
    df_clean['target'] = (df_clean['return'].rolling(21).sum().shift(-21) > 0).astype(int)
    
    if len(df_clean) < MIN_TRAIN_DAYS + 21:
        logging.debug(f"数据不足：{len(df_clean)}天")
        return pd.DataFrame()
    
    # <<<--- 滚动训练 + 样本权重 ---<<<
    all_factors = []
    scaler = StandardScaler()
    training_dates = df_clean.index.to_period('M').unique().to_timestamp(how='end')
    
    # <<<--- 🔥 新增 1：初始化重要性记录列表 ---<<<
    feature_importance_list = []

    for i, train_end_date in enumerate(training_dates):
        # 获取当前月份所有交易日
        valid_dates = df_clean.index[df_clean.index <= train_end_date]
        if valid_dates.empty:
            continue
        
        actual_train_end_date = valid_dates[-1]
        # 2. 【核心修复】设置隔离期（Gap），防止标签泄露
        # 必须回退 30 天（或 21 个交易日以上），因为你的 Target 是 shift(-21)
        safe_train_end_date = actual_train_end_date - pd.Timedelta(days=30)
        
        # 3. 【日期设定】计算训练开始日期
        # 逻辑：从“安全截止日”往前推 N 个月
        # 建议：A股一轮牛熊通常 3-5 年，建议设为 60 个月（5年）能让模型更稳健
        # 如果追求运算速度，维持 36 个月也可以
        train_window_months = 60  
        train_start_date = safe_train_end_date - pd.DateOffset(months=train_window_months)
        
        # 4. 切分数据
        train_df = df_clean.loc[train_start_date:safe_train_end_date]
        
        if len(train_df) < MIN_TRAIN_DAYS:
            continue
        
        X_train = train_df[features].dropna()
        y_train = train_df.loc[X_train.index, 'target'].dropna()
        
        if y_train.nunique() < 2:
            continue
        
        # 样本权重：近期样本权重更高
        date_weights = np.exp(-0.01 * np.arange(len(y_train))[::-1])
        sample_weights = date_weights / date_weights.sum() * len(date_weights)
        
        # 训练模型
        X_train_scaled = scaler.fit_transform(X_train)
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights[:len(X_train)])

        # <<<--- 🔥 新增 2：记录当前月份模型最看重什么 ---<<<
        # 提取当前模型的特征重要性
        imp_df = pd.DataFrame({
            'feature': features,
            'gain': model.booster_.feature_importance(importance_type='gain'), # gain表示贡献了多少收益
            'date': actual_train_end_date
        })
        feature_importance_list.append(imp_df)
        
        # 预测下月
        pred_start_date = actual_train_end_date + pd.Timedelta(days=1)
        pred_end_date = pred_start_date + pd.offsets.MonthEnd(1)
        pred_df = df_clean.loc[pred_start_date:pred_end_date]
        
        if pred_df.empty:
            continue
        
        X_pred = pred_df[features]
        pred_probas = model.predict_proba(scaler.transform(X_pred))[:, 1]
        month_factors = pd.DataFrame(pred_probas, index=X_pred.index, columns=['factor'])
        all_factors.append(month_factors)
    
        # <<<--- 🔥 新增 3：循环结束后，汇总并打印平均重要性 ---<<<
        if feature_importance_list:
            all_imp = pd.concat(feature_importance_list)
            # 按特征分组取平均值，从大到小排序
            summary = all_imp.groupby('feature')['gain'].mean().sort_values(ascending=False)
            
            print("\n" + "="*40)
            print(f"📊 模型心中的“选股秘籍” (Symbol: {df_sym['symbol'].iloc[0]})")
            print("="*40)
            # 计算百分比，更直观
            print((summary / summary.sum() * 100).apply(lambda x: f"{x:.1f}%"))
            print("="*40 + "\n")
    if not all_factors:
        return pd.DataFrame()
    
    final_factors = pd.concat(all_factors)
    return final_factors.reset_index()