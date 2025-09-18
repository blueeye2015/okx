import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

def generate_sample_data(days=500, n_assets=3):
    """
    生成模拟的多个资产的日K线数据用于演示。
    
    Args:
        days (int): 数据总天数。
        n_assets (int): 资产数量。
        
    Returns:
        pandas.DataFrame: 包含多个资产日K线数据的DataFrame。
    """
    dfs = []
    for i in range(n_assets):
        # 创建一个随机游走的价格序列
        price_changes = 1 + np.random.randn(days) / 100
        # 添加一个趋势，让数据更真实
        trend = np.linspace(1, 1 + i * 0.5, days)
        close = 100 * (price_changes * trend).cumprod()
        
        # 基于收盘价生成其他K线数据
        high = close * (1 + np.random.uniform(0, 0.05, days))
        low = close * (1 - np.random.uniform(0, 0.05, days))
        open_ = (high + low) / 2 # 简化开盘价
        volume = np.random.randint(1_000_000, 10_000_000, days)
        
        df = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        })
        df['asset'] = f'ASSET_{i}'
        df['date'] = pd.to_datetime(pd.to_datetime('2023-01-01') + pd.to_timedelta(np.arange(days), 'D'))
        
        dfs.append(df)
        
    return pd.concat(dfs).set_index(['date', 'asset'])

def calculate_features(df):
    """
    计算模型所需的自变量 (X)。
    
    Args:
        df (pandas.DataFrame): 包含K线数据的DataFrame。
        
    Returns:
        pandas.DataFrame: 增加了特征列的DataFrame。
    """
    # 按资产分组计算
    grouped = df.groupby(level='asset')
    
    # 核心：计算日收益率
    df['return'] = grouped['close'].pct_change()
    
    # --- 特征工程 ---
    # 1. 历史收益率 (为避免短期反转，剔除最近一天)
    # 例如，计算过去30天的收益率，但使用2天前到31天前的数据
    df['momentum_30d'] = grouped['return'].rolling(30, min_periods=20).sum().shift(2)
    df['momentum_60d'] = grouped['return'].rolling(60, min_periods=40).sum().shift(2)
    df['momentum_90d'] = grouped['return'].rolling(90, min_periods=60).sum().shift(2)
    
    # 2. 波动率：过去30天收益率的标准差
    df['volatility_30d'] = grouped['return'].rolling(30, min_periods=20).std().shift(1)
    
    # 3. 连续上涨/下跌天数 (收益持续期)
    df['return_sign'] = np.sign(df['return'])
    streaks = grouped['return_sign'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    df['streak_days'] = streaks.shift(1)

    # 4. 成交量指标
    df['volume_ma_20d'] = grouped['volume'].rolling(20, min_periods=15).mean().shift(1)
    df['log_volume_ma_20d'] = np.log1p(df['volume_ma_20d']) # 取对数平滑数据
    
    return df

def calculate_directional_momentum(df, window_size=120):
    """
    在滚动窗口上训练模型并计算方向动量因子。
    
    Args:
        df (pandas.DataFrame): 包含特征的DataFrame。
        window_size (int): 滚动窗口的大小（天数）。
        
    Returns:
        pandas.DataFrame: 增加了方向动量因子列的DataFrame。
    """
    # --- 目标变量 Y ---
    # Y = 1 如果下一天的收益率为正, 否则为 0
    df['target'] = (df.groupby(level='asset')['return'].shift(-1) > 0).astype(int)
    
    # 清理数据，去除包含NaN的行
    df_clean = df.dropna().copy()
    
    # 定义自变量和因变量
    features = [
        'momentum_30d', 'momentum_60d', 'momentum_90d',
        'volatility_30d', 'streak_days', 'log_volume_ma_20d'
    ]
    X = df_clean[features]
    X = sm.add_constant(X) # 添加截距项
    y = df_clean['target']
    
    # --- 滚动建模与预测 ---
    # 初始化一个空的Series来存储预测概率
    predictions = pd.Series(index=X.index, dtype=float)
    
    # 使用tqdm显示进度条
    # unique_dates = X.index.get_level_values('date').unique()
    # for i in tqdm(range(window_size, len(unique_dates)), desc="Rolling Window Prediction"):
    #     train_start_date = unique_dates[i - window_size]
    #     train_end_date = unique_dates[i - 1]
    #     predict_date = unique_dates[i]
        
    #     X_train = X.loc[train_start_date:train_end_date]
    #     y_train = y.loc[train_start_date:train_end_date]
    #     X_pred = X.loc[predict_date]
    # 在这个简化的版本中，我们将对每个时间点进行训练和预测
    # 更好的做法是按日期滚动，但为了代码简洁，我们逐行处理
    print("开始滚动窗口预测...（这可能需要一些时间）")
    for i in tqdm(range(window_size, len(df_clean))):
        # 定义训练和预测窗口
        X_train = X.iloc[i-window_size:i]
        y_train = y.iloc[i-window_size:i]
        X_pred = X.iloc[i:i+1] # 取当前行用于预测
        
        # 训练逻辑回归模型
        # 我们在这里使用try-except来处理可能无法收敛的情况
        try:
            model = sm.Logit(y_train, X_train).fit(disp=0) # disp=0 关闭拟合过程的输出
            # 预测下一期上涨的概率
            pred_proba = model.predict(X_pred)[0]
            predictions.iloc[i] = pred_proba
        except Exception:
            predictions.iloc[i] = np.nan # 如果模型失败，则标记为NaN
    
    # 将预测的概率（即我们的因子）合并回原始DataFrame
    df['d_mom_factor'] = predictions
    return df

# --- 主流程 ---
if __name__ == '__main__':
    # 1. 生成或加载您的数据
    # 请将下面这行替换为您自己的数据加载逻辑
    # 例如: my_data = pd.read_csv('your_kline_data.csv')
    daily_kline_data = generate_sample_data(days=500, n_assets=3)
    
    print("原始数据预览:")
    print(daily_kline_data.head())
    
    # 2. 计算特征
    featured_data = calculate_features(daily_kline_data)
    
    print("\n添加特征后的数据预览:")
    print(featured_data.tail())

    # 3. 计算方向动量因子
    final_data = calculate_directional_momentum(featured_data, window_size=120)

    print("\n最终结果（包含方向动量因子）:")
    # 打印每个资产的最新因子值
    print(final_data.groupby(level='asset')[['close', 'return', 'd_mom_factor']].tail(5))