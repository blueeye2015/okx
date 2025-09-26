import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

def generate_sample_data(days=500, n_assets=3):
    """
    Generates sample daily K-line data for multiple assets for demonstration.
    """
    dfs = []
    for i in range(n_assets):
        price_changes = 1 + np.random.randn(days) / 100
        trend = np.linspace(1, 1 + i * 0.5, days)
        close = 100 * (price_changes * trend).cumprod()
        
        high = close * (1 + np.random.uniform(0, 0.05, days))
        low = close * (1 - np.random.uniform(0, 0.05, days))
        open_ = (high + low) / 2
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
    Calculates the independent variables (X) for the model.
    """
    # Group by asset to perform calculations independently for each asset
    grouped = df.groupby(level='asset')
    
    # Calculate daily returns
    df['return'] = grouped['close'].pct_change()
    
    # --- Feature Engineering ---
    
    # 1. Historical returns (using .transform for safe index alignment)
    df['momentum_30d'] = grouped['return'].transform(lambda x: x.rolling(30, min_periods=20).sum().shift(2))
    df['momentum_60d'] = grouped['return'].transform(lambda x: x.rolling(60, min_periods=40).sum().shift(2))
    df['momentum_90d'] = grouped['return'].transform(lambda x: x.rolling(90, min_periods=60).sum().shift(2))
    
    # 2. Volatility: 30-day standard deviation of returns
    df['volatility_30d'] = grouped['return'].transform(lambda x: x.rolling(30, min_periods=20).std().shift(1))
    
    # 3. Consecutive up/down days (return streaks)
    df['return_sign'] = np.sign(df['return'])
    # This transform correctly calculates streaks within each group
    streaks = grouped['return_sign'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    df['streak_days'] = streaks.shift(1)

    # 4. Volume indicators
    df['volume_ma_20d'] = grouped['volume'].transform(lambda x: x.rolling(20, min_periods=15).mean().shift(1))
    df['log_volume_ma_20d'] = np.log1p(df['volume_ma_20d']) # Use log to smooth data
    
    return df

def calculate_directional_momentum(df, window_size=120):
    """
    Trains the model on a rolling window and calculates the directional momentum factor.
    """
    # Define the target variable Y (1 if next day's return is positive, else 0)
    df['target'] = (df.groupby(level='asset')['return'].shift(-1) > 0).astype(int)
    
    # Clean data by dropping rows with NaN values created by rolling windows
    df_clean = df.dropna().copy()
    
    # Define independent and dependent variables
    features = [
        'momentum_30d', 'momentum_60d', 'momentum_90d',
        'volatility_30d', 'streak_days', 'log_volume_ma_20d'
    ]
    X = df_clean[features]
    X = sm.add_constant(X) # Add an intercept
    y = df_clean['target']
    
    # --- Rolling Window Modeling and Prediction ---
    predictions = pd.Series(index=X.index, dtype=float)
    
    print("Starting rolling window prediction... (This may take some time)")
    # Loop through the data points, maintaining a fixed-size training window
    for i in tqdm(range(window_size, len(df_clean))):
        X_train = X.iloc[i - window_size : i]
        y_train = y.iloc[i - window_size : i]
        X_pred = X.iloc[i : i + 1] # The current data point to predict for
        
        try:
            # Train a logistic regression model
            model = sm.Logit(y_train, X_train).fit(disp=0) # disp=0 suppresses fitting output
            # Predict the probability of the next day's return being positive
            pred_proba = model.predict(X_pred)[0]
            predictions.iloc[i] = pred_proba
        except Exception:
            # If the model fails to converge, mark prediction as NaN
            predictions.iloc[i] = np.nan
    
    # Merge the predicted probabilities (our factor) back into the original DataFrame
    df['d_mom_factor'] = predictions
    return df

# --- Main execution flow ---
if __name__ == '__main__':
    # 1. Generate or load your data
    # Replace this line with your actual data loading logic
    # e.g., daily_kline_data = pd.read_csv('your_data.csv').set_index(['date', 'asset'])
    daily_kline_data = generate_sample_data(days=500, n_assets=3)
    
    print("Original data preview:")
    print(daily_kline_data.head())
    
    # 2. Calculate features
    featured_data = calculate_features(daily_kline_data)
    
    print("\nData preview after adding features:")
    print(featured_data.tail())

    # 3. Calculate the final directional momentum factor
    final_data = calculate_directional_momentum(featured_data, window_size=120)

    print("\nFinal results (including the directional momentum factor):")
    print(final_data.groupby(level='asset')[['close', 'return', 'd_mom_factor']].tail(5))