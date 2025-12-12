import clickhouse_connect
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import os
import csv

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'
LOG_FILE = "trading_signals.csv" # 结果存到这个文件

def append_to_log(timestamp, close_price, wall_shift, cvd, spoofing, signal, decision):
    """把信号写入 CSV 文件，方便后续用 Excel 分析"""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        # 如果是新文件，先写表头
        if not file_exists:
            writer.writerow(['Time', 'Price', 'Wall_Shift', 'CVD_Z', 'Spoofing', 'Signal', 'Decision'])
        writer.writerow([timestamp, close_price, wall_shift, cvd, spoofing, signal, decision])

def get_latest_signal():
    try:
        client = clickhouse_connect.get_client(**CLICKHOUSE)
        
        # 1. 拉取最近 7 天数据 (滚动训练)
        sql = f"""
        SELECT time, close_price, wall_shift_pct, net_cvd, spoofing_ratio
        FROM marketdata.features_15m
        WHERE symbol = '{SYMBOL}'
          AND time >= now() - INTERVAL 7 DAY
        ORDER BY time ASC
        """
        df = client.query_df(sql)
        
        if len(df) < 100:
            print("⚠️ 数据不足，跳过")
            return

        # 2. 数据清洗 (与训练逻辑严格一致)
        df = df.replace([np.inf, -np.inf], np.nan)
        df['wall_shift_pct'] = df['wall_shift_pct'].ffill().fillna(0).clip(-3, 3)
        df['spoofing_ratio'] = df['spoofing_ratio'].ffill().fillna(1.0)
        df['net_cvd'] = df['net_cvd'].fillna(0)
        
        # 归一化
        roll_mean = df['net_cvd'].rolling(20, min_periods=1).mean()
        roll_std = df['net_cvd'].rolling(20, min_periods=1).std().replace(0, 1)
        df['cvd_zscore'] = (df['net_cvd'] - roll_mean) / roll_std
        
        # 打标签
        df['next_return'] = (df['close_price'].shift(-1) - df['close_price']) / df['close_price'] * 100
        df_train = df.dropna(subset=['next_return'])
        df_train['label'] = (df_train['next_return'] > 0.1).astype(int)
        
        # 3. 训练
        X_train = df_train[['wall_shift_pct', 'cvd_zscore', 'spoofing_ratio']]
        y_train = df_train['label']
        
        clf = DecisionTreeClassifier(
            max_depth=4, criterion='entropy', random_state=42, 
            class_weight={0: 1, 1: 5}, min_samples_leaf=20
        )
        clf.fit(X_train, y_train)
        
        # 4. 预测最新 K 线
        latest = df.iloc[[-1]].copy()
        X_latest = latest[['wall_shift_pct', 'cvd_zscore', 'spoofing_ratio']]
        signal = clf.predict(X_latest)[0]
        
        # 5. 风控与决策
        wall_shift = latest['wall_shift_pct'].values[0]
        decision = "WAIT"
        
        if wall_shift < -0.2:
            decision = "STOP (Wall Collapsed)"
        elif signal == 1:
            decision = "BUY"
        else:
            decision = "WAIT"
            
        # 6. 记录日志
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        price = latest['close_price'].values[0]
        print(f"[{current_time}] 价格: {price} | 决策: {decision}")
        
        append_to_log(
            current_time, price, 
            f"{wall_shift:.2f}", 
            f"{latest['cvd_zscore'].values[0]:.2f}", 
            f"{latest['spoofing_ratio'].values[0]:.2f}", 
            signal, decision
        )
        
    except Exception as e:
        print(f"❌ 出错: {e}")

if __name__ == "__main__":
    get_latest_signal()