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
LOG_FILE = "/data/okx/accumulation_signals.csv"

def append_to_log(data_row):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Time', 'Price', 'Signal', 'Description', 'CVD_Cons', 'CVD_Cum', 'Wall_Cum'])
        writer.writerow(data_row)

def get_latest_signal():
    try:
        print(f"[{datetime.now()}] 🐋 潜伏者雷达启动...")
        client = clickhouse_connect.get_client(**CLICKHOUSE)
        
        # 1. 拉取足够长的数据 (至少2天，用于计算8小时窗口和训练)
        sql = f"""
        SELECT time, close_price, wall_shift_pct, net_cvd, spoofing_ratio
        FROM marketdata.features_15m
        WHERE symbol = '{SYMBOL}'
        ORDER BY time ASC
        """
        df = client.query_df(sql)
        
        # 2. 特征工程 (必须与回测一致: 8小时窗口)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        WINDOW = 32 # 8小时
        
        df['cvd_cum'] = df['net_cvd'].rolling(window=WINDOW).sum()
        df['cvd_consistency'] = (df['net_cvd'] > 0).rolling(window=WINDOW).mean()
        df['wall_cum_shift'] = df['wall_shift_pct'].rolling(window=WINDOW).sum()
        df['price_change_8h'] = df['close_price'].pct_change(periods=WINDOW) * 100
        
        # 3. 训练模型 (只用最新数据重新训练，保持敏锐)
        # 打标签: 未来24小时涨 3%
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=96)
        df['future_max_return'] = df['close_price'].rolling(window=indexer).max() / df['close_price'] - 1
        df['label'] = (df['future_max_return'] > 0.03).astype(int)
        
        df_train = df.dropna(subset=['future_max_return'])
        
        features = ['cvd_consistency', 'cvd_cum', 'wall_cum_shift', 'price_change_8h']
        X_train = df_train[features]
        y_train = df_train['label']
        
        # 使用你回测出的神参数
        clf = DecisionTreeClassifier(
            max_depth=3, 
            criterion='entropy', 
            random_state=42, 
            class_weight={0:1, 1:3}, 
            min_samples_leaf=20
        )
        clf.fit(X_train, y_train)
        
        # 4. 预测当下
        latest = df.iloc[[-1]].copy()
        X_latest = latest[features]
        signal = clf.predict(X_latest)[0]
        
        # [新增] 圣杯规则硬过滤
        cons_val = latest['cvd_consistency'].values[0]
        wall_val = latest['wall_cum_shift'].values[0]

        # 只有资金一致性够高，且墙没有大幅上移(压盘)时，才算数
        if signal == 1:
            if cons_val <= 0.33:
                signal = 0
                print(f"☕ 信号被过滤: 资金一致性不足 ({cons_val:.2f} <= 0.33)")
            elif wall_val > 5.0:  # 回测中的阈值是 4.97
                signal = 0
                print(f"☕ 信号被过滤: 墙上移过快，疑似拉升非吸筹 ({wall_val:.2f})")
        # 5. 硬规则过滤 (Price Suppression)
        # 如果价格已经涨起来了 (>1%)，就不算吸筹
        price_pumped = latest['price_change_8h'].values[0] > 1.0
        
        current_price = latest['close_price'].values[0]
        desc = "WAIT"
        
        print("\n" + "="*40)
        print(f"📊 当前价格: {current_price:.2f}")
        print(f"💧 资金一致性: {latest['cvd_consistency'].values[0]:.2f} (阈值 > 0.33)")
        print(f"🧱 墙累计移动: {latest['wall_cum_shift'].values[0]:.2f}")
        print("-" * 40)
        
        if signal == 1:
            if not price_pumped:
                desc = "🐋 WHALE ACCUMULATION DETECTED! (巨鲸吸筹)"
                print(f"🚀 {desc}")
                print("💡 建议操作: 现货/低倍做多，持有24小时，目标 +3% ~ +5%")
            else:
                desc = "WAIT (Signal but Price Pumped)"
                print(f"☕ {desc} - 价格已涨，错过最佳潜伏期")
        else:
            print("☕ 暂无吸筹迹象")
            
        print("="*40)
        
        # 记录
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if "ACCUMULATION" in desc:
             # 只有真正的信号才记录，避免日志太乱
            append_to_log([
                current_time, current_price, 1, desc,
                f"{latest['cvd_consistency'].values[0]:.2f}",
                f"{latest['cvd_cum'].values[0]:.2f}",
                f"{latest['wall_cum_shift'].values[0]:.2f}"
            ])

    except Exception as e:
        print(f"❌ 出错: {e}")

if __name__ == "__main__":
    get_latest_signal()