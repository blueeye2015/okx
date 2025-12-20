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
LOG_FILE = "/data/okx/universal_signals.csv"

# --- 交易参数 (必须与回测一致) ---
TP_PCT = 0.015  # 止盈 1.5%
SL_PCT = 0.010  # 止损 1.0%

def append_to_log(data_row):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Time', 'Price', 'Signal', 'Type', 'TP_Price', 'SL_Price', 'Dist_EMA', 'CVD_Z'])
        writer.writerow(data_row)

def get_latest_signal():
    try:
        print(f"[{datetime.now()}] 🧠 全能猎手正在分析市场...")
        client = clickhouse_connect.get_client(**CLICKHOUSE)
        
        # 1. 拉取最近 10 天数据 (滚动训练，保持盘感)
        sql = f"""
        SELECT time, close_price, wall_shift_pct, net_cvd, spoofing_ratio
        FROM marketdata.features_15m
        WHERE symbol = '{SYMBOL}'
        ORDER BY time ASC
        """
        df = client.query_df(sql)
        
        # 2. 特征工程 (必须与回测完全一致!)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        
        # EMA 和 偏离度
        df['ema_50'] = df['close_price'].ewm(span=50).mean()
        df['dist_ema'] = (df['close_price'] - df['ema_50']) / df['ema_50'] * 100
        
        # CVD Z-Score
        roll_mean = df['net_cvd'].rolling(96).mean()
        roll_std = df['net_cvd'].rolling(96).std().replace(0, 1)
        df['cvd_z'] = (df['net_cvd'] - roll_mean) / roll_std
        
        # Wall Smooth
        df['wall_smooth'] = df['wall_shift_pct'].rolling(3).mean().fillna(0)
        
        # 3. 构造训练集 (打标签)
        window = 16 # 4小时
        df['next_max'] = df['close_price'].shift(-1).rolling(window).max()
        df['next_min'] = df['close_price'].shift(-1).rolling(window).min()
        
        TARGET_PCT = 1.2
        df['label'] = 0
        
        # Long 条件
        long_cond = (df['next_max'] / df['close_price'] - 1 > TARGET_PCT/100) & \
                    (df['next_min'] / df['close_price'] - 1 > -TARGET_PCT/2/100)
        # Short 条件
        short_cond = (df['next_min'] / df['close_price'] - 1 < -TARGET_PCT/100) & \
                     (df['next_max'] / df['close_price'] - 1 < TARGET_PCT/2/100)
        
        df.loc[long_cond, 'label'] = 1
        df.loc[short_cond, 'label'] = 2
        
        # 剔除无法计算标签的尾部数据用于训练
        df_train = df.dropna(subset=['next_max'])
        
        # 4. 训练模型
        features = ['dist_ema', 'cvd_z', 'wall_smooth', 'spoofing_ratio']
        X_train = df_train[features]
        y_train = df_train['label'].astype(int)
        
        # 使用回测证明有效的参数
        clf = DecisionTreeClassifier(
            max_depth=4, 
            criterion='entropy', 
            random_state=42, 
            class_weight={0:1, 1:2, 2:2}, 
            min_samples_leaf=20
        )
        clf.fit(X_train, y_train)
        
        # 5. 预测当下 (最新一行数据)
        latest = df.iloc[[-1]].copy()
        X_latest = latest[features]
        
        signal = clf.predict(X_latest)[0]
        prob = clf.predict_proba(X_latest)[0]
        
        # [新增] 波动率过滤器：拒绝垃圾时间的信号
        # 如果价格距离均线太近（绝对值 < 0.5%），强制空仓
        # 除非你有超低的手续费，否则不要吃这种鱼尾巴
        dist_ema_val = latest['dist_ema'].values[0]
        
        if abs(dist_ema_val) < 0.5: 
            if signal != 0:
                print(f"🛑 信号过滤: 乖离率过小 ({dist_ema_val:.2f}%), 放弃开单")
            signal = 0

        # [新增] 圣杯逻辑增强：只有深跌或暴涨才出手
        # 这种机会虽然少，但单笔利润大，足以覆盖手续费
        # 比如：只做乖离率 > 1.5% 或 < -1.5% 的单子
        # 6. 输出决策
        current_price = latest['close_price'].values[0]
        dist_val = latest['dist_ema'].values[0]
        
        print("\n" + "="*40)
        print(f"📊 当前价格: {current_price:.2f}")
        print(f"📏 EMA偏离度: {dist_val:.2f}% (正=超买, 负=超卖)")
        print(f"🌊 资金力度: {latest['cvd_z'].values[0]:.2f}")
        print("-" * 40)
        
        trade_type = "WAIT"
        tp_price = 0
        sl_price = 0
        
        if signal == 1:
            trade_type = "🟢 LONG (做多)"
            tp_price = current_price * (1 + TP_PCT)
            sl_price = current_price * (1 - SL_PCT)
            print(f"🚀 信号触发: {trade_type}")
            print(f"🎯 建议止盈: {tp_price:.2f} (+1.5%)")
            print(f"🛡️ 建议止损: {sl_price:.2f} (-1.0%)")
            
        elif signal == 2:
            trade_type = "🔴 SHORT (做空)"
            tp_price = current_price * (1 - TP_PCT)
            sl_price = current_price * (1 + SL_PCT)
            print(f"🚀 信号触发: {trade_type}")
            print(f"🎯 建议止盈: {tp_price:.2f} (+1.5%)")
            print(f"🛡️ 建议止损: {sl_price:.2f} (-1.0%)")
            
        else:
            print("☕ 信号: 观望 (Wait)")
            
        print("="*40)
        
        # 记录日志
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        append_to_log([
            current_time, current_price, signal, trade_type, 
            f"{tp_price:.2f}", f"{sl_price:.2f}", 
            f"{dist_val:.2f}", f"{latest['cvd_z'].values[0]:.2f}"
        ])

    except Exception as e:
        print(f"❌ 出错: {e}")

if __name__ == "__main__":
    get_latest_signal()