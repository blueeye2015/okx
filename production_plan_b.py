import clickhouse_connect
import pandas as pd
import numpy as np
import joblib
import time
import os
from datetime import datetime

# ================= 配置区 =================
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'
MODEL_PATH = '/data/okx/universal_model.pkl'
SIGNAL_OUTPUT_PATH = '/data/okx/universal_signals.csv'

# 与训练保持一致的门槛
CONFIDENCE_THRESHOLD = 0.60 

# 止盈止损建议 (写入 CSV 供执行器读取)
# 这里的价格是动态计算的
TP_PCT = 0.006  # 0.6%
SL_PCT = 0.010  # 1.0%

def get_latest_data_and_features():
    """
    获取最新数据并计算复杂特征 (EMA96, Resonance)
    注意：我们需要拉取足够长的历史数据(比如200根)来计算 EMA 和 Rolling
    """
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    
    # 拉取过去 24小时的数据 (96 * 15m = 24h)，多拉点防备用
    query = f"""
    SELECT 
        time, close_price, 
        wall_shift_pct, spoofing_ratio, net_cvd
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time DESC
    LIMIT 200
    """
    df = client.query_df(query)
    
    # ClickHouse 查出来是倒序的，要转成正序计算指标
    df = df.sort_values('time').reset_index(drop=True)
    
    # ================= 特征工程 (必须与训练逻辑 100% 一致) =================
    
    # 1. 资金流 Z-Score
    df['cvd_mean'] = df['net_cvd'].rolling(window=96, min_periods=1).mean()
    df['cvd_std'] = df['net_cvd'].rolling(window=96, min_periods=1).std().replace(0, 1)
    df['cvd_zscore'] = (df['net_cvd'] - df['cvd_mean']) / df['cvd_std']
    
    # 2. 趋势乖离率 (EMA96)
    df['ema96'] = df['close_price'].ewm(span=96, adjust=False).mean()
    df['dist_ema96'] = (df['close_price'] - df['ema96']) / df['ema96'] * 100
    
    # 3. 趋势资金共振
    df['trend_flow_resonance'] = np.sign(df['dist_ema96']) * df['cvd_zscore']
    
    # 4. 填充
    df['wall_shift_pct'] = df['wall_shift_pct'].fillna(0)
    df['spoofing_ratio'] = df['spoofing_ratio'].fillna(1.0)
    df['cvd_zscore'] = df['cvd_zscore'].fillna(0)
    df['trend_flow_resonance'] = df['trend_flow_resonance'].fillna(0)
    
    # 取最后一行 (最新的 K 线)
    latest = df.iloc[[-1]].copy()
    
    return latest

def generate_signal():
    # 1. 准备特征
    try:
        df_latest = get_latest_data_and_features()
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        return

    current_time = df_latest['time'].iloc[0]
    current_price = df_latest['close_price'].iloc[0]

    # 1. 获取当前系统时间 (机器人的手表时间)
    system_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 必须保证特征顺序与训练时完全一致！
    feature_cols = ['wall_shift_pct', 'spoofing_ratio', 'cvd_zscore', 'dist_ema96', 'trend_flow_resonance']
    X = df_latest[feature_cols]
    
    # 2. 加载模型 & 预测
    try:
        clf = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 获取概率
    prob_up = clf.predict_proba(X)[0, 1]
    
    # 3. 生成信号
    signal = 0
    signal_type = "WAIT"
    
    # Plan B 逻辑：只有概率 > 0.60 才做多
    # 注意：我们目前训练的是"做多模型" (Target=1 代表涨)
    # 如果未来你想做空，需要单独训练一个做空模型，或者简单反转逻辑(但这不严谨)
    # 目前只做多 (Long Only) 或者 空仓
    if prob_up > CONFIDENCE_THRESHOLD:
        signal = 1
        signal_type = "🟢 LONG (爆发)"
    else:
        signal = 0
        signal_type = "WAIT"
        
    # 4. 计算建议止盈止损价格
    tp_price = 0.0
    sl_price = 0.0
    if signal == 1:
        tp_price = current_price * (1 + TP_PCT)
        sl_price = current_price * (1 - SL_PCT)

    # 5. 输出
    print(f"\n[{current_time}] 价格:{current_price:.2f} | 概率:{prob_up:.4f} | 信号:{signal_type}")
    print(f"   特征预览: Wall:{X['wall_shift_pct'].values[0]:.2f} | Reson:{X['trend_flow_resonance'].values[0]:.2f}")

    # 6. 保存到 CSV (追加模式)
    output_row = {
        'Log_Time': system_time, # [新增] 系统记录时间
        'Time': current_time,
        'Price': current_price,
        'Signal': signal,
        'Type': signal_type,
        'TP_Price': round(tp_price, 2),
        'SL_Price': round(sl_price, 2),
        'Prob': round(prob_up, 4),
        'Resonance': round(X['trend_flow_resonance'].values[0], 2)
    }
    
    df_out = pd.DataFrame([output_row])
    
    # 如果文件不存在，写入表头；否则追加
    if not os.path.exists(SIGNAL_OUTPUT_PATH):
        df_out.to_csv(SIGNAL_OUTPUT_PATH, index=False)
    else:
        df_out.to_csv(SIGNAL_OUTPUT_PATH, mode='a', header=False, index=False)
        
    print(f"✅ 信号已写入: {SIGNAL_OUTPUT_PATH}")

if __name__ == "__main__":
    print("🤖 Plan B 实盘信号生成器启动...")
    while True:
        try:
            generate_signal()
            # 每 15 分钟运行一次 (为了演示效果，这里设为 60秒 检查一次，实际应配合 Crontab 或 Sleep 900)
            # 建议：实际部署时，每分钟检查一下是否有新 K 线生成
            print("⏳ 等待下一轮...", end='\r')
            time.sleep(60) 
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ 主循环错误: {e}")
            time.sleep(10)