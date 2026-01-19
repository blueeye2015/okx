import pandas as pd
import numpy as np
import clickhouse_connect
# ==========================================
# ⚙️ 高级分析配置
# ==========================================
# 策略触发阈值
RATIO_THRESHOLD = 0.5 
CVD_THRESHOLD = 2_000_000 

# 观察窗口：信号发出后，观察未来多少根K线？
# 假设是 15分钟K线，观察 48根 (即 12小时) 内的表现
LOOK_FORWARD_CANDLES = 8 

CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'

def load_data():
    print("🚀 正在加载数据...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    sql = f"""
    SELECT time,symbol, close_price, net_cvd, cvd_ratio
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time ASC
    """
    df = client.query_df(sql)
    return df

# ==========================================
# 1. 核心分析引擎 (MFE/MAE 追踪器)
# ==========================================
def run_distribution_analysis(df):
    signals = []
    
    # --- 信号生成 ---
    for i, row in df.iterrows():
        signal_type = None
        if row['cvd_ratio'] < -RATIO_THRESHOLD and row['net_cvd'] < -CVD_THRESHOLD:
            signal_type = 'Long'
        elif row['cvd_ratio'] > RATIO_THRESHOLD and row['net_cvd'] > CVD_THRESHOLD:
            signal_type = 'Short'
            
        if signal_type:
            # 获取未来 N 根K线的数据
            future_data = df.iloc[i+1 : i+1+LOOK_FORWARD_CANDLES]
            
            if future_data.empty: continue
            
            entry_price = row['close_price']
            
            # --- 核心计算逻辑 ---
            if signal_type == 'Long':
                # 做多：最高价是潜在盈利，最低价是潜在亏损
                highest_price = future_data['close_price'].max()
                lowest_price = future_data['close_price'].min()
                
                # MFE (最大潜在涨幅 %)
                mfe = (highest_price - entry_price) / entry_price
                # MAE (最大潜在跌幅 %, 负数)
                mae = (lowest_price - entry_price) / entry_price
                
            else: # Short
                # 做空：最低价是潜在盈利，最高价是潜在亏损
                lowest_price = future_data['close_price'].min()
                highest_price = future_data['close_price'].max()
                
                # MFE (最大潜在跌幅 %, 正数代表赚)
                mfe = (entry_price - lowest_price) / entry_price
                # MAE (最大潜在涨幅 %, 负数代表亏)
                mae = (entry_price - highest_price) / entry_price

            signals.append({
                'time': row['time'],
                'symbol': row['symbol'],
                'type': signal_type,
                'entry_price': entry_price,
                'MFE_pct': round(mfe * 100, 2), # 转换成百分比
                'MAE_pct': round(mae * 100, 2)  # 转换成百分比
            })
            
    return pd.DataFrame(signals)

# ==========================================
# 2. 分布报告生成器
# ==========================================
def generate_distribution_report(stats_df):
    if stats_df.empty: return "无数据"

    print("="*50)
    print("       🎯 止盈/止损 深度分布报告       ")
    print("="*50)
    print(f"统计样本: {len(stats_df)} 笔交易")
    print(f"观察窗口: 信号发生后的 {LOOK_FORWARD_CANDLES} 根K线内")
    print("-" * 50)

    # --- 1. 潜在盈利分布 (MFE) ---
    print("\n💰 【止盈潜力分布】(如果拿到最高点能赚多少?)")
    # 设置分档: >0.5%, >1.0%, >1.5% ... >5.0%
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    for t in thresholds:
        # 统计有多少比例的交易触及了这个涨幅
        count = len(stats_df[stats_df['MFE_pct'] >= t])
        ratio = (count / len(stats_df)) * 100
        bar = "█" * int(ratio / 5)
        print(f"触及 +{t}% : {ratio:5.1f}% 概率 {bar}")

    print(f"\n👉 平均潜在盈利 (Avg MFE): {stats_df['MFE_pct'].mean():.2f}%")
    print(f"👉 最大潜在盈利 (Max MFE): {stats_df['MFE_pct'].max():.2f}%")

    # --- 2. 潜在回撤分布 (MAE) ---
    print("\n🛡️ 【回撤风险分布】(最深跌了多少?)")
    # 设置分档: <-0.5%, <-1.0% ...
    risk_thresholds = [-0.5, -0.8, -1.0, -1.2, -1.5, -2.0]
    
    for t in risk_thresholds:
        # 统计有多少比例的交易跌破了这个位置
        # 注意 MAE 是负数，所以用 <=
        count = len(stats_df[stats_df['MAE_pct'] <= t])
        ratio = (count / len(stats_df)) * 100
        safe_ratio = 100 - ratio # 没跌破的比例
        bar = "▒" * int(ratio / 5)
        print(f"跌破 {t}% : {ratio:5.1f}% 概率 {bar} (安全度: {safe_ratio:.1f}%)")

    print(f"\n👉 平均承受回撤 (Avg MAE): {stats_df['MAE_pct'].mean():.2f}%")
    
    # --- 3. 智能推荐 ---
    # 简单算法：寻找能够覆盖 80% 交易的止损点，和 50% 交易的止盈点
    # 计算 MAE 的 15分位数 (代表只有 15% 的单子会被打损)
    suggested_sl = stats_df['MAE_pct'].quantile(0.15)
    # 计算 MFE 的 50分位数 (代表有一半的单子能吃到这个利润)
    suggested_tp = stats_df['MFE_pct'].quantile(0.50)
    
    print("-" * 50)
    print("💡 【AI 参数优化建议】")
    print(f"建议止损位 (SL): {suggested_sl:.2f}% (可扛过 85% 的震荡)")
    print(f"建议止盈位 (TP): {suggested_tp:.2f}% (有 50% 的概率能达到)")
    print(f"预计盈亏比     : 1 : {abs(suggested_tp/suggested_sl):.2f}")
    print("="*50)

# ==========================================
# 3. 运行分析
# ==========================================
# 假设 df 已经有了 (从上一步加载)
# file_path = '新文件 11.txt' ... (略，同前文)
df = load_data()
print(f"🚀 开始分布分析...")
dist_df = run_distribution_analysis(df)
generate_distribution_report(dist_df)