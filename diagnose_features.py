import clickhouse_connect
import pandas as pd
import numpy as np

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'

def diagnose():
    print("🕵️‍♂️ 正在诊断数据质量...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    
    # 1. 加载原始特征数据
    sql = f"""
    SELECT 
        time,
        close_price,
        wall_shift_pct,
        net_cvd,
        spoofing_ratio
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time ASC
    """
    df = client.query_df(sql)
    print(f"📊 原始数据行数: {len(df)}")
    
    # 2. 模拟预处理过程
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 检查每一列的空值情况
    print("\n🔍 空值统计 (每列缺失多少):")
    print(df.isnull().sum())
    
    # 检查 0 值情况 (特别是 wall_shift 和 cvd)
    print("\n0️⃣ 零值统计 (每列有多少个 0):")
    print((df == 0).sum())
    
    # 3. 统计描述
    # 填充 0 之后再看分布，看看特征是否有区分度
    df_filled = df.fillna(0)
    print("\n📈 特征统计描述 (已填充 0):")
    print(df_filled[['wall_shift_pct', 'net_cvd', 'spoofing_ratio']].describe())
    
    # 4. 检查特征与未来的相关性
    # 计算未来收益率
    df_filled['next_return'] = (df_filled['close_price'].shift(-1) - df_filled['close_price']) / df_filled['close_price'] * 100
    df_filled = df_filled.dropna()
    
    print("\n🔗 特征与未来涨跌的相关性 (Correlation):")
    # 看看特征跟 next_return 到底有没有关系
    correlations = df_filled[['wall_shift_pct', 'net_cvd', 'spoofing_ratio', 'next_return']].corr()['next_return']
    print(correlations)

if __name__ == "__main__":
    try:
        diagnose()
    except Exception as e:
        print(f"❌ 诊断出错: {e}")