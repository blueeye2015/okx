import clickhouse_connect
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'

def load_data():
    print("🚀 加载数据...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    sql = f"""
    SELECT time, close_price, wall_shift_pct, net_cvd, spoofing_ratio
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time ASC
    """
    df = client.query_df(sql)
    return df

def feature_engineering_breakout(df):
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 1. 唐奇安通道 (Donchian Channel) - 寻找突破
    # 过去 24小时 (96根K线) 的最高价
    df['high_24h'] = df['close_price'].rolling(96).max().shift(1)
    # 距离前高的百分比 (如果是正数，说明突破了; 接近0说明在压力位)
    df['dist_to_high'] = (df['close_price'] - df['high_24h']) / df['high_24h'] * 100
    
    # 2. 资金爆发力 (CVD Acceleration)
    # 不看绝对值，看"加速度"。当前CVD和过去4小时均值的差
    cvd_mean_4h = df['net_cvd'].rolling(16).mean()
    cvd_std_4h = df['net_cvd'].rolling(16).std().replace(0, 1)
    df['cvd_accel'] = (df['net_cvd'] - cvd_mean_4h) / cvd_std_4h
    
    # 3. 墙的撤退 (Wall Retreat)
    # 如果墙大幅上移 (正值) 或者 卖墙撤单 (Spoofing高)，都是利好
    df['wall_impulse'] = df['wall_shift_pct'].rolling(3).sum()
    
    # 4. 目标：抓大鱼 (Big Pump)
    # 未来 4小时 涨幅 > 2.0% (波动率大的时候甚至可以定更高)
    TARGET_PUMP = 2.0
    
    # 计算未来最大涨幅
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=16)
    df['next_max_return'] = df['close_price'].rolling(window=indexer).max() / df['close_price'] - 1
    
    df['label'] = 0
    df.loc[df['next_max_return'] > TARGET_PUMP/100, 'label'] = 1
    
    df = df.dropna()
    print(f"🧹 数据准备完成: {len(df)} 条 | 🔥 暴涨样本: {sum(df['label']==1)}")
    return df

def run_breakout_bot(df):
    # 特征：是否突破、资金加速度、墙的动量
    features = ['dist_to_high', 'cvd_accel', 'wall_impulse', 'spoofing_ratio']
    X = df[features]
    y = df['label'].astype(int)
    
    # 划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    print("🧠 正在训练破浪者 (寻找大突破)...")
    # 暴涨样本很少，权重给高
    clf = DecisionTreeClassifier(
        max_depth=3, # 逻辑要简单粗暴
        criterion='entropy', 
        random_state=42, 
        class_weight={0:1, 1:5}, 
        min_samples_leaf=10
    )
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # --- 回测 ---
    # 突破策略的核心是盈亏比。
    # 我们假设：追突破，止损很窄 (-1%)，止盈很宽 (不设限，或者 +3% 以上)
    
    trade_count = 0
    wins = 0
    total_pnl = 0
    
    real_next_max = df.loc[X_test.index, 'next_max_return']
    
    print("\n⚔️ 开启突破回测...")
    
    for i in range(len(X_test)):
        if y_pred[i] == 1:
            # 过滤：只有真的接近前高，或者资金极其变态的时候才开
            # 这里用模型判断，我们可以加硬规则：
            # if X_test.iloc[i]['dist_to_high'] < -1.0: continue # 离前高太远的不做
            
            max_profit = real_next_max.iloc[i]
            
            # 简易模拟：
            # 如果最大涨幅能超过 2%，我们假设吃到了 2%
            # 如果没超过，但也没跌破止损... 比较难算
            # 我们简化：如果 max_profit > 2%，盈利 2%；否则亏损 0.5% (试错成本)
            
            if max_profit > 0.02:
                pnl = 0.02
                wins += 1
            else:
                pnl = -0.005 # 突破失败，快速止损
                
            total_pnl += pnl
            trade_count += 1
            
    print("\n" + "="*40)
    print(f"🌊 破浪者战报")
    print("="*40)
    print(f"🔥 尝试突破次数: {trade_count}")
    if trade_count > 0:
        print(f"🎯 成功爆发率: {wins/trade_count:.2%}")
        print(f"💰 累计回报 (单利): {total_pnl*100:.2f}%")
        print(f"⚖️ 盈亏比模拟: 赚2% vs 亏0.5% (需要 >20% 胜率即可盈利)")
    else:
        print("❄️ 没有发现突破机会")
        
    print("\n📜 突破密码 (Tree Rules):")
    print(export_text(clf, feature_names=features))

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        df = feature_engineering_breakout(df)
        run_breakout_bot(df)