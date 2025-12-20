import clickhouse_connect
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'

def load_data():
    print("🚀 正在加载数据 (寻找吸筹形态)...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    sql = f"""
    SELECT time, close_price, wall_shift_pct, net_cvd, spoofing_ratio
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time ASC
    """
    df = client.query_df(sql)
    return df

def feature_engineering_accumulation(df):
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # --- 1. 定义 "慢慢积累" 的时间窗口 ---
    # 我们看过去 4小时 (16根K线) 到 12小时 (48根K线)
    WINDOW = 32 # 8小时
    
    # --- 2. 资金累积特征 (CVD Trend) ---
    # 计算过去 8小时的 CVD 累积值
    # 如果这个值很大，说明主力买了很多，但如果是"缓缓"进入，我们看斜率
    df['cvd_cum'] = df['net_cvd'].rolling(window=WINDOW).sum()
    
    # 资金的一致性：过去 8小时里，有多少根K线 CVD 是正的？
    # 比如 32根K线里有 25根是正的，说明是持续买入，而不是突发买入
    df['cvd_consistency'] = (df['net_cvd'] > 0).rolling(window=WINDOW).mean()
    
    # --- 3. 墙的垫高特征 (Wall Build-up) ---
    # 累计的墙体移动。如果长期累积是正的，说明支撑位在不断上移
    df['wall_cum_shift'] = df['wall_shift_pct'].rolling(window=WINDOW).sum()
    
    # --- 4. 价格的压制特征 (Price Suppression) ---
    # 吸筹的一个关键点是：价格不能涨。如果价格已经涨飞了，那就不是吸筹而是拉升了。
    # 我们看过去 8小时的涨幅
    df['price_change_8h'] = df['close_price'].pct_change(periods=WINDOW) * 100
    
    # --- 5. 目标：随后的大爆发 ---
    # 如果满足吸筹，未来 24小时 (96根K线) 应该会有大涨
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=96)
    df['future_max_return'] = df['close_price'].rolling(window=indexer).max() / df['close_price'] - 1
    
    # 标记：未来24小时涨幅超过 3%，且不是假突破
    df['label'] = 0
    df.loc[df['future_max_return'] > 0.03, 'label'] = 1
    
    df = df.dropna()
    print(f"🧹 数据重构完成: {len(df)} 条 | 🐋 巨鲸吸筹样本: {sum(df['label']==1)}")
    return df

def run_accumulation_scan(df):
    # 特征：资金一致性、累计资金量、墙的累计移动、当前价格涨幅
    features = ['cvd_consistency', 'cvd_cum', 'wall_cum_shift', 'price_change_8h']
    X = df[features]
    y = df['label'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    print("🧠 正在训练潜伏者模型 (寻找爆发前夜)...")
    clf = DecisionTreeClassifier(
        max_depth=3, # 简单规则
        criterion='entropy', 
        random_state=42, 
        class_weight={0:1, 1:3}, 
        min_samples_leaf=20
    )
    clf.fit(X_train, y_train)
    
    # --- 规则解析与模拟 ---
    y_pred = clf.predict(X_test)
    
    # 获取真实回报
    real_returns = df.loc[X_test.index, 'future_max_return']
    
    print("\n⚔️ 开启吸筹扫描...")
    trade_count = 0
    wins = 0
    
    for i in range(len(X_test)):
        # 强制增加人工逻辑：真正的吸筹，价格必定是滞涨的
        # 如果过去8小时已经涨了 > 1%，那不算吸筹，那是追高
        price_pumped = X_test.iloc[i]['price_change_8h'] > 1.0
        
        if y_pred[i] == 1 and not price_pumped:
            # 这是一个信号！
            ret = real_returns.iloc[i]
            if ret > 0.03: wins += 1
            trade_count += 1
            
    print("\n" + "="*40)
    print(f"🐋 潜伏者战报")
    print("="*40)
    print(f"🔍 发现疑似吸筹: {trade_count} 次")
    if trade_count > 0:
        print(f"🎯 随后24h暴涨率: {wins/trade_count:.2%}")
    else:
        print("❄️ 也就是最近主力没有在吸筹 (或者已经拉升完了)")
        
    print("\n📜 巨鲸密码 (Tree Rules):")
    print(export_text(clf, feature_names=features))

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        df = feature_engineering_accumulation(df)
        run_accumulation_scan(df)