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
    print("🚀 加载数据中...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    # 依然加载 15m 数据，但我们在 Python 里把它合成大周期
    sql = f"""
    SELECT time, close_price, wall_shift_pct, net_cvd, spoofing_ratio
    FROM marketdata.features_15m
    WHERE symbol = '{SYMBOL}'
    ORDER BY time ASC
    """
    df = client.query_df(sql)
    return df

def feature_engineering_sniper(df):
    """
    狙击手特征工程：构造长周期、高质量特征
    """
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 1. 构造趋势特征 (Trend)
    # 计算 4小时均线 (15m * 16 = 4小时)
    df['ma_long'] = df['close_price'].rolling(window=16).mean()
    # 趋势判定: 1=牛市, 0=熊市/震荡
    df['trend_bullish'] = (df['close_price'] > df['ma_long']).astype(int)
    
    # 2. 构造资金蓄力特征 (Accumulation)
    # 过去 1小时 (4根K线) 的累计 CVD
    df['cvd_1h_sum'] = df['net_cvd'].rolling(window=4).sum()
    
    # CVD 归一化 (Z-Score) - 这里的窗口放大到 1天 (96根K线) 看相对强度
    cvd_mean = df['cvd_1h_sum'].rolling(window=96).mean()
    cvd_std = df['cvd_1h_sum'].rolling(window=96).std().replace(0, 1)
    df['cvd_zscore_long'] = (df['cvd_1h_sum'] - cvd_mean) / cvd_std
    
    # 3. 构造盘口异动特征 (Wall)
    # 过去 1小时内，墙是否曾经大幅上移？(取最大值)
    df['wall_shift_1h_max'] = df['wall_shift_pct'].rolling(window=4).max()
    
    # 4. [高标准] 定义目标 (Target)
    # 我们不再看下个 15分钟，我们看未来 4小时 (16根K线) 的回报
    df['future_return_4h'] = (df['close_price'].shift(-16) - df['close_price']) / df['close_price'] * 100
    
    # 清洗掉算不出指标的前面部分和后面部分
    df = df.dropna()
    
    # 5. [高标准] 打标签
    # 只有未来 4小时涨幅超过 0.8% (扣除手续费还有大赚) 才叫机会
    # 普通的小涨，我们看不上，标为 0
    df['label'] = 0
    df.loc[df['future_return_4h'] > 0.8, 'label'] = 1
    
    print(f"🧹 数据重构完成: 剩余 {len(df)} 条 | 正样本(大涨机会): {sum(df['label'])}")
    return df

def run_sniper_backtest(df):
    # 特征只选最硬核的
    features = ['wall_shift_1h_max', 'cvd_zscore_long', 'trend_bullish']
    X = df[features]
    y = df['label'].astype(int)
    
    # 划分 (不打乱)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # --- 训练阶段 ---
    print("🧠 狙击手正在校准瞄准镜 (Training)...")
    # 权重依然给高一点，因为大机会很少，不能漏
    clf = DecisionTreeClassifier(
        max_depth=3, # 树浅一点，逻辑越简单越可靠
        criterion='entropy', 
        random_state=42, 
        class_weight={0: 1, 1: 3.0}, 
        min_samples_leaf=30
    )
    clf.fit(X_train, y_train)
    
    # --- 预测阶段 ---
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] # 获取置信度
    
    # --- [核心] 狙击手开火逻辑 (Hard Filters) ---
    final_signals = np.zeros(len(X_test))
    
    for i in range(len(X_test)):
        # 提取当前时刻的特征
        idx = X_test.index[i]
        trend_ok = X_test.loc[idx, 'trend_bullish'] == 1
        cvd_strong = X_test.loc[idx, 'cvd_zscore_long'] > 0.5 # 资金必须是在流入的
        model_say_buy = y_pred[i] == 1
        confidence = y_prob[i]
        
        # 🔫 开火条件：
        # 1. 模型说买
        # 2. 必须顺势 (价格在均线之上)
        # 3. 资金面必须配合 (CVD > 0.5个标准差)
        if model_say_buy and trend_ok and cvd_strong:
            final_signals[i] = 1
            
    # --- 回测统计 ---
    # 获取未来 4小时 的真实回报
    real_returns_4h = df.loc[X_test.index, 'future_return_4h'] / 100
    
    # 计算资金曲线 (假设每次持有 4小时)
    # 注意：这里简化计算，假设信号不重叠。实际如果连续信号，相当于加仓。
    # 为了严谨，我们取信号点的回报。
    
    trade_count = sum(final_signals)
    win_count = sum((final_signals == 1) & (real_returns_4h > 0))
    
    print("\n" + "="*40)
    print(f"🔫 狙击手战报 (Test Set)")
    print("="*40)
    print(f"📊 总K线数: {len(X_test)}")
    print(f"🔥 开火次数: {int(trade_count)} 次 (频率大幅降低)")
    
    if trade_count > 0:
        win_rate = win_count / trade_count
        avg_ret = real_returns_4h[final_signals == 1].mean()
        print(f"🎯 命中率 (4小时后上涨): {win_rate:.2%}")
        print(f"💰 平均单笔回报: {avg_ret*100:.2f}%")
        
        # 累计回报 (简单累加模拟)
        total_return = real_returns_4h[final_signals == 1].sum()
        print(f"📈 累计净回报: {total_return*100:.2f}% (未计复利)")
    else:
        print("❄️ 没有扣动扳机 (没有符合高标准的机会)")
        
    print("\n📜 狙击手准则 (Tree Rules):")
    print(export_text(clf, feature_names=features))

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        df = feature_engineering_sniper(df)
        run_sniper_backtest(df)