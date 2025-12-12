import clickhouse_connect
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
SYMBOL = 'BTCUSDT'

def load_data_from_features_table():
    """
    直接从 features_15m 表加载数据，速度飞快。
    """
    print("🚀 正在从 features_15m 表加载数据...")
    client = clickhouse_connect.get_client(**CLICKHOUSE)
    
    # 我们需要按照时间排序，以便计算 Next Return
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
    print(f"📊 成功加载 {len(df)} 条 15m K线数据")
    return df

def prepare_training_data(df):
    """
    特征工程：计算标签 (Label) 并清洗异常数据
    """
    # 1. 计算目标变量 (Target): 下一个 15m 的收益率
    # 使用 shift(-1) 获取下一行的价格
    # 注意：如果 close_price 为 0，这里会产生 inf，所以后面要清洗
    df['next_return'] = (df['close_price'].shift(-1) - df['close_price']) / df['close_price'] * 100
    # [关键步骤] 计算 Z-Score (归一化)
    # 逻辑：(当前值 - 平均值) / 标准差
    # 结果通常落在 -3 到 +3 之间
    rolling_mean = df['net_cvd'].rolling(window=20, min_periods=1).mean()
    rolling_std = df['net_cvd'].rolling(window=20, min_periods=1).std()
    rolling_std = rolling_std.replace(0, 1) # 防止除以0
    
    # 生成新列：cvd_zscore
    df['cvd_zscore'] = (df['net_cvd'] - rolling_mean) / rolling_std
    # 2. [关键修复] 清洗数据 (清洗 NaN 和 Infinity)
    # 先把正负无穷大替换成 NaN，然后一次性丢弃所有包含 NaN 的行
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 3. [关键修复] 解决 SettingWithCopyWarning
    # dropna 返回的是一个视图，我们创建它的深拷贝，切断与原 df 的联系
    df = df.copy()
    
    # 4. 打标签 (Labeling)
    THRESHOLD = 0.2 
    df['label'] = 0
    df.loc[df['next_return'] > THRESHOLD, 'label'] = 1 
    
    # [可选] 再次检查特征列，确保没有遗漏的极大值
    # 有时候 spoofing_ratio 会变得极大但不是 inf，我们把它截断
    # df['spoofing_ratio'] = df['spoofing_ratio'].clip(upper=1000) 
    
    # [新增] 训练前剔除"暴跌"样本，防止 AI 学会接飞刀
    df = df[df['wall_shift_pct'] > -0.2]
    
    print(f"🧹 数据清洗完成，剩余有效样本: {len(df)} 条")
    return df

def train_and_optimize(df):
    # [修复 1] 确保标签存在
    y = df['label'].astype(int)
    
    # [修复 2] 特征矩阵必须用 Z-Score！
    # ❌ 错误: X = df[['wall_shift_pct', 'net_cvd', 'spoofing_ratio']]
    # ✅ 正确:
    X = df[['wall_shift_pct', 'cvd_zscore', 'spoofing_ratio']]
    
    print("🧠 正在训练决策树...")
    # 保持时间顺序划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # 权重调整：稍微降低做多门槛
    weights = {0: 1, 1: 2.0} 
    clf = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42, class_weight=weights)
    clf.fit(X_train, y_train)
    
    # 评估
    y_pred = clf.predict(X_test)
    print("\n🎓 模型评估报告 (Test Set):")
    print(classification_report(y_test, y_pred))
    
    # --- 回测修正 ---
    # [修复 3] 回报率计算必须除以 100
    # df['next_return'] 是百分数 (如 0.5 代表 0.5%)
    # 我们需要把它变成小数 (0.005) 才能计算复利
    real_returns = df.loc[X_test.index, 'next_return'] / 100 
    
    # 市场基准
    cum_market = (real_returns + 1).cumprod()
    
    # 策略收益 (加入人工熔断：墙塌了不买)
    # 模拟：如果 wall_shift_pct < -0.2，强制不买
    signals = y_pred.copy()
    mask_falling_knife = X_test['wall_shift_pct'] < -0.2
    signals[mask_falling_knife] = 0
    
    strategy_returns = real_returns * signals
    cum_strategy = (strategy_returns + 1).cumprod()
    
    print(f"💰 市场买入持有回报: {cum_market.iloc[-1]:.4f}x")
    print(f"🤖 AI 策略回报:     {cum_strategy.iloc[-1]:.4f}x")
    
    # 解析规则
    print("\n📜 散户最佳执行规则 (Human Readable Rules):")
    print(export_text(clf, feature_names=list(X.columns)))
    
    return clf

if __name__ == "__main__":
    try:
        # 1. 加载
        df_raw = load_data_from_features_table()
        
        if df_raw.empty:
            print("⚠️ 表 marketdata.features_15m 为空，请先运行 backfill 脚本或等待采集。")
        else:
            # 2. 预处理
            df_ready = prepare_training_data(df_raw)
            
            # 3. 训练
            train_and_optimize(df_ready)
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")