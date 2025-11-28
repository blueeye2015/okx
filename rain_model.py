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
    
    print(f"🧹 数据清洗完成，剩余有效样本: {len(df)} 条")
    return df

def train_and_optimize(df):
    # 只需要预测 "做多机会" (y=1)
    y = (df['label'] == 1).astype(int)
    
    # 特征矩阵
    X = df[['wall_shift_pct', 'net_cvd', 'spoofing_ratio']]
    
    # --- 2. 训练模型 ---
    print("🧠 正在训练决策树...")
    # 这里的 shuffle=False 很重要，因为是时间序列数据，测试集应该是"未来"的数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    if len(X_train) < 10:
        print("⚠️ 训练数据过少，无法进行有效训练。请等待更多数据积累。")
        return

    # 限制深度为 3，防止过拟合，并保证规则可读性
    clf = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # --- 3. 评估 ---
    y_pred = clf.predict(X_test)
    print("\n🎓 模型评估报告 (Test Set):")
    print(classification_report(y_test, y_pred))
    
    # --- 4. 回测 ---
    # 计算策略收益
    test_returns = df.loc[X_test.index, 'next_return'] / 100
    
    # 市场基准：买入持有 (Buy & Hold)
    cum_market = (test_returns + 1).cumprod()
    
    # 策略收益：只在模型预测为 1 时持有
    # 注意：这里假设信号出现立刻买入，持有15分钟
    strategy_returns = test_returns * y_pred 
    cum_strategy = (strategy_returns + 1).cumprod()
    
    market_final = cum_market.iloc[-1] if not cum_market.empty else 1.0
    strategy_final = cum_strategy.iloc[-1] if not cum_strategy.empty else 1.0

    print(f"💰 市场买入持有回报: {market_final:.4f}x")
    print(f"🤖 AI 策略回报:     {strategy_final:.4f}x")
    
    # --- 5. 解析规则 ---
    print("\n📜 散户最佳执行规则 (Human Readable Rules):")
    rules = export_text(clf, feature_names=list(X.columns))
    print(rules)

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