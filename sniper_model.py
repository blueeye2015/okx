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
    狙击手特征工程
    """
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 1. 趋势
    df['ma_long'] = df['close_price'].rolling(window=16).mean()
    df['trend_bullish'] = (df['close_price'] > df['ma_long']).astype(int)
    
    # 2. 资金
    df['cvd_1h_sum'] = df['net_cvd'].rolling(window=4).sum()
    cvd_mean = df['cvd_1h_sum'].rolling(window=96).mean()
    cvd_std = df['cvd_1h_sum'].rolling(window=96).std().replace(0, 1)
    df['cvd_zscore_long'] = (df['cvd_1h_sum'] - cvd_mean) / cvd_std
    
    # 3. 盘口
    df['wall_shift_1h_max'] = df['wall_shift_pct'].rolling(window=4).max()
    
    # 4. 目标 (未来 4小时回报)
    df['future_return_4h'] = (df['close_price'].shift(-16) - df['close_price']) / df['close_price'] * 100
    
    df = df.dropna()
    
    # 5. 打标签
    df['label'] = 0
    df.loc[df['future_return_4h'] > 0.8, 'label'] = 1
    
    print(f"🧹 数据重构完成: 剩余 {len(df)} 条 | 正样本: {sum(df['label'])}")
    return df

def run_sniper_backtest(df):
    features = ['wall_shift_1h_max', 'cvd_zscore_long', 'trend_bullish']
    X = df[features]
    y = df['label'].astype(int)
    
    # 划分 (X_test 将作为我们的模拟回测段)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    print("🧠 狙击手正在校准瞄准镜 (Training)...")
    clf = DecisionTreeClassifier(
        max_depth=3,
        criterion='entropy', 
        random_state=42, 
        class_weight={0: 1, 1: 3.0}, 
        min_samples_leaf=30
    )
    clf.fit(X_train, y_train)
    
    # --- 预测 ---
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # --- [核心修改] 详细回测记录 ---
    trade_records = [] # 用于存储每一笔交易的详情
    
    print("\n⚡ 开始逐行扫描开火点...")
    
    for i in range(len(X_test)):
        idx = X_test.index[i]
        
        # 提取特征
        trend_ok = X_test.loc[idx, 'trend_bullish'] == 1
        cvd_strong = X_test.loc[idx, 'cvd_zscore_long'] > 0.0
        model_say_buy = y_pred[i] == 1
        confidence = y_prob[i]
        
        # 🔫 开火逻辑
        if model_say_buy and trend_ok and cvd_strong:
            # 记录这一单的详细信息
            entry_time = df.loc[idx, 'time']
            entry_price = df.loc[idx, 'close_price']
            actual_ret_pct = df.loc[idx, 'future_return_4h'] # 这是百分比，如 1.2 代表 1.2%

            # [修改 2] 模拟带有硬止损的持仓
            # 假设止损是 -1.0%，止盈是 +2.0%
            # 这里我们用 simplified 逻辑：
            # 如果 future_return_4h < -1.0，就按 -1.0 算 (模拟盘中被打止损)
            # 如果 future_return_4h > 2.0，就按 2.0 算
            pnl_pct = actual_ret_pct
            
            if pnl_pct < -1.0: 
                pnl_pct = -1.1 # 给点滑点惩罚
                result_str = "❌ 止损离场"
            elif pnl_pct > 2.0:
                pnl_pct = 2.0
                result_str = "✅ 止盈离场"
            else:
                result_str = "⚖️ 时间到平仓"
            
            # 简单模拟：假设每单投入 1000 U
            position_size = 1000 
            pnl = position_size * (actual_ret_pct / 100)
            
            trade_records.append({
                '开火时间': entry_time,
                '入场价格': round(entry_price, 2),
                '模型置信度': round(confidence, 2),
                '持仓结果(4h)': f"{actual_ret_pct:.2f}%",
                '模拟盈亏(1000U)': round(pnl, 2),
                '胜负': '✅ 止盈' if actual_ret_pct > 0 else '❌ 止损'
            })
            
    # --- 结果展示 ---
    print("\n" + "="*60)
    print(f"🔫 狙击手详细战报")
    print("="*60)
    
    if len(trade_records) > 0:
        # 转为 DataFrame 方便展示
        df_trades = pd.DataFrame(trade_records)
        
        # 计算累计盈亏
        df_trades['账户累计盈亏'] = df_trades['模拟盈亏(1000U)'].cumsum()
        
        # 1. 打印详细流水 (设置显示选项以防省略)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.unicode.east_asian_width', True) # 对齐中文
        
        print(df_trades[['开火时间', '入场价格', '持仓结果(4h)', '模拟盈亏(1000U)', '胜负', '账户累计盈亏']])
        
        # 2. 统计摘要
        total_pnl = df_trades['模拟盈亏(1000U)'].sum()
        win_rate = len(df_trades[df_trades['模拟盈亏(1000U)'] > 0]) / len(df_trades)
        
        print("-" * 60)
        print(f"🔥 总开火次数: {len(df_trades)}")
        print(f"🎯 胜率: {win_rate:.2%}")
        print(f"💰 总盈亏 (每单1000U): {total_pnl:.2f} U")
        print(f"📈 盈亏比估算: {df_trades[df_trades['模拟盈亏(1000U)']>0]['模拟盈亏(1000U)'].mean() / abs(df_trades[df_trades['模拟盈亏(1000U)']<0]['模拟盈亏(1000U)'].mean()):.2f}")
        
    else:
        print("❄️ 本次测试区间内未触发任何开火信号。")

    print("\n📜 决策树规则:")
    print(export_text(clf, feature_names=features))

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        df = feature_engineering_sniper(df)
        run_sniper_backtest(df)