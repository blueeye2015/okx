import pandas as pd
import numpy as np
import os
import glob
from sqlalchemy import create_engine
import datetime

from dotenv import load_dotenv
load_dotenv('.env')
DSN = os.getenv('DB_DSN1')
# --- 配置部分 ---
FACTOR_DIR = "./factor_cache_global"  # 你的因子文件夹路径


# 🔥 核心配置：持仓数量
# 如果全市场约 5000 只股票，0.03 (3%) 大约是 150 只
# 如果你觉得还不够，可以调大这个比例，或者直接指定 TOP_N = 100
TOP_N_PCT = 0.03  

# 🔥 目标日期：对应文件名 factor_YYYY-MM.parquet
TARGET_DATE = '2025-12' 

# 🔥🔥🔥 新增：模拟实盘的初始本金 (例如 100 万)
INITIAL_CAPITAL = 1000000.0 

def load_target_factor(factor_dir, target_month):
    file_name = f"factor_{target_month}.parquet"
    file_path = os.path.join(factor_dir, file_name)
    
    if os.path.exists(file_path):
        print(f"✅ 找到指定因子文件: {file_path}")
        return pd.read_parquet(file_path)
    else:
        print(f"⚠️ 未找到 {file_name}，尝试查找该目录下最新的 parquet 文件...")
        files = sorted(glob.glob(os.path.join(factor_dir, "factor_*.parquet")))
        if not files:
            raise FileNotFoundError(f"❌ 目录下没有找到任何 factor_*.parquet 文件")
        latest_file = files[-1]
        print(f"👉 加载最新的文件代替: {latest_file}")
        return pd.read_parquet(latest_file)

def get_basic_info():
    engine = create_engine(DSN)
    df_basic = pd.read_sql("SELECT symbol, list_date, name FROM stock_basic", engine)
    df_basic['list_date'] = pd.to_datetime(df_basic['list_date'])
    return df_basic

def generate_buy_list():
    print(f"1. 加载 {TARGET_DATE} 的因子数据...")
    df_factor = load_target_factor(FACTOR_DIR, TARGET_DATE)
    
    if isinstance(df_factor.index, pd.MultiIndex):
        df_factor = df_factor.reset_index()
        
    latest_date = df_factor['trade_date'].max()
    print(f"   数据内最新交易日: {latest_date}")
    
    df_current = df_factor[df_factor['trade_date'] == latest_date].copy()
    
    print("2. 加载基础信息...")
    df_basic = get_basic_info()
    df_merge = pd.merge(df_current, df_basic, on='symbol', how='left')
    
    valid_stocks = []
    current_time = pd.Timestamp.now()
    
    for _, row in df_merge.iterrows():
        symbol = row['symbol']
        name = row['name'] if row['name'] else "Unknown"
        list_date = row['list_date']
        factor_val = row['factor']
        close_price = row['close']
        
        # 1. 新股过滤 (保留)
        if pd.isna(list_date) or (current_time - list_date).days < 60:
            continue
        
        # 2. ST 过滤 (已注释，保留ST)
        # if 'ST' in name: continue
            
        # 3. 数据完整性
        if pd.isna(factor_val) or pd.isna(close_price) or close_price <= 0:
            continue
            
        valid_stocks.append({
            'symbol': symbol,
            'name': name,
            'cost_price': close_price, # 🔥 重命名为 cost_price，作为模拟买入价
            'factor': factor_val
        })
    
    # --- 排序与截断 ---
    df_valid = pd.DataFrame(valid_stocks)
    df_valid = df_valid.sort_values(by='factor', ascending=False) 
    
    top_n = int(len(df_valid) * TOP_N_PCT)
    if top_n < 10: top_n = min(10, len(df_valid))
    
    df_buy = df_valid.head(top_n).copy()
    
    # --- 🔥🔥🔥 核心修改：计算持仓股数 (Volume) ---
    
    # 1. 计算单只股票分配资金 (等权)
    df_buy['target_weight'] = 1.0 / len(df_buy)
    target_amt_per_stock = INITIAL_CAPITAL * df_buy['target_weight']
    
    # 2. 计算股数 = 金额 / 股价 (向下取整到 100 股)
    # A股买入必须是 100 的整数倍
    df_buy['volume'] = (target_amt_per_stock / df_buy['cost_price']) // 100 * 100
    
    # 3. 过滤掉钱太少买不起 100 股的情况
    df_buy = df_buy[df_buy['volume'] > 0].copy()
    
    # 4. 记录日期
    df_buy['buy_date'] = datetime.datetime.now().strftime('%Y-%m-%d')

    # --- 输出结果 ---
    # 强制输出格式，确保与 track_portfolio.py 兼容
    output_cols = ['symbol', 'name', 'cost_price', 'volume', 'buy_date', 'factor', 'target_weight']
    
    # 保存为 my_holdings.csv (直接覆盖，方便下一步直接跑)
    output_file = "my_holdings.csv"
    df_buy[output_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"✅ 模拟持仓文件已生成: {output_file}")
    print(f"   模拟本金: {INITIAL_CAPITAL:,.0f}")
    print(f"   持仓股票: {len(df_buy)} 只")
    print(f"   实际占用资金: {(df_buy['volume'] * df_buy['cost_price']).sum():,.2f}")
    print("="*50)
    print(df_buy[['symbol', 'name', 'cost_price', 'volume']].head(5))

if __name__ == '__main__':
    generate_buy_list()