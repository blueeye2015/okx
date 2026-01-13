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

def get_next_open_batch(current_date_str, symbol_list):
    """
    🔥 极速版：只查 T+1 日的 Open，不需要算 LAG
    """
    if not symbol_list: return None
    
    engine = create_engine(DSN)
    symbols_str = "'" + "','".join(symbol_list) + "'"
    
    # SQL 只需要查 T+1 的开盘价
    sql = f"""
    SELECT DISTINCT ON (symbol)
        symbol, 
        open as next_open, 
        trade_date as next_date
    FROM stock_history
    WHERE trade_date > '{current_date_str}' 
      AND symbol IN ({symbols_str})
    ORDER BY symbol, trade_date ASC
    """
    
    try:
        df = pd.read_sql(sql, engine)
        return df
    except Exception as e:
        print(f"❌ SQL 查询失败: {e}")
        return pd.DataFrame()
    
def generate_buy_list():
    print(f"1. 加载 {TARGET_DATE} 的因子数据...")
    df_factor = load_target_factor(FACTOR_DIR, TARGET_DATE)
    if isinstance(df_factor.index, pd.MultiIndex): df_factor = df_factor.reset_index()
    
    # 因子产生的日期 (T日)
    factor_date = df_factor['trade_date'].max()
    # 转换为字符串 yyyy-mm-dd
    factor_date_str = pd.to_datetime(factor_date).strftime('%Y-%m-%d')
    print(f"   因子基准日: {factor_date_str}")
    
    df_current = df_factor[df_factor['trade_date'] == factor_date].copy()
    
    print("2. 加载基础信息...")
    df_basic = get_basic_info()
    df_merge = pd.merge(df_current, df_basic, on='symbol', how='left')
    
    # --- 初步筛选 ---
    candidates = []
    current_time = pd.Timestamp.now()
    
    for _, row in df_merge.iterrows():
        symbol = row['symbol']
        # 🔥 关键点：直接拿因子文件里的 close 作为“昨收价”
        # 因为因子文件是 T 日盘后生成的，这个 close 就是 T 日收盘价
        close_T = row['close']

        name = row['name'] if row['name'] else "Unknown"
        list_date = row['list_date']
        factor_val = row['factor']
        
        # 基础过滤
        if pd.isna(list_date) or (current_time - list_date).days < 60: continue
        if pd.isna(factor_val): continue
            
        candidates.append({
            'symbol': symbol,
            'name': name,
            'factor': factor_val,
            'pre_close': close_T # 🔥 这里直接存下来，这就是 T 日收盘价
        })
    
    df_candidates = pd.DataFrame(candidates)
    
    # --- 🔥🔥🔥 核心修改：T+1 日可买性检查 🔥🔥🔥 ---
    print("3. 获取 T+1 日开盘价并校验涨停...")
    symbol_list = df_candidates['symbol'].tolist()
    df_next = get_next_open_batch(factor_date_str, symbol_list)
    
    if df_next.empty:
        print("❌ 警告：未获取到 T+1 行情，无法剔除涨停")
        df_final = df_candidates.copy()
        df_final['cost_price'] = df_final['pre_close']
        buy_date = "UNKNOWN"
    else:
        # 🔥 Python 内存合并：Candidates (含 pre_close) + Next (含 open)
        df_final = pd.merge(df_candidates, df_next, on='symbol', how='inner')
        
        valid_buy_list = []
        blocked_count = 0
        
        if not df_final.empty:
            buy_date = df_final['next_date'].mode()[0]
        else:
            buy_date = "UNKNOWN"
            
        print(f"   锁定买入日期 (T+1日): {buy_date}")
        
        for _, row in df_final.iterrows():
            sym = row['symbol']
            name = row['name'] if row['name'] else "Unknown"
            
            # 🔥 核心计算：用 SQL 查出来的 Open 和 Python 存着的 Pre_Close 对比
            close_T = row['pre_close'] 
            open_T1 = row['next_open']
            
            # 涨幅计算
            pct_chg = (open_T1 - close_T) / close_T if close_T > 0 else 0
            
            # 涨停逻辑
            limit_ratio = 0.10
            if 'ST' in name: limit_ratio = 0.05
            elif sym.startswith(('688', '300')): limit_ratio = 0.20
            elif sym.startswith(('8', '4')): limit_ratio = 0.30
            
            # 剔除一字板
            if pct_chg > (limit_ratio - 0.005):
                blocked_count += 1
                continue
                
            valid_buy_list.append({
                'symbol': sym,
                'name': name,
                'factor': row['factor'],
                'cost_price': open_T1, # 真实的买入价
                'buy_date': row['next_date'],
                'target_weight': 0 # 占位
            })
            
        print(f"   🚫 因开盘涨停(买不进) 剔除: {blocked_count} 只")
        df_final = pd.DataFrame(valid_buy_list)

    # --- 4. 资金分配 ---
    if df_final.empty:
        print("❌ 筛选后无股票")
        return

    df_final = df_final.sort_values(by='factor', ascending=False)
    top_n = int(len(df_final) * TOP_N_PCT)
    if top_n < 10: top_n = min(10, len(df_final))
    
    df_buy = df_final.head(top_n).copy()
    
    # --- 计算资金 ---
    df_buy['target_weight'] = 1.0 / len(df_buy)
    target_amt_per_stock = INITIAL_CAPITAL * df_buy['target_weight']
    df_buy['volume'] = (target_amt_per_stock / df_buy['cost_price']) // 100 * 100
    df_buy = df_buy[df_buy['volume'] > 0].copy()

    # --- 输出 ---
    output_cols = ['symbol', 'name', 'cost_price', 'volume', 'buy_date', 'factor', 'target_weight']
    output_file = "my_holdings.csv"
    df_buy[output_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"✅ 模拟持仓文件已生成: {output_file}")
    print(f"   交易日期: {buy_date}")
    print(f"   持仓股票: {len(df_buy)} 只")
    print("="*50)
    print(df_buy[['symbol', 'name', 'cost_price', 'volume', 'buy_date']].head(5))

if __name__ == '__main__':
    generate_buy_list()