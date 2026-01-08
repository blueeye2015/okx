import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import os

# --- 配置部分 ---
# 修改为你的数据库连接
from dotenv import load_dotenv
load_dotenv('.env')
DSN = os.getenv('DB_DSN1')
HOLDINGS_FILE = "my_holdings.csv"       # 你的持仓记录文件
LOG_FILE = "paper_trading_log.csv"      # 净值历史记录

def get_latest_prices_from_db(symbol_list):
    """
    从本地数据库获取最新的收盘价
    使用 PostgreSQL 的 DISTINCT ON 语法，确保取到每个股票在库里的最后一条记录
    """
    engine = create_engine(DSN)
    
    if not symbol_list:
        return {}, None
        
    # 1. 格式化 symbol 列表用于 SQL IN 查询
    # 假设你的 stock_history 表里的 symbol 也是 '000001' 这种格式
    symbols_str = "'" + "','".join(symbol_list) + "'"
    
    # 2. 构造 SQL 查询
    # 逻辑：找出 stock_history 中这些股票最新的 trade_date 和 close
    # DISTINCT ON (symbol) ... ORDER BY symbol, trade_date DESC 
    # 这是 PG 数据库特有的高效去重语法，取最新一条
    sql = f"""
    SELECT DISTINCT ON (symbol)
        symbol, 
        close, 
        trade_date
    FROM stock_history
    WHERE symbol IN ({symbols_str})
    ORDER BY symbol, trade_date DESC
    """
    
    try:
        df = pd.read_sql(sql, engine)
        
        # 打印一下数据的日期，确认是不是最新的
        if not df.empty:
            max_date = df['trade_date'].max()
            min_date = df['trade_date'].min()
            print(f"✅ 数据库取价成功: 数据区间 {min_date} ~ {max_date}")
        
        # 转为字典: {'000001': 10.5, ...}
        price_map = df.set_index('symbol')['close'].to_dict()
        return price_map
        
    except Exception as e:
        print(f"❌ 数据库查询失败: {e}")
        return {}

def track_performance():
    print(f"📊 正在计算模拟盘收益 (本地数据库版)...")
    
    # 1. 读取持仓
    if not os.path.exists(HOLDINGS_FILE):
        print(f"❌ 找不到持仓文件 {HOLDINGS_FILE}，请先建立！")
        return
        
    df_hold = pd.read_csv(HOLDINGS_FILE)
    # 确保 symbol 是字符串类型 (防止 000001 被读成 1)
    df_hold['symbol'] = df_hold['symbol'].astype(str).str.zfill(6)

    # 2. 从数据库获取现价
    symbols = df_hold['symbol'].tolist()
    price_map = get_latest_prices_from_db(symbols) 

    # 3. 计算收益
    total_cost = 0
    total_value = 0
    
    print("\n{:<10} {:<8} {:<10} {:<10} {:<10} {:<10}".format(
        "代码", "名称", "成本价", "最新价", "持仓市值", "盈亏率"
    ))
    print("-" * 70)
    
    for index, row in df_hold.iterrows():
        sym = row['symbol']
        cost = row['cost_price']
        vol = row['volume']
        
        # 从数据库字典里取价
        curr_price = price_map.get(sym)
        
        # 如果数据库里没有（比如刚上市或者停牌太久或者代码不对）
        if curr_price is None:
            curr_price = cost 
            print(f"⚠️ 警告: 数据库中未找到 {sym} 的价格，暂按成本价计算")
            
        mkt_value = curr_price * vol
        cost_value = cost * vol
        pnl_pct = (curr_price - cost) / cost
        
        total_cost += cost_value
        total_value += mkt_value
        
        print("{:<10} {:<8} {:<10.2f} {:<10.2f} {:<10.0f} {:>.2%}".format(
            sym, row['name'], cost, curr_price, mkt_value, pnl_pct
        ))

    # 4. 汇总统计
    total_pnl = total_value - total_cost
    total_ret = total_pnl / total_cost if total_cost > 0 else 0
    
    print("-" * 70)
    print(f"💰 总投入: {total_cost:,.2f}")
    print(f"💎 总市值: {total_value:,.2f}")
    print(f"📈 总盈亏: {total_pnl:,.2f} ({total_ret:.2%})")
    
    # 5. 记录净值历史
    # 记录日期用 '今天'，因为你是今天在跑这个脚本看结果
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    log_df = pd.DataFrame([{
        'date': today_str,
        'total_value': total_value,
        'total_return': total_ret
    }])
    
    if not os.path.exists(LOG_FILE):
        log_df.to_csv(LOG_FILE, index=False)
    else:
        log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)
        print(f"✅ 净值已追加到 {LOG_FILE}")

if __name__ == '__main__':
    track_performance()