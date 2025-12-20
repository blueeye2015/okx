import os
import pandas as pd
from tqdm import tqdm

# ========= 1. 只改这里 =========
root_dir = r'factor_cache_per_stock'   # 你的顶层目录
# ================================

parquet_files = [os.path.join(dp, f)
                 for dp, _, files in os.walk(root_dir)
                 for f in files if f.lower().endswith('.parquet')]
if not parquet_files:
    raise SystemExit('未找到任何 .parquet 文件！')

print(f'共发现 {len(parquet_files)} 个 parquet 文件，开始读入内存…')

df_list = [pd.read_parquet(fp) for fp in tqdm(parquet_files, desc='读取')]
merged = pd.concat(df_list, ignore_index=True)

# ---------- 核心统计 ----------
num_symbols = merged['symbol'].nunique()
factor_0_or_neg1 = merged['factor'].isin([0, -1]).sum()
min_rows_symbol, min_rows_count = merged['symbol'].value_counts().agg(['idxmin', 'min'])
# 新增：最小、最大日期
min_date = merged['trade_date'].min()
max_date = merged['trade_date'].max()


# ---------- 结果 ----------
print('\n========== 内存分析结果 ==========')
print(f'symbol 总数：{num_symbols}')
print(f'factor = 0 或 -1 的记录数：{factor_0_or_neg1}')
print(f'记录行数最少的 symbol：{min_rows_symbol} （{min_rows_count} 行）')
print(f'日期范围：{min_date}  ～  {max_date}')
print('==================================')