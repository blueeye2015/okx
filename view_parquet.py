import pandas as pd
import os

# --- 1. 配置 ---
# 请将这里的文件名替换成您想要转换的具体文件名
parquet_file_path = 'factor_cache/daily_factor_chunk_3.parquet'

# 自动生成输出的CSV文件名
# 例如 'factor_cache_v2/daily_factor_chunk_0.parquet' -> 'daily_factor_chunk_0.csv'
base_name = os.path.basename(parquet_file_path)
csv_file_name = os.path.splitext(base_name)[0] + '.csv'

try:
    # --- 2. 读取 Parquet 文件 ---
    print(f"正在读取 Parquet 文件: {parquet_file_path}...")
    df = pd.read_parquet(parquet_file_path)
    print(f"读取成功，共 {len(df)} 行数据。")

    # --- 3. 保存为 CSV 文件 ---
    print(f"正在保存为 CSV 文件: {csv_file_name}...")
    # index=False: 不将DataFrame的行索引（0, 1, 2...）写入CSV文件，更整洁
    # encoding='utf-8-sig': 确保在Excel中打开时，中文不会显示为乱码
    df.to_csv(csv_file_name, index=False, encoding='utf-8-sig')

    print("\n--- 转换成功！ ---")
    print(f"文件 '{csv_file_name}' 已保存在当前目录下。")
    print("您现在可以用Excel或其他软件打开它进行分析。")

except FileNotFoundError:
    print(f"错误：找不到文件 '{parquet_file_path}'。")
    print("请检查文件名和路径是否正确。")
except Exception as e:
    print(f"转换过程中发生错误: {e}")