import pandas as pd

try:
    # --- 请确保 trade_log.csv 和本脚本在同一个文件夹 ---
    trade_log_path = 'trade_log.csv'
    df_trades = pd.read_csv(trade_log_path)

    # --- 开始分析 ---
    if df_trades.empty:
        print("交易日志为空，无法进行分析。")
    else:
        total_trades = len(df_trades)
        # 只计算盈利交易的总额，作为我们的基数
        winning_trades_df = df_trades[df_trades['pnl_net'] > 0]
        total_positive_pnl = winning_trades_df['pnl_net'].sum()
        
        # 按净利润降序排序
        df_trades_sorted = df_trades.sort_values(by='pnl_net', ascending=False)

        # 计算不同分位的交易数量
        # 这里我们按盈利交易的总数来计算百分比，更具说服力
        num_winning_trades = len(winning_trades_df)
        top_1_pct_count = max(1, int(num_winning_trades * 0.01))
        top_5_pct_count = max(1, int(num_winning_trades * 0.05))
        top_10_pct_count = max(1, int(num_winning_trades * 0.10))

        # 计算不同分位的利润贡献
        profit_top_1_pct = df_trades_sorted.head(top_1_pct_count)['pnl_net'].sum()
        profit_top_5_pct = df_trades_sorted.head(top_5_pct_count)['pnl_net'].sum()
        profit_top_10_pct = df_trades_sorted.head(top_10_pct_count)['pnl_net'].sum()

        # 计算贡献度百分比
        contribution_top_1_pct = (profit_top_1_pct / total_positive_pnl) * 100
        contribution_top_5_pct = (profit_top_5_pct / total_positive_pnl) * 100
        contribution_top_10_pct = (profit_top_10_pct / total_positive_pnl) * 100
        
        # 计算总净利润
        total_net_profit = df_trades['pnl_net'].sum()

        print("--- 交易利润集中度分析报告 ---")
        print(f"总交易笔数: {total_trades} | 盈利笔数: {num_winning_trades}")
        print(f"总净利润: {total_net_profit:,.2f} | 总盈利金额: {total_positive_pnl:,.2f}")
        print("-" * 40)
        print("【核心指标】盈利最高的交易对总盈利金额的贡献度：")
        print(f"盈利最高的 1% (约{top_1_pct_count}笔) 交易贡献了总盈利的: {contribution_top_1_pct:.2f}%")
        print(f"盈利最高的 5% (约{top_5_pct_count}笔) 交易贡献了总盈利的: {contribution_top_5_pct:.2f}%")
        print(f"盈利最高的 10% (约{top_10_pct_count}笔) 交易贡献了总盈利的: {contribution_top_10_pct:.2f}%")
        print("-" * 40)


except FileNotFoundError:
    print(f"错误：找不到文件 '{trade_log_path}'。请确保它与脚本在同一目录下。")
except Exception as e:
    print(f"分析时发生错误: {e}")