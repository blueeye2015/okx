import pandas as pd
import logging
import sys
import os

# 1. 设置路径，确保能找到主程序文件
sys.path.append('/data/okx')

# ========================================================
# 2. 核心操作：直接从主程序导入函数！
# ========================================================
try:
    # 假设你的主程序文件名是 auto_trader_v7_fixed.py
    # 如果文件名不同，请修改下面的 "auto_trader_v7_fixed"
    from auto_trader import (
        execute_trade,      # <--- 我们要测的核心函数
        get_current_price, 
        get_position, 
        SYMBOL,
        init_exchange       # 初始化杠杆等
    )
    print("✅ 成功导入主程序模块！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请检查：")
    print("1. 主程序文件名是否为 auto_trader_v7_fixed.py？")
    print("2. 两个文件是否在同一目录下？")
    exit()

# 配置简单的日志显示
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [TEST] - %(message)s')

def run_live_test():
    print("="*50)
    print("🧬 源代码级暴力测试 (Direct Source Code Test)")
    print("="*50)
    
    # 1. 初始化 (确保杠杆等设置正确)
    init_exchange()
    
    # 2. 获取当前状态
    price = get_current_price(SYMBOL)
    pos = get_position(SYMBOL)
    
    if not price:
        print("❌ 无法获取价格，测试中止。")
        return

    print(f"💰 当前价格: {price}")
    print(f"🎒 当前持仓: {pos}")

    # 3. 构造完美的 Pandas Series 信号
    # 这完全模拟了 get_signal 读取 CSV 后的结果
    now_str = pd.Timestamp.now()
    
    # === 场景判断 ===
    if not pos:
        print("\n[测试场景]: 当前空仓 -> 强行注入【做多】信号")
        
        # 构造信号行 (模拟 CSV 的一行)
        fake_signal = pd.Series({
            'Time': now_str,
            'Signal': 1,            # 1 = 开多
            'TP_Price': price * 1.04, # 模拟 CSV 里的 TP
            'SL_Price': price * 0.99  # 模拟 CSV 里的 SL
        })
        
        print(f"📝 注入数据: Signal=1 (Long)")
        print("⚡ 正在调用源代码 execute_trade()...")
        
        # --- 关键时刻：调用源代码 ---
        execute_trade(fake_signal, price)
        
    elif pos['side'] == 'LONG':
        print("\n[测试场景]: 当前持多 -> 强行注入【逃顶】信号")
        
        fake_signal = pd.Series({
            'Time': now_str,
            'Signal': -1,           # -1 = 平仓
            'TP_Price': 0,
            'SL_Price': 0
        })
        
        print(f"📝 注入数据: Signal=-1 (Escape)")
        print("⚡ 正在调用源代码 execute_trade()...")
        
        # --- 关键时刻：调用源代码 ---
        execute_trade(fake_signal, price)
        
    else:
        print("⚠️ 检测到空单，本测试脚本暂不处理平空逻辑。")

    print("\n✅ 测试结束。")
    print("👉 如果看到 '🚀' 或 '🏃‍♂️' 的日志，说明函数执行成功。")
    print("👉 请检查 bot_state.json 是否更新，以及 APP 是否成交。")

if __name__ == "__main__":
    run_live_test()