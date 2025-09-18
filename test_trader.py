# test_trader.py

import logging
import time
import uuid
# 从我们之前编写的文件中导入TradingClient类
from trade_executor import TradingClient 

# --- 日志基础配置 ---
# 让日志信息更清晰地显示在终端
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    """
    运行一个交互式的交易功能测试。
    """
    print("--- OKX交易API功能测试脚本 ---")
    print("重要提示: 请务必先在 trade_executor.py 文件中填入您的API密钥！")
    print("强烈建议首先使用模拟盘 (IS_DEMO = 1) 进行测试。\n")

    # 1. 初始化交易客户端
    # ==================================
    trader = TradingClient()
    if not trader.trade_api:
        print("\n测试终止：交易客户端初始化失败，请检查API密钥或网络连接。")
        return

    print("交易客户端已成功初始化。\n")

    # 2. 从用户获取交易参数
    # ==================================
    try:
        symbol = input("请输入交易对 (例如: BTC-USDT-SWAP): ").strip().upper()
        side = input("请输入交易方向 (buy 或 sell): ").strip().lower()
        ordType = input("请输入限价还是市价 (limit 或 market): ").strip().lower()
        usdt_amount_str = input("请输入希望交易的金额 (USDT, 例如: 10): ").strip()
        current_price_str = input(f"请输入 {symbol.split('-')[0]} 的大概当前市价 (用于计算下单数量): ").strip()

        # 参数校验
        if side not in ['buy', 'sell']:
            print("\n错误：交易方向必须是 'buy' 或 'sell'。")
            return
            
        usdt_amount = float(usdt_amount_str)
        current_price = float(current_price_str)
        
        if usdt_amount <= 0 or current_price <= 0:
            print("\n错误：金额和价格必须是正数。")
            return

    except ValueError:
        print("\n错误：输入的金额或价格无效，请输入数字。")
        return
    except Exception as e:
        print(f"\n发生未知错误: {e}")
        return

    # 3. 计算下单数量并生成唯一ID
    # ==================================
    size = usdt_amount / current_price
    client_order_id = f"test{uuid.uuid4().hex[:16]}" # 生成一个唯一的客户端订单ID

    print("\n--- 准备执行下单 ---")
    print(f"交易对: {symbol}")
    print(f"方向:   {side.upper()}")
    print(f"订单类型:   {ordType.upper()}")
    print(f"金额:   {usdt_amount} USDT")
    print(f"价格:   {current_price} USDT")
    print(f"计算数量: {size:.8f} {symbol.split('-')[0]}")
    print(f"客户端ID: {client_order_id}")
    
    confirm = input("确认以上信息并执行下单吗? (y/n): ").strip().lower()
    if confirm != 'y':
        print("操作已取消。")
        return

    # 4. 执行下单
    # ==================================
    print("\n--- 步骤1: 正在提交市价单... ---")
    order_result = trader.place_market_order(symbol, current_price, side, ordType, size, client_order_id)

    if not order_result:
        print("\n测试终止：下单请求失败，请检查日志输出获取详细错误信息。")
        return
        
    print("下单请求已成功发送至交易所。")
    time.sleep(5) # 等待几秒钟，给交易所处理订单的时间

    # 5. 查询订单状态
    # ==================================
    print(f"\n--- 步骤2: 正在查询订单 (ID: {client_order_id}) 的成交状态... ---")
    status_result = trader.check_order_status(symbol, client_order_id)

    if not status_result:
        print("\n测试结束：无法获取订单状态。订单可能未被创建或查询失败。")
        return

    print("\n--- 测试结果 ---")
    print("订单状态查询成功！详细信息如下:")
    for key, value in status_result.items():
        print(f"  {key}: {value}")

    # 对关键信息进行解读
    state = status_result.get('state')
    avg_price = status_result.get('avgPx')
    filled_size = status_result.get('accFillSz')

    print("\n--- 结果解读 ---")
    if state == 'filled':
        print(f"✅ 成功: 订单已完全成交！")
        print(f"   成交均价: {avg_price}")
        print(f"   成交数量: {filled_size}")
    elif state == 'partially_filled':
        print(f"⚠️  注意: 订单部分成交。")
        print(f"   成交均价: {avg_price}")
        print(f"   成交数量: {filled_size}")
    else:
        print(f"❌ 失败/待处理: 订单当前状态为 '{state}'，可能未成交或仍在处理中。")

if __name__ == '__main__':
    run_test()