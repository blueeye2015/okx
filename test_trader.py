# test_trader.py

import logging
import uuid
from trade_executor import TradingClient # 从我们之前的交易执行器文件中导入

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_interactive_test():
    """
    运行一个交互式的交易测试程序。
    """
    try:
        # 1. 初始化交易客户端
        # 确保 trade_executor.py 中的API Key等信息已配置
        trader = TradingClient()
        if not trader.trade_api or not trader.market_api:
            logging.error("交易客户端初始化失败，测试终止。")
            return
        print("\n交易客户端已成功初始化。\n")

        # 2. 获取用户输入
        symbol = input("请输入交易对 (例如: BTC-USDT-SWAP): ").strip().upper()
        side = input("请输入交易方向 (buy 或 sell): ").strip().lower()
        ord_type = input("请输入限价还是市价 (limit 或 market): ").strip().lower()
        
        amount_usdt_str = input("请输入希望交易的金额 (USDT, 例如: 10): ").strip()
        amount_usdt = float(amount_usdt_str)

        price_str = input(f"请输入 {symbol.split('-')[0]} 的大概当前市价 (用于计算下单数量): ").strip()
        price = float(price_str)
        
        if not all([symbol, side, ord_type, amount_usdt > 0, price > 0]):
            print("\n输入信息不完整或无效，测试终止。")
            return

        # 3. 计算下单数量
        size = amount_usdt / price
        
        # 4. 生成唯一的客户端订单ID
        client_order_id = f"test{uuid.uuid4().hex[:10]}"

        # 5. 显示确认信息
        print("\n--- 准备执行下单 ---")
        print(f"交易对: {symbol}")
        print(f"方向:   {side.upper()}")
        print(f"订单类型:   {ord_type.upper()}")
        print(f"金额:   {amount_usdt:.2f} USDT")
        print(f"价格:   {price:.4f} USDT")
        print(f"计算数量: {size:.8f} {symbol.split('-')[0]}")
        print(f"客户端ID: {client_order_id}")
        
        confirm = input("确认以上信息并执行下单吗? (y/n): ").strip().lower()

        if confirm != 'y':
            print("\n操作已取消。")
            return
            
        print("\n--- 步骤1: 正在提交订单... ---")
        
        # 6. 根据订单类型调用不同的函数
        px = trader.get_latest_price(symbol)
        size = amount_usdt/px
        order_result = None
        if ord_type == 'market':
            order_result = trader.place_market_order_by_amount(symbol, side, size, client_order_id)
        elif ord_type == 'limit':
            order_result = trader.place_limit_order(symbol, side, size, px, client_order_id)
        else:
            print(f"无效的订单类型: {ord_type}")
            return
            
        # 7. 检查结果
        if order_result:
             # 通常成功的 'code' 是 '0'
            if order_result.get('code') == '0':
                print("\n--- 测试成功: 订单已提交 ---")
                print(order_result)
            else:
                print("\n--- 测试失败: 订单提交失败 ---")
                print("API返回错误信息:")
                print(order_result)
        else:
            print("\n测试终止：下单请求失败，请检查日志输出获取详细错误信息。")

    except Exception as e:
        logging.error(f"测试过程中发生未知异常: {e}")

if __name__ == "__main__":
    run_interactive_test()