import ccxt
import time
import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('/data/okx/.env')

# ==========================================
# ⚠️⚠️⚠️ 测试配置区域 ⚠️⚠️⚠️
# ==========================================
USE_TESTNET = False # ⚠️ 如果是 True，请确保你的 API Key 是测试网的！如果是 False，将使用真金白银交易！

SYMBOL = 'BTC/USDT'
LEVERAGE = 5
TEST_USDT_AMOUNT = 6.0 # 币安最小下单金额通常在 5-10 USDT 左右，设 6 块比较稳妥 (杠杆后价值)
# 注意：币安合约最小下单名义价值通常是 5 USDT。
# 比如 100倍杠杆，你也得至少开 5 USDT 价值的仓位 (本金只需 0.05)。
# 这里为了稳妥，我们让 (本金 x 杠杆) > 5 USDT。

# API 配置
API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', 'YOUR_SECRET_KEY')

def check_api_keys():
    """检查 API Key 是否配置正确"""
    print("\n[0/5] 检查 API 配置...")
    
    if API_KEY == 'YOUR_API_KEY' or SECRET_KEY == 'YOUR_SECRET_KEY':
        print("❌ 错误: 未检测到有效的 API Key！")
        print("   原因: 环境变量 BINANCE_API_KEY 或 BINANCE_SECRET_KEY 未设置，程序正在使用默认占位符。")
        print("   解决: 请编辑 .env 文件或在终端 export 你的 Key。")
        return False
        
    if not API_KEY or not SECRET_KEY:
        print("❌ 错误: API Key 或 Secret 为空！")
        return False

    # 打印部分 Key 用于核对
    masked_key = f"{API_KEY[:4]}...{API_KEY[-4:]}" if len(API_KEY) > 8 else "****"
    print(f"✅ API Key 已加载: {masked_key} (长度: {len(API_KEY)})")
    print(f"✅ Secret 已加载: (长度: {len(SECRET_KEY)})")
    return True

def main():
    print("="*50)
    print("🦈 币安 API 下单功能测试程序")
    
    # 先检查 Key
    if not check_api_keys():
        return

    print(f"模式: {'🧪 测试网 (Testnet)' if USE_TESTNET else '💰 实盘 (Live)'}")
    print(f"交易对: {SYMBOL}")
    print(f"测试金额: {TEST_USDT_AMOUNT} USDT (名义价值)")
    print("="*50)
    
    if not USE_TESTNET:
        print("\n⚠️ 警告: 你正在使用实盘模式！将会产生真实交易费用和盈亏！")
        confirm = input("确认继续吗？(输入 yes 继续): ")
        if confirm.lower() != 'yes':
            print("测试已取消。")
            return

    # 1. 初始化
    try:
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
            'proxies': {
                'http': 'http://127.0.0.1:7890',
                'https': 'http://127.0.0.1:7890',
            },
            'options': {'defaultType': 'future'}
        })
        if USE_TESTNET:
            exchange.set_sandbox_mode(True)
            
        print("\n[1/5] 连接交易所...")
        exchange.load_markets()
        print("✅ 连接成功！")
        
        # 2. 检查余额
        print("\n[2/5] 检查 USDT 余额...")
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        print(f"✅ 可用余额: {usdt_balance} USDT")
        
        if usdt_balance < 1.0: # 稍微放宽一点限制，只要能付手续费就行，主要看保证金
            print("❌ 余额不足，无法测试。")
            return

        # 3. 设置杠杆和模式
        print("\n[3/5] 设置账户参数...")
        exchange.set_leverage(LEVERAGE, SYMBOL)
        try:
            exchange.set_position_mode(False, SYMBOL)
        except:
            pass
        print(f"✅ 杠杆 {LEVERAGE}x | 单向持仓模式")

        # 4. 执行开多测试
        print(f"\n[4/5] 测试开多单 (名义价值 {TEST_USDT_AMOUNT} USDT)...")
        
        # 计算数量
        price = exchange.fetch_ticker(SYMBOL)['last']
        amount = (TEST_USDT_AMOUNT * LEVERAGE) / price
        # 精度调整
        amount = float(exchange.amount_to_precision(SYMBOL, amount))
        
        print(f"   当前价格: {price}")
        print(f"   计划买入: {amount} BTC")
        
        # 下单
        order = exchange.create_market_buy_order(SYMBOL, amount)
        print(f"✅ 开单成功! 订单ID: {order['id']}")
        
        # 验证持仓
        time.sleep(2)
        positions = exchange.fetch_positions([SYMBOL])
        my_pos = next((p for p in positions if p['symbol'] == SYMBOL and float(p['contracts']) > 0), None)
        
        if my_pos:
            print(f"   当前持仓: {my_pos['side']} {my_pos['contracts']} BTC")
            print(f"   未实现盈亏: {my_pos['unrealizedPnl']}")
        else:
            print("❌ 未检测到持仓！可能下单失败或立刻成交了？")
            return

        # 5. 执行平仓测试
        print(f"\n[5/5] 测试平仓 (清理战场)...")
        time.sleep(2) # 稍微等一下让系统反应
        
        close_order = exchange.create_market_sell_order(SYMBOL, amount, params={'reduceOnly': True})
        print(f"✅ 平仓成功! 订单ID: {close_order['id']}")
        
        print("\n🎉🎉🎉 所有测试通过！你的 auto_trader.py 逻辑应该是没问题的。")

    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
