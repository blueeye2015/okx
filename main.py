import time
import logging
from config import Config
from exchange_api import BinanceClient
from strategy_runner import StrategyRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s', # 简化日志，具体内容由各模块加前缀
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

def main():
    logger.info("🤖 Multi-Strategy Bot Starting...")
    
    # 1. 初始化唯一的 API 客户端 (复用 TCP 连接)
    client = BinanceClient()
    
    # 2. 根据配置，批量创建策略实例
    runners = []
    for strat_cfg in Config.STRATEGIES:
        runner = StrategyRunner(strat_cfg, client)
        runners.append(runner)
        logger.info(f"✅ 策略已加载: {strat_cfg['name']} ({strat_cfg['symbol']})")

    # 3. 主循环
    while True:
        try:
            # 这一步是为了获取最新价格，可以优化为批量获取
            # 但为了简单，我们让每个策略自己去拿，或者在这里统一拿
            
            for runner in runners:
                # 获取该策略对应币种的价格
                price = client.get_market_price(runner.symbol)
                
                if price:
                    # 触发该策略的一次心跳
                    runner.tick(price)
                
                # 稍微歇一下，防止循环太快（如果有10个策略，这里不用 sleep 太久）
                time.sleep(1) 
            
            # 一轮跑完，休息几秒
            time.sleep(300)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            time.sleep(300)

if __name__ == "__main__":
    main()