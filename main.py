import logging
from time import sleep
from datetime import datetime
from config.settings import Config
from database.manager import DatabaseManager
from services.market_data import MarketDataService
import asyncio



async def main():
    logging.basicConfig(filename='klines.log', 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    config = Config()
    db_manager = DatabaseManager(config.DB_CONFIG)
    market_service = MarketDataService(config)
  
    
    try:
        if datetime.now().hour == 0 and datetime.now().minute == 0:
            config.update_symbols()
        
        #market_service.update_market_data()
        await market_service.run()
        ##await market_service.update_trade_data()          
        
    except KeyboardInterrupt:
        logging.info("程序正在退出...")
        
    except Exception as e:
        logging.error(f"发生错误: {str(e)}")
    finally:
        await db_manager.close()
            

if __name__ == "__main__":
    asyncio.run(main())
