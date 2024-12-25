import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import asyncio
import json
import okex.Market_api as Market 
from exchange.base import ExchangeBase
from config.settings import Config
from database.dao  import KlineDAO,TradeDAO
from database.manager import DatabaseManager
from models.kline import Kline
from models.trade import trade
from dotenv import load_dotenv
import os

class MarketDataService(ExchangeBase):
    """
    市场数据服务
    负责从交易所获取K线数据并存储到数据库
    """
    
    def __init__(self, config: Config):
        """
        初始化市场数据服务
        
        Args:
            config (Config): 配置对象，包含数据库配置和其他设置
        """
        super().__init__() 
        # 加载环境变量
        load_dotenv()

        self.config = config
        self.db_manager = DatabaseManager(config.DB_CONFIG)
        self.kline_dao = KlineDAO(self.db_manager)
        self.trade_dao = TradeDAO(self.db_manager)
        self._init_database()
        self.semaphore = asyncio.Semaphore(20)  # 限制并发请求数
        self._initialized_symbols = set()  # 只需要记录是否是首次执行
        self.proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
        } #
        self.api_key = os.getenv('API_KEY')
        self.secret_key = os.getenv('SECRET_KEY')
        self.passphrase = os.getenv('PASSPHRASE')
        
    def _init_database(self) -> None:
        """初始化数据库表"""
        try:
            self.kline_dao.create_table()
            self.trade_dao.create_table()
            logging.info("数据库表初始化成功")
        except Exception as e:
            logging.error(f"数据库表初始化失败: {e}")
            raise
        
    async def fetch_klines(self, symbol: str) -> List[Kline]:
        """
        从交易所获取K线数据
        
        Args:
            symbol (str): 交易对符号，例如 "BTC-USDT"
            start_time (datetime): 开始时间
            
        Returns:
            List[Kline]: K线数据列表
            
        Raises:
            Exception: 当获取数据失败时抛出异常
        """
        async with self.semaphore:  # 使用信号量控制并发
            try:
                # 将时间转换为毫秒时间戳
                # since = int(start_time.timestamp() * 1000)
                
                # 将交易对格式转换为交易所要求的格式（BTC-USDT -> BTC/USDT）
                exchange_symbol = symbol.replace('-', '/')
                
                # 根据是否首次执行决定获取数量
                limit = 300 if symbol not in self._initialized_symbols else 10
                
                # 获取K线数据
                ohlcv = self.exchange.fetch_ohlcv(
                    exchange_symbol,
                    timeframe=self.config.INTERVAL,
                    limit=limit
                )
                
                # 转换为 Kline 对象列表
                return [
                    Kline(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(data[0] / 1000),
                        open=float(data[1]),
                        high=float(data[2]),
                        low=float(data[3]),
                        close=float(data[4]),
                        volume=float(data[5])
                    )
                    for data in ohlcv
                ]
                    
            except Exception as e:
                logging.error(f"获取 {symbol} K线数据失败: {e}")
                raise
        
    async def update_single_symbol(self, symbol: str) -> None:
        """
        更新单个交易对的市场数据
        
        Args:
            symbol (str): 交易对符号
        """
        try:
            # 获取最新的K线数据
            # latest_kline = await self.kline_dao.get_latest_kline(symbol)
            
            # 如果没有历史数据，则从1000分钟前开始获取
            # start_time = datetime.now() - timedelta(minutes=1000)
            
            # if latest_kline:
            #     start_time = latest_kline.timestamp + timedelta(minutes=1)
            
            # 获取新数据
            new_klines = await self.fetch_klines(symbol)
            
            # 批量保存
            if new_klines:
                await self.kline_dao.save_klines(new_klines)
                logging.info(f"更新了 {symbol} 的 {len(new_klines)} 条K线数据")
            
            # 标记该交易对已初始化
            self._initialized_symbols.add(symbol)
                
        except Exception as e:
            logging.error(f"更新 {symbol} 时出错: {str(e)}")
            raise
    
    async def update_market_data(self) -> None:
        """
        更新所有交易对的市场数据
        """
        tasks = []
        for symbol in self.config.SYMBOLS:            
            task = asyncio.create_task(self.update_single_symbol(symbol))
            tasks.append(task)               
            # 添加适当的延迟以遵守API限制
            #time.sleep(1)  # 根据实际需要调整延迟时间
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # 处理结果和异常
            for symbol, result in zip(self.config.SYMBOLS, results):
                if isinstance(result, Exception):
                    logging.error(f"更新 {symbol} 失败: {str(result)}")
    
    async def fetch_trade(self, symbol: str) -> List[trade]:
        """
        从交易所获取成交数据
        
        Args:
            symbol (str): 交易对符号，例如 "BTC-USDT"
                        
        Returns:
            List[trade]: K线数据列表
            
        Raises:
            Exception: 当获取数据失败时抛出异常
        """
        async with self.semaphore:  # 使用信号量控制并发
            try:
                # 将时间转换为毫秒时间戳
                # since = int(start_time.timestamp() * 1000)
                
                # 将交易对格式转换为交易所要求的格式（BTC-USDT -> BTC/USDT）
                
                
                # 根据是否首次执行决定获取数量
                limit = 500 
                
                # 获取trade数据
                marketAPI = Market.MarketAPI(
                    self.api_key, 
                    self.secret_key, 
                    self.passphrase, 
                    False, 
                    '0', 
                    proxies=self.proxies)
                result = marketAPI.get_trades(symbol, limit)
                ohlcv = json.loads(json.dumps(result))
                
                # 转换为 Kline 对象列表
                return [
                    trade(
                        symbol=symbol,
                        side=data['side'],
                        sz=float(data['sz']),
                        px=float(data['px']),
                        tradeId=data['tradeId'],                        
                        timestamp=datetime.fromtimestamp(float(data['ts']) / 1000)                        
                    )
                    for data in ohlcv['data']
                ]
                    
            except Exception as e:
                logging.error(f"获取 {symbol} K线数据失败: {e}")
                raise
                            
    async def update_single_tradedata(self, symbol: str) -> None:
        """
        更新公共成交数据
        
        Args:
            symbol (str): 交易对符号
        """
        try:
            # 获取最新的K线数据
            # latest_kline = await self.kline_dao.get_latest_kline(symbol)
            
            # 如果没有历史数据，则从1000分钟前开始获取
            # start_time = datetime.now() - timedelta(minutes=1000)
            
            # if latest_kline:
            #     start_time = latest_kline.timestamp + timedelta(minutes=1)
            
            # 获取新数据
            new_trade = await self.fetch_trade(symbol)
            
            # 批量保存
            if new_trade:
                await self.trade_dao.save_trade(new_trade)
                logging.info(f"更新了 {symbol} 的 {len(new_trade)} 条K线数据")
            
            # 标记该交易对已初始化
            self._initialized_symbols.add(symbol)
                
        except Exception as e:
            logging.error(f"更新 {symbol} 时出错: {str(e)}")
            raise    
                    
    async def update_trade_data(self) -> None:
        """
        更新所有交易对的市场数据
        """
        tasks = []
        for symbol in self.config.SYMBOLS:            
            task = asyncio.create_task(self.update_single_tradedata(symbol))
            tasks.append(task)               
            # 添加适当的延迟以遵守API限制
            #time.sleep(1)  # 根据实际需要调整延迟时间
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # 处理结果和异常
            for symbol, result in zip(self.config.SYMBOLS, results):
                if isinstance(result, Exception):
                    logging.error(f"更新 {symbol} 失败: {str(result)}")    
            

    
    
    def close(self) -> None:
        """
        关闭服务，清理资源
        """
        try:
            self.db_manager.close()
            logging.info("市场数据服务已关闭")
        except Exception as e:
            logging.error(f"关闭市场数据服务时出错: {e}")
