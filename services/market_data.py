import logging
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
from models.trade import Trade
from dotenv import load_dotenv
import os
import ssl
import aiohttp
import platform

# Windows系统特殊处理
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
        
        self.kline_semaphore = asyncio.Semaphore(20)  # 限制并发请求数
        self.trade_semaphore = asyncio.Semaphore(20) 
        self._initialized_symbols = set()  # 只需要记录是否是首次执行
        self.proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
        } 
        #  API 
        self.api_key = os.getenv('API_KEY')
        self.secret_key = os.getenv('SECRET_KEY')
        self.passphrase = os.getenv('PASSPHRASE')

        # WebSocket配置
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self.ws_running = False
        
    
        
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
        async with self.kline_semaphore:  # 使用信号量控制并发
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
    

    async def handle_ws_message(self, trade_data: Dict) -> None:
            """处理WebSocket接收到的交易数据"""
            try:
                trade = Trade(
                    event_type=trade_data['e'],
                    event_time=datetime.fromtimestamp(trade_data['E'] / 1000),
                    symbol=trade_data['s'],
                    trade_id=trade_data['t'],
                    price=float(trade_data['p']),
                    quantity=float(trade_data['q']),
                    buyer_order_maker=trade_data['m'],
                    trade_time=datetime.fromtimestamp(trade_data['T'] / 1000)
                )
                await self.trade_dao.insert(trade)
                logging.info(f"保存交易数据: {trade.symbol} - {trade.trade_id}")
            except Exception as e:
                logging.error(f"处理交易数据时出错: {e}")

    async def start_ws_stream(self) -> None:
        """启动WebSocket数据流"""
        self.ws_running = True
        
        # 币安的订阅消息格式
        # 将交易对转换为小写并移除'-'
        formatted_symbols = ['btcusdt']
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol}@trade" for symbol in formatted_symbols],
            "id": 1
        }

        logging.debug(f"订阅消息: {subscribe_message}")  # 添加调试日志

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        while self.ws_running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        self.ws_url,
                        ssl=ssl_context,
                        proxy=self.proxies['http'] if self.proxies else None
                    ) as ws:
                        # 发送订阅请求
                        await ws.send_str(json.dumps(subscribe_message))
                        logging.info("WebSocket已连接并发送订阅请求")

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                logging.debug(f"收到原始消息: {data}")  # 添加调试日志
                                
                                # 处理订阅确认消息
                                if 'result' in data:
                                    logging.info(f"订阅确认: {data}")
                                    continue
                                
                                # 处理交易数据
                                await self.handle_ws_message(data)
                                    
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                logging.error(f"WebSocket连接关闭或错误: {msg.type}")
                                break
                                
            except Exception as e:
                logging.error(f"WebSocket连接错误: {e}")
                if self.ws_running:
                    await asyncio.sleep(5)  # 重连延迟

    async def run(self) -> None:
        """运行市场数据服务"""
        try:
            # 创建定时更新K线数据的任务
            kline_task = asyncio.create_task(self._run_kline_updates())
            # 创建WebSocket数据流任务
            #ws_task = asyncio.create_task(self.start_ws_stream())
            
            # 等待两个任务
            await asyncio.gather(kline_task)
        except Exception as e:
            logging.error(f"运行市场数据服务时出错: {e}")
        finally:
            self.close()

    async def _run_kline_updates(self) -> None:
        """定时运行K线数据更新"""
        while True:
            try:
                await self.update_market_data()
                await asyncio.sleep(60)
            except Exception as e:
                logging.error(f"K线数据更新出错: {e}")
                await asyncio.sleep(5)

    def close(self) -> None:
        """
        关闭服务，清理资源
        """
        try:
            self.db_manager.close()
            logging.info("市场数据服务已关闭")
        except Exception as e:
            logging.error(f"关闭市场数据服务时出错: {e}")
