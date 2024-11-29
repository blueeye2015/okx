import logging
import time
from datetime import datetime, timedelta
from typing import List

from exchange import ExchangeBase
from config import Config
from database import DatabaseManager, KlineDAO
from models import Kline

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
        self.config = config
        self.db_manager = DatabaseManager(config.DB_CONFIG)
        self.kline_dao = KlineDAO(self.db_manager)
        self._init_database()
        
    def _init_database(self) -> None:
        """初始化数据库表"""
        try:
            self.kline_dao.create_table()
            logging.info("数据库表初始化成功")
        except Exception as e:
            logging.error(f"数据库表初始化失败: {e}")
            raise
        
    def fetch_klines(self, symbol: str, start_time: datetime) -> List[Kline]:
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
        try:
            # 将时间转换为毫秒时间戳
            since = int(start_time.timestamp() * 1000)
            
            # 将交易对格式转换为交易所要求的格式（BTC-USDT -> BTC/USDT）
            exchange_symbol = symbol.replace('-', '/')
            
            # 获取K线数据
            ohlcv = self.exchange.fetch_ohlcv(
                exchange_symbol,
                timeframe=self.config.INTERVAL,
                since=since,
                limit=self.config.BATCH_SIZE
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
        
    def update_single_symbol(self, symbol: str) -> None:
        """
        更新单个交易对的市场数据
        
        Args:
            symbol (str): 交易对符号
        """
        try:
            # 获取最新的K线数据
            latest_kline = self.kline_dao.get_latest_kline(symbol)
            
            # 如果没有历史数据，则从1000分钟前开始获取
            start_time = datetime.now() - timedelta(minutes=1000)
            
            if latest_kline:
                start_time = latest_kline.timestamp + timedelta(minutes=1)
            
            # 获取新数据
            new_klines = self.fetch_klines(symbol, start_time)
            
            # 批量保存
            if new_klines:
                self.kline_dao.save_klines(new_klines)
                logging.info(f"更新了 {symbol} 的 {len(new_klines)} 条K线数据")
                
        except Exception as e:
            logging.error(f"更新 {symbol} 时出错: {str(e)}")
            raise
    
    def update_market_data(self) -> None:
        """
        更新所有交易对的市场数据
        """
        for symbol in self.config.SYMBOLS:
            try:
                self.update_single_symbol(symbol)
                # 添加适当的延迟以遵守API限制
                time.sleep(1)  # 根据实际需要调整延迟时间
                    
            except Exception as e:
                logging.error(f"更新 {symbol} 时出错: {str(e)}")
                continue
    
    def close(self) -> None:
        """
        关闭服务，清理资源
        """
        try:
            self.db_manager.close()
            logging.info("市场数据服务已关闭")
        except Exception as e:
            logging.error(f"关闭市场数据服务时出错: {e}")
