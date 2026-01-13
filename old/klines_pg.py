import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import ccxt
import requests
## 首先添加必要的导入
import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession


# 1. 添加交易所基类
class ExchangeBase:
    _instance = None
    _exchange = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExchangeBase, cls).__new__(cls)
        return cls._instance
    
    @property
    def exchange(self) -> ccxt.Exchange:
        
        proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据您的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据您的实际代理地址修改
            }
        if self._exchange is None:
            self._exchange = ccxt.okx({
            'apiKey': 'ba7f444f-e83e-4dd1-8507-bf8dd9033cbc',
            'secret': 'D5474EF76B0A7397BFD26B9656006480',
            'password': 'TgTB+pJoM!d20F',
            'enableRateLimit': True,
            'proxies': proxies,  # 添加代理设置
            'timeout': 30000,    # 设置超时时间为30秒
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
            })
        # try:
        #     self.exchange.load_markets()
        #     logging.info("交易所连接成功")
        # except Exception as e:
        #     logging.info(f"交易所连接失败: {str(e)}")
        #     raise e
        return self._exchange
    
    def close(self):
        if self._exchange:
            self._exchange = None

# 1. 配置管理
@dataclass
class DBConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    
class Config:
    # PostgreSQL配置
    DB_CONFIG = DBConfig(
        host="localhost",
        port=5432,
        database="market_data",
        user="postgres",
        password="12"
    )
    INTERVAL = "1m"
    BATCH_SIZE = 1000

    def __init__(self):
            self.market_analyzer = MarketAnalyzer()
            self.update_symbols()
        
    def update_symbols(self):
            self.SYMBOLS = self.market_analyzer.get_valid_symbols()
            logging.info(f"更新交易对列表，共 {len(self.SYMBOLS)} 个")

# 定义基类
class Base(DeclarativeBase):
    pass

# 3. 定义ORM模型
class KlineModel(Base):
    __tablename__ = 'klines'
    
    symbol = sa.Column(sa.String, primary_key=True)
    timestamp = sa.Column(sa.DateTime, primary_key=True)
    open = sa.Column(sa.Float)
    high = sa.Column(sa.Float)
    low = sa.Column(sa.Float)
    close = sa.Column(sa.Float)
    volume = sa.Column(sa.Float)
    created_at = sa.Column(sa.DateTime, server_default=sa.func.now())

# 4. 修改数据库管理器
class DatabaseManager:
    _instance = None
    _engine = None
    _session_factory = None
    
    def __new__(cls, config: DBConfig):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.config = config
            cls._instance._init_db()
        return cls._instance
    
    def _init_db(self):
        """初始化数据库连接"""
        url = f"postgresql://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
        self._engine = sa.create_engine(url, pool_size=5, max_overflow=10)
        self._session_factory = sessionmaker(bind=self._engine)
        
        # 创建表
        Base.metadata.create_all(self._engine)
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        return self._session_factory()
    
    def close(self):
        """关闭数据库连接"""
        if self._engine:
            self._engine.dispose()
            self._engine = None

@dataclass
class Kline:
    """K线数据类"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_exchange_data(cls, symbol: str, data: list) -> 'Kline':
        """
        从交易所数据创建Kline对象
        data格式: [timestamp, open, high, low, close, volume]
        """
        return cls(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(data[0] / 1000),  # 转换毫秒时间戳
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5])
        )
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }
        
# 5. 数据访问层基类
class BaseDAO(ABC):
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    @abstractmethod
    def create_table(self):
        pass
    
    @abstractmethod
    def insert(self, data):
        pass
    
    @abstractmethod
    def query(self, **kwargs):
        pass
                
# 6. 修改KlineDAO
class KlineDAO(BaseDAO):
    def create_table(self):
        """表已经通过SQLAlchemy自动创建"""
        pass
    
    def insert(self, kline: Kline):
        """插入单条数据"""
        session = self.db_manager.get_session()
        try:
            kline_model = KlineModel(
                symbol=kline.symbol,
                timestamp=kline.timestamp,
                open=kline.open,
                high=kline.high,
                low=kline.low,
                close=kline.close,
                volume=kline.volume
            )
            
            stmt = insert(KlineModel).values(
                vars(kline_model)
            ).on_conflict_do_update(
                index_elements=['symbol', 'timestamp'],
                set_={
                    'open': kline_model.open,
                    'high': kline_model.high,
                    'low': kline_model.low,
                    'close': kline_model.close,
                    'volume': kline_model.volume
                }
            )
            
            session.execute(stmt)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def save_klines(self, kline_models: List[KlineModel]):
        session = self.db_manager.get_session()
        try:
            # 创建 insert 语句
            stmt = insert(KlineModel)
            # 准备插入的数据
            values = [vars(model) for model in kline_models]
            
            # 创建 on conflict do update 语句
            stmt = stmt.values(values).on_conflict_do_update(
                index_elements=['symbol', 'timestamp'],
                set_={
                    'open': stmt.excluded.open,
                    'high': stmt.excluded.high,
                    'low': stmt.excluded.low,
                    'close': stmt.excluded.close,
                    'volume': stmt.excluded.volume
                }
            )
            
            session.execute(stmt)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    

    def get_latest_kline(self, symbol: str) -> Optional[Kline]:
        """获取指定交易对的最新K线数据（同步方式）"""
        session = self.db_manager.get_session()
        try:
            query = session.query(KlineModel)
            query = query.filter(KlineModel.symbol == symbol)
            query = query.order_by(KlineModel.timestamp.desc()).first()
            
            if query:
                return Kline(
                    symbol=query.symbol,
                    timestamp=query.timestamp,
                    open=query.open,
                    high=query.high,
                    low=query.low,
                    close=query.close,
                    volume=query.volume
                )
            return None
        finally:
            session.close()

    def query(self, symbol: str = None, 
              start_time: datetime = None, 
              end_time: datetime = None) -> List[Kline]:
        """查询数据"""
        session = self.db_manager.get_session()
        try:
            query = session.query(KlineModel)
            
            if symbol:
                query = query.filter(KlineModel.symbol == symbol)
            if start_time:
                query = query.filter(KlineModel.timestamp >= start_time)
            if end_time:
                query = query.filter(KlineModel.timestamp <= end_time)
                
            query = query.order_by(KlineModel.timestamp)
            
            return [
                Kline(
                    symbol=row.symbol,
                    timestamp=row.timestamp,
                    open=row.open,
                    high=row.high,
                    low=row.low,
                    close=row.close,
                    volume=row.volume
                ) for row in query.all()
            ]
        finally:
            session.close()

# 6. 添加索引创建函数
# def create_indexes(engine):
#     """创建必要的索引"""
#     with engine.connect() as conn:
#         # 创建symbol和timestamp的联合索引
#         conn.execute("""
#             CREATE INDEX IF NOT EXISTS idx_klines_symbol_timestamp 
#             ON klines (symbol, timestamp)
#         """)
        
#         # 创建timestamp索引用于范围查询
#         conn.execute("""
#             CREATE INDEX IF NOT EXISTS idx_klines_timestamp 
#             ON klines (timestamp)
#         """)



# 6. 业务逻辑层
class MarketDataService(ExchangeBase):
    def __init__(self, config: Config):
        super().__init__() 
        self.config = config
        self.db_manager = DatabaseManager(config.DB_CONFIG)
        self.kline_dao = KlineDAO(self.db_manager)
        self.kline_dao.create_table()
        
    def fetch_klines(self, symbol: str, start_time: datetime) -> List[Kline]:
        """
        从交易所获取K线数据
        """           
            # 将时间转换为毫秒时间戳
        since = int(start_time.timestamp() * 1000)
            
            # 获取K线数据
        ohlcv = self.exchange.fetch_ohlcv(
            symbol,  # 将 BTC-USDT 转换为 BTC/USDT
            timeframe='1m',
            since=since,
            limit=1000
        )
            
        # 转换为 Kline 对象列表
        klines = []
        for data in ohlcv:
            timestamp = datetime.fromtimestamp(data[0] / 1000)  # 转换毫秒时间戳为datetime
            kline = Kline(
                symbol=symbol,
                timestamp=timestamp,
                open=float(data[1]),
                high=float(data[2]),
                low=float(data[3]),
                close=float(data[4]),
                volume=float(data[5])
            )
            klines.append(kline)
                
        return klines        
    
    def update_market_data(self):
        """更新所有交易对的市场数据"""
        for symbol in self.config.SYMBOLS:
            try:
                # 获取最新的K线数据
                latest_kline = self.kline_dao.get_latest_kline(symbol)
                start_time = datetime.now() - timedelta(minutes=1000)
                
                if latest_kline:
                    start_time = latest_kline.timestamp + timedelta(minutes=1)
                
                # 获取新数据
                new_klines = self.fetch_klines(symbol, start_time)
                
                # 批量保存
                if new_klines:
                    self.kline_dao.save_klines(new_klines)
                    logging.info(f"更新了 {symbol} 的 {len(new_klines)} 条K线数据")
                    
                # 添加适当的延迟以遵守API限制
                time.sleep(1)  # 根据实际需要调整延迟时间
                    
            except Exception as e:
                logging.error(f"更新 {symbol} 时出错: {str(e)}")

class MarketAnalyzer(ExchangeBase):
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 3600  # 缓存1小时   
        self.proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据你的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据你的实际代理地址修改
        } 
                    
    def get_market_cap_data(self) -> Dict[str, float]:
        """
        从 CoinGecko 获取市值数据
        返回格式: {'BTC': 800000000000, 'ETH': 200000000000, ...}
        """
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 250,  # 获取前250个币种
                'page': 1,
                'sparkline': False
            }
            response = requests.get(url,proxies=self.proxies, params=params)
            data = response.json()
            
            return {
                item['symbol'].upper(): {
                    'market_cap': item['market_cap'],
                    'first_listed': item.get('genesis_date')  # 上市日期
                }
                for item in data if item['market_cap'] is not None
            }
        except Exception as e:
            logging.error(f"获取市值数据失败: {str(e)}")
            return {}

    def get_valid_symbols(self, min_market_cap: float = 20000000, min_age_months: int = 1) -> List[str]:
        """
        获取符合条件的交易对
        :param min_market_cap: 最小市值（美元）
        :param min_age_months: 最小上市月数
        :return: 符合条件的交易对列表
        """
        try:
            # 获取交易所支持的所有交易对
            markets = self.exchange.load_markets()
            
            # 获取市值数据
            market_cap_data = self.get_market_cap_data()
            
            # 当前时间
            current_time = datetime.now()
            min_list_date = current_time - timedelta(days=30 * min_age_months)
            
            valid_symbols = []
            
            for symbol, market in markets.items():
                try:
                    # 只考虑USDT交易对
                    if not symbol.endswith('/USDT'):
                        continue
                        
                    base_currency = market['base']  # 基础货币 (例如 BTC, ETH)
                    
                    # 检查是否有市值数据
                    if base_currency not in market_cap_data:
                        continue
                        
                    market_info = market_cap_data[base_currency]
                    
                    # 检查市值
                    if market_info['market_cap'] < min_market_cap:
                        continue
                        
                    # 检查上市时间
                    if market_info['first_listed']:
                        list_date = datetime.strptime(market_info['first_listed'], '%Y-%m-%d')
                        if list_date > min_list_date:
                            continue
                            
                    # 将交易所格式转换为我们的格式 (BTC/USDT -> BTC-USDT)
                    formatted_symbol = symbol.replace('/', '-')
                    valid_symbols.append(formatted_symbol)
                    
                except Exception as e:
                    logging.warning(f"处理交易对 {symbol} 时出错: {str(e)}")
                    continue
            
            logging.info(f"找到 {len(valid_symbols)} 个符合条件的交易对")
            return valid_symbols
            
        except Exception as e:
            logging.error(f"获取有效交易对时出错: {str(e)}")
            return []
        

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    config = Config()
    db_manager = DatabaseManager(config.DB_CONFIG)
    # create_indexes(db_manager._engine)
    
    market_service = MarketDataService(config)
    
    while True:
        try:
            if datetime.now().hour == 0 and datetime.now().minute == 0:
                config.update_symbols()
            
            market_service.update_market_data()
            time.sleep(60)
            
        except KeyboardInterrupt:
            logging.info("程序正在退出...")
            db_manager.close()
            break
        except Exception as e:
            logging.error(f"发生错误: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    main()

