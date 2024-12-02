from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from config.settings import DBConfig
from database.models import Base

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
        url = f"postgresql+asyncpg://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
        self._engine = create_async_engine(
            url, 
            pool_size=100,               # 增加连接池大小
            pool_pre_ping=True,         # 自动检查连接是否有效
            pool_timeout=30,            # 连接超时时间
            pool_recycle=1800          # 连接回收时间
            )
        self.async_session = sessionmaker(            
            self._engine, 
            class_=AsyncSession, 
            expire_on_commit=False)
        
    
    def get_session(self):
        return self.async_session()
    
    async def close(self):
        if self._engine:
            self._engine.dispose()
            self._engine = None
