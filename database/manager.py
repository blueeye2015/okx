from sqlalchemy import create_engine
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
        url = f"postgresql://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
        self._engine = create_engine(url, pool_size=5, max_overflow=10)
        self._session_factory = sessionmaker(bind=self._engine)
        Base.metadata.create_all(self._engine)
    
    def get_session(self):
        return self._session_factory()
    
    def close(self):
        if self._engine:
            self._engine.dispose()
            self._engine = None
