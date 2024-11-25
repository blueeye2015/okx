import duckdb
import pandas as pd
from datetime import datetime
from typing import Dict, List

class MarketDataStore:
    def __init__(self, db_path):
        self.conn = duckdb.connect(db_path)
        self.initialize_tables()
        
    def initialize_tables(self):
        """初始化数据表"""
        # 创建K线数据表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS klines (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # 创建交易对信息表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                symbol VARCHAR PRIMARY KEY,
                listing_date TIMESTAMP,
                market_cap DOUBLE,
                last_updated TIMESTAMP
            )
        """)
        
        # 创建突破事件表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS breakout_events (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                price DOUBLE,
                volume DOUBLE,
                breakout_type VARCHAR,  -- 'UP' or 'DOWN'
                strength DOUBLE,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
    def save_klines(self, symbol: str, df: pd.DataFrame):
        """保存K线数据
        
        Parameters:
        -----------
        symbol : str
            交易对符号
        df : pd.DataFrame
            包含OHLCV数据的DataFrame
        """
        # 确保DataFrame包含所有必要的列
        df['symbol'] = symbol  # 添加symbol列
        
        # 确保列的顺序与表结构匹配
        df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # 执行插入操作
        self.conn.execute("""
            INSERT INTO klines 
            SELECT * FROM df
            ON CONFLICT (symbol, timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """)
        
    def get_recent_klines(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """获取最近的K线数据"""
        return self.conn.execute("""
            SELECT * FROM klines
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, [symbol, limit]).df()
        
    def analyze_breakout(self, symbol: str) -> pd.DataFrame:
        """分析突破情况"""
        return self.conn.execute("""
            WITH price_stats AS (
                SELECT 
                    symbol,
                    timestamp,
                    close,
                    volume,
                    AVG(close) OVER (
                        PARTITION BY symbol 
                        ORDER BY timestamp 
                        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                    ) as avg_price,
                    AVG(volume) OVER (
                        PARTITION BY symbol 
                        ORDER BY timestamp 
                        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                    ) as avg_volume
                FROM klines
                WHERE symbol = ?
            )
            SELECT 
                *,
                (close - avg_price) / avg_price as price_change,
                volume / NULLIF(avg_volume, 0) as volume_ratio
            FROM price_stats
            WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 day'
            ORDER BY timestamp DESC
        """, [symbol]).df()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


class TradingDataStore:
    def __init__(self, db_path):
        self.conn = duckdb.connect(db_path)
        self.initialize_tables()
        
    def initialize_tables(self):
        """初始化交易相关的表"""
        # 订单表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                side VARCHAR,
                type VARCHAR,
                amount DOUBLE,
                price DOUBLE,
                status VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)
        
        # 持仓表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol VARCHAR PRIMARY KEY,
                amount DOUBLE,
                entry_price DOUBLE,
                current_price DOUBLE,
                unrealized_pnl DOUBLE,
                entry_time TIMESTAMP,
                last_updated TIMESTAMP
            )
        """)
        
        # 交易记录表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR PRIMARY KEY,
                order_id VARCHAR,
                symbol VARCHAR,
                side VARCHAR,
                amount DOUBLE,
                price DOUBLE,
                fee DOUBLE,
                realized_pnl DOUBLE,
                timestamp TIMESTAMP
            )
        """)

    def update_position(self, position_data: Dict):
        """更新持仓信息"""
        self.conn.execute("""
            INSERT INTO positions 
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (symbol) DO UPDATE SET
                amount = EXCLUDED.amount,
                current_price = EXCLUDED.current_price,
                unrealized_pnl = EXCLUDED.unrealized_pnl,
                last_updated = CURRENT_TIMESTAMP
        """, [
            position_data['symbol'],
            position_data['amount'],
            position_data['entry_price'],
            position_data['current_price'],
            position_data['unrealized_pnl'],
            position_data['entry_time']
        ])

    def get_position_analysis(self) -> pd.DataFrame:
        """获取持仓分析"""
        return self.conn.execute("""
            SELECT 
                p.*,
                t.total_trades,
                t.win_trades,
                t.total_pnl
            FROM positions p
            LEFT JOIN (
                SELECT 
                    symbol,
                    COUNT(*) as total_trades,
                    COUNT(*) FILTER (WHERE realized_pnl > 0) as win_trades,
                    SUM(realized_pnl) as total_pnl
                FROM trades
                GROUP BY symbol
            ) t ON p.symbol = t.symbol
        """).df()

# 使用示例
def main():
    path = 'D:\docker\duckdb\marketdata.duckdb'
    market_db = MarketDataStore(path)
    trading_db = TradingDataStore(path)
    
    # 模拟数据获取和存储
    symbols = ['BTC/USDT', 'ETH/USDT']
    for symbol in symbols:
        # 假设这是从交易所获取的数据
        df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [50000],
            'high': [51000],
            'low': [49000],
            'close': [50500],
            'volume': [1000]
        })
        market_db.save_klines(symbol, df)
        
        # 分析突破
        breakouts = market_db.analyze_breakout(symbol)
        if not breakouts.empty:
            # 处理突破信号...
            pass
        
        # 更新持仓信息
        position_data = {
            'symbol': symbol,
            'amount': 1.0,
            'entry_price': 50000,
            'current_price': 50500,
            'unrealized_pnl': 500,
            'entry_time': datetime.now()
        }
        trading_db.update_position(position_data)
    
    # 获取持仓分析
    position_analysis = trading_db.get_position_analysis()
    print(position_analysis)

if __name__ == "__main__":
    main()
