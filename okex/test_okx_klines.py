import json
import psycopg2
from datetime import datetime, timezone
from typing import List, Dict
import logging

class KlineDataImporter:
    def __init__(self, db_config: Dict[str, str]):
        """
        初始化导入器
        
        Args:
            db_config: 数据库配置字典，包含：
                host: 数据库主机
                database: 数据库名
                user: 用户名
                password: 密码
                port: 端口
        """
        self.db_config = db_config
        
    def create_table(self, conn) -> None:
        """创建K线数据表"""
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kline_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE,
                    open_price DECIMAL(20, 8),
                    high_price DECIMAL(20, 8),
                    low_price DECIMAL(20, 8),
                    close_price DECIMAL(20, 8),
                    volume DECIMAL(20, 8),
                    volume_currency DECIMAL(20, 8),
                    volume_currency_quote DECIMAL(20, 8),
                    is_confirmed BOOLEAN,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                
                -- 创建时间索引
                CREATE INDEX IF NOT EXISTS idx_kline_timestamp ON kline_data(timestamp);
            """)
            conn.commit()
            
    def convert_timestamp(self, ts: str) -> datetime:
        """
        将毫秒时间戳转换为UTC时间
        
        Args:
            ts: 毫秒时间戳字符串
            
        Returns:
            datetime: UTC时间对象
        """
        return datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
        
    def import_data(self, json_data: str) -> None:
        """
        导入JSON数据到PostgreSQL
        
        Args:
            json_data: JSON格式的K线数据
        """
        try:
            # 解析JSON数据
            data = json.loads(json_data)
            if data['code'] != '0':
                raise ValueError(f"数据响应错误: {data['msg']}")
                
            # 连接数据库
            with psycopg2.connect(**self.db_config) as conn:
                # 创建表（如果不存在）
                self.create_table(conn)
                
                # 准备插入语句
                insert_query = """
                    INSERT INTO kline_data (
                        timestamp, open_price, high_price, low_price, close_price,
                        volume, volume_currency, volume_currency_quote, is_confirmed
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ;
                """
                
                with conn.cursor() as cur:
                    for kline in data['data']:
                        # 转换数据
                        timestamp = self.convert_timestamp(kline[0])
                        values = (
                            timestamp,
                            float(kline[1]),  # open
                            float(kline[2]),  # high
                            float(kline[3]),  # low
                            float(kline[4]),  # close
                            float(kline[5]),  # volume
                            float(kline[6]),  # volCcy
                            float(kline[7]),  # volCcyQuote
                            kline[8] == '1'   # confirm (转换为boolean)
                        )
                        
                        cur.execute(insert_query, values)
                        
                conn.commit()
                logging.info(f"成功导入 {len(data['data'])} 条K线数据")
                
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析错误: {e}")
            raise
        except psycopg2.Error as e:
            logging.error(f"数据库错误: {e}")
            raise
        except Exception as e:
            logging.error(f"导入数据时发生错误: {e}")
            raise

# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 数据库配置
    db_config = {
        'host': 'localhost',
        'database': 'market_data',
        'user': 'postgres',
        'password': '12',
        'port': '5432'
    }
    
    # JSON数据
    with open('klines.json', 'r') as file:
        data = json.load(file)
        json_data=json.dumps(data)
    
    try:
        # 创建导入器并导入数据
        importer = KlineDataImporter(db_config)
        importer.import_data(json_data)
        
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
