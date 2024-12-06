# market_analyzer.py
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum
from sqlalchemy.orm import declarative_base  # 新的导入方式
import enum

Base = declarative_base()

class TrendType(enum.Enum):
    UP = "up"
    DOWN = "down"
    SHOCK = "shock"

class TradeSignal(Base):
    __tablename__ = 'trade_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)  # buy/sell
    vol_beishu = Column(Float, nullable=False)  # 量比
    vol_eq_forward_minutes = Column(Integer, nullable=False)  # 相当于前多少分钟的总量
    trend = Column(Enum(TrendType), nullable=False)
    timestamp_occur = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False)




class MarketAnalyzer:
    def __init__(self):
        # 数据库配置
        self.engine = create_engine('postgresql://postgres:12@localhost:5432/market_data')
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def get_symbol(self, minutes=120):
        """获取所有symbol"""
        query = text("""
            SELECT DISTINCT symbol
            FROM klines
            WHERE timestamp >= NOW() - :minutes * INTERVAL '1 minute'
        """)

        df = pd.read_sql(
            query,
            self.engine,
            params={'minutes': minutes}
        )
        # 返回纯列表，不包含表头
        return df['symbol'].tolist() 

    def get_kline_data(self, symbol, minutes=120):
        """获取最近n分钟的K线数据"""
        query = text("""
            SELECT timestamp, volume, close
            FROM klines
            WHERE symbol = :symbol
            AND timestamp >= NOW() - :minutes * INTERVAL '1 minute'
            ORDER BY timestamp DESC
        """)
        
        df = pd.read_sql(
            query,
            self.engine,
            params={'symbol': symbol, 'minutes': minutes}
        )
        return df
    
    def analyze_volume(self, df):
        """分析交易量"""
        if len(df) < 2:
            return None
            
        # 获取当前时间的前两条记录，因为数据更新完整有延迟
        current = df.iloc[2]
        previous = df.iloc[3]
        
        # 添加零值检查
        if ['volume'] == 0 or pd.isna(previous['volume']) or pd.isna(current['volume']):
            return None
        
        # 计算量比
        vol_ratio = current['volume'] / previous['volume']
        
        # 如果量比在2-20倍之间
        if 2 <= vol_ratio <= 20:
            # 计算当前成交量相当于之前多少分钟的总量
            cumsum = 0
            minutes_count = 0
            
            for i in range(1, len(df)):
                cumsum += df.iloc[i]['volume']
                minutes_count += 1
                if cumsum >= current['volume']:
                    break

            # logging.info(
            #             f"vol_ratio: {vol_ratio}, current_volume: {current['volume']}, previous_volume: {previous['volume']}, "
            #             f"current_timestamp: {current['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}, "
            #             f"previous_timestamp: {previous['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            #         )


            return {
                'vol_ratio': float(vol_ratio),  # 确保转换为Python原生float
                'equivalent_minutes': minutes_count,
                'timestamp_occur': df.iloc[0]['timestamp'] 
            }

        
        return None
    
    def analyze_trend(self, df):
        """分析价格趋势"""
        if len(df) < 60:
            return None
        
        # 使用1小时数据计算趋势
        hour_data = df.head(60).copy()
        
        if len(hour_data) < 60:
            return None
            
        # 计算移动平均线
        hour_data.loc[:, 'MA5'] = hour_data['close'].rolling(window=5).mean()
        hour_data.loc[:, 'MA20'] = hour_data['close'].rolling(window=20).mean()
        hour_data.loc[:, 'MA60'] = hour_data['close'].rolling(window=60).mean()
        
        # 去除NaN
        hour_data = hour_data.dropna()
        
        if len(hour_data) < 20:
            return None
            
        # 计算趋势
        ma5_slope = (hour_data['MA5'].iloc[0] - hour_data['MA5'].iloc[-1]) / len(hour_data)
        ma20_slope = (hour_data['MA20'].iloc[0] - hour_data['MA20'].iloc[-1]) / len(hour_data)
        ma60_slope = (hour_data['MA60'].iloc[0] - hour_data['MA60'].iloc[-1]) / len(hour_data)
        
        # 定义趋势判断标准
        threshold = 0.0001  # 可以根据实际情况调整
        
        if ma5_slope > threshold and ma20_slope > threshold:
            return TrendType.UP
        elif ma5_slope < -threshold and ma20_slope < -threshold:
            return TrendType.DOWN
        else:
            return TrendType.SHOCK
    
    def save_signal(self, symbol, analysis_result, trend):
        try:
            """保存分析结果到数据库"""
            signal = TradeSignal(
                    symbol=symbol,
                    side='buy',
                    vol_beishu=float(analysis_result['vol_ratio']),  # 确保转换为Python原生float
                    vol_eq_forward_minutes=int(analysis_result['equivalent_minutes']),  # 确保转换为Python原生int
                    trend=trend,
                    timestamp_occur = analysis_result['timestamp_occur'], #保留发生的timestamp
                    created_at=datetime.now()
                )
        except Exception as e:
            self.session.rollback()
            logging.error(f"Error saving signal for {symbol}: {str(e)}")
            raise
        
        self.session.add(signal)
        self.session.commit()

    
    async def analyze_symbol(self, symbol):
        """分析单个交易对"""
        try:
            # 获取K线数据
            df = self.get_kline_data(symbol)
            
            # 确保数据至少有2分钟前的
            if len(df) < 2:
                logging.warning(f"Insufficient data for {symbol}")
                return
                
            # 分析交易量
            volume_analysis = self.analyze_volume(df)
                        
            if volume_analysis:
                # 分析趋势
                trend = self.analyze_trend(df)
                
                if trend:
                    # 保存信号
                    self.save_signal(symbol, volume_analysis, trend)
                    logging.info(f"Signal generated for {symbol}: volume ratio {volume_analysis['vol_ratio']:.2f}, "
                               f"equivalent to {volume_analysis['equivalent_minutes']} minutes, trend: {trend.value}")
                    
        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {str(e)}")

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    analyzer = MarketAnalyzer()
    
    # 获取需要分析的交易对列表
    symbols = analyzer.get_symbol()  # 示例交易对
    
    while True:
        try:
            # 并发分析所有交易对
            tasks = [analyzer.analyze_symbol(symbol) for symbol in symbols]
            await asyncio.gather(*tasks)
            
            # 等待下一分钟
            await asyncio.sleep(60)
            
        except Exception as e:
            logging.error(f"Main loop error: {str(e)}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
