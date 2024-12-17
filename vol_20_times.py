# market_analyzer.py
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum, Boolean
from sqlalchemy.orm import declarative_base  # 新的导入方式
import enum
from scipy import stats

Base = declarative_base()

class TrendType(enum.Enum):
    UP = "up"
    DOWN = "down"
    SHOCK = "shock"

class TradeSignal(Base):
    __tablename__ = 'trend_records'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    ma5 = Column(Float, nullable=False)  
    ma20 = Column(Float, nullable=False) 
    ma60 = Column(Float, nullable=False) 
    ma5_slope  = Column(Float, nullable=False) 
    ma20_slope  = Column(Float, nullable=False) 
    ma60_slope  = Column(Float, nullable=False) 
    ma5_r2  = Column(Float, nullable=False) 
    ma20_r2  = Column(Float, nullable=False) 
    ma60_r2  = Column(Float, nullable=False) 
    trend = Column(Enum(TrendType), nullable=False)
    significant_trends = Column(Integer, nullable=False) 
    ma_alignment_up = Column(Boolean, nullable=False)
    ma_alignment_down = Column(Boolean, nullable=False)
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
    
    def get_slopes_r2(self, symbol):
            """获取最近的斜率和r2"""
            query = text("""
            SELECT * FROM get_moving_average_slopes(:symbol)
            """)

            df = pd.read_sql(
            query,
            self.engine,
            params={'symbol': symbol}
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
        
        # # 使用足够长的数据来计算MA60
        hour_data = df.sort_values('date', ascending=False).head(60).copy()
        
        # # 计算移动平均线 - 现在使用MA5, MA20, MA60
        # hour_data.loc[:, 'ma5'] = hour_data['close'].rolling(window=5).mean()
        # hour_data.loc[:, 'ma20'] = hour_data['close'].rolling(window=20).mean()
        # hour_data.loc[:, 'ma60'] = hour_data['close'].rolling(window=60).mean()
        
        # # 去除NaN值
        # hour_data = hour_data.dropna()
        
        # if len(hour_data) < 60:
        #     return None
            
        # def calculate_trend(data):
        #     """计算趋势的斜率和R²值"""
        #     x = np.arange(len(data))
        #     y = data.values
        #     slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        #     return slope, r_value**2
        
        # # 计算MA5、MA20和MA60的趋势
        # ma5_slope, ma5_r2 = calculate_trend(hour_data['ma5'])
        # ma20_slope, ma20_r2 = calculate_trend(hour_data['ma20'])
        # ma60_slope, ma60_r2 = calculate_trend(hour_data['ma60'])
        ma5_r2 = float(hour_data['r_value_ma5'][0])
        ma20_r2 = float(hour_data['r_value_ma20'][0])
        ma60_r2 = float(hour_data['r_value_ma60'][0])
        ma5_slope = float(hour_data['slope_ma5'][0])
        ma20_slope = float(hour_data['slope_ma20'][0])
        ma60_slope = float(hour_data['slope_ma60'][0])
        
        # # 打印调试信息
        # # print(f"MA5 trend - Slope: {ma5_slope:.6f}, R²: {ma5_r2:.4f}")
        # # print(f"MA20 trend - Slope: {ma20_slope:.6f}, R²: {ma20_r2:.4f}")
        # # print(f"MA60 trend - Slope: {ma60_slope:.6f}, R²: {ma60_r2:.4f}")
        
        # # 趋势判断标准
        slope_threshold_ma5 = 0.02    # 短期可以设置大一点
        slope_threshold_ma20 = 0.015  # 中期适中
        slope_threshold_ma60 = 0.01   # 长期可以设置小一点  # 斜率阈值
        r2_threshold = 0.6      # R²值阈值
        
        # # 判断趋势
        # # 1. 三条均线都要朝同一个方向
        # # 2. 至少有两条均线的R²值要大于阈值
        def count_significant_r2(r2_values):
            return sum(1 for r2 in r2_values if r2 > r2_threshold)
        
        r2_values = [ma5_r2, ma20_r2, ma60_r2]
        significant_trends = count_significant_r2(r2_values)
        
        # 检查均线位置关系
        latest_prices = hour_data.iloc[0]
        ma_alignment_up = (latest_prices['ma5'] > latest_prices['ma20'] > latest_prices['ma60'])
        ma_alignment_down = (latest_prices['ma5'] < latest_prices['ma20'] < latest_prices['ma60'])
        
        # 确定最终趋势
        if (ma5_slope > slope_threshold_ma5 and 
            ma20_slope > slope_threshold_ma20 and 
            ma60_slope > slope_threshold_ma60 and 
            significant_trends >= 2 and
            ma_alignment_up):
            trend = TrendType.UP
        elif (ma5_slope < -slope_threshold_ma5 and 
            ma20_slope < -slope_threshold_ma20 and 
            ma60_slope < -slope_threshold_ma60 and 
            significant_trends >= 2 and
            ma_alignment_down):
            trend = TrendType.DOWN
        else:
            trend = TrendType.SHOCK

        # 创建要返回的趋势记录
        trend_record = {
            'timestamp': df['date'][0],  # 记录当前时间
            'ma5': float(latest_prices['ma5']),
            'ma20': float(latest_prices['ma20']),
            'ma60': float(latest_prices['ma60']),
            'ma5_slope': float(ma5_slope),
            'ma20_slope': float(ma20_slope),
            'ma60_slope': float(ma60_slope),
            'ma5_r2': float(ma5_r2),
            'ma20_r2': float(ma20_r2),
            'ma60_r2': float(ma60_r2),
            'trend': trend.value,  # 假设TrendType是Enum类型
            'significant_trends': significant_trends,
            'ma_alignment_up': ma_alignment_up,
            'ma_alignment_down': ma_alignment_down
        }
        
        return trend_record

    
    def save_signal(self, symbol, trend):
        try:
            """保存分析结果到数据库"""
            signal = TradeSignal(
                    symbol=symbol,
                    timestamp=trend['timestamp'],
                    ma5=trend['ma5'],  
                    ma20=trend['ma20'],  
                    ma60=trend['ma60'], 
                    ma5_slope=trend['ma5_slope'], 
                    ma20_slope=trend['ma20_slope'],
                    ma60_slope=trend['ma60_slope'], 
                    ma5_r2=trend['ma5_r2'], 
                    ma20_r2=trend['ma20_r2'],
                    ma60_r2=trend['ma60_r2'], 
                    trend=trend['trend'],
                    significant_trends = trend['significant_trends'], 
                    ma_alignment_up = trend['ma_alignment_up'],
                    ma_alignment_down = trend['ma_alignment_down'],
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
            df = self.get_slopes_r2(symbol)
            
            # 确保数据至少有2分钟前的
            # if len(df) < 2:
            #     logging.warning(f"Insufficient data for {symbol}")
            #     return
                
            # 分析交易量
            # volume_analysis = self.analyze_volume(df)
                        
            # if volume_analysis:
            # 分析趋势
            trend = self.analyze_trend(df)
                
            if trend:
                # 保存信号
                self.save_signal(symbol, trend)
                # logging.info(f"Signal generated for {symbol}: volume ratio {volume_analysis['vol_ratio']:.2f}, "
                #             f"equivalent to {volume_analysis['equivalent_minutes']} minutes, trend: {trend.value}")
                
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
