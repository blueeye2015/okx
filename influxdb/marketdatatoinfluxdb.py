from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone
import pytz
import okex.Market_api as Market
import json
from api_parser import parse_historytrades 



tz = pytz.timezone('Asia/Shanghai')  # GMT+8

if __name__ == '__main__':
    api_key = ""
    secret_key = ""
    passphrase = ""

    # 设置代理
    proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
    }
    # flag是实盘与模拟盘的切换参数
    # flag = '1'  # 模拟盘
    flag = '0'  # 实盘
    
    # market api
    marketAPI = Market.MarketAPI(api_key, secret_key, passphrase, False, flag, proxies=proxies)
    result = marketAPI.history_trades('BTC-USDT',limit='100',before='587627525',type=1)
    
    api_response = json.dumps(result)  # 这里应该是完整的API响应
    
    try:
        history_trades= parse_historytrades(api_response)
        for his_trade in history_trades:          
            ts_int = int(his_trade['ts'])
            # 假设 ts 是毫秒级时间戳
            trade_time = datetime.fromtimestamp(ts_int / 1000, tz=tz)
            point = Point("test") \
            .tag("instId", his_trade['symbol']) \
            .tag("side", his_trade['side']) \
            .tag("tradeId", his_trade['tradeId']) \
            .field("size", float(his_trade['size'])) \
            .field("price", float(his_trade['price'])) \
            .time(trade_time)  # 使用当前时间而不是 trade['ts']
            
            with InfluxDBClient.from_config_file("config.ini") as client:
                with client.write_api(write_options=SYNCHRONOUS) as writer:
                    try:
                        writer.write(bucket="history_trades", record=[point])
                        print(f"Successfully wrote point: {point}")
                        print(f"Writing point: {point.to_line_protocol()}")
                    except Exception as e:
                        print(e)
            
        print("All data has been processed")
        
    except Exception as e: 
        print(f"Error: {e}")  
    
    
    finally:
        client.close()
        
    
    
