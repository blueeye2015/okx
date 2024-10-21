from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta
import pytz
import okex.Market_api as Market
import time

tz = pytz.timezone('Asia/Shanghai')  # GMT+8

def get_and_write_trades(marketAPI, inst_id,  writer, bucket):
    
    batch_size = 500
    points = []

    while 1 > 0:
        #result = marketAPI.history_trades(inst_id, limit=str(batch_size), after=str(int(current_time.timestamp() * 1000)),type=2)
        result = marketAPI.market_trades(inst_id, limit=str(batch_size))
        
        if result['code'] == '0':
            trades = result['data']
            
            if not trades:
                break  # 没有更多数据

            for trade in trades:
                ts_int = int(trade['ts'])
                trade_time = datetime.fromtimestamp(ts_int / 1000, tz=tz)
                
                point = Point("TNSR") \
                    .tag("instId", inst_id) \
                    .tag("side", trade['side']) \
                    .tag("tradeId", trade['tradeId']) \
                    .field("size", float(trade['sz'])) \
                    .time(trade_time)
                
                points.append(point)

            # 批量写入数据
            if len(points) >= batch_size:
                try:
                    writer.write(bucket=bucket, record=points)
                    print(f"Successfully wrote {len(points)} points")
                    points = []  # 清空列表
                except Exception as e:
                    print(f"Error writing points: {e}")

            current_time = datetime.fromtimestamp(int(trades[-1]["ts"]) / 1000, tz=pytz.UTC)
        else:
            print(f"API错误: {result['msg']}")
            break

        # 遵守限速规则
        time.sleep(0.2)  # 100ms 延迟，确保不超过 20次/2s 的限制

    # 写入剩余的点
    if points:
        try:
            writer.write(bucket=bucket, record=points)
            print(f"Successfully wrote remaining {len(points)} points")
        except Exception as e:
            print(f"Error writing remaining points: {e}")

if __name__ == '__main__':
    api_key = ""
    secret_key = ""
    passphrase = ""

    # 设置代理
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }
    flag = '0'  # 实盘
    
    # market api
    marketAPI = Market.MarketAPI(api_key, secret_key, passphrase, False, flag, proxies=proxies)

    # 设置时间范围
    #end_time = datetime.now(tz) - timedelta(days=1)
    #start_time = end_time - timedelta(days=6)  # 7天前到1天前

    inst_id = 'TNSR-USDT-SWAP'

    try:
        with InfluxDBClient.from_config_file("config.ini") as client:
            with client.write_api(write_options=SYNCHRONOUS) as writer:
                get_and_write_trades(marketAPI, inst_id, writer, "history_trades")

        print("All data has been processed")
        
    except Exception as e: 
        print(f"Error: {e}")
    
    finally:
        client.close()
