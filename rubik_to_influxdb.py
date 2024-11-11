from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta
import pytz
import okex.rubik_api as rubik
import time

tz = pytz.timezone('Asia/Shanghai')  # GMT+8

def get_and_write_trades(rubikapi, inst_id,  writer, bucket):
    
    points = []

    while 1 > 0:
        #result = marketAPI.history_trades(inst_id, limit=str(batch_size), after=str(int(current_time.timestamp() * 1000)),type=2)
        result = rubikapi.take_volume(inst_id, 'SPOT')
        
        if result['code'] == '0':
            volumes = result['data']
            
            if not volumes:
                break  # 没有更多数据

            for volume in volumes:
                ts_int = int(volume[0])
                trade_time = datetime.fromtimestamp(ts_int / 1000, tz=tz)
                
                point = Point('takevolume') \
                    .tag("instId", inst_id) \
                    .field("sellVol", float(volume[1])) \
                    .field("buyVol", float(volume[2])) \
                    .time(trade_time)
                
                points.append(point)

            # 批量写入数据
            
            try:
                writer.write(bucket=bucket, record=points)
                print(f"Successfully wrote {len(points)} points")
                points = []  # 清空列表
            except Exception as e:
                print(f"Error writing points: {e}")

            
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
    rubikapi = rubik.RubikApi(api_key, secret_key, passphrase, False, flag, proxies=proxies)

    # 设置时间范围
    #end_time = datetime.now(tz) - timedelta(days=1)
    #start_time = end_time - timedelta(days=6)  # 7天前到1天前

    inst_id = 'BTC'

    try:
        with InfluxDBClient.from_config_file("config.ini") as client:
            with client.write_api(write_options=SYNCHRONOUS) as writer:
                get_and_write_trades(rubikapi, inst_id, writer, "history_trades")

        print("All data has been processed")
        
    except Exception as e: 
        print(f"Error: {e}")
    
    finally:
        client.close()
