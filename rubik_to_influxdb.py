from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta
import pytz
import okex.rubik_api as rubik
import time
import threading
import signal
import sys
import queue
import random

tz = pytz.timezone('Asia/Shanghai')  # GMT+8

# 创建一个事件标志来控制线程
stop_event = threading.Event()

# 创建一个限速器类
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                # 移除过期的调用记录
                self.calls = [call for call in self.calls if call > now - self.period]
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.calls[0] - (now - self.period)
                    time.sleep(sleep_time)
                self.calls.append(time.time())
            return func(*args, **kwargs)
        return wrapper

# 创建一个限速器实例
rate_limiter = RateLimiter(max_calls=5, period=2)

# 重试装饰器
def retry_with_backoff(retries=5, backoff_in_seconds=1):
    def rwb(f):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise
                    sleep = (backoff_in_seconds * 2 ** x +
                             random.uniform(0, 1))
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return rwb

def signal_handler(signum, frame):
    print("\n正在停止所有线程，请稍候...")
    stop_event.set()  # 设置停止标志

@rate_limiter
@retry_with_backoff(retries=3, backoff_in_seconds=1)
def get_take_volume(rubikapi, inst_id):
    return rubikapi.take_volume(inst_id, 'SPOT')

def get_and_write_trades(rubikapi, inst_id, writer, bucket):
    points = []

    while not stop_event.is_set():
        try:
            result = get_take_volume(rubikapi, inst_id)
            
            if result['code'] == '0':
                volumes = result['data']
                
                if not volumes:
                    break

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
                    print(f"{inst_id}: Successfully wrote {len(points)} points")
                    points = []  # 清空列表
                except Exception as e:
                    print(f"{inst_id}: Error writing points: {e}")
            else:
                print(f"{inst_id}: API错误: {result['msg']}")
                break

        except Exception as e:
            print(f"{inst_id}: Unexpected error: {e}")
            break

    # 写入剩余的点
    if points:
        try:
            writer.write(bucket=bucket, record=points)
            print(f"{inst_id}: Successfully wrote remaining {len(points)} points")
        except Exception as e:
            print(f"{inst_id}: Error writing remaining points: {e}")

def process_coin(rubikapi, inst_id, writer, bucket):
    print(f"Starting to process {inst_id}")
    try:
        get_and_write_trades(rubikapi, inst_id, writer, bucket)
    except Exception as e:
        print(f"{inst_id}: Error in process_coin: {e}")
    finally:
        print(f"Finished processing {inst_id}")

if __name__ == '__main__':
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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

    # 币种列表
    coin_list = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'TRX', 'OP', 'UNI', 'DOT', 'TON', 'ARB']

    client = None
    try:
        client = InfluxDBClient.from_config_file("D:\OKex-API\config.ini")
        with client.write_api(write_options=SYNCHRONOUS) as writer:
            # 创建线程列表
            threads = []
            
            # 为每个币种创建一个线程
            for coin in coin_list:
                thread = threading.Thread(target=process_coin, args=(rubikapi, coin, writer, "history_trades"))
                thread.daemon = True  # 设置为守护线程
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成或接收到停止信号
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=1.0)  # 每秒检查一次线程状态
                if stop_event.is_set():
                    break

        print("所有数据处理已完成或程序被终止")
        
    except Exception as e: 
        print(f"Error: {e}")
    
    finally:
        if client:
            client.close()
        sys.exit(0)
