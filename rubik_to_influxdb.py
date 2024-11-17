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
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

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

class TooManyRequestsError(Exception):
    pass
# 重试装饰器
def retry_with_backoff(retries=5, backoff_in_seconds=1):
    def rwb(f):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return f(*args, **kwargs)
                except TooManyRequestsError as e:
                    if x == retries:
                        raise
                    sleep = (backoff_in_seconds * 2 ** x +
                             random.uniform(0, 1))
                    logging.warning(f"Too many requests, retrying in {sleep:.2f} seconds...")
                    time.sleep(sleep)
                    x += 1
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    raise
        return wrapper
    return rwb

def signal_handler(signum, frame):
    logging.info("\n正在停止所有线程，请稍候...")
    stop_event.set()  # 设置停止标志


def should_retry(exception):
    """判断是否应该重试的函数"""
    if isinstance(exception, Exception):
        if hasattr(exception, 'args') and len(exception.args) > 0:
            error_msg = str(exception.args[0])
            return "Too Many Requests" in error_msg or "50011" in error_msg
    return False

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=should_retry,
    before_sleep=lambda retry_state: logging.info(
        f"Retrying in {retry_state.next_action.sleep} seconds due to rate limit..."
    )
)

def get_take_volume(rubikapi, inst_id):
    """获取交易量数据，带重试机制"""
    try:
        result = rubikapi.take_volume(inst_id, 'SPOT')
        if result['code'] == '50011':
            logging.warning(f"{inst_id}: Rate limit hit, will retry...")
            raise Exception(f"API Request Error(code=50011): Too Many Requests")
        elif result['code'] != '0':
            raise Exception(f"API错误: {result['msg']}")
        return result
    except Exception as e:
        logging.warning(f"{inst_id}: Error in get_take_volume: {str(e)}")
        raise

def get_and_write_trades(rubikapi, inst_id, writer, bucket):
    """获取并写入交易数据"""
    points = []
    
    while not stop_event.is_set():
        try:
            result = get_take_volume(rubikapi, inst_id)
            volumes = result['data']
            
            if not volumes:
                logging.info(f"{inst_id}: No more data to process")
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
            if points:
                try:
                    writer.write(bucket=bucket, record=points)
                    logging.info(f"{inst_id}: Successfully wrote {len(points)} points")
                    points = []  # 清空列表
                except Exception as e:
                    logging.error(f"{inst_id}: Error writing points: {e}")
                    raise

            # 添加小延迟，避免请求过于频繁
            time.sleep(1)

        except Exception as e:
            if should_retry(e):
                logging.warning(f"{inst_id}: Rate limit hit, retrying...")
                continue
            else:
                logging.error(f"{inst_id}: Fatal error: {e}")
                break

    # 写入剩余的点
    if points:
        try:
            writer.write(bucket=bucket, record=points)
            logging.info(f"{inst_id}: Successfully wrote remaining {len(points)} points")
        except Exception as e:
            logging.error(f"{inst_id}: Error writing remaining points: {e}")

def process_coin(rubikapi, inst_id, writer, bucket):
    logging.info(f"Starting to process {inst_id}")
    try:
        get_and_write_trades(rubikapi, inst_id, writer, bucket)
    except Exception as e:
        logging.error(f"{inst_id}: Error in process_coin: {e}")
    finally:
        logging.info(f"Finished processing {inst_id}")

if __name__ == '__main__':
    # 设置日志配置
    logging.basicConfig(filename='rubik.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('脚本开始执行')

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
                    t.join(timeout=2.0)  # 每秒检查一次线程状态
                if stop_event.is_set():
                    break

        logging.info("所有数据处理已完成或程序被终止")
        
    except Exception as e: 
        logging.error(f"Error: {e}")
    
    finally:
        if client:
            client.close()
        sys.exit(0)
