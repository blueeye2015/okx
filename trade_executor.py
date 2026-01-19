import time
import csv
import os
import pandas as pd
import requests
import base64
from urllib.parse import urlencode, quote
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# 加载环境变量
load_dotenv('/data/okx/.env')

# ================= 配置区域 =================
API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY') # Ed25519 私钥内容
BASE_URL = 'https://fapi.binance.com'
COMMAND_FILE = '/data/okx/trade_commands.csv'
PROXIES = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
# ===========================================

# --- 签名工具函数 (复用你验证过的成功代码) ---
def get_signature(payload):
    private_key_str = SECRET_KEY.strip()
    if not private_key_str.startswith("-----BEGIN"):
        private_key_str = f"-----BEGIN PRIVATE KEY-----\n{private_key_str}\n-----END PRIVATE KEY-----"
    try:
        private_key = load_pem_private_key(private_key_str.encode('utf-8'), password=None)
        signature = private_key.sign(payload.encode('utf-8'))
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        print(f"❌ 签名失败: {e}")
        return None

def send_request(method, endpoint, params=None):
    if params is None: params = {}
    params['timestamp'] = int(time.time() * 1000)
    query_string = urlencode(params)
    signature = get_signature(query_string)
    full_url = f"{BASE_URL}{endpoint}?{query_string}&signature={quote(signature)}"
    headers = {'X-MBX-APIKEY': API_KEY}
    try:
        resp = requests.request(method, full_url, headers=headers, proxies=PROXIES, timeout=10)
        return resp.json()
    except Exception as e:
        print(f"❌ API 请求异常: {e}")
        return None

# --- 交易动作执行 ---
def execute_order(action, symbol, qty):
    print(f"⚡ 收到指令: {action} | 数量: {qty}")
    
    side = None
    reduce_only = 'false'
    
    # 场景 1: 开首单
    if action == 'OPEN_LONG':
        side = 'BUY'
        reduce_only = 'false'
        
    # 场景 2: 马丁补仓 (DCA)
    # 在交易所眼里，补仓就是继续买入，不开 reduceOnly
    elif action == 'DCA_BUY':
        side = 'BUY'
        reduce_only = 'false'
        
    # 场景 3: 平仓 (止盈/止损)
    elif action == 'CLOSE_LONG':
        side = 'SELL'
        reduce_only = 'true' # 关键！只减仓，防止变成反向开空
        
    if side:
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': qty,
            'reduceOnly': reduce_only
        }
        
        # 发送请求
        res = send_request('POST', '/fapi/v1/order', params)
        
        if res and 'orderId' in res:
            print(f"✅ 执行成功! {action} 成交. ID: {res['orderId']}")
            return True
        else:
            print(f"❌ 执行失败: {res}")
            return False
    return False

# --- 主循环监听 ---
def run_executor():
    print("🤖 交易执行器启动... 正在监听指令文件")
    last_processed_idx = -1
    
    # 初次启动先定位到文件末尾，避免重复执行历史指令
    if os.path.exists(COMMAND_FILE):
        df = pd.read_csv(COMMAND_FILE)
        last_processed_idx = len(df) - 1
        print(f"   已忽略历史指令，从第 {last_processed_idx + 1} 行开始监听")

    while True:
        try:
            if not os.path.exists(COMMAND_FILE):
                time.sleep(1)
                continue
                
            # 读取文件
            df = pd.read_csv(COMMAND_FILE)
            current_len = len(df)
            
            # 如果有新行
            if current_len > (last_processed_idx + 1):
                # 获取所有新指令
                new_commands = df.iloc[last_processed_idx + 1:]
                
                for idx, row in new_commands.iterrows():
                    # 检查状态，防止重复 (虽然按行号读已经防了，但双重保险)
                    if row.get('status', '') == 'DONE':
                        continue
                        
                    print(f"\n📩 收到指令 [{row['timestamp']}]: {row['action']} {row['reason']}")
                    
                    # 执行下单
                    success = execute_order(row['action'], row['symbol'], row['quantity'])
                    
                    # (可选) 回写状态到 CSV，标记为 DONE
                    # 这里为了简化，我们只在内存里更新索引，实际工程中最好回写数据库或文件
                    
                # 更新指针
                last_processed_idx = current_len - 1
            
            time.sleep(1) # 每秒轮询一次
            
        except Exception as e:
            print(f"⚠️ 监听循环出错: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_executor()