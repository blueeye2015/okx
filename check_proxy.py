import requests
import time

# 确保端口和你 config.py 里一致
PROXIES = {
    'http': 'socks5h://127.0.0.1:7890',
    'https': 'socks5h://127.0.0.1:7890'
}

PROXIES_ = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }

def test_url(name, url):
    print(f"正在测试 {name} 连接 ({url})...")
    start = time.time()
    try:
        resp = requests.get(url, proxies=PROXIES_, timeout=15)
        latency = (time.time() - start) * 1000
        print(f"✅ {name} 通畅! 状态码: {resp.status_code}, 耗时: {latency:.0f}ms")
    except Exception as e:
        print(f"❌ {name} 失败: {e}")

if __name__ == "__main__":
    # 1. 先测 Google (验证梯子本身是否好用)
    test_url("Google", "https://www.google.com")
    
    # 2. 再测 Binance API (验证节点是否屏蔽了币安或者被墙)
    test_url("Binance API", "https://fapi.binance.com/fapi/v1/time")