import requests

# 配置 SOCKS5 代理
proxies = {
    "http": "socks5://ufo9_2010:9pgjAaWtzC@172.120.69.176:50101",
    "https": "socks5://ufo9_2010:9pgjAaWtzC@172.120.69.176:50101"
}

try:
    resp = requests.get("https://api.binance.com/api/v3/time", proxies=proxies, timeout=10)
    print("✅ 成功！当前时间:", resp.json())
except Exception as e:
    print("❌ 失败:", str(e))