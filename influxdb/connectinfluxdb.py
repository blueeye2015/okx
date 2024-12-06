from influxdb_client import InfluxDBClient

url = "http://localhost:8086"
token = "uFRmk92XY1hfR7Los_xyfnO-nKvwR2CtjjvYqh5vwvt1YVBCXSKB2C4Ma1qIFZfw4rQKac4H_gH0mEzZL_Z7Ww=="  # 使用初始化时获得的令牌
org = "marketdata"
bucket = "history_trades"

client = InfluxDBClient(url=url, token=token, org=org)

try:
    health = client.health()
    if health.status == "pass":
        print("成功连接到 Docker 版 InfluxDB！")
    else:
        print("连接失败。状态:", health.status)
except Exception as e:
    print("连接错误:", str(e))

client.close()
