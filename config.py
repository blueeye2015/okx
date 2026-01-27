import os
from dotenv import load_dotenv

load_dotenv('/data/okx/.env')

class Config:
    # --- 全局账户配置 (所有策略共用) ---
    API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
    SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
    BASE_URL = 'https://fapi.binance.com'
    
    PROXIES = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }

    # --- 策略清单 (在这里扩展任意数量的策略) ---
    # config.py
    STRATEGIES = [
        {
            "name": "ETH_Ladder",
            "symbol": "ETHUSDT",
            "signal_file": "/data/okx/eth_signals.csv",
            "base_qty": 0.2,          # 每次购买的固定个数
            "leverage": 20,           # 你之前提到的 20x 杠杆
            "first_gap": 0.035,        # 第一笔补仓间距 3.5%
            "subsequent_gap": 0.02,    # 之后每笔间距 2%
            "ladder_count": 8,         # 总共 1+7 笔
            "tp_rate": 0.25,           # 整体止盈 25%
            "reversal_tp": 0.03        # 补仓后反弹 3% 即撤离
        }
    ]