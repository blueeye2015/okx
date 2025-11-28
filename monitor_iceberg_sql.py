#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Market Radar] 全视角市场监控系统 (修正版)
修正说明：
- 适配 trades 表 Symbol 格式为 "BTC-USDT"
- 适配 depth 表 Symbol 格式为 "BTCUSDT"
"""
import time
import clickhouse_connect
import pandas as pd
from datetime import datetime, timedelta
import logging

# --- 配置 ---
CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')

# [!! 关键修改 !!] 分别定义两个表的 Symbol 格式
SYMBOL_DEPTH = 'BTCUSDT'   # depth 表通常是这种 (来自 Binance 原始流)
SYMBOL_TRADE = 'BTC-USDT'  # trades 表您指定是这种

REFRESH_RATE = 3  # 每 3 秒刷新一次

# 阈值设置
FAKE_WALL_THRESHOLD = 2.0   # 撤单量 > 2 BTC 且无成交 -> 判定为假墙
ICEBERG_RATIO = 3.0         # 成交量 > 3倍可见量 -> 判定为冰山
STALE_WALL_TIME = 30        # 墙存在超过 30 秒价格未动 -> 判定为失效

# --- 日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Radar")

class MarketRadar:
    def __init__(self):
        self.client = clickhouse_connect.get_client(**CLICKHOUSE)
        # 用于 Strategy 3: 记录我们关注的墙 {key: info}
        self.active_walls = {} 

    def get_current_price(self):
        """获取最新成交价 (查询 trades 表)"""
        try:
            # 使用 SYMBOL_TRADE
            res = self.client.query(f"SELECT argMax(price, event_time) FROM trades WHERE symbol='{SYMBOL_TRADE}'")
            if res.result_rows:
                return res.result_rows[0][0]
            return 0
        except Exception as e:
            logger.error(f"获取价格失败: {e}")
            return 0

    # ==========================================
    # 策略 1: 监控“墙的撤单率” (Fake Walls / Spoofing)
    # ==========================================
    def detect_cancellations(self):
        """
        逻辑：在最近 10 秒内，寻找 Delta_Qty (深度变化) 为负，
        且 Abs(Delta_Qty) >> Trade_Volume 的点。
        """
        sql = f"""
        WITH 
            -- 1. 计算深度净变化 (查询 depth 表 -> 使用 SYMBOL_DEPTH)
            DepthDiff AS (
                SELECT 
                    price,
                    side,
                    argMax(quantity, event_time) - argMin(quantity, event_time) as delta_qty,
                    max(quantity) as max_qty
                FROM depth
                WHERE symbol = '{SYMBOL_DEPTH}' AND event_time >= now() - INTERVAL 10 SECOND
                GROUP BY price, side
                HAVING delta_qty < -{FAKE_WALL_THRESHOLD} -- 只看大幅减少的
            ),
            -- 2. 计算同期的成交量 (查询 trades 表 -> 使用 SYMBOL_TRADE)
            TradeVol AS (
                SELECT 
                    price,
                    sum(quantity) as traded_qty
                FROM trades
                WHERE symbol = '{SYMBOL_TRADE}' AND event_time >= now() - INTERVAL 10 SECOND
                GROUP BY price
            )
        
        SELECT 
            D.price,
            D.side,
            D.delta_qty,
            T.traded_qty,
            D.max_qty
        FROM DepthDiff AS D
        LEFT JOIN TradeVol AS T ON D.price = T.price
        -- 核心判断：减少的量(绝对值) > 成交量 * 2 (说明大部分是撤单，不是被吃)
        WHERE abs(D.delta_qty) > (ifNull(T.traded_qty, 0) * 2 + 0.0001)
        ORDER BY D.delta_qty ASC
        LIMIT 5
        """
        
        try:
            df = self.client.query_df(sql)
            if not df.empty:
                print(f"\n🚨 [策略 1] 发现假墙撤单 (Spoofing) - 过去10秒")
                for _, row in df.iterrows():
                    direction = "🟢买单支撑" if row['side'] == 'bid' else "🔴卖单压制"
                    traded = row['traded_qty'] if row['traded_qty'] > 0 else 0
                    print(f"   {direction} @ {row['price']}: 消失了 {abs(row['delta_qty']):.2f} BTC | 仅成交 {traded:.4f} BTC | ⚠️ 纯撤单!")
        except Exception as e:
            logger.error(f"撤单检测出错: {e}")

    # ==========================================
    # 策略 2: 寻找“冰山订单” (Icebergs)
    # ==========================================
    def detect_icebergs(self):
        """
        逻辑：最近 5 分钟，成交量 > 可见挂单 * 3
        """
        sql = f"""
        WITH 
            -- 查询 trades 表 -> 使用 SYMBOL_TRADE
            TradeStats AS (
                SELECT 
                    price,
                    if(buyer_order_maker = 1, 'bid', 'ask') as side,
                    sum(quantity) AS total_traded
                FROM trades
                WHERE symbol = '{SYMBOL_TRADE}' AND event_time >= now() - INTERVAL 5 MINUTE
                GROUP BY price, side
                HAVING total_traded > 5.0
            ),
            -- 查询 depth 表 -> 使用 SYMBOL_DEPTH
            DepthStats AS (
                SELECT 
                    price,
                    side,
                    max(quantity) AS max_visible
                FROM depth
                WHERE symbol = '{SYMBOL_DEPTH}' AND event_time >= now() - INTERVAL 5 MINUTE
                GROUP BY price, side
            )
        SELECT 
            T.side,
            T.price,
            T.total_traded,
            D.max_visible,
            T.total_traded / (D.max_visible + 0.0001) as ratio
        FROM TradeStats AS T
        INNER JOIN DepthStats AS D ON T.price = D.price AND T.side = D.side
        WHERE ratio > {ICEBERG_RATIO}
        ORDER BY T.total_traded DESC
        LIMIT 3
        """
        try:
            df = self.client.query_df(sql)
            if not df.empty:
                print(f"\n🧊 [策略 2] 发现冰山订单 (Icebergs) - 过去5分钟")
                for _, row in df.iterrows():
                    icon = "🚢支撑" if row['side'] == 'bid' else "🏔️压制"
                    print(f"   {icon} {row['side'].upper()} @ {row['price']}: 已成交 {row['total_traded']:.2f} BTC (可见仅 {row['max_visible']:.2f}) | 隐藏倍数 {row['ratio']:.1f}x")
                    
                    # 将冰山加入“关注列表”，用于策略3的监控
                    key = f"{row['side']}_{row['price']}"
                    if key not in self.active_walls:
                        self.active_walls[key] = {
                            'price': row['price'], 
                            'side': row['side'], 
                            'first_seen': datetime.now(),
                            'type': 'iceberg'
                        }
        except Exception as e:
            logger.error(f"冰山检测出错: {e}")

    # ==========================================
    # 策略 3: 动态止损与时效验证 (Time-Based Validation)
    # ==========================================
    def monitor_stale_walls(self):
        """
        逻辑：检查 self.active_walls 中的墙。
        如果 (当前时间 - 发现时间 > 30s) 且 (当前价格 依然离墙很近)，报警。
        """
        if not self.active_walls:
            return

        current_price = self.get_current_price()
        if current_price == 0: return

        keys_to_remove = []
        
        print(f"\n⏱️ [策略 3] 墙体时效监控 (当前价: {current_price})")
        
        for key, wall in self.active_walls.items():
            duration = (datetime.now() - wall['first_seen']).total_seconds()
            price_diff = (current_price - wall['price']) / wall['price'] * 100
            
            # 格式化输出状态
            status = "🟢有效"
            msg = ""

            # 逻辑：如果是买单墙 (支撑)
            if wall['side'] == 'bid':
                # 1. 价格已经涨上去了 (> 0.1%) -> 成功反弹
                if price_diff > 0.1:
                    status = "✅成功"
                    msg = "支撑有效，价格已弹开"
                    keys_to_remove.append(key) # 任务完成，移除监控
                # 2. 价格跌破了 (< -0.05%) -> 支撑失效
                elif price_diff < -0.05:
                    status = "❌击穿"
                    msg = "支撑已被击穿！止损！"
                    keys_to_remove.append(key)
                # 3. 时间久了还在磨蹭 -> 动能衰竭
                elif duration > STALE_WALL_TIME:
                    status = "⚠️危险"
                    msg = f"耗时 {duration:.0f}s 仍未弹开，支撑变弱"
                else:
                    status = "⏳观察"
                    msg = f"已持续 {duration:.0f}s..."

            # 逻辑：如果是卖单墙 (阻力)
            elif wall['side'] == 'ask':
                if price_diff < -0.1:
                    status = "✅成功"
                    msg = "阻力有效，价格已回落"
                    keys_to_remove.append(key)
                elif price_diff > 0.05:
                    status = "❌突破"
                    msg = "阻力已被突破！止损！"
                    keys_to_remove.append(key)
                elif duration > STALE_WALL_TIME:
                    status = "⚠️危险"
                    msg = f"耗时 {duration:.0f}s 仍未回落，阻力变弱"
                else:
                    status = "⏳观察"
                    msg = f"已持续 {duration:.0f}s..."

            print(f"   {wall['side'].upper()} @ {wall['price']} | {status} | {msg}")

        # 清理已完成或失效的墙
        for k in keys_to_remove:
            del self.active_walls[k]

    def run(self):
        print(f"🚀 市场雷达已启动 | 深度Symbol: {SYMBOL_DEPTH} | 成交Symbol: {SYMBOL_TRADE}")
        print("-" * 60)
        while True:
            print(f"\n--- 扫描时间: {datetime.now().strftime('%H:%M:%S')} ---")
            self.detect_cancellations()
            self.detect_icebergs()
            self.monitor_stale_walls()
            time.sleep(REFRESH_RATE)

if __name__ == "__main__":
    radar = MarketRadar()
    try:
        radar.run()
    except KeyboardInterrupt:
        print("监控停止")