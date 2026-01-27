import pandas as pd
import logging
import time
import math
from datetime import datetime, timedelta
from config import Config
from state_manager import ClickHouseStateManager

logger = logging.getLogger(__name__)

class StrategyRunner:
    def __init__(self, strategy_config, client):
        self.cfg = strategy_config
        self.client = client
        self.name = self.cfg['name']
        self.symbol = self.cfg['symbol']
        
        # 初始化 ClickHouse 状态管理器
        self.state = ClickHouseStateManager(self.name, self.client, self.symbol)
        
        # 初始化杠杆 (只设一次)
        try:
            self.client.send_request('POST', '/fapi/v1/leverage', 
                                    {'symbol': self.symbol, 'leverage': self.cfg['leverage']})
        except: pass

        # === 核心逻辑移植 ===
        # 初始化 last_processed_time，防止程序重启时把旧信号当新信号
        self.last_signal_time = self._get_initial_signal_time()
        logger.info(f"[{self.name}] 初始化完成，最新已处理信号时间: {self.last_signal_time}")

    def _get_initial_signal_time(self):
        """启动时先读一次最后一行，作为基准"""
        sig = self.get_latest_signal()
        if sig is not None:
            return sig['Time']
        return None

    def get_latest_signal(self):
        """读取 CSV 最后一行"""
        try:
            # 你的代码里用的是 pd.read_csv，这里沿用
            path = self.cfg['signal_file']
            # 优化：只读最后几行，避免文件大了读得慢
            # 但为了兼容性，先保持全读 (如果文件超过100MB请改成 tail 模式)
            df = pd.read_csv(path)
            if df.empty: return None
            return df.iloc[-1]
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"[{self.name}] 读取信号文件失败: {e}")
            return None


    def execute_ladder_orders(self, current_price):
        """
        核心逻辑：生成1个市价单 + 7个阶梯限价单
        """
        qty = self.cfg['base_qty']
        symbol = self.symbol
        
        # 1. 撤销该币种所有现有挂单
        self.client._close_all_and_cancel(symbol)
        
        # 2. 发送第1笔：市价开多
        first_order = self.client.place_order(symbol, 'BUY', qty, 'MARKET')
        if not first_order:
            logger.error("❌ 首笔市价单失败，取消后续挂单")
            return

        logger.info(f"🚀 首笔市价开仓成功: {qty} ETH @ {current_price}")
        
        # 3. 循环发送后续 7 笔限价单 (Limit Orders)
        ladder_prices = []
        for i in range(1, self.cfg['ladder_count']):
            # 价格每跌 drop_step (2.5%) 挂一单
            target_price = round(current_price * (1 - self.cfg['drop_step'] * i), 2)
            
            # 发送限价买单
            res = self.client.send_request('POST', '/fapi/v1/order', {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'LIMIT',
                'quantity': qty,
                'price': str(target_price),
                'timeInForce': 'GTC'
            })
            if res:
                logger.info(f"📦 阶梯挂单 {i} 已放置: {target_price}")
                ladder_prices.append(target_price)

        # 4. 更新状态到 ClickHouse
        # 注意：这里存的是初始状态，后续需要同步均价
        self.state.update_open_position('LONG', qty, current_price, 0, 0)

    def check_risk_management(self, current_price):
        """
        移植自 check_breakeven，逻辑完全一致
        """
        pos = self.state.state
        if pos['position_side'] != 'LONG' or pos['amount'] <= 0: return
        
        entry = pos['entry_price']
        if entry <= 0: return

        pnl_pct = (current_price - entry) / entry
        
        # BREAKEVEN_TRIGGER = 0.012
        if not pos['is_breakeven_triggered'] and pnl_pct > 0.012:
            new_sl = round(entry * 1.002, 2) # 保本微利
            
            # 只有当新止损优于旧止损才修改
            if new_sl > pos['sl_price']:
                logger.info(f"💰 浮盈 {pnl_pct*100:.2f}%，触发保本 (SL -> {new_sl})")
                
                # 撤单 + 挂新单
                self.client.cancel_all_orders(self.symbol)
                self.client.place_order(self.symbol, 'SELL', pos['amount'], 'STOP_MARKET', stop_price=new_sl)
                self.client.place_order(self.symbol, 'SELL', pos['amount'], 'TAKE_PROFIT_MARKET', stop_price=pos['tp_price'])
                
                # 更新状态
                self.state.update_sl_tp(sl=new_sl, breakeven_triggered=True)

    def execute_dynamic_ladder(self, current_price):
        symbol = self.symbol
        qty = self.cfg['base_qty']
        
        # 1. 撤销该币种所有旧挂单，腾出位置
        self.client.cancel_all_orders(symbol)
        
        # 2. 第一笔：市价买入
        logger.info(f"🚀 [动态阶梯] 信号触发，首笔市价买入 {qty} {symbol}")
        self.client.place_order(symbol, 'BUY', qty, 'MARKET')
        
        # 3. 动态计算并提前挂出 7 个限价单
        # 逻辑：第1个间距 3.5%，之后每个 2%
        accumulated_gap = 0
        for i in range(1, 8):
            if i == 1:
                accumulated_gap += self.cfg['first_gap']
            else:
                accumulated_gap += self.cfg['subsequent_gap']
            
            target_price = round(current_price * (1 - accumulated_gap), 2)
            
            # 提前在订单簿“占位”
            res = self.client.place_order(symbol, 'BUY', qty, 'LIMIT', price=target_price)
            if res:
                logger.info(f"📍 阶梯单 {i} 挂出成功: {target_price} (距首笔 -{accumulated_gap*100:.1f}%)")
                
    def _close_all_and_cancel(self):
        """一键清理：全清仓位 + 撤销所有挂单"""
        self.client.cancel_all_orders(self.symbol)
        pos = self.client.get_position_risk(self.symbol)
        if pos and pos['amount'] > 0:
            # 市价全平
            self.client.place_order(self.symbol, 'SELL', pos['amount'], 'MARKET', reduce_only=True)
            self.state.clear_position()
            logger.info("✅ 账户已清空，等待下一轮信号。")

    # def check_overall_tp(self, current_price):
    #     """
    #     监控整体仓位：如果买到第二次补仓或达到25%利润，全部清仓
    #     """
    #     # 从交易所获取真实持仓情况（最准确）
    #     pos_info = self.client.send_request('GET', '/fapi/v2/positionRisk', {'symbol': self.symbol})
    #     if not pos_info: return
        
    #     target_pos = next((p for p in pos_info if p['symbol'] == self.symbol), None)
    #     if not target_pos: return

    #     amt = abs(float(target_pos['positionAmt']))
    #     entry_price = float(target_pos['entryPrice'])
        
    #     if amt == 0: return

    #     # 计算已触发了几次补仓
    #     # 初始 0.2, 如果 amt >= 0.4 说明至少触发了一次补仓 (即买到了第2个单子)
    #     num_orders_filled = round(amt / self.cfg['base_qty'])
        
    #     pnl_pct = (current_price - entry_price) / entry_price
        
    #     # 触发条件：
    #     # 1. 利润达到 25%
    #     # 2. 或者买到了第二次补仓 (即总数达到 base_qty * 2) 且有盈利
    #     # (你可以根据需求调整：是只要买到第二个就跑，还是买到第二个且反弹才跑)
    #     should_exit = False
    #     if pnl_pct >= self.cfg['tp_rate']:
    #         logger.info(f"💰 达到整体止盈目标 {self.cfg['tp_rate']*100}%，执行全清")
    #         should_exit = True
    #     elif num_orders_filled >= 2 and pnl_pct > 0.02: # 买到第二个且有微利
    #         logger.info(f"🔄 已触发第二次补仓 ({amt}) 且反弹，执行阶梯获利全清")
    #         should_exit = True

    #     if should_exit:
    #         self._close_all_and_cancel()

    def check_overall_tp(self, current_price):
        # 1. 从交易所获取实时仓位信息
        pos = self.client.get_position_risk(self.symbol)
        if not pos or pos['amount'] == 0:
            return

        total_amt = pos['amount']
        avg_entry = pos['entry_price']
        pnl_pct = (current_price - avg_entry) / avg_entry
        
        # 2. 计算当前成交了几倍的基础数量 (base_qty = 0.2)
        filled_count = round(total_amt / self.cfg['base_qty'])

        # 3. 动态止盈映射表 (关键：仓位越高，目标越低)
        tp_map = {
            1: 0.25,    # 1笔：25%
            2: 0.05,    # 2笔：5%
            3: 0.03,    # 3笔：3%
            4: 0.015,   # 4笔：1.5%
            5: 0.01,    # 5笔：1%
            6: 0.008,   # 6笔：0.8%
            7: 0.005,   # 7笔：0.5%
            8: 0.003    # 8笔：0.3% (保命离场)
        }

        # 获取当前层级对应的止盈目标，如果超过8笔则按第8笔算
        current_tp_target = tp_map.get(filled_count, 0.003)

        # 4. 判定是否执行清仓
        if pnl_pct >= current_tp_target:
            logger.info(f"💰 动态止盈触发！当前层级: {filled_count}, 目标: {current_tp_target*100}%, 当前利润: {pnl_pct*100:.2f}%")
            self._close_all_and_cancel()

    def tick(self, current_price):
        """
        主循环每次调用的入口
        """
        try:
            # 1. 信号检测 (Time-Stamp Check)
            sig = self.get_latest_signal()
            
            if sig is not None:
                new_time = sig['Time']
                
                # === 核心判定逻辑 ===
                if new_time != self.last_signal_time:
                    
                    # 移植你的时效性检查 (1800秒 = 30分钟)
                    # 注意：CSV里的时间是字符串，需要转 datetime
                    try:
                        # 假设 CSV 时间格式是 '2025-01-22 12:00:00'
                        sig_dt = pd.to_datetime(new_time)
                        # 如果是 naive time (无时区)，需要小心。这里简单做差
                        time_diff = (datetime.now() - sig_dt).total_seconds()
                        
                        if time_diff < 1800:
                            logger.info(f"⚡ [{self.name}] 新信号确认: {new_time} (延迟 {time_diff:.1f}s)")
                            self.execute_ladder_orders(sig, current_price)
                        else:
                            logger.warning(f"⚠️ [{self.name}] 信号过期 (延迟 {time_diff:.0f}s)，忽略。")
                    except Exception as e:
                        logger.error(f"时间解析错误: {e}")
                    
                    # 无论是否过期，都更新 last_time，避免重复处理
                    self.last_signal_time = new_time
            
            # 2. 风控检测
            self.check_overall_tp(current_price)

        except Exception as e:
            logger.error(f"[{self.name}] Tick Error: {e}")