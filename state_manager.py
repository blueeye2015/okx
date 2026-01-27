import json
import logging
import time
from datetime import datetime
import clickhouse_connect 
from config import Config

logger = logging.getLogger(__name__)

class ClickHouseStateManager:
    def __init__(self, strategy_name, client, symbol):
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.client = client  # 交易所API client，不是CK client
        
        # 默认初始状态
        self.state = {
            "strategy": strategy_name,
            "symbol": symbol,
            "position_side": "NONE",
            "amount": 0.0,
            "entry_price": 0.0,
            "sl_price": 0.0,
            "tp_price": 0.0,
            "is_breakeven_triggered": False
        }
        
        # 初始化 CK 连接
        # 建议把 CK 配置放入 Config 类: 
        # Config.CK_HOST='localhost', Config.CK_USER='default'...
        CLICKHOUSE = dict(host='localhost', port=8123, database='marketdata', username='default', password='12')
        self.ck = clickhouse_connect.get_client(**CLICKHOUSE)

        
        self.load_state()
        self.sync_with_exchange()

    def load_state(self):
        """
        从 ClickHouse 读取最新的一条日志
        """
        try:
            # 核心逻辑：按时间倒序取 LIMIT 1
            query = """
            SELECT state_json 
            FROM bot_state_log 
            WHERE strategy_name = %(name)s 
            ORDER BY event_time DESC 
            LIMIT 1
            """
            result = self.ck.query(query, {'name': self.strategy_name})
            
            if result:
                # ClickHouse 返回的是 tuple list，result[0][0] 是 JSON 字符串
                json_str = result.result_rows[0][0]
                self.state = json.loads(json_str)
                logger.info(f"[{self.strategy_name}] 状态已从 ClickHouse 恢复")
            else:
                logger.info(f"[{self.strategy_name}] CK 中无历史记录，初始化新状态")
                self.save_state() # 插入第一条初始记录
                
        except Exception as e:
            logger.error(f"ClickHouse 读取失败: {e}")

    def save_state(self):
        """
        核心逻辑：Append-Only (只插入，不修改)
        """
        try:
            now = datetime.now()
            json_str = json.dumps(self.state)
            
            # 修正：使用 insert 方法，数据格式为 list of lists
            data = [[now, self.strategy_name, self.symbol, json_str]]
            self.ck.insert('bot_state_log', data, column_names=['event_time', 'strategy_name', 'symbol', 'state_json'])
        except Exception as e:
            logger.error(f"ClickHouse 写入失败: {e}")

    # 以下逻辑完全不变，只要调用 save_state() 就会触发插入
    def update_open_position(self, side, amount, entry, sl, tp):
        self.state.update({
            "position_side": side,
            "amount": float(amount),
            "entry_price": float(entry),
            "sl_price": float(sl),
            "tp_price": float(tp),
            "is_breakeven_triggered": False
        })
        self.save_state()

    def update_sl_tp(self, sl=None, tp=None, breakeven_triggered=False):
        if sl: self.state['sl_price'] = sl
        if tp: self.state['tp_price'] = tp
        if breakeven_triggered: self.state['is_breakeven_triggered'] = True
        self.save_state()

    def clear_position(self):
        self.state.update({
            "position_side": "NONE",
            "amount": 0.0,
            "entry_price": 0.0,
            "sl_price": 0.0,
            "tp_price": 0.0,
            "is_breakeven_triggered": False
        })
        self.save_state()
        
    def sync_with_exchange(self):
        """
        启动时调用：从交易所拉取真实数据，防止本地文件和链上不一致
        """
        logger.info("🔄 正在与交易所同步仓位状态...")
        res = self.client.send_request('GET', '/fapi/v2/positionRisk', {'symbol': self.symbol})
        found_pos = False
        if res:
            for pos in res:
                amt = float(pos['positionAmt'])
                if amt != 0:
                    found_pos = True
                    side = 'LONG' if amt > 0 else 'SHORT'
                    self.state['position_side'] = side
                    self.state['amount'] = abs(amt)
                    self.state['entry_price'] = float(pos['entryPrice'])
                    # 注意：交易所API查不到当前的止损价，除非查openOrders
                    # 这里为了安全，如果是重启后发现有仓位，建议手动重置一下SL/TP逻辑，或者保持文件原样
                    logger.info(f"✅ 检测到远程持仓: {side} {abs(amt)} @ {pos['entryPrice']}")
                    break
        
        if not found_pos:
            logger.info("✅ 远程无持仓，重置本地状态。")
            self.clear_position()
        
        self.save_state()