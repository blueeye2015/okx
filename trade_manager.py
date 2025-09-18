import logging
import datetime
import pandas as pd
from trade_executor import TradingClient  # 假设您已将之前的交易执行器代码保存为 trade_executor.py
import clickhouse_connect # 用于获取市值

class TradeManager:
    def __init__(self, ch_client):
        """
        初始化交易管理器
        :param ch_client: 一个已连接的ClickHouse客户端实例
        """
        logging.info("初始化交易管理器...")
        self.trading_client = TradingClient()
        self.ch_client = ch_client

        # --- 策略核心参数 ---
        self.position_size_usdt = 100.0  # 规则2: 每个币买100 USDT
        self.stop_loss_pct = 0.08        # 规则1: 止损比例 -8%
        
        # 规则3: 最大持仓配置 (可根据回测调整)
        self.max_positions_config = {
            'small': 3,
            'mid': 1,
            'large': 1
        }
        self.total_max_positions = sum(self.max_positions_config.values())

        # 市值分层定义 (可调整)
        self.market_cap_tiers = {
            'small': (0, 500_000_000),             # 0 到 5亿
            'mid':   (500_000_000, 10_000_000_000), # 5亿 到 100亿
            'large': (10_000_000_000, float('inf')) # 100亿以上
        }

        # --- 运行时状态 ---
        self.open_positions = {}  # 用于存储当前持仓信息
        self.market_cap_data = self.load_market_caps() # 加载市值数据

    def load_market_caps(self):
        """从ClickHouse加载最新的市值数据"""
        logging.info("正在从ClickHouse加载市值数据...")
        try:
            df = self.ch_client.query_df('SELECT symbol, market_cap FROM marketdata.coin_info')
            # 将DataFrame转换为 符号 -> 市值 的字典，便于快速查找
            return pd.Series(df.market_cap.values, index=df.symbol).to_dict()
        except Exception as e:
            logging.error(f"加载市值数据失败: {e}")
            return {}

    def get_market_cap_category(self, symbol):
        """根据市值判断币种分类"""
        market_cap = self.market_cap_data.get(symbol.split('-')[0], 0)
        for category, (lower_bound, upper_bound) in self.market_cap_tiers.items():
            if lower_bound <= market_cap < upper_bound:
                return category
        return 'unknown'

    def process_signals(self, signals_df):
        """处理扫描器发现的新信号"""
        if signals_df.empty:
            return

        logging.warning(f"接收到 {len(signals_df)} 个新信号，开始处理...")

        # 1. 检查当前总持仓是否已满
        if len(self.open_positions) >= self.total_max_positions:
            logging.info(f"总持仓已达上限 ({self.total_max_positions})，忽略新信号。")
            return

        # 2. 为信号添加市值分类并排序
        signals_df['category'] = signals_df['symbol'].apply(self.get_market_cap_category)
        signals_df = signals_df[signals_df['category'] != 'unknown'] # 过滤掉未知市值的币
        signals_df = signals_df.sort_values('RVol', ascending=False) # 按RVol（信号强度）排序

        logging.info("信号按强度排序分类完成:\n" + signals_df.to_string())

        # 3. 计算各分类的持仓空位
        open_counts = {'small': 0, 'mid': 0, 'large': 0}
        for pos_symbol in self.open_positions:
            cat = self.get_market_cap_category(pos_symbol)
            if cat in open_counts:
                open_counts[cat] += 1
        
        # 4. 遍历排序后的信号，填补空位
        for _, signal in signals_df.iterrows():
            symbol = signal['symbol']
            category = signal['category']

            if symbol in self.open_positions:
                logging.info(f"[{symbol}] 已持有该币种，忽略信号。")
                continue

            # 检查该分类是否有空位，以及总持仓是否已满
            if open_counts[category] < self.max_positions_config[category] and \
               len(self.open_positions) < self.total_max_positions:
                
                logging.warning(f"[{symbol}] 符合建仓条件 (分类: {category})，准备执行买入。")
                self.execute_buy(signal)
                
                # 更新持仓计数
                open_counts[category] += 1
                if len(self.open_positions) >= self.total_max_positions:
                    logging.warning("总持仓已满，停止处理更多信号。")
                    break

    def execute_buy(self, signal):
        """执行买入操作"""
        symbol = signal['symbol']
        price = signal['current_price']
        
        # 计算下单数量
        size = self.position_size_usdt / price
        
        # 规则4: 执行市价单
        order_result = self.trading_client.place_market_order(symbol, 'buy', size)
        
        # 规则4: 如果API报错，此函数会返回None并记录错误，交易终止
        if order_result:
            # 简化处理：假设订单立即以当前价格成交
            # 生产环境需要查询订单确保完全成交，并获取真实成交价
            self.open_positions[symbol] = {
                'entry_price': price,
                'size': size,
                'entry_time': datetime.datetime.now()
            }
            logging.warning(f"成功建立仓位: {symbol} at ~{price:.4f}, size: {size}")

    def manage_open_positions(self, current_prices_df):
        """监控所有持仓，执行止损"""
        if not self.open_positions:
            return

        logging.info(f"开始监控 {len(self.open_positions)} 个持仓...")
        current_prices = pd.Series(current_prices_df.lastPx.values, index=current_prices_df.instId).to_dict()

        for symbol, position in list(self.open_positions.items()):
            current_price = float(current_prices.get(symbol, 0))
            if current_price == 0:
                logging.warning(f"未能获取到 [{symbol}] 的最新价格，本次跳过监控。")
                continue
            
            entry_price = position['entry_price']
            
            # 规则1: 检查 -8% 硬止损
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            
            if current_price <= stop_loss_price:
                logging.warning(f"[{symbol}] 触发止损! "
                                f"入场价: {entry_price:.4f}, 当前价: {current_price:.4f}, "
                                f"止损价: {stop_loss_price:.4f}. 准备平仓...")
                self.execute_sell(symbol, position['size'])

    def execute_sell(self, symbol, size):
        """执行卖出平仓操作"""
        order_result = self.trading_client.place_market_order(symbol, 'sell', size)
        if order_result:
            # 成功平仓后，从持仓字典中移除
            if symbol in self.open_positions:
                del self.open_positions[symbol]
            logging.warning(f"成功平仓: {symbol}")