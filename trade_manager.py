# trade_manager.py (最终再平衡增强版)
import logging
import datetime
import pandas as pd
import time
#from trade_executor import TradingClient
import uuid

class TradeManager:
    def __init__(self, ch_client, trading_client, dry_run=True):
        # ... (初始化部分与之前完全相同)
        logging.info("初始化交易管理器...")
        self.trading_client = trading_client  ##不再自己创建，而是使用传入的实例
        self.ch_client = ch_client
        self.dry_run = dry_run
        self.position_size_usdt = 100.0
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.25
        self.max_positions_config = {'small': 3, 'mid': 1, 'large': 1}
        self.total_max_positions = sum(self.max_positions_config.values())
        self.market_cap_tiers = {
            'small': (0, 500_000_000),
            'mid': (500_000_000, 10_000_000_000),
            'large': (10_000_000_000, float('inf'))
        }
        self.open_positions = {}
        self.market_cap_data = self.load_market_caps()

        # --- 【新!】再平衡参数 ---
        # 如果一个持仓币种的新排名低于这个名次，就主动卖出
        self.rebalance_rank_threshold = 100 

        # --- 【核心升级】运行时状态 ---
        self.cooldown_period_hours = 24*2
        # self.cooldown_list 不再是字典，而是一个set，用于快速内存查找
        self.cooldown_list = set()

        self.load_open_positions_from_db() ##获取仓位
        self.load_active_cooldowns_from_db() # 【新!】启动时加载冷却列表


    def load_active_cooldowns_from_db(self):
        """【新!】在启动时从数据库加载所有仍在冷却期内的币种。"""
        logging.warning("正在从数据库恢复有效的冷却列表...")
        try:
            # now() 函数需要 ClickHouse 21.8 或更高版本
            query = "SELECT DISTINCT symbol FROM marketdata.cooldown_list WHERE cooldown_expiry > now('Asia/Shanghai')"
            cooldown_df = self.ch_client.query_df(query)
            
            if not cooldown_df.empty:
                self.cooldown_list = set(cooldown_df['symbol'].tolist())
                logging.warning(f"成功恢复了 {len(self.cooldown_list)} 个仍在冷却期的币种: {self.cooldown_list}")
            else:
                logging.warning("数据库中没有需要恢复的冷却币种。")
        except Exception as e:
            logging.error(f"从数据库恢复冷却列表失败: {e}", exc_info=True)
            # 即使恢复失败，程序也可以继续，只是冷却机制暂时失效

    def load_open_positions_from_db(self):
        """
        【最终修正版】在启动时只加载 side='buy' 且 status='OPEN' 的订单。
        这是最可靠的状态恢复方式。
        """
        logging.warning("正在从数据库恢复未平仓头寸...")
        try:
            query = """
            SELECT symbol, entry_price, size, created_at, client_order_id
            FROM marketdata.orders
            WHERE status = 'OPEN' AND side = 'buy'
            """
            open_orders_df = self.ch_client.query_df(query)

            if open_orders_df.empty:
                logging.warning("数据库中没有需要恢复的未平仓头寸。")
                return

            for _, row in open_orders_df.iterrows():
                symbol = row['symbol']
                self.open_positions[symbol] = {
                    'entry_price': float(row['entry_price']),
                    'size': float(row['size']),
                    'entry_time': pd.to_datetime(row['created_at']),
                    'client_order_id': row['client_order_id']
                }
            logging.warning(f"成功恢复了 {len(self.open_positions)} 个未平仓头寸: {list(self.open_positions.keys())}")

        except Exception as e:
            logging.error(f"从数据库恢复头寸失败: {e}", exc_info=True)
            raise SystemExit("无法恢复持仓状态，程序终止以防意外交易。")
        
    def load_market_caps(self):
        # ... (此函数无需修改)
        logging.info("正在从ClickHouse加载市值数据...")
        try:
            df = self.ch_client.query_df('SELECT symbol, market_cap FROM marketdata.coin_info')
            return pd.Series(df.market_cap.values, index=df.symbol).to_dict()
        except Exception as e:
            logging.error(f"加载市值数据失败: {e}")
            return {}

    def get_market_cap_category(self, symbol):
        # ... (此函数无需修改)
        base_currency = symbol.split('-')[0]
        market_cap = self.market_cap_data.get(base_currency, 0)
        for category, (lower_bound, upper_bound) in self.market_cap_tiers.items():
            if lower_bound <= market_cap < upper_bound:
                return category
        return 'unknown'

    def process_new_signals(self, signals_df):
        """
        【最终增强版】处理新信号，包含"再平衡"逻辑。
        """
        if signals_df.empty:
            logging.warning("信号列表为空，跳过处理。")
            return
        logging.warning(f"接收到 {len(signals_df)} 个新信号，开始进行投资组合再平衡...")

        # ###################### 1. 执行“末位淘汰”（再平衡） ######################
        if self.open_positions:
            # 创建一个币种->排名的字典，便于快速查找
            rank_map = {row['symbol']: index for index, row in signals_df.iterrows()}
            
            # 必须使用 list(self.open_positions.items()) 来创建一个副本进行遍历，因为我们可能会在循环中删除字典元素
            for symbol, position in list(self.open_positions.items()):
                new_rank = rank_map.get(symbol)
                
                if new_rank is None or new_rank > self.rebalance_rank_threshold:
                    reason = f"新排名({new_rank})低于阈值({self.rebalance_rank_threshold})"
                    logging.warning(f"[{symbol}] 触发再平衡卖出条件: {reason}")
                    self.execute_sell(symbol, position, reason) # 调用统一的卖出函数

        # ###################### 2. 执行“择优上车”（与之前逻辑相同） ######################
        logging.info("开始用最优信号填补可用仓位...")
        if len(self.open_positions) >= self.total_max_positions:
            logging.info(f"总持仓已达上限 ({self.total_max_positions})，无需买入新币种。")
            return

        signals_df['category'] = signals_df['symbol'].apply(self.get_market_cap_category)
        signals_df = signals_df[signals_df['category'] != 'unknown']
        
        open_counts = {cat: 0 for cat in self.max_positions_config}
        for pos_symbol in self.open_positions:
            cat = self.get_market_cap_category(pos_symbol)
            if cat in open_counts: open_counts[cat] += 1
        
        for _, signal in signals_df.iterrows():
            symbol = signal['symbol']
            category = signal['category']

            if symbol in self.open_positions:
                continue
            
            # 【核心新增】建仓前的冷却检查 (现在查询内存中的set)
            if symbol in self.cooldown_list:
                logging.info(f"[{symbol}] 处于止损冷却期，忽略买入信号。")
                continue

            can_open_position = (
                open_counts[category] < self.max_positions_config[category] and
                len(self.open_positions) < self.total_max_positions
            )

            if can_open_position:
                logging.warning(f"[{symbol}] 符合建仓条件 (分类: {category}, 排名: {signals_df[signals_df.symbol==symbol].index[0]})，准备执行买入。")
                self.execute_buy(signal)
                open_counts[category] += 1

            if len(self.open_positions) >= self.total_max_positions:
                logging.warning("总持仓已满，停止处理本轮剩余信号。")
                break
    
    def execute_buy(self, signal):
        """执行买入操作，并更新持仓字典"""
        symbol = signal['symbol']
        entry_price = self.trading_client.get_latest_price(symbol)
        # 下单前余额检查
        if not self.dry_run:
            available_balance = self.trading_client.get_usdt_balance()
            if available_balance < self.position_size_usdt:
                logging.error(f"[{symbol}] 下单取消: 余额不足。需要: {self.position_size_usdt} USDT, 可用: {available_balance} USDT")
                return
            
        size = self.position_size_usdt / entry_price
        client_order_id = f"buy{symbol.split('-')[0]}{uuid.uuid4().hex[:10]}"

        
        if self.dry_run:
            order_result = {'code': '0', 'data': [{'sCode': '0'}]} 
            logging.warning(f"--- [空跑模式] 模拟买入 {self.position_size_usdt} USDT worth of {symbol} ---")
        else:
            order_result = self.trading_client.place_limit_order(
                symbol, 'buy', size, entry_price, client_order_id
            )

        if order_result and order_result.get('data') and order_result['data'][0].get('sCode') == '0':
            self.open_positions[symbol] = {
                'entry_price': signal['current_price'],
                'size': size,
                'entry_time': datetime.datetime.now()
            }
            logging.warning(f"成功建立仓位 (或模拟): {symbol}")

            # 2. 【核心新增】将新订单写入数据库
            try:
                # ###################### 核心修正点在这里 ######################
                # 1. 定义列的顺序，必须与数据库表结构完全一致
                column_names = [
                    'client_order_id', 'exchange_order_id', 'symbol', 'status', 'side', 'order_type',
                    'usdt_amount', 'entry_price', 'exit_price', 'size', 'created_at', 'updated_at',
                    'pnl_usdt', 'pnl_pct', 'notes'
                ]
                order_values = [
                    client_order_id, order_result['data'][0].get('ordId', ''), symbol,
                    'OPEN', 'buy', 'market', size, entry_price,
                    0.0, size, datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))),
                    datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))),
                    0.0, 0.0, 'New position opened by momentum strategy'
                ]

                # 3. 使用 column_names 参数进行插入
                self.ch_client.insert('marketdata.orders', [order_values], column_names=column_names)
                # #############################################################
                logging.info(f"[{symbol}] 新订单记录已成功写入数据库。")
            except Exception as e:
                logging.error(f"[{symbol}] 严重错误：下单成功但写入数据库失败！: {e}", exc_info=True)
                # 此处应加入警报机制，例如发送邮件或钉钉消息
        else:
            logging.error(f"[{symbol}] 下单失败或API返回错误: {order_result.get('msg', '未知错误')}")


    def execute_sell(self, symbol, position_data, reason):
        """【新!】统一的卖出函数，并更新持仓字典
        【已升级】平仓时，如果是止损，则向数据库插入一条冷却记录。
        """
        sell_client_id = f"close{symbol.split('-')[0]}{uuid.uuid4().hex[:10]}"
        open_client_id = position_data['client_order_id']

         # 获取最新价格以计算盈亏
        latest_prices = self.trading_client.get_latest_price(symbol)
        exit_price = latest_prices.get(symbol, position_data['entry_price']) # 如果获取失败，用成本价计算
        
        if self.dry_run:
            order_result = {'code': '0', 'data': [{'sCode': '0'}]}
            logging.warning(f"--- [空跑模式] 模拟卖出 {position_data['size']:.6f} of {symbol} ---")
        else:
            order_result = self.trading_client.place_market_order_by_size(
                symbol, 'sell', position_data['size'], sell_client_id
            ) 
        
        if order_result and order_result.get('data') and order_result['data'][0].get('sCode') == '0':
            logging.warning(f"成功平仓 (或模拟): {symbol} | 原因: {reason}")

            # ###################### 【核心新增】触发持久化冷却 ######################
            if '止损' in reason:
                cooldown_expiry = datetime.datetime.now() + datetime.timedelta(hours=self.cooldown_period_hours)
                try:
                    self.ch_client.insert('marketdata.cooldown_list', 
                                          [[symbol, cooldown_expiry]], 
                                          column_names=['symbol', 'cooldown_expiry'])
                    # 同时更新内存中的set
                    self.cooldown_list.add(symbol)
                    logging.warning(f"[{symbol}] 已被加入持久化冷却列表，直到 {cooldown_expiry.strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception as e:
                    logging.error(f"[{symbol}] 严重错误：写入冷却记录到数据库失败！: {e}", exc_info=True)

            pnl_usdt = (exit_price - position_data['entry_price']) * position_data['size']
            pnl_pct = (exit_price / position_data['entry_price']) - 1 if position_data['entry_price'] > 0 else 0
            try:
                column_names = [
                    'client_order_id', 'exchange_order_id', 'symbol', 'status', 'side', 'order_type',
                    'usdt_amount', 'entry_price', 'exit_price', 'size', 'created_at', 'updated_at',
                    'pnl_usdt', 'pnl_pct', 'notes'
                ]
                order_values = [
                    sell_client_id, order_result['data'][0].get('ordId', ''), symbol,
                    'CLOSED', 'sell', 'market', exit_price * position_data['size'],
                    position_data['entry_price'], exit_price, position_data['size'],
                    datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))),
                    datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))),
                    pnl_usdt, pnl_pct, f"Closing position opened by {open_client_id}. Reason: {reason}"
                ]
                self.ch_client.insert('marketdata.orders', [order_values], column_names=column_names)
                logging.info(f"[{symbol}] 新的 'sell' 订单记录已成功写入数据库。")
            except Exception as e:
                 logging.error(f"[{symbol}] 严重错误：平仓成功但写入'sell'记录失败！: {e}", exc_info=True)

            # 从持仓字典中移除
            if symbol in self.open_positions:
                del self.open_positions[symbol]            
        else:
            logging.error(f"[{symbol}] 平仓失败或API返回错误: {order_result.get('msg', '未知错误')}")

    def manage_open_positions(self):
        """监控所有持仓，仅执行止盈止损。"""
        # ... (此函数无需修改，它的职责很清晰)
        if not self.open_positions: return

        symbols_to_check = list(self.open_positions.keys())
        logging.info(f"开始监控 {len(symbols_to_check)} 个持仓的止盈止损: {symbols_to_check}")
        # ###################### 核心修正点在这里 ######################
        # 1. 创建一个空字典来存储价格
        current_prices = {}
        # 2. 循环遍历持仓列表，逐个获取价格
        for symbol in symbols_to_check:
            price = self.trading_client.get_latest_price(symbol)
            if price is not None:
                current_prices[symbol] = price
            # 增加一个微小的延时，以避免在持仓很多时触发API限频
            time.sleep(0.1) 
        # #############################################################
        if not current_prices: return

        for symbol, position in list(self.open_positions.items()):
            current_price = current_prices.get(symbol)
            if not current_price: continue

            pnl_pct = (current_price / position['entry_price']) - 1
            should_close, close_reason = False, ""

            if pnl_pct >= self.take_profit_pct:
                should_close, close_reason = True, f"止盈({pnl_pct:.2%})"
            if pnl_pct <= -self.stop_loss_pct:
                should_close, close_reason = True, f"止损({pnl_pct:.2%})"

            if should_close:
                logging.warning(f"[{symbol}] 触发平仓条件: {close_reason}")
                self.execute_sell(symbol, position, close_reason)