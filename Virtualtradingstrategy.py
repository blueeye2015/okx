import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List, Dict
import ccxt
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

class VirtualTradingStrategy:
    def __init__(self, 
                 db_path: str,
                 initial_balance: float = 100000,  # 虚拟账户初始资金
                 stop_loss_threshold: float = 0.4,
                 take_profit_threshold: float = 0.3,  # 添加止盈阈值
                 max_holding_days: int = 30,
                 leverage: int = 3,
                 top_n: int = 10,
                 position_size_per_trade: float = 0.1):  # 每笔交易占总资金的比例
        
        self.db_path = db_path
        self.initial_balance = initial_balance
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.max_holding_days = max_holding_days
        self.leverage = leverage
        self.top_n = top_n
        self.position_size_per_trade = position_size_per_trade
        self.engine = create_engine('postgresql://postgres:12@localhost:5432/market_data')
        Session = sessionmaker(bind=self.engine)
        self.session = Session()     
                          

    def get_eligible_symbols(self) -> List[str]:
        """获取符合条件的交易对"""
                
        # 1. 获取正资金费率的币种
        funding_query = """
        SELECT symbol, fundingrate 
        FROM fundingrate 
        WHERE fundingtime = (select max(fundingtime) from fundingrate)
        and fundingrate >0
        """
        funding_df = pd.read_sql(funding_query, self.engine)
        
        # 2. 获取trend_records中连续20天为0的币种
        trend_query = """
        SELECT symbol
        FROM trend_records 
        WHERE timestamp >=NOW()-INTERVAL '21 days'
		AND consecutive_count=0
        GROUP BY symbol 
        HAVING COUNT(*) >= 20
        """
        trend_df = pd.read_sql(trend_query, self.engine)
        
        # 3. 从klines获取价格回落超过50%的币种
        klines_query = """
        SELECT symbol,highest_price,close,highest_price*1.0/close -1 as drop_ratio 
        FROM (
            SELECT symbol,close,timestamp,
                MAX(high) OVER (PARTITION BY symbol) as highest_price,
                FIRST_VALUE(timestamp) 
                OVER (PARTITION BY symbol ORDER BY high DESC) as highest_price_timestamp,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp desc) cnt
            FROM klines
            WHERE TIMESTAMP >= NOW()-INTERVAL '365 days'
            ) a
        WHERE cnt = 1
        """
        klines_df = pd.read_sql(klines_query, self.engine)
        price_drop_symbols = klines_df[
            (klines_df['drop_ratio'] ) > 0.5
        ]['symbol'].unique()
        
        # 找出满足所有条件的币种
        eligible_symbols = set(funding_df['symbol']) & set(trend_df['symbol']) & set(price_drop_symbols)
        
        # 按资金费率排序，返回top N
        if eligible_symbols:
            top_symbols = funding_df[
                funding_df['symbol'].isin(eligible_symbols)
            ].nlargest(self.top_n, 'funding_rate')['symbol'].tolist()
            return top_symbols
        return []

    def place_virtual_orders(self, symbols: List[str]):
        """创建虚拟订单"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for symbol in symbols:
            try:
                # 检查是否已经有相同币种的持仓
                cursor.execute('''
                SELECT COUNT(*) FROM virtual_active_positions WHERE symbol = ?
                ''', (symbol,))
                
                if cursor.fetchone()[0] > 0:
                    continue
                
                # 获取当前市场价格
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # 计算虚拟仓位大小
                position_size = self.initial_balance * self.position_size_per_trade
                
                # 记录虚拟订单
                cursor.execute('''
                INSERT INTO virtual_active_positions 
                (symbol, entry_price, entry_time, position_size, leverage, 
                last_price, last_update_time, unrealized_pnl, unrealized_roi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, current_price, datetime.now(), position_size, 
                    self.leverage, current_price, datetime.now(), 0.0, 0.0
                ))
                
            except Exception as e:
                print(f"虚拟下单失败 {symbol}: {str(e)}")
        
        conn.commit()
        conn.close()

    def monitor_virtual_positions(self):
        """监控虚拟持仓"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取所有活跃持仓
        cursor.execute('SELECT * FROM virtual_active_positions')
        active_positions = cursor.fetchall()
        
        for position in active_positions:
            position_id = position[0]
            symbol = position[1]
            entry_price = position[2]
            entry_time = datetime.strptime(position[3], '%Y-%m-%d %H:%M:%S.%f')
            position_size = position[4]
            
            try:
                # 获取当前价格
                current_price = self.exchange.fetch_ticker(symbol)['last']
                
                # 计算未实现盈亏 (空单)
                unrealized_pnl = (entry_price - current_price) * position_size * self.leverage
                unrealized_roi = unrealized_pnl / (position_size / self.leverage)
                
                # 更新持仓信息
                cursor.execute('''
                UPDATE virtual_active_positions 
                SET last_price = ?, last_update_time = ?, 
                    unrealized_pnl = ?, unrealized_roi = ?
                WHERE id = ?
                ''', (current_price, datetime.now(), unrealized_pnl, unrealized_roi, position_id))
                
                # 检查止损条件
                if unrealized_roi < -self.stop_loss_threshold:
                    self._close_virtual_position(position_id, current_price, "stop_loss")
                    continue
                
                # 检查止盈条件
                if unrealized_roi > self.take_profit_threshold:
                    self._close_virtual_position(position_id, current_price, "take_profit")
                    continue
                
                # 检查持仓时间
                holding_time = datetime.now() - entry_time
                if holding_time.days >= self.max_holding_days:
                    self._close_virtual_position(position_id, current_price, "timeout")
                    continue
                
            except Exception as e:
                print(f"监控失败 {symbol}: {str(e)}")
        
        conn.commit()
        conn.close()

    def _close_virtual_position(self, position_id: int, exit_price: float, reason: str):
        """平掉虚拟持仓"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取持仓信息
        cursor.execute('SELECT * FROM virtual_active_positions WHERE id = ?', (position_id,))
        position = cursor.fetchone()
        
        if position:
            symbol = position[1]
            entry_price = position[2]
            entry_time = position[3]
            position_size = position[4]
            leverage = position[5]
            
            # 计算最终盈亏
            pnl = (entry_price - exit_price) * position_size * leverage
            roi = pnl / (position_size / leverage)
            success = 1 if pnl > 0 else 0
            
            # 记录交易结果
            cursor.execute('''
            INSERT INTO virtual_trade_records 
            (symbol, entry_price, entry_time, exit_price, exit_time, 
            position_size, leverage, pnl, roi, status, close_reason, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, entry_price, entry_time, exit_price, datetime.now(),
                position_size, leverage, pnl, roi, 'closed', reason, success
            ))
            
            # 删除活跃持仓
            cursor.execute('DELETE FROM virtual_active_positions WHERE id = ?', (position_id,))
            
            # 更新策略性能记录
            self._update_strategy_performance()
        
        conn.commit()
        conn.close()

    def _update_strategy_performance(self):
        """更新策略性能统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        # 获取今日交易统计
        cursor.execute('''
        SELECT COUNT(*) as total,
               SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as wins,
               SUM(pnl) as daily_pnl
        FROM virtual_trade_records 
        WHERE date(exit_time) = date('now')
        ''')
        
        result = cursor.fetchone()
        total_trades, winning_trades, daily_pnl = result
        
        if total_trades:
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0
            
        # 获取当前余额
        cursor.execute('''
        SELECT SUM(pnl) FROM virtual_trade_records
        ''')
        total_pnl = cursor.fetchone()[0] or 0
        current_balance = self.initial_balance + total_pnl
        
        # 更新性能记录
        cursor.execute('''
        INSERT OR REPLACE INTO strategy_performance 
        (date, balance, daily_pnl, total_positions, winning_positions, win_rate)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (today, current_balance, daily_pnl, total_trades, winning_trades, win_rate))
        
        conn.commit()
        conn.close()

    def get_strategy_performance(self) -> Dict:
        """获取策略性能统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取总体统计
        cursor.execute('''
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as winning_trades,
            SUM(pnl) as total_pnl,
            AVG(CASE WHEN success = 1 THEN roi ELSE NULL END) as avg_win,
            AVG(CASE WHEN success = 0 THEN roi ELSE NULL END) as avg_loss
        FROM virtual_trade_records
        ''')
        
        result = cursor.fetchone()
        total_trades, winning_trades, total_pnl, avg_win, avg_loss = result
        
        # 获取当前活跃持仓数
        cursor.execute('SELECT COUNT(*) FROM virtual_active_positions')
        active_positions = cursor.fetchone()[0]
        
        # 计算其他指标
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        current_balance = self.initial_balance + (total_pnl or 0)
        total_return = (current_balance / self.initial_balance - 1) * 100
        
        conn.close()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'current_balance': current_balance,
            'active_positions': active_positions,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

        def run(self, check_interval: int = 300):
        """运行策略"""
        print("开始运行虚拟交易策略...")
        
        while True:
            try:
                # 获取符合条件的交易对
                eligible_symbols = self.get_eligible_symbols()
                print(f"找到 {len(eligible_symbols)} 个符合条件的交易对")
                
                # 创建虚拟订单
                self.place_virtual_orders(eligible_symbols)
                
                # 监控虚拟持仓
                self.monitor_virtual_positions()
                
                # 打印策略表现
                performance = self.get_strategy_performance()
                print(f"\n策略表现统计:")
                print(f"总交易次数: {performance['total_trades']}")
                print(f"盈利交易数: {performance['winning_trades']}")
                print(f"胜率: {performance['win_rate']*100:.2f}%")
                print(f"总盈亏: {performance['total_pnl']:.2f} USDT")
                print(f"当前余额: {performance['current_balance']:.2f} USDT")
                print(f"当前活跃持仓数: {performance['active_positions']}")
                
                # 打印活跃持仓详情
                self.print_active_positions()
                
                print(f"\n等待 {check_interval} 秒进行下一次检查...\n")
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"策略运行错误: {str(e)}")
                time.sleep(60)  # 发生错误时等待1分钟后继续

    def print_active_positions(self):
        """打印当前活跃持仓详情"""
        conn = sqlite3.connect(self.db_path)
        
        # 获取所有活跃持仓
        active_positions_df = pd.read_sql('''
        SELECT 
            symbol,
            entry_price,
            entry_time,
            position_size,
            leverage,
            last_price,
            last_update_time,
            unrealized_pnl,
            unrealized_pnl_percentage
        FROM virtual_active_positions
        ''', conn)
        
        if not active_positions_df.empty:
            print("\n当前活跃持仓:")
            for _, position in active_positions_df.iterrows():
                holding_time = datetime.now() - datetime.strptime(position['entry_time'], 
                                                                '%Y-%m-%d %H:%M:%S.%f')
                print(f"\n{position['symbol']}:")
                print(f"  入场价格: {position['entry_price']:.4f}")
                print(f"  当前价格: {position['last_price']:.4f}")
                print(f"  持仓规模: {position['position_size']:.2f}")
                print(f"  杠杆倍数: {position['leverage']}x")
                print(f"  未实现盈亏: {position['unrealized_pnl']:.2f} USDT ({position['unrealized_pnl_percentage']:.2f}%)")
                print(f"  持仓时间: {holding_time.days}天 {holding_time.seconds//3600}小时")
        else:
            print("\n当前没有活跃持仓")
        
        conn.close()

    def get_detailed_report(self, start_date=None, end_date=None):
        """获取详细的策略报告"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT 
            date(exit_time) as trade_date,
            COUNT(*) as daily_trades,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(pnl) as daily_pnl,
            AVG(pnl) as avg_trade_pnl,
            MAX(pnl) as best_trade,
            MIN(pnl) as worst_trade,
            AVG(CASE WHEN success = 1 THEN pnl ELSE NULL END) as avg_win,
            AVG(CASE WHEN success = 0 THEN pnl ELSE NULL END) as avg_loss
        FROM virtual_trade_records
        '''
        
        if start_date and end_date:
            query += f" WHERE date(exit_time) BETWEEN '{start_date}' AND '{end_date}'"
        
        query += " GROUP BY date(exit_time) ORDER BY date(exit_time)"
        
        daily_stats = pd.read_sql(query, conn)
        
        # 计算累计统计
        total_stats = {
            'total_trades': daily_stats['daily_trades'].sum(),
            'total_winning_trades': daily_stats['winning_trades'].sum(),
            'total_losing_trades': daily_stats['losing_trades'].sum(),
            'total_pnl': daily_stats['daily_pnl'].sum(),
            'avg_daily_trades': daily_stats['daily_trades'].mean(),
            'best_day_pnl': daily_stats['daily_pnl'].max(),
            'worst_day_pnl': daily_stats['daily_pnl'].min(),
            'best_trade': daily_stats['best_trade'].max(),
            'worst_trade': daily_stats['worst_trade'].min(),
            'avg_win': daily_stats['avg_win'].mean(),
            'avg_loss': daily_stats['avg_loss'].mean(),
        }
        
        total_stats['win_rate'] = (total_stats['total_winning_trades'] / 
                                 total_stats['total_trades'] if total_stats['total_trades'] > 0 else 0)
        
        total_stats['profit_factor'] = (abs(daily_stats[daily_stats['daily_pnl'] > 0]['daily_pnl'].sum()) / 
                                      abs(daily_stats[daily_stats['daily_pnl'] < 0]['daily_pnl'].sum())
                                      if abs(daily_stats[daily_stats['daily_pnl'] < 0]['daily_pnl'].sum()) > 0 else float('inf'))
        
        conn.close()
        
        return {
            'daily_stats': daily_stats,
            'total_stats': total_stats
        }


