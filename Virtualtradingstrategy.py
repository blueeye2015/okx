import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List, Dict
import ccxt
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import logging

class VirtualTradingStrategy:
    def __init__(self, 
                 db_path: str,
                 initial_balance: float = 10000,  # 虚拟账户初始资金
                 stop_loss_threshold: float = 0.4,
                 take_profit_threshold: float = 1,  # 添加止盈阈值
                 max_holding_days: int = 30,
                 leverage: int = 3,
                 top_n: int = 20,
                 position_size_per_trade: float = 0.05):  # 每笔交易占总资金的比例
        
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
        load_dotenv('D:\OKex-API\.env')
        
        proxies = {
            'http': 'http://127.0.0.1:7890',  # 根据您的实际代理地址修改
            'https': 'http://127.0.0.1:7890'  # 根据您的实际代理地址修改
        }
        # 初始化交易所API（这里以binance为例）
        self.exchange = ccxt.okx({
            'apiKey': os.getenv('API_KEY'),
            'secret': os.getenv('SECRET_KEY'),
            'password': os.getenv('PASSPHRASE'),
            'enableRateLimit': True,
            'proxies': proxies,  # 添加代理设置
            'timeout': 30000,    # 设置超时时间为30秒
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })     
                          

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
                
        for symbol in symbols:
            try:
                # 检查是否已经有相同币种的持仓
                query = text('''
                SELECT COUNT(*) FROM virtual_active_positions WHERE symbol = :symbol
                ''')
                df = pd.read_sql(query, self.engine, params={'symbol': symbol})
                if df.empty:
                    logging.error(f"未找到订单信息:  symbol={self.symbol}")
                # 获取当前市场价格
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # 计算虚拟仓位大小
                position_size = self.initial_balance * self.position_size_per_trade
                
                # 记录虚拟订单
                sql = text('''
                INSERT INTO virtual_active_positions 
                (symbol, entry_price, entry_time, position_size, leverage, 
                last_price, last_update_time, unrealized_pnl, unrealized_roi)
                VALUES (:symbol, :current_price, :entry_time, :position_size, 
                    :leverage, :current_price, :last_update_time, :unrealized_pnl, :unrealized_roi)
                ''')
                
                self.session.execute(
                    sql,
                    {
                        'symbol': symbol,
                        'entry_price': current_price,
                        'entry_time': datetime.now(),
                        'position_size': position_size,
                        'leverage': self.leverage,
                        'last_price': current_price,
                        'last_update_time': datetime.now(),
                        'unrealized_pnl': 0.0,
                        'unrealized_roi': 0.0
                    }
                )
                self.session.commit()
                
            except Exception as e:
                print(f"虚拟下单失败 {symbol}: {str(e)}")
        
        

    def monitor_virtual_positions(self):
        """监控虚拟持仓"""
        
        
        # 获取所有活跃持仓
        query = text('SELECT * FROM virtual_active_positions')
        df = pd.read_sql(query, self.engine)
        
        for position in df:
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
                current_time = datetime.now()
                sql = '''
                UPDATE virtual_active_positions 
                SET last_price = :current_price, last_update_time = :current_time, 
                    unrealized_pnl = :unrealized_pnl, unrealized_roi = :unrealized_roi
                WHERE id = :position_id
                '''
                
                self.session.execute
                (
                    text(sql),
                    {
                        'symbol': symbol,
                        'updated_at': current_time
                    }
                )
                
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
        
      

    def _close_virtual_position(self, position_id: int, exit_price: float, reason: str):
        """平掉虚拟持仓"""
        
        # 获取持仓信息
        query = text('SELECT * FROM virtual_active_positions WHERE id = :position_id')
        df = pd.read_sql(query, self.engine, params={'position_id': position_id})
        
        if df:
            symbol = df[1]
            entry_price = df[2]
            entry_time = df[3]
            position_size = df[4]
            leverage = df[5]
            
            # 计算最终盈亏
            pnl = (entry_price - exit_price) * position_size * leverage
            roi = pnl / (position_size / leverage)
            success = 1 if pnl > 0 else 0
            
            # 记录交易结果
            sql ='''
            INSERT INTO virtual_trade_records 
            (symbol, entry_price, entry_time, exit_price, exit_time, 
            position_size, leverage, pnl, roi, status, close_reason, success)
            VALUES (:symbol, :entry_price, :entry_time, :exit_price, :exit_time,
            :position_size, :leverage, :pnl, :roi, :status, :reason, :success
            )
            '''
            
            self.session.execute(
                    sql,
                    {
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'entry_time': entry_time,
                        'exit_price': exit_price,
                        'exit_time': datetime.now(),
                        'position_size': position_size,
                        'leverage': leverage,
                        'pnl': pnl,
                        'roi': roi,
                        'status': 'closed',
                        'reason': reason,
                        'success': success
                    }
                )
            self.session.commit()
            # 删除活跃持仓
            sql = '''DELETE FROM virtual_active_positions WHERE id = :position_id'''
            self.session.execute(
                sql,
                {position_id}
            )
            
            # 更新策略性能记录
            self._update_strategy_performance()
        
    def _update_strategy_performance(self):
        """更新策略性能统计"""
        
        today = datetime.now().date()
        
        # 获取今日交易统计
        sql = '''
        SELECT COUNT(*) as total,
               SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as wins,
               SUM(pnl) as daily_pnl
        FROM virtual_trade_records 
        WHERE date(exit_time) = date('now')
        '''
        
        df = pd.read_sql(sql, self.engine)
        total, wins, daily_pnl = df
        
        if total:
            win_rate = wins / total
        else:
            win_rate = 0
            
        # 获取当前余额
        query = '''
        SELECT SUM(pnl) FROM virtual_trade_records
        '''
        total_pnl = pd.read_sql(query, self.engine)
        current_balance = self.initial_balance + total_pnl
        
        # 更新性能记录
        query1 = '''
        INSERT INTO strategy_performance 
        (strategy_name, date, balance, daily_pnl, total_positions, winning_positions, win_rate)
        VALUES (:strategy_name, :today, :balance, :daily_pnl, :total, :wins, :win_rate)
        ON CONFLICT (strategy_name,date) 
        DO UPDATE SET 
            balance = EXCLUDED.balance
            daily_pnl = EXCLUDED.daily_pnl,
            total_positions = EXCLUDED.total_positions,
            winning_positions = EXCLUDED.winning_positions,
            win_rate = EXCLUDED.win_rate
        '''
        self.session.execute(
                    query1,
                    {
                        'strategy_name': 'ma60short',
                        'today': today,
                        'balance': current_balance,
                        'daily_pnl': daily_pnl,
                        'total': total,
                        'wins': wins,
                        'win_rate': win_rate
                    }
                )
      

    def get_strategy_performance(self) -> Dict:
        """获取策略性能统计"""
        
        # 获取总体统计
        query = '''
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as winning_trades,
            SUM(pnl) as total_pnl,
            AVG(CASE WHEN success = 1 THEN roi ELSE NULL END) as avg_win,
            AVG(CASE WHEN success = 0 THEN roi ELSE NULL END) as avg_loss
        FROM virtual_trade_records
        '''
        df = pd.read_sql(query, self.engine)
        
        total_trades, winning_trades, total_pnl, avg_win, avg_loss = df
        
        # 获取当前活跃持仓数
        query1 = 'SELECT COUNT(*) FROM virtual_active_positions'
        active_positions = pd.read_sql(query1, self.engine)
        
        # 计算其他指标
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        current_balance = self.initial_balance + (total_pnl or 0)
        total_return = (current_balance / self.initial_balance - 1) * 100
        
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
        logging.basicConfig(filename=f'ma60short_strategy.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('脚本开始执行')
        
        while True:
            try:
                # 获取符合条件的交易对
                eligible_symbols = self.get_eligible_symbols()
                logging.info(f"找到 {len(eligible_symbols)} 个符合条件的交易对")
                
                # 创建虚拟订单
                self.place_virtual_orders(eligible_symbols)
                
                # 监控虚拟持仓
                self.monitor_virtual_positions()
                
                # 打印策略表现
                performance = self.get_strategy_performance()
                logging.info(f"\n策略表现统计:")
                logging.info(f"总交易次数: {performance['total_trades']}")
                logging.info(f"盈利交易数: {performance['winning_trades']}")
                logging.info(f"胜率: {performance['win_rate']*100:.2f}%")
                logging.info(f"总盈亏: {performance['total_pnl']:.2f} USDT")
                logging.info(f"当前余额: {performance['current_balance']:.2f} USDT")
                logging.info(f"当前活跃持仓数: {performance['active_positions']}")
                
                # 打印活跃持仓详情
                self.print_active_positions()
                
                logging.info(f"\n等待 {check_interval} 秒进行下一次检查...\n")
                time.sleep(check_interval)
                
            except Exception as e:
                logging.info(f"策略运行错误: {str(e)}")
                time.sleep(60)  # 发生错误时等待1分钟后继续

    def print_active_positions(self):
        """打印当前活跃持仓详情"""
        
        
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
        ''', self.engine)
        
        if not active_positions_df.empty:
            logging.info("\n当前活跃持仓:")
            for _, position in active_positions_df.iterrows():
                holding_time = datetime.now() - datetime.strptime(position['entry_time'], 
                                                                '%Y-%m-%d %H:%M:%S.%f')
                logging.info(f"\n{position['symbol']}:")
                logging.info(f"  入场价格: {position['entry_price']:.4f}")
                logging.info(f"  当前价格: {position['last_price']:.4f}")
                logging.info(f"  持仓规模: {position['position_size']:.2f}")
                logging.info(f"  杠杆倍数: {position['leverage']}x")
                logging.info(f"  未实现盈亏: {position['unrealized_pnl']:.2f} USDT ({position['unrealized_pnl_percentage']:.2f}%)")
                logging.info(f"  持仓时间: {holding_time.days}天 {holding_time.seconds//3600}小时")
        else:
            logging.info("\n当前没有活跃持仓")
        

    def get_detailed_report(self, start_date=None, end_date=None):
        """获取详细的策略报告"""
                
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
        
        daily_stats = pd.read_sql(query, self.engine)
        
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
        
        
        return {
            'daily_stats': daily_stats,
            'total_stats': total_stats
        }


