import json

class BaseParser:
    def __init__(self, api_response):
        self.data = json.loads(api_response)
    
    def check_success(self):
        if self.data[self.SUCCESS_CODE_KEY] != self.SUCCESS_CODE_VALUE:
            raise Exception(f"API调用失败,错误代码: {self.data[self.SUCCESS_CODE_KEY]}")
    
    def parse(self):
        self.check_success()
        return [self.parse_item(item) for item in self.data[self.DATA_KEY]]
    
    def parse_item(self, item):
        raise NotImplementedError("Subclasses must implement parse_item method")

##持仓解析JSON字符串##
class PositionsParser(BaseParser):
    SUCCESS_CODE_KEY = 'code'
    SUCCESS_CODE_VALUE = '0'
    DATA_KEY = 'data'
    
    def parse_item(self, position):
        return {
            'symbol': position['instId'],
            'size': float(position['pos']),
            'side': 'buy' if float(position['pos']) > 0 else 'sell',
            'avg_price': float(position['avgPx']),
            'mark_price': float(position['markPx']),
            'liquidation_price': float(position['liqPx']),
            'unrealized_pnl': float(position['upl']),
            'uplRatio': float(position['uplRatio']),
            'margin': float(position['margin']),
            'leverage': float(position['lever']),
            'mgnRatio': float(position['mgnRatio'])            
        }

##未完成订单解析字符串json
class OrderlistParser(BaseParser):
    SUCCESS_CODE_KEY = 'code'
    SUCCESS_CODE_VALUE = '0'
    DATA_KEY = 'data'
    
    def parse_item(self, orderlist):
        return {
            'symbol': orderlist['instId'],
            'ordId': orderlist['ordId'],
            'size': float(orderlist['sz']),
            'side': orderlist['side'],
            'price': float(orderlist['px'])
        }

##成交
class TradesParser(BaseParser):
    SUCCESS_CODE_KEY = 'code'
    SUCCESS_CODE_VALUE = '0'
    DATA_KEY = 'data'
    
    def parse_item(self, trade):
        return {
            'symbol': trade['instId'],
            'tradeId': trade['tradeId'],
            'size': float(trade['sz']),
            'side': trade['side'],
            'price': float(trade['px']),
            'ts': trade['ts']
        }        

#获取保证金余额
class BalanceParser(BaseParser):
    SUCCESS_CODE_KEY = 'code'
    SUCCESS_CODE_VALUE = '0'
    DATA_KEY = 'data'
    
    def parse_item(self, balance):
        # 取出 details 中的第一个元素
        details = balance['details'][0]
        return {
            'availBal': float(details['availBal']),
            'totalEq': float(balance['totalEq']),
            'upl': float(details['upl']),
            'isoEq': float(balance['isoEq']),
            # 你可以根据需要添加更多字段
        }

# 余额调用
def parse_balance(api_response):
    parser = BalanceParser(api_response)
    return parser.parse()

# 持仓调用
def parse_positions(api_response):
    parser = PositionsParser(api_response)
    return parser.parse()

# 未完成订单调用
def parse_orderlist(api_response):
    parser = OrderlistParser(api_response)
    return parser.parse()

# 成交历史
def parse_historytrades(api_response):
    parser = TradesParser(api_response)
    return parser.parse()
