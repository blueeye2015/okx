# event/event.py

class Event:
    """
    所有事件类型的基类 (父类)。
    """
    pass


class SignalEvent(Event):
    """
    交易信号事件。
    
    这是我们最重要的事件，用于从策略模块 (momentum_trader) 
    向执行模块 (trade_executor) 发送具体的交易指令。
    """
    def __init__(self, instId, side, size):
        """
        初始化一个交易信号。

        Args:
            instId (str): 产品ID, e.g., 'BTC-USDT-SWAP'.
            side (str): 交易方向, 'buy' or 'sell'.
            size (float): 委托数量.
        """
        self.type = 'SIGNAL'
        self.instId = instId
        self.side = side
        self.size = size