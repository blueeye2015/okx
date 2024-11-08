import json
import random
import time

class OrderBook:
    def __init__(self):
        self.bids = {}
        self.asks = {}

    def update_from_json(self, json_data):
        data = json.loads(json_data)
        book_data = data['data'][0]
        
        self.asks.clear()
        self.bids.clear()
        
        for ask in book_data['asks']:
            price, quantity = float(ask[0]), float(ask[1])
            self.asks[price] = quantity

        for bid in book_data['bids']:
            price, quantity = float(bid[0]), float(bid[1])
            self.bids[price] = quantity

    def add_order(self, side, price, quantity):
        if side == 'bid':
            if price in self.bids:
                self.bids[price] += quantity
            else:
                self.bids[price] = quantity
        elif side == 'ask':
            if price in self.asks:
                self.asks[price] += quantity
            else:
                self.asks[price] = quantity

    def remove_order(self, side, price, quantity):
        if side == 'bid' and price in self.bids:
            self.bids[price] -= quantity
            if self.bids[price] <= 0:
                del self.bids[price]
        elif side == 'ask' and price in self.asks:
            self.asks[price] -= quantity
            if self.asks[price] <= 0:
                del self.asks[price]

    def get_best_bid(self):
        return max(self.bids.keys()) if self.bids else None

    def get_best_ask(self):
        return min(self.asks.keys()) if self.asks else None

class HighFrequencyTrader:
    def __init__(self, order_book):
        self.order_book = order_book
        self.orders = []

    def calculate_order_price(self, side):
        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()
        if not best_bid or not best_ask:
            return None

        mid_price = (best_bid + best_ask) / 2

        if side == 'bid':
            return mid_price - (mid_price * 0.0001)  # 略低于中间价
        elif side == 'ask':
            return mid_price + (mid_price * 0.0001)  # 略高于中间价
        else:
            return mid_price

    def place_orders(self):
        self.cancel_orders()  # 先取消所有旧订单

        bid_price = self.calculate_order_price('bid')
        ask_price = self.calculate_order_price('ask')

        if bid_price and ask_price:
            bid_quantity = random.uniform(0.001, 0.01)
            ask_quantity = random.uniform(0.001, 0.01)

            self.order_book.add_order('bid', bid_price, bid_quantity)
            self.order_book.add_order('ask', ask_price, ask_quantity)

            self.orders.append(('bid', bid_price, bid_quantity))
            self.orders.append(('ask', ask_price, ask_quantity))

            print(f"下单: 买单 价格={bid_price:.2f}, 数量={bid_quantity:.6f}")
            print(f"下单: 卖单 价格={ask_price:.2f}, 数量={ask_quantity:.6f}")

    def cancel_orders(self):
        for order in self.orders:
            side, price, quantity = order
            self.order_book.remove_order(side, price, quantity)
            print(f"撤单: {side} 价格={price:.2f}, 数量={quantity:.6f}")
        self.orders.clear()

def simulate_market_movement(order_book):
    movement = random.uniform(-10, 10)
    for price in list(order_book.bids.keys()):
        new_price = price + movement
        order_book.bids[new_price] = order_book.bids.pop(price)
    for price in list(order_book.asks.keys()):
        new_price = price + movement
        order_book.asks[new_price] = order_book.asks.pop(price)

def main():
    json_data = '''{
        "arg": {
            "channel": "books5",
            "instId": "BTC-USDT"
        },
        "data": [
            {
                "asks": [
                    ["75863.9", "0.19995487", "0", "6"],
                    ["75864", "0.0691", "0", "1"],
                    ["75866.3", "0.01970546", "0", "2"],
                    ["75867", "0.01669247", "0", "1"],
                    ["75867.9", "0.11821371", "0", "2"]
                ],
                "bids": [
                    ["75863.8", "0.14206789", "0", "2"],
                    ["75863.6", "0.06967886", "0", "2"],
                    ["75863.4", "0.02633812", "0", "1"],
                    ["75861.3", "0.00001", "0", "1"],
                    ["75860.3", "0.02336859", "0", "1"]
                ],
                "instId": "BTC-USDT",
                "ts": "1731032227500",
                "seqId": 35957872352
            }
        ]
    }'''

    order_book = OrderBook()
    order_book.update_from_json(json_data)
    trader = HighFrequencyTrader(order_book)

    for i in range(10):
        print(f"\n时间步骤 {i + 1}")
        print(f"最佳买价: {order_book.get_best_bid():.2f}")
        print(f"最佳卖价: {order_book.get_best_ask():.2f}")

        trader.place_orders()
        time.sleep(0.5)  # 模拟高频交易的时间间隔

        simulate_market_movement(order_book)

if __name__ == "__main__":
    main()
