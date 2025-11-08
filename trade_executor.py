class TradeExecutor:
    def __init__(self):
        self.orders = []

    def execute_order(self, symbol, signal, price, quantity):
        order = {
            'symbol': symbol,
            'action': signal,
            'price': price,
            'quantity': quantity
        }
        self.orders.append(order)
        print(f"Executando ordem: {order}")
        return order
