class PortfolioManager:
    def __init__(self, capital):
        self.capital = capital
        self.exposure = 0.03  # 3% por operação

    def calculate_position_size(self, price):
        position_size = (self.capital * self.exposure) / price
        return int(position_size)

    def update_capital(self, profit):
        self.capital += profit
        print(f"Capital atualizado: {self.capital}")
