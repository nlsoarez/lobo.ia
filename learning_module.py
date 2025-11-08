class LearningModule:
    def __init__(self):
        self.history = []

    def record_trade(self, trade_result):
        self.history.append(trade_result)

    def evaluate_performance(self):
        total = sum([t['profit'] for t in self.history])
        print(f"Lucro total: {total}")
        return total

    def adjust_strategy(self):
        print("Ajustando estratégia com base no desempenho...")
