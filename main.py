from data_collector import DataCollector
from signal_analyzer import SignalAnalyzer
from trade_executor import TradeExecutor
from learning_module import LearningModule
from portfolio_manager import PortfolioManager
from logger import Logger

collector = DataCollector(symbol='PETR4', interval='1d', range_days='30d', token='jvAXu6Lhiwtavx9oZNNv5s')
data = collector.fetch_data()

analyzer = SignalAnalyzer(data)
signals = analyzer.generate_signals()

executor = TradeExecutor()
learner = LearningModule()
portfolio = PortfolioManager(capital=100000)
logger = Logger()

for index, row in signals.iterrows():
    signal = row['signal']
    if signal in ['buy', 'sell']:
        qty = portfolio.calculate_position_size(row['close'])
        order = executor.execute_order('PETR4', signal, row['close'], qty)
        profit = qty * (row['close'] * (0.01 if signal == 'buy' else -0.01))
        learner.record_trade({'profit': profit})
        portfolio.update_capital(profit)
        logger.log_trade({
            'symbol': order['symbol'],
            'date': row['date'].strftime('%Y-%m-%d'),
            'action': order['action'],
            'price': order['price'],
            'quantity': order['quantity'],
            'profit': profit,
            'indicators': f"RSI: {row['rsi']}, MACD: {row['macd']}",
            'notes': 'Simulação de operação'
        })

learner.evaluate_performance()
learner.adjust_strategy()
