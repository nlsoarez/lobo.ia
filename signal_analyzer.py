import ta

class SignalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.data = self._add_indicators()

    def _add_indicators(self):
        self.data['rsi'] = ta.momentum.RSIIndicator(close=self.data['close']).rsi()
        self.data['macd'] = ta.trend.macd_diff(self.data['close'])
        self.data['ema'] = ta.trend.EMAIndicator(close=self.data['close'], window=14).ema_indicator()
        self.data['sma'] = ta.trend.SMAIndicator(close=self.data['close'], window=14).sma_indicator()
        return self.data

    def generate_signals(self):
        signals = []
        for i in range(1, len(self.data)):
            if self.data['rsi'][i] < 30 and self.data['macd'][i] > 0:
                signals.append('buy')
            elif self.data['rsi'][i] > 70 and self.data['macd'][i] < 0:
                signals.append('sell')
            else:
                signals.append('hold')
        signals.insert(0, 'hold')
        self.data['signal'] = signals
        return self.data
