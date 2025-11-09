import ta

class SignalAnalyzer:
    def __init__(self, data):
        self.data = data

    def _add_indicators(self):
        # Verifica se os dados possuem a coluna 'close'
        if 'close' not in self.data.columns:
            raise ValueError(f"Colunas disponíveis: {self.data.columns}. Esperado: 'close'.")
        
        # Calcula indicadores técnicos
        self.data['rsi'] = ta.momentum.RSIIndicator(close=self.data['close']).rsi()
        self.data['ema'] = ta.trend.EMAIndicator(close=self.data['close'], window=9).ema_indicator()

    def generate_signal(self):
        # Adiciona indicadores
        self._add_indicators()

        # Pega a última linha para análise
        last_row = self.data.iloc[-1]

        # Estratégia simples baseada em RSI e EMA
        if last_row['rsi'] < 30 and last_row['close'] > last_row['ema']:
            return {
                "symbol": "PETR4",
                "action": "BUY",
                "price": last_row['close'],
                "quantity": 10
            }
        elif last_row['rsi'] > 70 and last_row['close'] < last_row['ema']:
            return {
                "symbol": "PETR4",
                "action": "SELL",
                "price": last_row['close'],
                "quantity": 10
            }
        return None
