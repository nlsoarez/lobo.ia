import pandas as pd
import yfinance as yf

class DataCollector:
    def __init__(self, symbol="PETR4.SA", period="1d", interval="5m"):
        self.symbol = symbol
        self.period = period
        self.interval = interval

    def get_data(self):
        df = yf.download(self.symbol, period=self.period, interval=self.interval)
        if df.empty:
            raise ValueError("Nenhum dado retornado. Verifique o símbolo ou conexão.")
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        df.dropna(inplace=True)
        return df
