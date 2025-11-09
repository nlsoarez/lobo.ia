import pandas as pd
import yfinance as yf

class DataCollector:
    def __init__(self, symbol="PETR4.SA", period="1d", interval="5m"):
        self.symbol = symbol
        self.period = period
        self.interval = interval

    def get_data(self):
        # Coleta dados do Yahoo Finance
        df = yf.download(self.symbol, period=self.period, interval=self.interval)

        # Verifica se retornou dados
        if df.empty:
            raise ValueError(f"Nenhum dado retornado para {self.symbol}. Verifique o símbolo ou conexão.")

        # Renomeia colunas para padrão esperado
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        # Remove linhas com valores nulos
        df.dropna(inplace=True)

        return df
