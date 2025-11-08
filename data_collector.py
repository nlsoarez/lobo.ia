import requests
import pandas as pd

class DataCollector:
    def __init__(self, symbol, interval='1d', range_days='30d', token=None):
        self.symbol = symbol
        self.interval = interval
        self.range_days = range_days
        self.token = token
        self.base_url = f"https://brapi.dev/api/quote/{self.symbol}?range={self.range_days}&interval={self.interval}"
        if self.token:
            self.base_url += f"&token={self.token}"

    def fetch_data(self):
        response = requests.get(self.base_url)
        if response.status_code == 200:
            data = response.json()
            historical_data = data['results'][0]['historicalDataPrice']
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            return df
        else:
            print(f"Erro ao buscar dados: {response.status_code}")
            return pd.DataFrame()
