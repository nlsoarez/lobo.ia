"""
Cliente CoinMarketCap API para dados de criptomoedas em tempo real.
Fornece precos, market cap, volume e metricas de mercado.
"""

import os
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time

from system_logger import system_logger


class CoinMarketCapClient:
    """
    Cliente para API do CoinMarketCap.

    Limites do plano gratuito:
    - 10.000 creditos/mes
    - 30 requisicoes/minuto
    - Sem dados historicos
    """

    BASE_URL = "https://pro-api.coinmarketcap.com/v1"

    # Mapeamento de simbolos Yahoo Finance -> CMC
    SYMBOL_MAP = {
        'BTC-USD': 'BTC',
        'ETH-USD': 'ETH',
        'BNB-USD': 'BNB',
        'XRP-USD': 'XRP',
        'SOL-USD': 'SOL',
        'ADA-USD': 'ADA',
        'DOGE-USD': 'DOGE',
        'AVAX-USD': 'AVAX',
        'DOT-USD': 'DOT',
        'TRX-USD': 'TRX',
        'LINK-USD': 'LINK',
        'MKR-USD': 'MKR',
        'AAVE-USD': 'AAVE',
        'ARB-USD': 'ARB',
        'OP-USD': 'OP',
        'ATOM-USD': 'ATOM',
        'LTC-USD': 'LTC',
        'FIL-USD': 'FIL',
        'XLM-USD': 'XLM',
        'NEAR-USD': 'NEAR',
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa o cliente CoinMarketCap.

        Args:
            api_key: Chave da API (ou usa variavel de ambiente CMC_API_KEY)
        """
        self.api_key = api_key or os.environ.get('CMC_API_KEY', '')

        if not self.api_key:
            system_logger.warning("CMC_API_KEY nao configurada")

        self.headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.api_key,
        }

        # Cache para reduzir requisicoes
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=2)  # Cache de 2 minutos

        # Rate limiting
        self._last_request_time = None
        self._min_request_interval = 2.0  # 2 segundos entre requisicoes (30/min)

        # Estatisticas
        self.requests_made = 0
        self.credits_used = 0

    def _rate_limit(self):
        """Aplica rate limiting para respeitar limite de 30 req/min."""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = datetime.now()

    def _is_cache_valid(self, key: str) -> bool:
        """Verifica se cache ainda e valido."""
        if key not in self._cache_time:
            return False
        return datetime.now() - self._cache_time[key] < self._cache_duration

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Faz requisicao para API do CMC.

        Args:
            endpoint: Endpoint da API
            params: Parametros da query

        Returns:
            Resposta JSON ou None em caso de erro
        """
        if not self.api_key:
            return None

        # Verifica cache
        cache_key = f"{endpoint}_{str(params)}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        # Aplica rate limiting
        self._rate_limit()

        try:
            url = f"{self.BASE_URL}{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=10)

            self.requests_made += 1

            if response.status_code == 200:
                data = response.json()

                # Atualiza creditos usados (estimativa)
                if 'status' in data:
                    self.credits_used = data['status'].get('credit_count', 0)

                # Atualiza cache
                self._cache[cache_key] = data
                self._cache_time[cache_key] = datetime.now()

                return data
            else:
                system_logger.warning(
                    f"CMC API erro {response.status_code}: {response.text[:200]}"
                )
                return None

        except requests.exceptions.Timeout:
            system_logger.warning("CMC API timeout")
            return None
        except Exception as e:
            system_logger.error(f"CMC API erro: {e}")
            return None

    def get_latest_listings(self, limit: int = 20, convert: str = 'USD') -> List[Dict]:
        """
        Obtem listagem das principais criptomoedas.

        Args:
            limit: Numero de criptos (max 5000)
            convert: Moeda de conversao

        Returns:
            Lista de criptomoedas com dados de mercado
        """
        params = {
            'start': 1,
            'limit': limit,
            'convert': convert,
            'sort': 'market_cap',
            'sort_dir': 'desc',
        }

        response = self._make_request('/cryptocurrency/listings/latest', params)

        if not response or 'data' not in response:
            return []

        cryptos = []
        for item in response['data']:
            quote = item.get('quote', {}).get(convert, {})

            cryptos.append({
                'id': item.get('id'),
                'name': item.get('name'),
                'symbol': item.get('symbol'),
                'slug': item.get('slug'),
                'rank': item.get('cmc_rank'),
                'price': quote.get('price', 0),
                'volume_24h': quote.get('volume_24h', 0),
                'market_cap': quote.get('market_cap', 0),
                'change_1h': quote.get('percent_change_1h', 0),
                'change_24h': quote.get('percent_change_24h', 0),
                'change_7d': quote.get('percent_change_7d', 0),
                'change_30d': quote.get('percent_change_30d', 0),
                'market_cap_dominance': quote.get('market_cap_dominance', 0),
                'last_updated': quote.get('last_updated'),
            })

        system_logger.info(f"CMC: {len(cryptos)} criptos carregadas")
        return cryptos

    def get_quotes(self, symbols: List[str], convert: str = 'USD') -> Dict[str, Dict]:
        """
        Obtem cotacoes para simbolos especificos.

        Args:
            symbols: Lista de simbolos (ex: ['BTC', 'ETH'])
            convert: Moeda de conversao

        Returns:
            Dicionario {simbolo: dados}
        """
        # Converte simbolos Yahoo Finance para CMC
        cmc_symbols = []
        for sym in symbols:
            if sym in self.SYMBOL_MAP:
                cmc_symbols.append(self.SYMBOL_MAP[sym])
            elif '-USD' in sym:
                cmc_symbols.append(sym.replace('-USD', ''))
            else:
                cmc_symbols.append(sym)

        params = {
            'symbol': ','.join(cmc_symbols),
            'convert': convert,
        }

        response = self._make_request('/cryptocurrency/quotes/latest', params)

        if not response or 'data' not in response:
            return {}

        quotes = {}
        for symbol, data in response['data'].items():
            quote = data.get('quote', {}).get(convert, {})

            # Mapeia de volta para formato Yahoo Finance
            yf_symbol = f"{symbol}-USD"

            quotes[yf_symbol] = {
                'id': data.get('id'),
                'name': data.get('name'),
                'symbol': symbol,
                'price': quote.get('price', 0),
                'volume_24h': quote.get('volume_24h', 0),
                'market_cap': quote.get('market_cap', 0),
                'change_1h': quote.get('percent_change_1h', 0),
                'change_24h': quote.get('percent_change_24h', 0),
                'change_7d': quote.get('percent_change_7d', 0),
                'change_30d': quote.get('percent_change_30d', 0),
                'volume_change_24h': quote.get('volume_change_24h', 0),
                'last_updated': quote.get('last_updated'),
            }

        return quotes

    def get_global_metrics(self) -> Optional[Dict]:
        """
        Obtem metricas globais do mercado crypto.

        Returns:
            Dicionario com metricas globais
        """
        response = self._make_request('/global-metrics/quotes/latest')

        if not response or 'data' not in response:
            return None

        data = response['data']
        quote = data.get('quote', {}).get('USD', {})

        return {
            'active_cryptocurrencies': data.get('active_cryptocurrencies'),
            'total_cryptocurrencies': data.get('total_cryptocurrencies'),
            'active_exchanges': data.get('active_exchanges'),
            'total_market_cap': quote.get('total_market_cap'),
            'total_volume_24h': quote.get('total_volume_24h'),
            'btc_dominance': data.get('btc_dominance'),
            'eth_dominance': data.get('eth_dominance'),
            'defi_volume_24h': quote.get('defi_volume_24h'),
            'defi_market_cap': quote.get('defi_market_cap'),
            'stablecoin_volume_24h': quote.get('stablecoin_volume_24h'),
            'last_updated': quote.get('last_updated'),
        }

    def get_fear_greed_index(self) -> Optional[Dict]:
        """
        Calcula indice de medo/ganancia baseado em metricas.
        (Aproximacao - CMC nao tem endpoint oficial para isso)

        Returns:
            Dicionario com indice e classificacao
        """
        # Busca BTC como proxy do mercado
        quotes = self.get_quotes(['BTC-USD'])

        if 'BTC-USD' not in quotes:
            return None

        btc = quotes['BTC-USD']

        # Calcula indice baseado em variacao (simplificado)
        change_24h = btc.get('change_24h', 0)
        change_7d = btc.get('change_7d', 0)

        # Score de 0 a 100
        # Variacao positiva = ganancia, negativa = medo
        score = 50 + (change_24h * 2) + (change_7d * 0.5)
        score = max(0, min(100, score))

        if score < 25:
            classification = 'Extreme Fear'
            emoji = '😱'
        elif score < 40:
            classification = 'Fear'
            emoji = '😨'
        elif score < 60:
            classification = 'Neutral'
            emoji = '😐'
        elif score < 75:
            classification = 'Greed'
            emoji = '😀'
        else:
            classification = 'Extreme Greed'
            emoji = '🤑'

        return {
            'score': round(score),
            'classification': classification,
            'emoji': emoji,
            'btc_change_24h': change_24h,
            'btc_change_7d': change_7d,
        }

    def get_market_overview(self) -> Dict:
        """
        Retorna visao geral do mercado crypto.

        Returns:
            Dicionario com overview completo
        """
        overview = {
            'timestamp': datetime.now().isoformat(),
            'global_metrics': None,
            'fear_greed': None,
            'top_cryptos': [],
            'market_sentiment': 'NEUTRAL',
        }

        # Metricas globais
        global_metrics = self.get_global_metrics()
        if global_metrics:
            overview['global_metrics'] = global_metrics

        # Fear & Greed Index
        fear_greed = self.get_fear_greed_index()
        if fear_greed:
            overview['fear_greed'] = fear_greed

            # Define sentimento baseado no indice
            score = fear_greed['score']
            if score < 35:
                overview['market_sentiment'] = 'BEARISH'
            elif score > 65:
                overview['market_sentiment'] = 'BULLISH'

        # Top criptos
        top_cryptos = self.get_latest_listings(limit=10)
        overview['top_cryptos'] = top_cryptos

        return overview

    def get_api_status(self) -> Dict:
        """
        Retorna status de uso da API.

        Returns:
            Dicionario com estatisticas de uso
        """
        return {
            'api_key_configured': bool(self.api_key),
            'requests_made': self.requests_made,
            'credits_used_estimate': self.credits_used,
            'monthly_limit': 10000,
            'rate_limit': '30 req/min',
            'cache_duration': str(self._cache_duration),
        }

    def print_market_report(self):
        """Imprime relatorio do mercado."""
        print("\n" + "=" * 70)
        print("COINMARKETCAP - RELATORIO DO MERCADO CRYPTO")
        print("=" * 70)

        # Status da API
        status = self.get_api_status()
        if not status['api_key_configured']:
            print("\n⚠️ API Key nao configurada!")
            print("Configure a variavel CMC_API_KEY ou passe no construtor.")
            return

        # Fear & Greed
        fg = self.get_fear_greed_index()
        if fg:
            print(f"\n{fg['emoji']} Fear & Greed Index: {fg['score']} ({fg['classification']})")

        # Metricas globais
        metrics = self.get_global_metrics()
        if metrics:
            print(f"\n📊 METRICAS GLOBAIS:")
            print(f"   Market Cap Total: ${metrics['total_market_cap']/1e12:.2f}T")
            print(f"   Volume 24h: ${metrics['total_volume_24h']/1e9:.2f}B")
            print(f"   Dominancia BTC: {metrics['btc_dominance']:.1f}%")
            print(f"   Dominancia ETH: {metrics['eth_dominance']:.1f}%")

        # Top 10
        top = self.get_latest_listings(10)
        if top:
            print(f"\n🏆 TOP 10 CRIPTOMOEDAS:")
            print(f"{'Rank':<5} {'Simbolo':<8} {'Preco':<12} {'24h':<10} {'7d':<10} {'Market Cap'}")
            print("-" * 65)
            for c in top:
                print(
                    f"{c['rank']:<5} {c['symbol']:<8} "
                    f"${c['price']:<11,.2f} "
                    f"{c['change_24h']:>+8.2f}% "
                    f"{c['change_7d']:>+8.2f}% "
                    f"${c['market_cap']/1e9:.1f}B"
                )

        print("\n" + "=" * 70)
        print(f"Requisicoes: {self.requests_made} | Creditos usados: ~{self.credits_used}")
        print("=" * 70)


# Instancia global (configurada via ambiente)
cmc_client = CoinMarketCapClient()


if __name__ == "__main__":
    # Teste do cliente
    client = CoinMarketCapClient()
    client.print_market_report()
