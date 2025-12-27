"""
Scanner de Criptomoedas - Analisa BTC, ETH e outras criptos.
Mercado 24/7 - Funciona mesmo quando B3 está fechada.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import ta

from config_loader import config
from system_logger import system_logger

# Importa CoinMarketCap client (opcional)
try:
    from coinmarketcap_client import CoinMarketCapClient
    HAS_CMC = True
except ImportError:
    HAS_CMC = False


# Lista de criptomoedas suportadas (Yahoo Finance usa -USD)
# Nota: MATIC-USD, UNI-USD e APT-USD foram substituidas por nao funcionarem no Yahoo Finance
CRYPTOCURRENCIES = {
    # Principais
    'BTC-USD': {'name': 'Bitcoin', 'category': 'major'},
    'ETH-USD': {'name': 'Ethereum', 'category': 'major'},

    # Top 10
    'BNB-USD': {'name': 'Binance Coin', 'category': 'top10'},
    'XRP-USD': {'name': 'Ripple', 'category': 'top10'},
    'SOL-USD': {'name': 'Solana', 'category': 'top10'},
    'ADA-USD': {'name': 'Cardano', 'category': 'top10'},
    'DOGE-USD': {'name': 'Dogecoin', 'category': 'top10'},
    'AVAX-USD': {'name': 'Avalanche', 'category': 'top10'},
    'DOT-USD': {'name': 'Polkadot', 'category': 'top10'},
    'TRX-USD': {'name': 'Tron', 'category': 'top10'},  # Substituiu MATIC

    # DeFi
    'LINK-USD': {'name': 'Chainlink', 'category': 'defi'},
    'MKR-USD': {'name': 'Maker', 'category': 'defi'},  # Substituiu UNI
    'AAVE-USD': {'name': 'Aave', 'category': 'defi'},

    # Layer 2
    'ARB-USD': {'name': 'Arbitrum', 'category': 'layer2'},
    'OP-USD': {'name': 'Optimism', 'category': 'layer2'},

    # Outras
    'ATOM-USD': {'name': 'Cosmos', 'category': 'other'},
    'LTC-USD': {'name': 'Litecoin', 'category': 'other'},
    'FIL-USD': {'name': 'Filecoin', 'category': 'other'},
    'XLM-USD': {'name': 'Stellar', 'category': 'other'},  # Substituiu APT
    'NEAR-USD': {'name': 'NEAR Protocol', 'category': 'other'},
}


class CryptoScanner:
    """
    Scanner de criptomoedas com analise tecnica.
    Funciona 24/7 - ideal para quando B3 esta fechada.
    """

    def __init__(self):
        """Inicializa o scanner de criptomoedas."""
        crypto_config = config.get('crypto', {})

        self.enabled = crypto_config.get('enabled', True)
        self.symbols = crypto_config.get('symbols', ['BTC-USD', 'ETH-USD'])
        self.top_n = crypto_config.get('top_coins', 10)
        self.max_workers = crypto_config.get('max_workers', 5)
        self.interval = crypto_config.get('interval', '1h')  # Cripto usa intervalos maiores
        self.period = crypto_config.get('period', '7d')

        # Pesos para ranking
        self.weights = crypto_config.get('weights', {
            'signal_strength': 0.30,
            'volume_score': 0.20,
            'trend_score': 0.25,
            'volatility_score': 0.15,
            'momentum_score': 0.10,
        })

        # Cache
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(minutes=crypto_config.get('cache_minutes', 5))

        # CoinMarketCap client (dados em tempo real mais precisos)
        self.cmc_enabled = HAS_CMC
        self.cmc_client = CoinMarketCapClient() if HAS_CMC else None
        self._cmc_data = {}  # Cache de dados CMC

        cmc_status = "ATIVO" if self.cmc_enabled and self.cmc_client and self.cmc_client.api_key else "DESATIVADO"
        system_logger.info(f"Crypto Scanner inicializado - {len(CRYPTOCURRENCIES)} criptomoedas disponiveis")
        system_logger.info(f"CoinMarketCap API: {cmc_status}")

    def scan_crypto_market(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Escaneia mercado de criptomoedas.

        Args:
            force_refresh: Forçar atualização.

        Returns:
            Lista de criptomoedas com análise.
        """
        if not self.enabled:
            return []

        # Verifica cache
        if not force_refresh and self._is_cache_valid():
            return self._cache.get('results', [])

        system_logger.info("Iniciando scan do mercado de criptomoedas...")

        # Busca dados do CoinMarketCap (se disponivel)
        self._fetch_cmc_data()
        start_time = datetime.now()

        results = []
        failed = []

        symbols_to_scan = list(CRYPTOCURRENCIES.keys())

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._analyze_crypto, symbol): symbol
                for symbol in symbols_to_scan
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    failed.append(symbol)
                    system_logger.debug(f"Erro ao analisar {symbol}: {e}")

        # Ordena por score
        results.sort(key=lambda x: x['total_score'], reverse=True)

        # Atualiza cache
        self._cache = {'results': results}
        self._cache_time = datetime.now()

        elapsed = (datetime.now() - start_time).total_seconds()
        system_logger.info(
            f"Scan crypto completo em {elapsed:.1f}s - "
            f"{len(results)} criptos analisadas, {len(failed)} falhas"
        )

        return results

    def _is_cache_valid(self) -> bool:
        """Verifica se cache é válido."""
        if self._cache_time is None:
            return False
        return datetime.now() - self._cache_time < self._cache_duration

    def _fetch_cmc_data(self):
        """Busca dados do CoinMarketCap se disponivel."""
        if not self.cmc_enabled or not self.cmc_client or not self.cmc_client.api_key:
            return

        try:
            # Busca cotacoes para todas as criptos
            symbols = list(CRYPTOCURRENCIES.keys())
            quotes = self.cmc_client.get_quotes(symbols)

            if quotes:
                self._cmc_data = quotes
                system_logger.info(f"CMC: {len(quotes)} cotacoes carregadas")

            # Busca overview do mercado
            overview = self.cmc_client.get_market_overview()
            if overview:
                self._cmc_data['_market_overview'] = overview
                sentiment = overview.get('market_sentiment', 'NEUTRAL')
                fear_greed = overview.get('fear_greed', {})
                if fear_greed:
                    system_logger.info(
                        f"CMC: Mercado {sentiment} | "
                        f"Fear&Greed: {fear_greed.get('score', 'N/A')} ({fear_greed.get('classification', 'N/A')})"
                    )

        except Exception as e:
            system_logger.debug(f"Erro ao buscar dados CMC: {e}")

    def _analyze_crypto(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Analisa uma criptomoeda.

        Args:
            symbol: Símbolo (ex: BTC-USD).

        Returns:
            Dicionário com análise ou None.
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=self.period, interval=self.interval)

            if hist.empty or len(hist) < 20:
                return None

            current_price = hist['Close'].iloc[-1]
            avg_volume = hist['Volume'].mean()

            # Calcula indicadores tecnicos (Yahoo Finance)
            analysis = self._calculate_indicators(hist)
            if analysis is None:
                return None

            # Calcula scores
            scores = self._calculate_scores(hist, analysis)

            # Score total
            total_score = sum(
                scores.get(key, 0) * weight
                for key, weight in self.weights.items()
            )

            crypto_info = CRYPTOCURRENCIES.get(symbol, {'name': symbol, 'category': 'other'})

            # Dados base do Yahoo Finance
            result = {
                'symbol': symbol,
                'name': crypto_info['name'],
                'category': crypto_info['category'],
                'price': current_price,
                'avg_volume': avg_volume,
                'change_1h': self._calculate_change(hist, 1),
                'change_24h': self._calculate_change(hist, 24),
                'change_7d': self._calculate_change(hist, 168),  # 7 * 24
                **analysis,
                **scores,
                'total_score': total_score,
                'signal': self._determine_signal(analysis, scores),
                'analyzed_at': datetime.now().isoformat(),
                'data_source': 'yahoo_finance',
            }

            # Enriquece com dados do CoinMarketCap (se disponivel)
            if symbol in self._cmc_data:
                cmc = self._cmc_data[symbol]
                result['price'] = cmc.get('price', current_price)  # Preco mais preciso
                result['change_1h'] = cmc.get('change_1h', result['change_1h'])
                result['change_24h'] = cmc.get('change_24h', result['change_24h'])
                result['change_7d'] = cmc.get('change_7d', result['change_7d'])
                result['change_30d'] = cmc.get('change_30d', 0)
                result['market_cap'] = cmc.get('market_cap', 0)
                result['volume_24h'] = cmc.get('volume_24h', 0)
                result['volume_change_24h'] = cmc.get('volume_change_24h', 0)
                result['data_source'] = 'coinmarketcap+yahoo'

            # Ajusta score baseado no sentimento de mercado CMC
            if '_market_overview' in self._cmc_data:
                overview = self._cmc_data['_market_overview']
                sentiment = overview.get('market_sentiment', 'NEUTRAL')
                fear_greed = overview.get('fear_greed', {})

                # Ajusta score baseado no Fear & Greed Index
                fg_score = fear_greed.get('score', 50)
                if fg_score < 30:  # Extreme Fear - bom para comprar
                    total_score = min(100, total_score + 10)
                elif fg_score > 70:  # Extreme Greed - cuidado
                    total_score = max(0, total_score - 10)

                result['total_score'] = total_score
                result['market_sentiment'] = sentiment
                result['fear_greed_index'] = fg_score
                result['signal'] = self._determine_signal(analysis, scores)

            return result

        except Exception as e:
            system_logger.debug(f"Erro ao analisar {symbol}: {e}")
            return None

    def _calculate_indicators(self, hist: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calcula indicadores técnicos para cripto."""
        try:
            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']

            # RSI
            rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
            current_rsi = rsi.iloc[-1]

            # EMAs
            ema_12 = ta.trend.EMAIndicator(close, window=12).ema_indicator()
            ema_26 = ta.trend.EMAIndicator(close, window=26).ema_indicator()
            ema_50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()

            # MACD
            macd = ta.trend.MACD(close)
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()
            macd_hist = macd.macd_diff()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close)
            bb_high = bb.bollinger_hband()
            bb_low = bb.bollinger_lband()
            bb_pct = (close - bb_low) / (bb_high - bb_low)

            # ATR
            atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

            # Volume ratio
            vol_sma = volume.rolling(window=20).mean()
            vol_ratio = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1

            return {
                'rsi': current_rsi,
                'ema_12': ema_12.iloc[-1],
                'ema_26': ema_26.iloc[-1],
                'ema_50': ema_50.iloc[-1] if len(ema_50.dropna()) > 0 else ema_26.iloc[-1],
                'macd': macd_line.iloc[-1],
                'macd_signal': macd_signal.iloc[-1],
                'macd_hist': macd_hist.iloc[-1],
                'bb_pct': bb_pct.iloc[-1],
                'atr': atr.iloc[-1],
                'atr_pct': (atr.iloc[-1] / close.iloc[-1]) * 100,
                'volume_ratio': vol_ratio,
            }

        except Exception as e:
            system_logger.debug(f"Erro ao calcular indicadores: {e}")
            return None

    def _calculate_scores(self, hist: pd.DataFrame, analysis: Dict[str, float]) -> Dict[str, float]:
        """Calcula scores para ranking."""
        scores = {}

        # Signal Strength Score
        rsi = analysis['rsi']
        signal_score = 50

        if rsi < 30:
            signal_score += 30  # Oversold
        elif rsi < 40:
            signal_score += 15
        elif rsi > 70:
            signal_score -= 20  # Overbought

        if analysis['macd_hist'] > 0:
            signal_score += 20
        if analysis['macd'] > analysis['macd_signal']:
            signal_score += 15

        if analysis['bb_pct'] < 0.2:
            signal_score += 15

        scores['signal_strength'] = max(0, min(100, signal_score))

        # Volume Score
        vol_ratio = analysis['volume_ratio']
        if vol_ratio > 2:
            scores['volume_score'] = 100
        elif vol_ratio > 1.5:
            scores['volume_score'] = 80
        elif vol_ratio > 1:
            scores['volume_score'] = 60
        else:
            scores['volume_score'] = vol_ratio * 60

        # Trend Score
        price = hist['Close'].iloc[-1]
        ema_12 = analysis['ema_12']
        ema_26 = analysis['ema_26']

        trend_score = 50
        if price > ema_12 > ema_26:
            trend_score += 30
        elif price > ema_26:
            trend_score += 15
        if ema_12 > ema_26:
            trend_score += 10

        scores['trend_score'] = max(0, min(100, trend_score))

        # Volatility Score (cripto é mais volátil, ajuste diferente)
        atr_pct = analysis['atr_pct']
        if 2 <= atr_pct <= 8:
            scores['volatility_score'] = 100
        elif 1 <= atr_pct <= 12:
            scores['volatility_score'] = 70
        else:
            scores['volatility_score'] = 40

        # Momentum Score
        change_24h = self._calculate_change(hist, 24)
        momentum = 50
        if change_24h > 0:
            momentum += min(30, change_24h * 3)
        else:
            momentum += max(-30, change_24h * 3)
        scores['momentum_score'] = max(0, min(100, momentum))

        return scores

    def _calculate_change(self, hist: pd.DataFrame, periods: int) -> float:
        """Calcula variação percentual."""
        if len(hist) < periods + 1:
            return 0.0
        current = hist['Close'].iloc[-1]
        past = hist['Close'].iloc[-periods-1]
        if past == 0:
            return 0.0
        return ((current - past) / past) * 100

    def _determine_signal(self, analysis: Dict[str, float], scores: Dict[str, float]) -> str:
        """Determina sinal de trading."""
        signal_strength = scores.get('signal_strength', 0)
        rsi = analysis.get('rsi', 50)

        if signal_strength >= 70 and rsi < 35:
            return 'STRONG_BUY'
        elif signal_strength >= 55 and rsi < 50:
            return 'BUY'
        elif signal_strength <= 30 or rsi > 75:
            return 'SELL'
        else:
            return 'HOLD'

    def get_btc_eth_analysis(self) -> Dict[str, Any]:
        """
        Retorna análise específica de BTC e ETH.

        Returns:
            Dicionário com análise de BTC e ETH.
        """
        results = self.scan_crypto_market()

        btc = next((r for r in results if r['symbol'] == 'BTC-USD'), None)
        eth = next((r for r in results if r['symbol'] == 'ETH-USD'), None)

        return {
            'btc': btc,
            'eth': eth,
            'market_sentiment': self._get_market_sentiment(results)
        }

    def _get_market_sentiment(self, results: List[Dict]) -> str:
        """Calcula sentimento geral do mercado crypto."""
        if not results:
            return 'NEUTRAL'

        buy_signals = len([r for r in results if 'BUY' in r.get('signal', '')])
        sell_signals = len([r for r in results if r.get('signal') == 'SELL'])

        if buy_signals > len(results) * 0.6:
            return 'BULLISH'
        elif sell_signals > len(results) * 0.4:
            return 'BEARISH'
        return 'NEUTRAL'

    def get_top_opportunities(self, n: int = None) -> List[Dict[str, Any]]:
        """Retorna top N oportunidades de cripto."""
        results = self.scan_crypto_market()
        n = n or self.top_n
        return [r for r in results if 'BUY' in r.get('signal', '')][:n]

    def get_market_overview(self) -> Dict[str, Any]:
        """
        Retorna visao geral do mercado crypto (CMC + analise tecnica).

        Returns:
            Dicionario com overview do mercado
        """
        overview = {
            'timestamp': datetime.now().isoformat(),
            'source': 'yahoo_finance',
        }

        # Se tiver dados CMC
        if '_market_overview' in self._cmc_data:
            cmc_overview = self._cmc_data['_market_overview']
            overview['source'] = 'coinmarketcap'
            overview['market_sentiment'] = cmc_overview.get('market_sentiment', 'NEUTRAL')
            overview['fear_greed'] = cmc_overview.get('fear_greed', {})
            overview['global_metrics'] = cmc_overview.get('global_metrics', {})

        # Analise propria
        results = self.scan_crypto_market()
        buy_signals = len([r for r in results if 'BUY' in r.get('signal', '')])
        sell_signals = len([r for r in results if r.get('signal') == 'SELL'])

        overview['total_analyzed'] = len(results)
        overview['buy_signals'] = buy_signals
        overview['sell_signals'] = sell_signals
        overview['technical_sentiment'] = self._get_market_sentiment(results)

        return overview

    def print_report(self, top_n: int = 10):
        """Imprime relatório do mercado crypto."""
        results = self.scan_crypto_market()[:top_n]

        print("\n" + "=" * 80)
        print("CRYPTO SCANNER - MELHORES OPORTUNIDADES")
        print("=" * 80)

        # Mostra dados CMC se disponiveis
        if '_market_overview' in self._cmc_data:
            overview = self._cmc_data['_market_overview']
            fear_greed = overview.get('fear_greed', {})
            global_metrics = overview.get('global_metrics', {})

            if fear_greed:
                print(f"\n{fear_greed.get('emoji', '')} Fear & Greed Index: {fear_greed.get('score', 'N/A')} ({fear_greed.get('classification', 'N/A')})")

            if global_metrics:
                total_mc = global_metrics.get('total_market_cap', 0)
                btc_dom = global_metrics.get('btc_dominance', 0)
                if total_mc:
                    print(f"   Market Cap Total: ${total_mc/1e12:.2f}T | BTC Dominancia: {btc_dom:.1f}%")

        print(f"\n{'Rank':<5} {'Simbolo':<12} {'Nome':<15} {'Preco':<12} {'Score':<8} {'Sinal':<12} {'24h':<10}")
        print("-" * 80)

        for i, r in enumerate(results, 1):
            source_icon = "🔷" if 'coinmarketcap' in r.get('data_source', '') else "📊"
            print(
                f"{i:<5} {r['symbol']:<12} "
                f"{r['name'][:14]:<15} "
                f"${r['price']:<11,.2f} "
                f"{r['total_score']:<8.1f} "
                f"{r['signal']:<12} "
                f"{r['change_24h']:>+8.2f}%"
            )

        print("=" * 80)

        # BTC e ETH destacados
        btc_eth = self.get_btc_eth_analysis()
        print(f"\nSentimento do mercado: {btc_eth['market_sentiment']}")
        if btc_eth['btc']:
            print(f"BTC: ${btc_eth['btc']['price']:,.2f} ({btc_eth['btc']['change_24h']:+.2f}%) - {btc_eth['btc']['signal']}")
        if btc_eth['eth']:
            print(f"ETH: ${btc_eth['eth']['price']:,.2f} ({btc_eth['eth']['change_24h']:+.2f}%) - {btc_eth['eth']['signal']}")

        # Mostra fonte de dados
        print(f"\nFonte de dados: {'CoinMarketCap + Yahoo Finance' if self._cmc_data else 'Yahoo Finance'}")


# Instância global
crypto_scanner = CryptoScanner()


if __name__ == "__main__":
    scanner = CryptoScanner()
    scanner.print_report(15)
