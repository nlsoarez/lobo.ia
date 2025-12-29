"""
Market Scanner - Escaneia todo o mercado B3 para identificar melhores oportunidades.
Analisa volume, tendência, volatilidade e indicadores técnicos.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import ta

from config_loader import config
from system_logger import system_logger


def get_close_column(df):
    """
    Retorna a coluna de preço de fechamento (case-insensitive).
    Resolve problema 'Close' vs 'close'.
    """
    if df is None or df.empty:
        return None

    for col in ['Close', 'close', 'CLOSE', 'last', 'Last']:
        if col in df.columns:
            return df[col]

    # Fallback case-insensitive
    for col in df.columns:
        if isinstance(col, str) and col.lower() == 'close':
            return df[col]

    return None


def get_volume_column(df):
    """Retorna a coluna de volume (case-insensitive)."""
    if df is None or df.empty:
        return None

    for col in ['Volume', 'volume', 'VOLUME']:
        if col in df.columns:
            return df[col]
    return None


def get_high_column(df):
    """Retorna a coluna high (case-insensitive)."""
    if df is None or df.empty:
        return None

    for col in ['High', 'high', 'HIGH']:
        if col in df.columns:
            return df[col]
    return None


def get_low_column(df):
    """Retorna a coluna low (case-insensitive)."""
    if df is None or df.empty:
        return None

    for col in ['Low', 'low', 'LOW']:
        if col in df.columns:
            return df[col]
    return None


# Lista de ações da B3 (principais + small caps)
# Atualizada periodicamente - inclui ações do Ibovespa e outras líquidas
B3_STOCKS = [
    # Ibovespa - Blue Chips
    "PETR4.SA", "PETR3.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBDC3.SA",
    "BBAS3.SA", "B3SA3.SA", "ABEV3.SA", "ITSA4.SA", "WEGE3.SA", "RENT3.SA",
    "SUZB3.SA", "JBSS3.SA", "GGBR4.SA", "BPAC11.SA", "RADL3.SA", "RAIL3.SA",
    "EQTL3.SA", "LREN3.SA", "HAPV3.SA", "PRIO3.SA", "CSAN3.SA", "VIVT3.SA",
    "RDOR3.SA", "TOTS3.SA", "SBSP3.SA", "ENEV3.SA", "CCRO3.SA", "UGPA3.SA",
    "CMIG4.SA", "BBSE3.SA", "ENGI11.SA", "ELET3.SA", "ELET6.SA", "KLBN11.SA",
    "VBBR3.SA", "BRAP4.SA", "CPLE6.SA", "TAEE11.SA", "CSNA3.SA", "EGIE3.SA",
    "EMBR3.SA", "NTCO3.SA", "MRFG3.SA", "CPFE3.SA", "ASAI3.SA", "BRFS3.SA",
    "USIM5.SA", "CMIN3.SA", "MULT3.SA", "GOAU4.SA", "MRVE3.SA", "PETZ3.SA",
    "CRFB3.SA", "COGN3.SA", "BEEF3.SA", "CYRE3.SA", "MGLU3.SA", "IRBR3.SA",

    # Financeiro
    "SANB11.SA", "BIDI11.SA", "BMGB4.SA", "BPAN4.SA", "BRSR6.SA", "ABCB4.SA",
    "MODL11.SA", "XPBR31.SA",

    # Varejo
    "VIIA3.SA", "AMER3.SA", "BTOW3.SA", "LAME4.SA", "SOMA3.SA", "ARZZ3.SA",
    "GRND3.SA", "ALPA4.SA", "CEAB3.SA", "LJQQ3.SA", "AMAR3.SA", "GUAR3.SA",

    # Energia
    "NEOE3.SA", "AURE3.SA", "AESB3.SA", "TRPL4.SA", "CPLE3.SA", "COCE5.SA",
    "ALUP11.SA", "OMGE3.SA", "LIGT3.SA", "CESP6.SA",

    # Saúde
    "FLRY3.SA", "QUAL3.SA", "HYPE3.SA", "PNVL3.SA", "DASA3.SA", "MATD3.SA",
    "ONCO3.SA", "AALR3.SA",

    # Tecnologia
    "LWSA3.SA", "CASH3.SA", "INTB3.SA", "MLAS3.SA", "NINJ3.SA", "BMOB3.SA",
    "SQIA3.SA", "POSI3.SA",

    # Construção
    "EZTC3.SA", "EVEN3.SA", "DIRR3.SA", "CURY3.SA", "PLPL3.SA", "TEND3.SA",
    "TRIS3.SA", "LAVV3.SA", "MTRE3.SA", "MDNE3.SA",

    # Indústria
    "TUPY3.SA", "RAIZ4.SA", "UNIP6.SA", "FESA4.SA", "KEPL3.SA", "ROMI3.SA",
    "METAL3.SA", "SHUL4.SA", "POMO4.SA",

    # Commodities
    "SLCE3.SA", "SMTO3.SA", "AGRO3.SA", "CAML3.SA", "TTEN3.SA",

    # Seguros
    "PSSA3.SA", "SULA11.SA", "CXSE3.SA", "WIZC3.SA",

    # Outros
    "AZUL4.SA", "GOLL4.SA", "CVCB3.SA", "TGMA3.SA", "SIMH3.SA", "RECV3.SA",
    "RRRP3.SA", "VAMO3.SA", "MOVI3.SA", "LCAM3.SA", "STBP3.SA", "HBSA3.SA",
    "PTBL3.SA", "ENJU3.SA", "SEER3.SA", "YDUQ3.SA",
]


class MarketScanner:
    """
    Escaneia o mercado B3 para identificar as melhores oportunidades de trading.
    Analisa múltiplos critérios: volume, volatilidade, tendência e sinais técnicos.
    """

    def __init__(self):
        """Inicializa o scanner com configurações."""
        scanner_config = config.get('scanner', {})

        # Configurações de filtro
        self.min_volume = scanner_config.get('min_avg_volume', 1_000_000)  # Volume mínimo diário
        self.min_price = scanner_config.get('min_price', 1.0)  # Preço mínimo
        self.max_price = scanner_config.get('max_price', 500.0)  # Preço máximo
        self.top_n = scanner_config.get('top_stocks', 20)  # Quantas ações retornar
        self.max_workers = scanner_config.get('max_workers', 10)  # Threads paralelas

        # Pesos para ranking
        self.weights = scanner_config.get('weights', {
            'signal_strength': 0.30,  # Força do sinal técnico
            'volume_score': 0.20,     # Score de volume
            'trend_score': 0.25,      # Score de tendência
            'volatility_score': 0.15, # Score de volatilidade (moderada é melhor)
            'momentum_score': 0.10,   # Score de momentum
        })

        # Cache de análise
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(minutes=scanner_config.get('cache_minutes', 15))

        system_logger.info(f"Market Scanner inicializado - Analisando {len(B3_STOCKS)} ações")

    def scan_market(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Escaneia todo o mercado e retorna as melhores oportunidades.

        Args:
            force_refresh: Força atualização mesmo com cache válido.

        Returns:
            Lista ordenada das melhores ações com scores.
        """
        # Verifica cache
        if not force_refresh and self._is_cache_valid():
            system_logger.debug("Usando cache do scanner")
            return self._cache.get('results', [])

        system_logger.info("Iniciando scan do mercado B3...")
        start_time = datetime.now()

        results = []
        failed = []

        # Analisa ações em paralelo
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._analyze_stock, symbol): symbol
                for symbol in B3_STOCKS
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

        # Ordena por score total
        results.sort(key=lambda x: x['total_score'], reverse=True)

        # Filtra top N
        top_results = results[:self.top_n]

        # Atualiza cache
        self._cache = {'results': top_results}
        self._cache_time = datetime.now()

        elapsed = (datetime.now() - start_time).total_seconds()
        system_logger.info(
            f"Scan completo em {elapsed:.1f}s - "
            f"{len(results)} ações válidas, {len(failed)} falhas, "
            f"Top {len(top_results)} selecionadas"
        )

        return top_results

    def _is_cache_valid(self) -> bool:
        """Verifica se o cache ainda é válido."""
        if self._cache_time is None:
            return False
        return datetime.now() - self._cache_time < self._cache_duration

    def _analyze_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Analisa uma ação individual.

        Args:
            symbol: Símbolo da ação (ex: PETR4.SA).

        Returns:
            Dicionário com análise ou None se não passar nos filtros.
        """
        try:
            # Baixa dados históricos
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo", interval="1d")

            if hist.empty or len(hist) < 20:
                return None

            # V4.0: Obtém colunas de forma segura (case-insensitive)
            close_col = get_close_column(hist)
            volume_col = get_volume_column(hist)

            if close_col is None:
                system_logger.debug(f"{symbol}: Coluna 'close' não encontrada")
                return None

            # Verifica filtros básicos
            current_price = close_col.iloc[-1]
            avg_volume = volume_col.mean() if volume_col is not None else 0

            if pd.isna(current_price) or current_price < self.min_price or current_price > self.max_price:
                return None

            if avg_volume < self.min_volume:
                return None

            # Calcula indicadores
            analysis = self._calculate_indicators(hist)

            if analysis is None:
                return None

            # Calcula scores
            scores = self._calculate_scores(hist, analysis)

            # Calcula score total ponderado
            total_score = sum(
                scores.get(key, 0) * weight
                for key, weight in self.weights.items()
            )

            return {
                'symbol': symbol,
                'price': current_price,
                'avg_volume': avg_volume,
                'change_1d': self._calculate_change(hist, 1),
                'change_5d': self._calculate_change(hist, 5),
                'change_20d': self._calculate_change(hist, 20),
                **analysis,
                **scores,
                'total_score': total_score,
                'signal': self._determine_signal(analysis, scores),
                'analyzed_at': datetime.now().isoformat()
            }

        except Exception as e:
            system_logger.debug(f"Erro ao analisar {symbol}: {e}")
            return None

    def _calculate_indicators(self, hist: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calcula indicadores técnicos."""
        try:
            # V4.0: Obtém colunas de forma segura (case-insensitive)
            close = get_close_column(hist)
            high = get_high_column(hist)
            low = get_low_column(hist)
            volume = get_volume_column(hist)

            if close is None:
                return None

            # Fallbacks se colunas não existirem
            if high is None:
                high = close
            if low is None:
                low = close
            if volume is None:
                volume = pd.Series([0] * len(close), index=close.index)

            # RSI
            rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
            current_rsi = rsi.iloc[-1]

            # EMAs
            ema_9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
            ema_21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
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

            # ATR (volatilidade)
            atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

            # Volume médio relativo
            vol_sma = volume.rolling(window=20).mean()
            vol_ratio = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1

            # ADX (força da tendência)
            adx = ta.trend.ADXIndicator(high, low, close)
            adx_value = adx.adx().iloc[-1]

            return {
                'rsi': current_rsi,
                'ema_9': ema_9.iloc[-1],
                'ema_21': ema_21.iloc[-1],
                'ema_50': ema_50.iloc[-1] if len(ema_50.dropna()) > 0 else ema_21.iloc[-1],
                'macd': macd_line.iloc[-1],
                'macd_signal': macd_signal.iloc[-1],
                'macd_hist': macd_hist.iloc[-1],
                'bb_pct': bb_pct.iloc[-1],
                'atr': atr.iloc[-1],
                'atr_pct': (atr.iloc[-1] / close.iloc[-1]) * 100,
                'volume_ratio': vol_ratio,
                'adx': adx_value if not np.isnan(adx_value) else 25,
            }

        except Exception as e:
            system_logger.debug(f"Erro ao calcular indicadores: {e}")
            return None

    def _calculate_scores(self, hist: pd.DataFrame, analysis: Dict[str, float]) -> Dict[str, float]:
        """Calcula scores para ranking."""
        scores = {}

        # Signal Strength Score (0-100)
        # Baseado em RSI, MACD e Bollinger
        rsi = analysis['rsi']
        signal_score = 0

        # RSI oversold = oportunidade de compra
        if rsi < 30:
            signal_score += 40
        elif rsi < 40:
            signal_score += 25
        elif rsi > 70:
            signal_score -= 20
        else:
            signal_score += 10

        # MACD bullish crossover
        if analysis['macd_hist'] > 0:
            signal_score += 30
        if analysis['macd'] > analysis['macd_signal']:
            signal_score += 20

        # Bollinger Band position
        if analysis['bb_pct'] < 0.2:  # Perto da banda inferior
            signal_score += 20
        elif analysis['bb_pct'] > 0.8:  # Perto da banda superior
            signal_score -= 10

        scores['signal_strength'] = max(0, min(100, signal_score))

        # Volume Score (0-100)
        vol_ratio = analysis['volume_ratio']
        if vol_ratio > 2:
            scores['volume_score'] = 100
        elif vol_ratio > 1.5:
            scores['volume_score'] = 80
        elif vol_ratio > 1:
            scores['volume_score'] = 60
        else:
            scores['volume_score'] = vol_ratio * 60

        # Trend Score (0-100)
        close_col = get_close_column(hist)
        price = close_col.iloc[-1] if close_col is not None else 0
        ema_9 = analysis['ema_9']
        ema_21 = analysis['ema_21']
        ema_50 = analysis['ema_50']

        trend_score = 50  # Neutro

        # Preço acima das EMAs = tendência de alta
        if price > ema_9 > ema_21:
            trend_score += 30
        elif price > ema_21:
            trend_score += 15

        # EMAs alinhadas para alta
        if ema_9 > ema_21 > ema_50:
            trend_score += 20
        elif ema_9 > ema_21:
            trend_score += 10

        # ADX indica força da tendência
        adx = analysis['adx']
        if adx > 25:
            trend_score += 10

        scores['trend_score'] = max(0, min(100, trend_score))

        # Volatility Score (0-100)
        # Volatilidade moderada é ideal (não muito alta, não muito baixa)
        atr_pct = analysis['atr_pct']
        if 1.5 <= atr_pct <= 4:
            scores['volatility_score'] = 100
        elif 1 <= atr_pct <= 5:
            scores['volatility_score'] = 70
        elif atr_pct < 1:
            scores['volatility_score'] = 40  # Muito baixa
        else:
            scores['volatility_score'] = max(20, 100 - (atr_pct - 5) * 10)  # Muito alta

        # Momentum Score (0-100)
        change_5d = self._calculate_change(hist, 5)
        change_20d = self._calculate_change(hist, 20)

        momentum = 50
        if change_5d > 0:
            momentum += min(25, change_5d * 5)
        else:
            momentum += max(-25, change_5d * 5)

        if change_20d > 0 and change_5d > change_20d / 4:
            momentum += 15  # Acelerando

        scores['momentum_score'] = max(0, min(100, momentum))

        return scores

    def _calculate_change(self, hist: pd.DataFrame, days: int) -> float:
        """Calcula variação percentual em N dias."""
        if len(hist) < days + 1:
            return 0.0

        close_col = get_close_column(hist)
        if close_col is None:
            return 0.0

        current = close_col.iloc[-1]
        past = close_col.iloc[-days-1]

        if pd.isna(past) or past == 0:
            return 0.0

        return ((current - past) / past) * 100

    def _determine_signal(self, analysis: Dict[str, float], scores: Dict[str, float]) -> str:
        """Determina o sinal de trading."""
        signal_strength = scores.get('signal_strength', 0)
        trend_score = scores.get('trend_score', 0)
        rsi = analysis.get('rsi', 50)

        if signal_strength >= 70 and rsi < 40:
            return 'STRONG_BUY'
        elif signal_strength >= 50 and trend_score >= 60:
            return 'BUY'
        elif signal_strength <= 30 or rsi > 70:
            return 'SELL'
        else:
            return 'HOLD'

    def get_top_symbols(self, n: int = None, signal_filter: str = None) -> List[str]:
        """
        Retorna os N melhores símbolos para trading.

        Args:
            n: Número de símbolos (default: configuração).
            signal_filter: Filtrar por sinal ('BUY', 'STRONG_BUY', etc).

        Returns:
            Lista de símbolos.
        """
        results = self.scan_market()

        if signal_filter:
            results = [r for r in results if signal_filter in r.get('signal', '')]

        n = n or self.top_n
        return [r['symbol'] for r in results[:n]]

    def get_opportunities(self, min_score: float = 50) -> List[Dict[str, Any]]:
        """
        Retorna oportunidades com score acima do mínimo.

        Args:
            min_score: Score mínimo para considerar oportunidade.

        Returns:
            Lista de oportunidades.
        """
        results = self.scan_market()
        return [r for r in results if r['total_score'] >= min_score]

    def get_buy_signals(self) -> List[Dict[str, Any]]:
        """Retorna ações com sinal de compra."""
        results = self.scan_market()
        return [r for r in results if 'BUY' in r.get('signal', '')]

    def print_report(self, top_n: int = 10):
        """Imprime relatório das melhores oportunidades."""
        results = self.scan_market()[:top_n]

        print("\n" + "=" * 80)
        print("MARKET SCANNER - MELHORES OPORTUNIDADES B3")
        print("=" * 80)
        print(f"\n{'Rank':<5} {'Símbolo':<12} {'Preço':<10} {'Score':<8} {'Sinal':<12} {'RSI':<6} {'Var 5d':<8}")
        print("-" * 80)

        for i, r in enumerate(results, 1):
            print(
                f"{i:<5} {r['symbol']:<12} "
                f"R${r['price']:<8.2f} "
                f"{r['total_score']:<8.1f} "
                f"{r['signal']:<12} "
                f"{r['rsi']:<6.1f} "
                f"{r['change_5d']:>+7.2f}%"
            )

        print("=" * 80)


# Instância global
scanner = MarketScanner()


if __name__ == "__main__":
    # Teste standalone
    scanner = MarketScanner()
    scanner.print_report(20)
