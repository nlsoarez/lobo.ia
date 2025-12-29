"""
V4.0 Phase 2: Volume Analyzer
Analisa volume e liquidez para filtrar ativos de qualidade.
Detecta spikes de volume, volume profile e liquidez.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from system_logger import system_logger
from data_utils import normalize_dataframe_columns, safe_get_ohlcv_arrays


@dataclass
class VolumeAnalysis:
    """Resultado da análise de volume."""
    symbol: str
    current_volume: float
    avg_volume_20: float
    avg_volume_50: float
    volume_ratio_20: float
    volume_ratio_50: float
    is_volume_spike: bool
    volume_trend: str  # INCREASING, DECREASING, STABLE
    volume_score: float  # 0-100
    quality: str  # LOW, MEDIUM, HIGH


@dataclass
class LiquidityAnalysis:
    """Resultado da análise de liquidez."""
    symbol: str
    estimated_liquidity_usd: float
    spread_percent: float
    sufficient_liquidity: bool
    liquidity_score: float  # 0-100
    quality: str  # LOW, MEDIUM, HIGH


class VolumeAnalyzer:
    """
    V4.0 Phase 2: Analisador de volume e liquidez.
    Filtra ativos de qualidade baseado em volume e liquidez.
    """

    def __init__(self):
        """Inicializa o analisador de volume."""
        # Requisitos de volume
        self.min_volume_ratio = 1.5       # Volume atual deve ser 1.5x > média
        self.volume_spike_threshold = 3.0  # 3x para considerar spike
        self.min_liquidity_usd = 500000    # $500K mínimo (ajustado para cryptos menores)
        self.max_spread_percent = 0.3      # 0.3% spread máximo

        # Períodos para médias
        self.short_period = 20
        self.long_period = 50

        # Pesos para score
        self.volume_weights = {
            'ratio': 0.35,
            'trend': 0.25,
            'consistency': 0.20,
            'spike_bonus': 0.20
        }

        system_logger.info("VolumeAnalyzer V4.0 inicializado")

    def calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula métricas de volume a partir de dados históricos.
        """
        if len(df) < 10:
            return {
                'current_volume': 0,
                'avg_20': 0,
                'avg_50': 0,
                'ratio_20': 0,
                'ratio_50': 0,
                'std_20': 0
            }

        # V4.1 FIX: Usa extração segura de volume
        df_norm = normalize_dataframe_columns(df)
        data = safe_get_ohlcv_arrays(df_norm)
        volumes = data['volume']

        if len(volumes) == 0:
            return {
                'current_volume': 0,
                'avg_20': 0,
                'avg_50': 0,
                'ratio_20': 0,
                'ratio_50': 0,
                'std_20': 0
            }

        current_volume = volumes[-1]
        avg_20 = np.mean(volumes[-self.short_period:]) if len(volumes) >= self.short_period else np.mean(volumes)
        avg_50 = np.mean(volumes[-self.long_period:]) if len(volumes) >= self.long_period else np.mean(volumes)
        std_20 = np.std(volumes[-self.short_period:]) if len(volumes) >= self.short_period else np.std(volumes)

        ratio_20 = current_volume / avg_20 if avg_20 > 0 else 0
        ratio_50 = current_volume / avg_50 if avg_50 > 0 else 0

        return {
            'current_volume': current_volume,
            'avg_20': avg_20,
            'avg_50': avg_50,
            'ratio_20': ratio_20,
            'ratio_50': ratio_50,
            'std_20': std_20
        }

    def detect_volume_trend(self, df: pd.DataFrame, periods: int = 10) -> str:
        """
        Detecta tendência de volume: INCREASING, DECREASING ou STABLE.
        """
        if len(df) < periods + 5:
            return 'STABLE'

        # V4.1 FIX: Usa extração segura de volume
        df_norm = normalize_dataframe_columns(df)
        data = safe_get_ohlcv_arrays(df_norm)
        volumes = data['volume']

        if len(volumes) == 0:
            return 'STABLE'

        # Calcula médias móveis de volume
        recent_avg = np.mean(volumes[-periods:])
        older_avg = np.mean(volumes[-(periods*2):-periods])

        change_pct = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

        if change_pct > 0.15:
            return 'INCREASING'
        elif change_pct < -0.15:
            return 'DECREASING'
        else:
            return 'STABLE'

    def is_volume_spike(self, current_volume: float, avg_volume: float, std_volume: float) -> bool:
        """
        Verifica se há um spike de volume significativo.
        Spike = Volume atual > média + 2 desvios padrão OU > 3x média
        """
        if avg_volume <= 0:
            return False

        ratio = current_volume / avg_volume
        z_score = (current_volume - avg_volume) / std_volume if std_volume > 0 else 0

        return ratio >= self.volume_spike_threshold or z_score >= 2.5

    def calculate_volume_score(self, metrics: Dict[str, float], trend: str, is_spike: bool) -> float:
        """
        Calcula score de volume (0-100) baseado em múltiplos fatores.
        """
        score = 0

        # Score baseado no ratio (35%)
        ratio = metrics.get('ratio_20', 0)
        if ratio >= 3.0:
            ratio_score = 100
        elif ratio >= 2.0:
            ratio_score = 80
        elif ratio >= 1.5:
            ratio_score = 60
        elif ratio >= 1.0:
            ratio_score = 40
        else:
            ratio_score = 20

        # Score baseado na tendência (25%)
        trend_scores = {
            'INCREASING': 100,
            'STABLE': 60,
            'DECREASING': 30
        }
        trend_score = trend_scores.get(trend, 50)

        # Score de consistência (20%)
        avg_20 = metrics.get('avg_20', 0)
        avg_50 = metrics.get('avg_50', 0)
        if avg_50 > 0:
            consistency_ratio = avg_20 / avg_50
            if 0.8 <= consistency_ratio <= 1.5:
                consistency_score = 80
            elif 0.6 <= consistency_ratio <= 2.0:
                consistency_score = 60
            else:
                consistency_score = 40
        else:
            consistency_score = 50

        # Bonus por spike (20%)
        spike_score = 100 if is_spike else 40

        # Calcula score total ponderado
        total_score = (
            ratio_score * self.volume_weights['ratio'] +
            trend_score * self.volume_weights['trend'] +
            consistency_score * self.volume_weights['consistency'] +
            spike_score * self.volume_weights['spike_bonus']
        )

        return min(100, total_score)

    def analyze_volume(self, symbol: str, df: pd.DataFrame) -> VolumeAnalysis:
        """
        Análise completa de volume para um símbolo.
        """
        try:
            metrics = self.calculate_volume_metrics(df)
            trend = self.detect_volume_trend(df)
            is_spike = self.is_volume_spike(
                metrics['current_volume'],
                metrics['avg_20'],
                metrics['std_20']
            )
            score = self.calculate_volume_score(metrics, trend, is_spike)

            # Determina qualidade
            if score >= 75:
                quality = 'HIGH'
            elif score >= 50:
                quality = 'MEDIUM'
            else:
                quality = 'LOW'

            return VolumeAnalysis(
                symbol=symbol,
                current_volume=metrics['current_volume'],
                avg_volume_20=metrics['avg_20'],
                avg_volume_50=metrics['avg_50'],
                volume_ratio_20=metrics['ratio_20'],
                volume_ratio_50=metrics['ratio_50'],
                is_volume_spike=is_spike,
                volume_trend=trend,
                volume_score=score,
                quality=quality
            )

        except Exception as e:
            system_logger.debug(f"Erro analisando volume de {symbol}: {e}")
            return VolumeAnalysis(
                symbol=symbol,
                current_volume=0,
                avg_volume_20=0,
                avg_volume_50=0,
                volume_ratio_20=0,
                volume_ratio_50=0,
                is_volume_spike=False,
                volume_trend='STABLE',
                volume_score=0,
                quality='LOW'
            )

    def estimate_liquidity(self, symbol: str, df: pd.DataFrame, current_price: float) -> LiquidityAnalysis:
        """
        Estima liquidez baseado em volume e preço.
        Nota: Em produção, usar dados de order book se disponíveis.
        """
        try:
            if len(df) < 5:
                return LiquidityAnalysis(
                    symbol=symbol,
                    estimated_liquidity_usd=0,
                    spread_percent=1.0,
                    sufficient_liquidity=False,
                    liquidity_score=0,
                    quality='LOW'
                )

            # V4.1 FIX: Usa extração segura de OHLCV
            df_norm = normalize_dataframe_columns(df)
            data = safe_get_ohlcv_arrays(df_norm)

            # Estima liquidez como média de volume * preço
            volumes = data['volume']
            avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes) if len(volumes) > 0 else 0
            estimated_liquidity = avg_volume * current_price

            # Estima spread baseado na volatilidade do candle
            highs = data['high']
            lows = data['low']
            if len(highs) >= 10 and len(lows) >= 10:
                avg_range = np.mean(highs[-10:] - lows[-10:])
            else:
                avg_range = 0
            spread_estimate = (avg_range / current_price) * 100 if current_price > 0 else 1.0

            # Calcula score de liquidez
            liquidity_score = 0

            # Score baseado em liquidez USD
            if estimated_liquidity >= 10000000:  # $10M+
                liquidity_score += 50
            elif estimated_liquidity >= 1000000:  # $1M+
                liquidity_score += 35
            elif estimated_liquidity >= 500000:   # $500K+
                liquidity_score += 20
            else:
                liquidity_score += 5

            # Score baseado em spread
            if spread_estimate <= 0.1:
                liquidity_score += 50
            elif spread_estimate <= 0.2:
                liquidity_score += 35
            elif spread_estimate <= 0.3:
                liquidity_score += 20
            else:
                liquidity_score += 5

            sufficient = (
                estimated_liquidity >= self.min_liquidity_usd and
                spread_estimate <= self.max_spread_percent
            )

            if liquidity_score >= 75:
                quality = 'HIGH'
            elif liquidity_score >= 50:
                quality = 'MEDIUM'
            else:
                quality = 'LOW'

            return LiquidityAnalysis(
                symbol=symbol,
                estimated_liquidity_usd=estimated_liquidity,
                spread_percent=spread_estimate,
                sufficient_liquidity=sufficient,
                liquidity_score=liquidity_score,
                quality=quality
            )

        except Exception as e:
            system_logger.debug(f"Erro estimando liquidez de {symbol}: {e}")
            return LiquidityAnalysis(
                symbol=symbol,
                estimated_liquidity_usd=0,
                spread_percent=1.0,
                sufficient_liquidity=False,
                liquidity_score=0,
                quality='LOW'
            )

    def check_volume_requirements(self, symbol: str, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Verifica se o símbolo atende aos requisitos de volume.
        Retorna (passed, reason).
        """
        analysis = self.analyze_volume(symbol, df)

        if analysis.volume_ratio_20 < self.min_volume_ratio:
            return False, f"Volume baixo: {analysis.volume_ratio_20:.2f}x (min: {self.min_volume_ratio}x)"

        if analysis.quality == 'LOW':
            return False, f"Qualidade de volume baixa: score {analysis.volume_score:.0f}"

        return True, f"Volume OK: {analysis.volume_ratio_20:.2f}x, score {analysis.volume_score:.0f}"

    def check_liquidity_requirements(self, symbol: str, df: pd.DataFrame, current_price: float) -> Tuple[bool, str]:
        """
        Verifica se o símbolo atende aos requisitos de liquidez.
        Retorna (passed, reason).
        """
        analysis = self.estimate_liquidity(symbol, df, current_price)

        if not analysis.sufficient_liquidity:
            reasons = []
            if analysis.estimated_liquidity_usd < self.min_liquidity_usd:
                reasons.append(f"Liquidez ${analysis.estimated_liquidity_usd/1000:.0f}K < ${self.min_liquidity_usd/1000:.0f}K")
            if analysis.spread_percent > self.max_spread_percent:
                reasons.append(f"Spread {analysis.spread_percent:.2f}% > {self.max_spread_percent}%")
            return False, "; ".join(reasons)

        return True, f"Liquidez OK: ${analysis.estimated_liquidity_usd/1000:.0f}K, spread {analysis.spread_percent:.2f}%"

    def get_volume_multiplier(self, analysis: VolumeAnalysis) -> float:
        """
        Retorna multiplicador de exposição baseado no volume.
        Maior volume = maior confiança = pode usar mais exposição.
        """
        if analysis.is_volume_spike and analysis.volume_ratio_20 >= 3.0:
            return 1.25  # +25% exposição para volume excepcional
        elif analysis.volume_ratio_20 >= 2.0:
            return 1.15  # +15% para volume alto
        elif analysis.volume_ratio_20 >= 1.5:
            return 1.0   # Exposição normal
        elif analysis.volume_ratio_20 >= 1.0:
            return 0.85  # -15% para volume médio
        else:
            return 0.70  # -30% para volume baixo

    def analyze_volume_profile(self, df: pd.DataFrame, price_levels: int = 20) -> Dict[str, Any]:
        """
        Cria perfil de volume simplificado.
        Identifica níveis de preço com maior volume (POC - Point of Control).
        """
        try:
            if len(df) < 20:
                return {'poc_price': 0, 'value_area_high': 0, 'value_area_low': 0}

            # V4.1 FIX: Usa extração segura de OHLCV
            df_norm = normalize_dataframe_columns(df)
            data = safe_get_ohlcv_arrays(df_norm)
            closes = data['close']
            volumes = data['volume']

            if len(closes) == 0 or len(volumes) == 0:
                return {'poc_price': 0, 'value_area_high': 0, 'value_area_low': 0}

            # Divide o range de preço em níveis
            price_min = np.min(closes)
            price_max = np.max(closes)
            price_range = price_max - price_min

            if price_range <= 0:
                return {'poc_price': closes[-1], 'value_area_high': closes[-1], 'value_area_low': closes[-1]}

            level_size = price_range / price_levels

            # Distribui volume por nível de preço
            volume_by_level = {}
            for i in range(len(closes)):
                level = int((closes[i] - price_min) / level_size)
                level = min(level, price_levels - 1)
                if level not in volume_by_level:
                    volume_by_level[level] = 0
                volume_by_level[level] += volumes[i]

            # Encontra POC (nível com maior volume)
            poc_level = max(volume_by_level, key=volume_by_level.get)
            poc_price = price_min + (poc_level + 0.5) * level_size

            # Calcula Value Area (70% do volume)
            total_volume = sum(volume_by_level.values())
            target_volume = total_volume * 0.7

            # Expande a partir do POC até atingir 70%
            included_levels = [poc_level]
            current_volume = volume_by_level.get(poc_level, 0)

            while current_volume < target_volume:
                candidates = []
                min_level = min(included_levels)
                max_level = max(included_levels)

                if min_level > 0:
                    candidates.append(min_level - 1)
                if max_level < price_levels - 1:
                    candidates.append(max_level + 1)

                if not candidates:
                    break

                best_candidate = max(candidates, key=lambda x: volume_by_level.get(x, 0))
                included_levels.append(best_candidate)
                current_volume += volume_by_level.get(best_candidate, 0)

            value_area_low = price_min + min(included_levels) * level_size
            value_area_high = price_min + (max(included_levels) + 1) * level_size

            return {
                'poc_price': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'total_volume': total_volume
            }

        except Exception as e:
            system_logger.debug(f"Erro calculando volume profile: {e}")
            return {'poc_price': 0, 'value_area_high': 0, 'value_area_low': 0}

    def get_comprehensive_analysis(self, symbol: str, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Análise completa de volume e liquidez para um símbolo.
        """
        volume_analysis = self.analyze_volume(symbol, df)
        liquidity_analysis = self.estimate_liquidity(symbol, df, current_price)
        volume_profile = self.analyze_volume_profile(df)

        volume_ok, volume_reason = self.check_volume_requirements(symbol, df)
        liquidity_ok, liquidity_reason = self.check_liquidity_requirements(symbol, df, current_price)

        # Score combinado
        combined_score = (volume_analysis.volume_score * 0.6 + liquidity_analysis.liquidity_score * 0.4)

        # Multiplier de exposição
        exposure_multiplier = self.get_volume_multiplier(volume_analysis)

        return {
            'symbol': symbol,
            'volume': {
                'current': volume_analysis.current_volume,
                'ratio': volume_analysis.volume_ratio_20,
                'trend': volume_analysis.volume_trend,
                'is_spike': volume_analysis.is_volume_spike,
                'score': volume_analysis.volume_score,
                'quality': volume_analysis.quality,
                'passed': volume_ok,
                'reason': volume_reason
            },
            'liquidity': {
                'estimated_usd': liquidity_analysis.estimated_liquidity_usd,
                'spread_pct': liquidity_analysis.spread_percent,
                'score': liquidity_analysis.liquidity_score,
                'quality': liquidity_analysis.quality,
                'passed': liquidity_ok,
                'reason': liquidity_reason
            },
            'volume_profile': volume_profile,
            'combined_score': combined_score,
            'exposure_multiplier': exposure_multiplier,
            'overall_passed': volume_ok and liquidity_ok
        }
