"""
V4.0 Phase 2: Technical Pattern Scanner
Detecta padrões gráficos para aumentar win rate.
Suporta: Bull/Bear Flag, Triangles, Double Top/Bottom, Head & Shoulders
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from system_logger import system_logger
from data_utils import normalize_dataframe_columns, safe_get_ohlcv_arrays, validate_ohlcv_data


class PatternType(Enum):
    """Tipos de padrões suportados."""
    BULL_FLAG = "BULL_FLAG"
    BEAR_FLAG = "BEAR_FLAG"
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    SYMMETRICAL_TRIANGLE = "SYMMETRICAL_TRIANGLE"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    DOUBLE_TOP = "DOUBLE_TOP"
    HEAD_SHOULDERS = "HEAD_SHOULDERS"
    INVERSE_HEAD_SHOULDERS = "INVERSE_HEAD_SHOULDERS"
    WEDGE_UP = "WEDGE_UP"
    WEDGE_DOWN = "WEDGE_DOWN"


@dataclass
class PatternResult:
    """Resultado da detecção de padrão."""
    detected: bool
    pattern_type: PatternType
    score: float  # 0-100
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward: float
    confidence: str  # LOW, MEDIUM, HIGH
    description: str


class TechnicalPatternScanner:
    """
    V4.0 Phase 2: Scanner de padrões técnicos.
    Detecta padrões gráficos com alta probabilidade de sucesso.
    """

    def __init__(self):
        """Inicializa o scanner de padrões."""
        self.min_pattern_score = 70
        self.confirmation_bars = 2
        self.lookback_periods = 50  # Períodos para análise

        # Pesos para cálculo de score
        self.pattern_weights = {
            'clarity': 0.30,       # Clareza do padrão
            'volume': 0.25,        # Volume durante formação
            'trend_alignment': 0.25,  # Alinhamento com tendência
            'breakout_strength': 0.20  # Força do breakout
        }

        system_logger.info("TechnicalPatternScanner V4.0 inicializado")

    def _get_ohlcv_safe(self, df: pd.DataFrame) -> tuple:
        """
        Extrai dados OHLCV de forma segura (case-insensitive).

        Returns:
            Tuple (prices, highs, lows, volumes) ou (None, None, None, None) se falhar
        """
        try:
            # Normaliza colunas para lowercase
            df_norm = normalize_dataframe_columns(df)

            # Valida dados
            validation = validate_ohlcv_data(df_norm)
            if not validation['valid']:
                return None, None, None, None

            # Extrai arrays
            data = safe_get_ohlcv_arrays(df_norm)

            if len(data['close']) == 0:
                return None, None, None, None

            return data['close'], data['high'], data['low'], data['volume']
        except Exception as e:
            system_logger.error(f"Erro extraindo OHLCV: {e}")
            return None, None, None, None

    def detect_bull_flag(self, df: pd.DataFrame, min_score: int = 70) -> PatternResult:
        """
        Detecta padrão Bull Flag:
        1. Forte movimento de alta (polo) - mínimo 5%
        2. Consolidação em canal descendente/lateral (bandeira) - 3-10 barras
        3. Volume decrescente durante consolidação
        4. Breakout acima da resistência da bandeira
        """
        if len(df) < 20:
            return PatternResult(
                detected=False, pattern_type=PatternType.BULL_FLAG,
                score=0, entry_price=0, target_price=0, stop_loss=0,
                risk_reward=0, confidence='LOW', description='Dados insuficientes'
            )

        try:
            # V4.1 FIX: Usa extração segura de OHLCV
            prices, highs, lows, volumes = self._get_ohlcv_safe(df)
            if prices is None:
                return PatternResult(
                    detected=False, pattern_type=PatternType.BULL_FLAG,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Erro ao extrair dados OHLCV'
                )

            # Procura movimento de alta de 5%+ nas últimas 20 barras
            pole_found = False
            pole_start_idx = 0
            pole_end_idx = 0
            pole_gain = 0

            for i in range(len(df) - 15, max(0, len(df) - 30), -1):
                for j in range(i + 3, min(i + 10, len(df) - 5)):
                    gain = (prices[j] - prices[i]) / prices[i]
                    if gain >= 0.05:  # 5% mínimo
                        pole_found = True
                        pole_start_idx = i
                        pole_end_idx = j
                        pole_gain = gain
                        break
                if pole_found:
                    break

            if not pole_found:
                return PatternResult(
                    detected=False, pattern_type=PatternType.BULL_FLAG,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Polo não encontrado'
                )

            # Verifica consolidação (bandeira)
            flag_start = pole_end_idx
            flag_end = len(df) - 1
            flag_bars = flag_end - flag_start

            if flag_bars < 3 or flag_bars > 15:
                return PatternResult(
                    detected=False, pattern_type=PatternType.BULL_FLAG,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Bandeira inválida'
                )

            # Calcula canal da bandeira
            flag_highs = highs[flag_start:flag_end + 1]
            flag_lows = lows[flag_start:flag_end + 1]
            flag_closes = prices[flag_start:flag_end + 1]

            # Verifica se é canal descendente ou lateral
            high_slope = (flag_highs[-1] - flag_highs[0]) / len(flag_highs)
            low_slope = (flag_lows[-1] - flag_lows[0]) / len(flag_lows)

            # Canal deve ser descendente ou lateral (não ascendente forte)
            if high_slope > 0.002 * prices[-1]:  # Inclinação máxima
                return PatternResult(
                    detected=False, pattern_type=PatternType.BULL_FLAG,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Canal não descendente'
                )

            # Verifica volume decrescente na bandeira
            pole_volume = np.mean(volumes[pole_start_idx:pole_end_idx + 1])
            flag_volume = np.mean(volumes[flag_start:flag_end + 1])
            volume_decreasing = flag_volume < pole_volume * 0.8

            # Calcula níveis
            resistance = np.max(flag_highs)
            support = np.min(flag_lows)
            current_price = prices[-1]

            # Verifica breakout
            breakout_confirmed = current_price > resistance

            # Calcula score
            clarity_score = min(100, (pole_gain / 0.05) * 50 + 50)  # Mais ganho = mais claro
            volume_score = 80 if volume_decreasing else 40
            trend_score = 70  # Assumindo tendência de alta
            breakout_score = 100 if breakout_confirmed else 30

            total_score = (
                clarity_score * self.pattern_weights['clarity'] +
                volume_score * self.pattern_weights['volume'] +
                trend_score * self.pattern_weights['trend_alignment'] +
                breakout_score * self.pattern_weights['breakout_strength']
            )

            # Calcula targets
            pole_height = prices[pole_end_idx] - prices[pole_start_idx]
            entry_price = resistance * 1.001  # Ligeiramente acima da resistência
            target_price = entry_price + pole_height  # Projeção do polo
            stop_loss = support * 0.995  # Ligeiramente abaixo do suporte

            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0

            confidence = 'HIGH' if total_score >= 80 else 'MEDIUM' if total_score >= 60 else 'LOW'

            return PatternResult(
                detected=total_score >= min_score,
                pattern_type=PatternType.BULL_FLAG,
                score=total_score,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward=risk_reward,
                confidence=confidence,
                description=f'Bull Flag: Polo +{pole_gain*100:.1f}%, RR {risk_reward:.2f}'
            )

        except Exception as e:
            system_logger.debug(f"Erro detectando Bull Flag: {e}")
            return PatternResult(
                detected=False, pattern_type=PatternType.BULL_FLAG,
                score=0, entry_price=0, target_price=0, stop_loss=0,
                risk_reward=0, confidence='LOW', description=f'Erro: {str(e)}'
            )

    def detect_ascending_triangle(self, df: pd.DataFrame, min_score: int = 70) -> PatternResult:
        """
        Detecta Triângulo Ascendente:
        - Resistência horizontal (topo plano)
        - Suporte ascendente (fundos mais altos)
        - Breakout acima da resistência
        """
        if len(df) < 20:
            return PatternResult(
                detected=False, pattern_type=PatternType.ASCENDING_TRIANGLE,
                score=0, entry_price=0, target_price=0, stop_loss=0,
                risk_reward=0, confidence='LOW', description='Dados insuficientes'
            )

        try:
            # V4.1 FIX: Usa extração segura de OHLCV
            closes, highs, lows, volumes = self._get_ohlcv_safe(df)
            if closes is None:
                return PatternResult(
                    detected=False, pattern_type=PatternType.ASCENDING_TRIANGLE,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Erro ao extrair dados OHLCV'
                )

            # Encontra topos (resistência horizontal)
            peaks = []
            for i in range(2, len(df) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    peaks.append((i, highs[i]))

            if len(peaks) < 2:
                return PatternResult(
                    detected=False, pattern_type=PatternType.ASCENDING_TRIANGLE,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Topos insuficientes'
                )

            # Verifica se topos são aproximadamente horizontais (tolerância 1%)
            peak_prices = [p[1] for p in peaks[-3:]]
            resistance_level = np.mean(peak_prices)
            peak_variance = np.std(peak_prices) / resistance_level

            if peak_variance > 0.015:  # 1.5% de tolerância
                return PatternResult(
                    detected=False, pattern_type=PatternType.ASCENDING_TRIANGLE,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Resistência não horizontal'
                )

            # Encontra fundos (suporte ascendente)
            troughs = []
            for i in range(2, len(df) - 2):
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                   lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    troughs.append((i, lows[i]))

            if len(troughs) < 2:
                return PatternResult(
                    detected=False, pattern_type=PatternType.ASCENDING_TRIANGLE,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Fundos insuficientes'
                )

            # Verifica se fundos são ascendentes
            trough_prices = [t[1] for t in troughs[-3:]]
            ascending = all(trough_prices[i] <= trough_prices[i+1] for i in range(len(trough_prices)-1))

            if not ascending:
                return PatternResult(
                    detected=False, pattern_type=PatternType.ASCENDING_TRIANGLE,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Suporte não ascendente'
                )

            # Calcula níveis
            support_level = trough_prices[-1]
            triangle_height = resistance_level - support_level
            current_price = closes[-1]

            # Verifica breakout
            breakout_confirmed = current_price > resistance_level

            # Calcula score
            clarity_score = 80 if peak_variance < 0.01 else 60
            volume_score = 70  # Volume típico
            trend_score = 80  # Padrão bullish
            breakout_score = 100 if breakout_confirmed else 40

            total_score = (
                clarity_score * self.pattern_weights['clarity'] +
                volume_score * self.pattern_weights['volume'] +
                trend_score * self.pattern_weights['trend_alignment'] +
                breakout_score * self.pattern_weights['breakout_strength']
            )

            # Targets
            entry_price = resistance_level * 1.002
            target_price = entry_price + triangle_height
            stop_loss = support_level * 0.99

            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0

            confidence = 'HIGH' if total_score >= 80 else 'MEDIUM' if total_score >= 60 else 'LOW'

            return PatternResult(
                detected=total_score >= min_score,
                pattern_type=PatternType.ASCENDING_TRIANGLE,
                score=total_score,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward=risk_reward,
                confidence=confidence,
                description=f'Ascending Triangle: Height {triangle_height:.2f}, RR {risk_reward:.2f}'
            )

        except Exception as e:
            system_logger.debug(f"Erro detectando Ascending Triangle: {e}")
            return PatternResult(
                detected=False, pattern_type=PatternType.ASCENDING_TRIANGLE,
                score=0, entry_price=0, target_price=0, stop_loss=0,
                risk_reward=0, confidence='LOW', description=f'Erro: {str(e)}'
            )

    def detect_double_bottom(self, df: pd.DataFrame, min_score: int = 70) -> PatternResult:
        """
        Detecta Double Bottom (W):
        - Dois fundos aproximadamente no mesmo nível
        - Recuperação entre os fundos
        - Breakout acima do neckline
        """
        if len(df) < 25:
            return PatternResult(
                detected=False, pattern_type=PatternType.DOUBLE_BOTTOM,
                score=0, entry_price=0, target_price=0, stop_loss=0,
                risk_reward=0, confidence='LOW', description='Dados insuficientes'
            )

        try:
            # V4.1 FIX: Usa extração segura de OHLCV
            closes, highs, lows, volumes = self._get_ohlcv_safe(df)
            if closes is None:
                return PatternResult(
                    detected=False, pattern_type=PatternType.DOUBLE_BOTTOM,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Erro ao extrair dados OHLCV'
                )

            # Encontra os dois fundos mais baixos recentes
            window = min(30, len(df) - 5)
            recent_lows = lows[-window:]

            # Encontra fundos locais
            local_bottoms = []
            for i in range(2, len(recent_lows) - 2):
                if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and \
                   recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]:
                    local_bottoms.append((len(df) - window + i, recent_lows[i]))

            if len(local_bottoms) < 2:
                return PatternResult(
                    detected=False, pattern_type=PatternType.DOUBLE_BOTTOM,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Fundos insuficientes'
                )

            # Pega os dois fundos mais recentes
            bottom1_idx, bottom1_price = local_bottoms[-2]
            bottom2_idx, bottom2_price = local_bottoms[-1]

            # Verifica se fundos estão no mesmo nível (tolerância 2%)
            price_diff = abs(bottom1_price - bottom2_price) / bottom1_price
            if price_diff > 0.02:
                return PatternResult(
                    detected=False, pattern_type=PatternType.DOUBLE_BOTTOM,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Fundos não alinhados'
                )

            # Encontra o pico entre os fundos (neckline)
            between_start = bottom1_idx
            between_end = bottom2_idx
            if between_end <= between_start:
                return PatternResult(
                    detected=False, pattern_type=PatternType.DOUBLE_BOTTOM,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Índices inválidos'
                )

            neckline = np.max(highs[between_start:between_end + 1])
            pattern_depth = neckline - min(bottom1_price, bottom2_price)

            current_price = closes[-1]
            breakout_confirmed = current_price > neckline

            # Calcula score
            clarity_score = 80 if price_diff < 0.01 else 60
            volume_score = 75  # Volume típico
            trend_score = 85  # Padrão de reversão bullish
            breakout_score = 100 if breakout_confirmed else 35

            total_score = (
                clarity_score * self.pattern_weights['clarity'] +
                volume_score * self.pattern_weights['volume'] +
                trend_score * self.pattern_weights['trend_alignment'] +
                breakout_score * self.pattern_weights['breakout_strength']
            )

            # Targets
            entry_price = neckline * 1.002
            target_price = entry_price + pattern_depth
            stop_loss = min(bottom1_price, bottom2_price) * 0.99

            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0

            confidence = 'HIGH' if total_score >= 80 else 'MEDIUM' if total_score >= 60 else 'LOW'

            return PatternResult(
                detected=total_score >= min_score,
                pattern_type=PatternType.DOUBLE_BOTTOM,
                score=total_score,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward=risk_reward,
                confidence=confidence,
                description=f'Double Bottom: Depth {pattern_depth:.2f}, RR {risk_reward:.2f}'
            )

        except Exception as e:
            system_logger.debug(f"Erro detectando Double Bottom: {e}")
            return PatternResult(
                detected=False, pattern_type=PatternType.DOUBLE_BOTTOM,
                score=0, entry_price=0, target_price=0, stop_loss=0,
                risk_reward=0, confidence='LOW', description=f'Erro: {str(e)}'
            )

    def detect_symmetrical_triangle(self, df: pd.DataFrame, min_score: int = 70) -> PatternResult:
        """
        Detecta Triângulo Simétrico:
        - Topos descendentes (resistência descendente)
        - Fundos ascendentes (suporte ascendente)
        - Convergência dos dois lados
        """
        if len(df) < 20:
            return PatternResult(
                detected=False, pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                score=0, entry_price=0, target_price=0, stop_loss=0,
                risk_reward=0, confidence='LOW', description='Dados insuficientes'
            )

        try:
            # V4.1 FIX: Usa extração segura de OHLCV
            closes, highs, lows, _ = self._get_ohlcv_safe(df)
            if closes is None:
                return PatternResult(
                    detected=False, pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Erro ao extrair dados OHLCV'
                )

            # Encontra topos
            peaks = []
            for i in range(2, len(df) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))

            # Encontra fundos
            troughs = []
            for i in range(2, len(df) - 2):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    troughs.append((i, lows[i]))

            if len(peaks) < 2 or len(troughs) < 2:
                return PatternResult(
                    detected=False, pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Pontos insuficientes'
                )

            # Verifica topos descendentes
            peak_prices = [p[1] for p in peaks[-3:]]
            descending_highs = all(peak_prices[i] >= peak_prices[i+1] for i in range(len(peak_prices)-1))

            # Verifica fundos ascendentes
            trough_prices = [t[1] for t in troughs[-3:]]
            ascending_lows = all(trough_prices[i] <= trough_prices[i+1] for i in range(len(trough_prices)-1))

            if not (descending_highs and ascending_lows):
                return PatternResult(
                    detected=False, pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                    score=0, entry_price=0, target_price=0, stop_loss=0,
                    risk_reward=0, confidence='LOW', description='Não é triângulo simétrico'
                )

            # Calcula níveis
            resistance = peak_prices[-1]
            support = trough_prices[-1]
            apex = (resistance + support) / 2
            triangle_height = peak_prices[0] - trough_prices[0]
            current_price = closes[-1]

            # Verifica breakout (pode ser para cima ou para baixo)
            breakout_up = current_price > resistance
            breakout_down = current_price < support

            # Score
            clarity_score = 75
            volume_score = 70
            trend_score = 60  # Neutro (pode ir para qualquer lado)
            breakout_score = 90 if (breakout_up or breakout_down) else 30

            total_score = (
                clarity_score * self.pattern_weights['clarity'] +
                volume_score * self.pattern_weights['volume'] +
                trend_score * self.pattern_weights['trend_alignment'] +
                breakout_score * self.pattern_weights['breakout_strength']
            )

            # Targets (assumindo breakout para cima)
            if breakout_up:
                entry_price = resistance * 1.002
                target_price = entry_price + triangle_height * 0.7
                stop_loss = support * 0.995
            else:
                entry_price = support * 0.998
                target_price = entry_price - triangle_height * 0.7
                stop_loss = resistance * 1.005

            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward = reward / risk if risk > 0 else 0

            confidence = 'HIGH' if total_score >= 80 else 'MEDIUM' if total_score >= 60 else 'LOW'

            return PatternResult(
                detected=total_score >= min_score,
                pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                score=total_score,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward=risk_reward,
                confidence=confidence,
                description=f'Symmetrical Triangle: Height {triangle_height:.2f}, RR {risk_reward:.2f}'
            )

        except Exception as e:
            system_logger.debug(f"Erro detectando Symmetrical Triangle: {e}")
            return PatternResult(
                detected=False, pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                score=0, entry_price=0, target_price=0, stop_loss=0,
                risk_reward=0, confidence='LOW', description=f'Erro: {str(e)}'
            )

    def scan_all_patterns(self, symbol: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Escaneia todos os padrões para um símbolo.
        Retorna lista de padrões detectados ordenados por score.
        """
        detected_patterns = []

        # Lista de funções de detecção
        detectors = [
            ('BULL_FLAG', self.detect_bull_flag),
            ('ASCENDING_TRIANGLE', self.detect_ascending_triangle),
            ('DOUBLE_BOTTOM', self.detect_double_bottom),
            ('SYMMETRICAL_TRIANGLE', self.detect_symmetrical_triangle),
        ]

        for pattern_name, detector_func in detectors:
            try:
                result = detector_func(df, self.min_pattern_score)
                if result.detected:
                    detected_patterns.append({
                        'symbol': symbol,
                        'pattern': pattern_name,
                        'score': result.score,
                        'entry_price': result.entry_price,
                        'target_price': result.target_price,
                        'stop_loss': result.stop_loss,
                        'risk_reward': result.risk_reward,
                        'confidence': result.confidence,
                        'description': result.description
                    })
            except Exception as e:
                system_logger.debug(f"Erro escaneando {pattern_name} para {symbol}: {e}")

        # Ordena por score
        detected_patterns.sort(key=lambda x: x['score'], reverse=True)

        return detected_patterns

    def get_best_pattern(self, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Retorna o melhor padrão detectado para um símbolo.
        """
        patterns = self.scan_all_patterns(symbol, df)
        return patterns[0] if patterns else None

    def calculate_pattern_bonus(self, patterns: List[Dict[str, Any]]) -> float:
        """
        Calcula bonus de score baseado em padrões detectados.
        Retorna multiplicador de 1.0 a 1.5.
        """
        if not patterns:
            return 1.0

        best_score = patterns[0]['score']
        best_rr = patterns[0]['risk_reward']

        # Base bonus de 10% para qualquer padrão detectado
        bonus = 1.10

        # Bonus adicional por score alto
        if best_score >= 85:
            bonus += 0.20
        elif best_score >= 75:
            bonus += 0.10

        # Bonus adicional por RR alto
        if best_rr >= 3.0:
            bonus += 0.15
        elif best_rr >= 2.0:
            bonus += 0.05

        return min(bonus, 1.50)  # Cap em 50% bonus
