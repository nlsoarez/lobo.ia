"""
V4.0 Phase 2: Breakout Detector
Detecta e confirma breakouts de suporte/resistência.
Calcula força do breakout e probabilidade de sucesso.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from system_logger import system_logger


class BreakoutType(Enum):
    """Tipos de breakout."""
    RESISTANCE_BREAK = "resistance_break"
    SUPPORT_BREAK = "support_break"
    RANGE_BREAK_UP = "range_break_up"
    RANGE_BREAK_DOWN = "range_break_down"
    FALSE_BREAKOUT = "false_breakout"


@dataclass
class SupportResistanceLevel:
    """Nível de suporte ou resistência."""
    price: float
    strength: int  # Número de toques
    level_type: str  # 'support' ou 'resistance'
    last_touch: int  # Índice do último toque
    is_valid: bool


@dataclass
class BreakoutResult:
    """Resultado da detecção de breakout."""
    detected: bool
    breakout_type: BreakoutType
    breakout_price: float
    level_broken: float
    strength_score: float  # 0-100
    volume_confirmation: bool
    bars_since_break: int
    is_confirmed: bool
    target_price: float
    stop_loss: float
    risk_reward: float
    confidence: str  # LOW, MEDIUM, HIGH


class BreakoutDetector:
    """
    V4.0 Phase 2: Detector de breakouts.
    Identifica e confirma breakouts com alta probabilidade.
    """

    def __init__(self, confirmation_bars: int = 2):
        """
        Inicializa o detector de breakouts.

        Args:
            confirmation_bars: Número de barras para confirmar breakout
        """
        self.confirmation_bars = confirmation_bars
        self.min_touches = 2  # Mínimo de toques para nível válido
        self.tolerance_pct = 0.005  # 0.5% tolerância para definir nível
        self.breakout_threshold = 0.002  # 0.2% acima/abaixo do nível

        system_logger.info(f"BreakoutDetector V4.0 inicializado (confirmação: {confirmation_bars} barras)")

    def find_support_resistance_levels(
        self,
        df: pd.DataFrame,
        lookback: int = 50,
        num_levels: int = 5
    ) -> List[SupportResistanceLevel]:
        """
        Encontra níveis de suporte e resistência baseado em pivôs.
        """
        if len(df) < lookback:
            return []

        highs = df['high'].values[-lookback:]
        lows = df['low'].values[-lookback:]
        closes = df['close'].values[-lookback:]

        levels = []

        # Encontra pivôs de alta (resistências)
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                levels.append({
                    'price': highs[i],
                    'type': 'resistance',
                    'index': len(df) - lookback + i
                })

        # Encontra pivôs de baixa (suportes)
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                levels.append({
                    'price': lows[i],
                    'type': 'support',
                    'index': len(df) - lookback + i
                })

        # Agrupa níveis próximos
        grouped_levels = self._group_nearby_levels(levels, tolerance_pct=self.tolerance_pct)

        # Converte para SupportResistanceLevel
        result = []
        current_price = closes[-1]

        for level in grouped_levels[:num_levels * 2]:
            sr_level = SupportResistanceLevel(
                price=level['price'],
                strength=level.get('touches', 1),
                level_type=level['type'],
                last_touch=level.get('last_index', 0),
                is_valid=level.get('touches', 1) >= self.min_touches
            )
            result.append(sr_level)

        # Ordena por proximidade do preço atual
        result.sort(key=lambda x: abs(x.price - current_price))

        return result[:num_levels]

    def _group_nearby_levels(self, levels: List[Dict], tolerance_pct: float) -> List[Dict]:
        """
        Agrupa níveis de preço próximos.
        """
        if not levels:
            return []

        # Ordena por preço
        sorted_levels = sorted(levels, key=lambda x: x['price'])

        grouped = []
        current_group = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            last_price = current_group[-1]['price']
            if abs(level['price'] - last_price) / last_price <= tolerance_pct:
                current_group.append(level)
            else:
                # Finaliza grupo atual
                avg_price = np.mean([l['price'] for l in current_group])
                level_type = 'resistance' if current_group[0]['type'] == 'resistance' else 'support'
                last_index = max(l['index'] for l in current_group)
                grouped.append({
                    'price': avg_price,
                    'type': level_type,
                    'touches': len(current_group),
                    'last_index': last_index
                })
                current_group = [level]

        # Finaliza último grupo
        if current_group:
            avg_price = np.mean([l['price'] for l in current_group])
            level_type = 'resistance' if current_group[0]['type'] == 'resistance' else 'support'
            last_index = max(l['index'] for l in current_group)
            grouped.append({
                'price': avg_price,
                'type': level_type,
                'touches': len(current_group),
                'last_index': last_index
            })

        # Ordena por força (número de toques)
        grouped.sort(key=lambda x: x['touches'], reverse=True)

        return grouped

    def detect_breakout(
        self,
        df: pd.DataFrame,
        resistance_level: Optional[float] = None,
        support_level: Optional[float] = None
    ) -> BreakoutResult:
        """
        Detecta breakout de níveis de suporte ou resistência.
        """
        if len(df) < 10:
            return self._empty_result()

        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            current_price = closes[-1]

            # Se níveis não fornecidos, encontra automaticamente
            if resistance_level is None or support_level is None:
                levels = self.find_support_resistance_levels(df)
                if len(levels) >= 2:
                    # Pega resistência mais próxima acima e suporte mais próximo abaixo
                    for level in levels:
                        if level.level_type == 'resistance' and level.price > current_price:
                            resistance_level = level.price
                            break
                    for level in levels:
                        if level.level_type == 'support' and level.price < current_price:
                            support_level = level.price
                            break

            if resistance_level is None and support_level is None:
                return self._empty_result()

            # Detecta breakout de resistência
            if resistance_level is not None:
                resistance_break = self._check_resistance_breakout(
                    closes, highs, resistance_level
                )
                if resistance_break['detected']:
                    return self._create_breakout_result(
                        df, BreakoutType.RESISTANCE_BREAK,
                        resistance_level, support_level, resistance_break
                    )

            # Detecta breakout de suporte
            if support_level is not None:
                support_break = self._check_support_breakout(
                    closes, lows, support_level
                )
                if support_break['detected']:
                    return self._create_breakout_result(
                        df, BreakoutType.SUPPORT_BREAK,
                        support_level, resistance_level, support_break
                    )

            return self._empty_result()

        except Exception as e:
            system_logger.debug(f"Erro detectando breakout: {e}")
            return self._empty_result()

    def _check_resistance_breakout(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        resistance: float
    ) -> Dict[str, Any]:
        """
        Verifica breakout de resistência.
        """
        threshold = resistance * (1 + self.breakout_threshold)
        recent_closes = closes[-self.confirmation_bars - 1:]
        recent_highs = highs[-self.confirmation_bars - 1:]

        # Verifica se fechou acima da resistência nas últimas N barras
        closes_above = sum(1 for c in recent_closes if c > resistance)
        highs_above = sum(1 for h in recent_highs if h > threshold)

        detected = closes_above >= self.confirmation_bars and highs_above >= 1
        bars_since = 0

        if detected:
            # Conta barras desde o breakout
            for i in range(len(closes) - 1, -1, -1):
                if closes[i] <= resistance:
                    break
                bars_since += 1

        return {
            'detected': detected,
            'closes_above': closes_above,
            'bars_since': bars_since
        }

    def _check_support_breakout(
        self,
        closes: np.ndarray,
        lows: np.ndarray,
        support: float
    ) -> Dict[str, Any]:
        """
        Verifica breakout de suporte.
        """
        threshold = support * (1 - self.breakout_threshold)
        recent_closes = closes[-self.confirmation_bars - 1:]
        recent_lows = lows[-self.confirmation_bars - 1:]

        # Verifica se fechou abaixo do suporte nas últimas N barras
        closes_below = sum(1 for c in recent_closes if c < support)
        lows_below = sum(1 for l in recent_lows if l < threshold)

        detected = closes_below >= self.confirmation_bars and lows_below >= 1
        bars_since = 0

        if detected:
            for i in range(len(closes) - 1, -1, -1):
                if closes[i] >= support:
                    break
                bars_since += 1

        return {
            'detected': detected,
            'closes_below': closes_below,
            'bars_since': bars_since
        }

    def _create_breakout_result(
        self,
        df: pd.DataFrame,
        breakout_type: BreakoutType,
        level_broken: float,
        opposite_level: Optional[float],
        break_info: Dict
    ) -> BreakoutResult:
        """
        Cria resultado de breakout com targets e scores.
        """
        closes = df['close'].values
        volumes = df['volume'].values
        current_price = closes[-1]

        # Verifica confirmação de volume
        avg_volume = np.mean(volumes[-20:])
        recent_volume = np.mean(volumes[-3:])
        volume_confirmation = recent_volume > avg_volume * 1.3

        # Calcula força do breakout
        strength_score = self._calculate_breakout_strength(
            df, level_broken, breakout_type, volume_confirmation
        )

        # Calcula targets
        if breakout_type == BreakoutType.RESISTANCE_BREAK:
            # Long trade
            entry = current_price
            range_size = level_broken - (opposite_level or level_broken * 0.95)
            target = entry + range_size  # Projeção do range
            stop_loss = level_broken * 0.995  # Ligeiramente abaixo da resistência
        else:
            # Short trade (ou evitar)
            entry = current_price
            range_size = (opposite_level or level_broken * 1.05) - level_broken
            target = entry - range_size
            stop_loss = level_broken * 1.005

        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        risk_reward = reward / risk if risk > 0 else 0

        # Determina confiança
        if strength_score >= 80 and volume_confirmation:
            confidence = 'HIGH'
        elif strength_score >= 60 or volume_confirmation:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        return BreakoutResult(
            detected=True,
            breakout_type=breakout_type,
            breakout_price=current_price,
            level_broken=level_broken,
            strength_score=strength_score,
            volume_confirmation=volume_confirmation,
            bars_since_break=break_info['bars_since'],
            is_confirmed=break_info['bars_since'] >= self.confirmation_bars,
            target_price=target,
            stop_loss=stop_loss,
            risk_reward=risk_reward,
            confidence=confidence
        )

    def _calculate_breakout_strength(
        self,
        df: pd.DataFrame,
        level: float,
        breakout_type: BreakoutType,
        volume_confirmed: bool
    ) -> float:
        """
        Calcula força do breakout (0-100).
        """
        closes = df['close'].values
        current_price = closes[-1]

        # Distância do nível (40%)
        distance_pct = abs(current_price - level) / level
        if distance_pct >= 0.03:
            distance_score = 100
        elif distance_pct >= 0.02:
            distance_score = 80
        elif distance_pct >= 0.01:
            distance_score = 60
        else:
            distance_score = 40

        # Confirmação de volume (30%)
        volume_score = 100 if volume_confirmed else 40

        # Fechamento acima/abaixo (30%)
        if breakout_type == BreakoutType.RESISTANCE_BREAK:
            close_score = 100 if current_price > level else 50
        else:
            close_score = 100 if current_price < level else 50

        total_score = (
            distance_score * 0.40 +
            volume_score * 0.30 +
            close_score * 0.30
        )

        return min(100, total_score)

    def _empty_result(self) -> BreakoutResult:
        """Retorna resultado vazio."""
        return BreakoutResult(
            detected=False,
            breakout_type=BreakoutType.FALSE_BREAKOUT,
            breakout_price=0,
            level_broken=0,
            strength_score=0,
            volume_confirmation=False,
            bars_since_break=0,
            is_confirmed=False,
            target_price=0,
            stop_loss=0,
            risk_reward=0,
            confidence='LOW'
        )

    def detect_range_breakout(self, df: pd.DataFrame, range_periods: int = 20) -> BreakoutResult:
        """
        Detecta breakout de range (consolidação).
        """
        if len(df) < range_periods + 5:
            return self._empty_result()

        try:
            # Define o range das últimas N barras
            range_data = df.iloc[-(range_periods + 5):-5]
            recent_data = df.iloc[-5:]

            range_high = range_data['high'].max()
            range_low = range_data['low'].min()
            range_size = range_high - range_low

            current_price = df['close'].iloc[-1]

            # Verifica breakout do range
            if current_price > range_high * 1.002:
                # Breakout para cima
                break_info = {'detected': True, 'bars_since': 1}
                result = self._create_breakout_result(
                    df, BreakoutType.RANGE_BREAK_UP,
                    range_high, range_low, break_info
                )
                result.target_price = current_price + range_size
                return result

            elif current_price < range_low * 0.998:
                # Breakout para baixo
                break_info = {'detected': True, 'bars_since': 1}
                result = self._create_breakout_result(
                    df, BreakoutType.RANGE_BREAK_DOWN,
                    range_low, range_high, break_info
                )
                result.target_price = current_price - range_size
                return result

            return self._empty_result()

        except Exception as e:
            system_logger.debug(f"Erro detectando range breakout: {e}")
            return self._empty_result()

    def is_false_breakout(self, df: pd.DataFrame, level: float, breakout_type: BreakoutType) -> bool:
        """
        Verifica se é um falso breakout (price returned to range).
        """
        if len(df) < 5:
            return False

        closes = df['close'].values
        current_price = closes[-1]

        if breakout_type == BreakoutType.RESISTANCE_BREAK:
            # Falso se preço voltou para abaixo da resistência
            broke_above = any(c > level for c in closes[-5:-1])
            now_below = current_price < level
            return broke_above and now_below

        elif breakout_type == BreakoutType.SUPPORT_BREAK:
            # Falso se preço voltou para acima do suporte
            broke_below = any(c < level for c in closes[-5:-1])
            now_above = current_price > level
            return broke_below and now_above

        return False

    def get_best_breakout(self, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Retorna o melhor breakout detectado para um símbolo.
        """
        # Tenta detectar breakout de S/R
        sr_breakout = self.detect_breakout(df)

        # Tenta detectar breakout de range
        range_breakout = self.detect_range_breakout(df)

        # Retorna o melhor
        breakouts = []
        if sr_breakout.detected:
            breakouts.append(sr_breakout)
        if range_breakout.detected:
            breakouts.append(range_breakout)

        if not breakouts:
            return None

        # Ordena por score
        best = max(breakouts, key=lambda x: x.strength_score)

        return {
            'symbol': symbol,
            'type': best.breakout_type.value,
            'level_broken': best.level_broken,
            'current_price': best.breakout_price,
            'target': best.target_price,
            'stop_loss': best.stop_loss,
            'risk_reward': best.risk_reward,
            'strength': best.strength_score,
            'volume_confirmed': best.volume_confirmation,
            'confidence': best.confidence,
            'is_confirmed': best.is_confirmed
        }

    def calculate_breakout_bonus(self, breakout_result: BreakoutResult) -> float:
        """
        Calcula bonus de score baseado em breakout.
        Retorna multiplicador de 1.0 a 1.4.
        """
        if not breakout_result.detected:
            return 1.0

        bonus = 1.0

        # Bonus base por breakout detectado
        bonus += 0.10

        # Bonus por confirmação de volume
        if breakout_result.volume_confirmation:
            bonus += 0.10

        # Bonus por força
        if breakout_result.strength_score >= 80:
            bonus += 0.15
        elif breakout_result.strength_score >= 60:
            bonus += 0.05

        # Bonus por confirmação
        if breakout_result.is_confirmed:
            bonus += 0.05

        return min(bonus, 1.40)
