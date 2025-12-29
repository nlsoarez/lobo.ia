"""
V4.0 Phase 4: Market Regime Detector
Detecta regimes de mercado usando diferentes métodos estatísticos.
Implementação simplificada sem dependências de HMM externas.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from system_logger import system_logger


class MarketRegime(Enum):
    """Tipos de regime de mercado."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


@dataclass
class RegimeDetection:
    """Resultado de detecção de regime."""
    regime: MarketRegime
    confidence: float
    probabilities: Dict[str, float]
    features: Dict[str, float]
    timestamp: datetime


@dataclass
class RegimeStrategy:
    """Estratégia específica para um regime."""
    aggressiveness: float
    position_sizing: float
    take_profit_multiplier: float
    stop_loss_multiplier: float
    max_positions: int


class MarketRegimeDetector:
    """
    V4.0 Phase 4: Detector de regime de mercado.
    Usa análise estatística para identificar regimes.
    """

    def __init__(self, n_regimes: int = 5):
        """Inicializa o detector."""
        self.n_regimes = n_regimes
        self.regime_labels = list(MarketRegime)
        self.current_regime: Optional[MarketRegime] = None
        self.regime_history: deque = deque(maxlen=1000)
        self.feature_history: deque = deque(maxlen=500)

        # Thresholds para detecção
        self.thresholds = {
            'trend_strong': 0.02,      # 2% para tendência forte
            'trend_weak': 0.005,       # 0.5% para tendência fraca
            'vol_high': 0.03,          # 3% para alta volatilidade
            'vol_low': 0.01,           # 1% para baixa volatilidade
            'vol_ratio_high': 1.5,     # Ratio acima de 1.5 = alta vol
            'vol_ratio_low': 0.7,      # Ratio abaixo de 0.7 = baixa vol
        }

        # Estratégias por regime
        self.regime_strategies = {
            MarketRegime.BULL: RegimeStrategy(
                aggressiveness=1.2,
                position_sizing=1.1,
                take_profit_multiplier=1.0,
                stop_loss_multiplier=0.8,
                max_positions=6
            ),
            MarketRegime.BEAR: RegimeStrategy(
                aggressiveness=0.6,
                position_sizing=0.7,
                take_profit_multiplier=0.8,
                stop_loss_multiplier=1.2,
                max_positions=3
            ),
            MarketRegime.SIDEWAYS: RegimeStrategy(
                aggressiveness=0.9,
                position_sizing=0.9,
                take_profit_multiplier=0.7,
                stop_loss_multiplier=1.0,
                max_positions=4
            ),
            MarketRegime.HIGH_VOL: RegimeStrategy(
                aggressiveness=0.7,
                position_sizing=0.8,
                take_profit_multiplier=1.3,
                stop_loss_multiplier=1.5,
                max_positions=3
            ),
            MarketRegime.LOW_VOL: RegimeStrategy(
                aggressiveness=1.0,
                position_sizing=1.0,
                take_profit_multiplier=0.9,
                stop_loss_multiplier=0.9,
                max_positions=5
            ),
        }

        # Matriz de transição (estimada)
        self.transition_counts = np.ones((n_regimes, n_regimes))  # Laplace smoothing

        system_logger.info(f"MarketRegimeDetector V4.0 inicializado com {n_regimes} regimes")

    def prepare_features(self, price_data: List[Dict]) -> Dict[str, float]:
        """
        Prepara features para detecção de regime.

        Args:
            price_data: Lista de candles com 'close', 'high', 'low', 'volume'
        """
        if len(price_data) < 50:
            return {}

        closes = np.array([d['close'] for d in price_data])
        volumes = np.array([d.get('volume', 1) for d in price_data])
        highs = np.array([d.get('high', d['close']) for d in price_data])
        lows = np.array([d.get('low', d['close']) for d in price_data])

        features = {}

        # 1. Tendência de preço (20d e 50d)
        features['price_trend_20d'] = (closes[-1] - closes[-20]) / closes[-20] if closes[-20] > 0 else 0
        features['price_trend_50d'] = (closes[-1] - closes[-50]) / closes[-50] if closes[-50] > 0 else 0

        # 2. Volatilidade
        returns = np.diff(closes) / closes[:-1]
        features['volatility_20d'] = np.std(returns[-20:])
        features['volatility_50d'] = np.std(returns[-50:])
        features['volatility_ratio'] = features['volatility_20d'] / (features['volatility_50d'] + 0.001)

        # 3. Volume trend
        vol_ma_20 = np.mean(volumes[-20:])
        vol_ma_50 = np.mean(volumes[-50:])
        features['volume_trend'] = (vol_ma_20 - vol_ma_50) / (vol_ma_50 + 0.001)
        features['volume_ratio'] = vol_ma_20 / (vol_ma_50 + 0.001)

        # 4. Range (ATR simplificado)
        tr = np.maximum(highs[-20:] - lows[-20:],
                       np.abs(highs[-20:] - np.roll(closes[-20:], 1)),
                       np.abs(lows[-20:] - np.roll(closes[-20:], 1)))
        features['atr_20d'] = np.mean(tr[1:])  # Ignora primeiro (roll artifact)
        features['atr_pct'] = features['atr_20d'] / closes[-1] if closes[-1] > 0 else 0

        # 5. RSI
        features['rsi_14'] = self._calculate_rsi(closes, 14)

        # 6. MACD
        ema_12 = self._calculate_ema(closes, 12)
        ema_26 = self._calculate_ema(closes, 26)
        features['macd'] = (ema_12 - ema_26) / closes[-1] if closes[-1] > 0 else 0

        # 7. Momentum
        features['momentum_10d'] = (closes[-1] - closes[-10]) / closes[-10] if closes[-10] > 0 else 0

        # 8. Higher highs / Lower lows
        features['higher_highs'] = self._count_higher_highs(highs[-20:])
        features['lower_lows'] = self._count_lower_lows(lows[-20:])

        self.feature_history.append({
            'timestamp': datetime.now(),
            'features': features.copy()
        })

        return features

    def detect_regime(self, features: Dict[str, float]) -> RegimeDetection:
        """
        Detecta regime atual baseado em features.
        """
        if not features:
            # Default para sideways se sem dados
            return RegimeDetection(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.5,
                probabilities={r.value: 0.2 for r in MarketRegime},
                features={},
                timestamp=datetime.now()
            )

        # Calcula probabilidades para cada regime
        probabilities = self._calculate_regime_probabilities(features)

        # Determina regime principal
        regime_idx = np.argmax(list(probabilities.values()))
        regime = self.regime_labels[regime_idx]
        confidence = list(probabilities.values())[regime_idx]

        # Atualiza histórico
        detection = RegimeDetection(
            regime=regime,
            confidence=confidence,
            probabilities=probabilities,
            features=features,
            timestamp=datetime.now()
        )

        # Atualiza matriz de transição
        if self.current_regime is not None:
            prev_idx = self.regime_labels.index(self.current_regime)
            curr_idx = self.regime_labels.index(regime)
            self.transition_counts[prev_idx][curr_idx] += 1

        self.current_regime = regime
        self.regime_history.append(detection)

        return detection

    def _calculate_regime_probabilities(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calcula probabilidades para cada regime."""
        scores = {r.value: 0.0 for r in MarketRegime}

        trend_20d = features.get('price_trend_20d', 0)
        trend_50d = features.get('price_trend_50d', 0)
        vol_20d = features.get('volatility_20d', 0.02)
        vol_ratio = features.get('volatility_ratio', 1.0)
        rsi = features.get('rsi_14', 50)
        macd = features.get('macd', 0)
        momentum = features.get('momentum_10d', 0)
        hh = features.get('higher_highs', 0)
        ll = features.get('lower_lows', 0)

        # BULL: Tendência positiva forte
        if trend_20d > self.thresholds['trend_strong']:
            scores['bull'] += 3.0
        if trend_50d > self.thresholds['trend_weak']:
            scores['bull'] += 2.0
        if rsi > 50 and rsi < 70:
            scores['bull'] += 1.5
        if macd > 0:
            scores['bull'] += 1.0
        if hh > ll:
            scores['bull'] += 1.0

        # BEAR: Tendência negativa forte
        if trend_20d < -self.thresholds['trend_strong']:
            scores['bear'] += 3.0
        if trend_50d < -self.thresholds['trend_weak']:
            scores['bear'] += 2.0
        if rsi < 50 and rsi > 30:
            scores['bear'] += 1.5
        if macd < 0:
            scores['bear'] += 1.0
        if ll > hh:
            scores['bear'] += 1.0

        # SIDEWAYS: Sem tendência clara
        if abs(trend_20d) < self.thresholds['trend_weak']:
            scores['sideways'] += 3.0
        if abs(trend_50d) < self.thresholds['trend_weak']:
            scores['sideways'] += 2.0
        if 40 < rsi < 60:
            scores['sideways'] += 1.5
        if abs(macd) < 0.001:
            scores['sideways'] += 1.0
        if abs(hh - ll) < 3:
            scores['sideways'] += 1.0

        # HIGH_VOL: Volatilidade alta
        if vol_20d > self.thresholds['vol_high']:
            scores['high_vol'] += 4.0
        if vol_ratio > self.thresholds['vol_ratio_high']:
            scores['high_vol'] += 2.0
        if rsi > 70 or rsi < 30:
            scores['high_vol'] += 1.0

        # LOW_VOL: Volatilidade baixa
        if vol_20d < self.thresholds['vol_low']:
            scores['low_vol'] += 4.0
        if vol_ratio < self.thresholds['vol_ratio_low']:
            scores['low_vol'] += 2.0

        # Normaliza para probabilidades
        total = sum(scores.values())
        if total > 0:
            probabilities = {k: v / total for k, v in scores.items()}
        else:
            probabilities = {r.value: 0.2 for r in MarketRegime}

        return probabilities

    def get_regime_strategy(self, regime: Optional[MarketRegime] = None) -> RegimeStrategy:
        """Retorna estratégia para o regime."""
        if regime is None:
            regime = self.current_regime or MarketRegime.SIDEWAYS

        return self.regime_strategies.get(regime, self.regime_strategies[MarketRegime.SIDEWAYS])

    def predict_next_regime(self) -> Optional[Dict[str, Any]]:
        """Prediz próximo regime baseado em matriz de transição."""
        if self.current_regime is None:
            return None

        current_idx = self.regime_labels.index(self.current_regime)

        # Normaliza linha da matriz de transição
        row = self.transition_counts[current_idx]
        probabilities = row / row.sum()

        next_regime_idx = np.argmax(probabilities)
        next_regime = self.regime_labels[next_regime_idx]

        return {
            'current_regime': self.current_regime.value,
            'next_regime': next_regime.value,
            'probability': probabilities[next_regime_idx],
            'all_probabilities': {
                self.regime_labels[i].value: p
                for i, p in enumerate(probabilities)
            }
        }

    def get_transition_matrix(self) -> np.ndarray:
        """Retorna matriz de transição normalizada."""
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        return self.transition_counts / row_sums

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas dos regimes."""
        if not self.regime_history:
            return {'total_detections': 0}

        regime_counts = {r.value: 0 for r in MarketRegime}
        regime_durations = {r.value: [] for r in MarketRegime}

        current_regime = None
        current_start = None

        for detection in self.regime_history:
            regime_counts[detection.regime.value] += 1

            if detection.regime != current_regime:
                if current_regime is not None and current_start is not None:
                    duration = (detection.timestamp - current_start).total_seconds() / 3600
                    regime_durations[current_regime.value].append(duration)

                current_regime = detection.regime
                current_start = detection.timestamp

        # Calcula estatísticas
        stats = {
            'total_detections': len(self.regime_history),
            'regime_distribution': {
                k: v / len(self.regime_history) * 100
                for k, v in regime_counts.items()
            },
            'avg_regime_duration_hours': {
                k: np.mean(v) if v else 0
                for k, v in regime_durations.items()
            },
            'current_regime': self.current_regime.value if self.current_regime else None,
            'transition_matrix': self.get_transition_matrix().tolist()
        }

        return stats

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calcula RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calcula EMA."""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _count_higher_highs(self, highs: np.ndarray) -> int:
        """Conta higher highs em uma série."""
        count = 0
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                count += 1
        return count

    def _count_lower_lows(self, lows: np.ndarray) -> int:
        """Conta lower lows em uma série."""
        count = 0
        for i in range(1, len(lows)):
            if lows[i] < lows[i-1]:
                count += 1
        return count

    def log_regime_status(self):
        """Loga status do regime atual."""
        if self.current_regime is None:
            system_logger.info("\n📊 REGIME: Não detectado")
            return

        strategy = self.get_regime_strategy()
        prediction = self.predict_next_regime()

        system_logger.info(f"\n📊 REGIME DE MERCADO:")
        system_logger.info(f"   Regime atual: {self.current_regime.value.upper()}")
        system_logger.info(f"   Agressividade: {strategy.aggressiveness:.1f}x")
        system_logger.info(f"   Position sizing: {strategy.position_sizing:.1f}x")
        system_logger.info(f"   TP multiplier: {strategy.take_profit_multiplier:.1f}x")
        system_logger.info(f"   SL multiplier: {strategy.stop_loss_multiplier:.1f}x")

        if prediction:
            system_logger.info(f"   Próximo regime: {prediction['next_regime']} "
                             f"({prediction['probability']*100:.0f}%)")

