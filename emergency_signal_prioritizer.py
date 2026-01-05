"""
Emergency Signal Prioritizer - Priorização de Sinais V4.2
Sistema para priorizar sinais durante modo emergência.

Features:
- Classificação por múltiplos critérios
- Identificação de sinais CRITICAL
- Bypass automático de limites para sinais excepcionais
- Análise de tendência de curto prazo
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from config_loader import config
from system_logger import system_logger


class SignalPriority(Enum):
    """Níveis de prioridade de sinais."""
    CRITICAL = "critical"      # Bypass de limites permitido
    HIGH = "high"              # Prioridade máxima dentro de limites
    MEDIUM = "medium"          # Prioridade normal
    LOW = "low"                # Baixa prioridade
    SKIP = "skip"              # Ignorar


@dataclass
class PrioritizedSignal:
    """Sinal com dados de priorização."""
    symbol: str
    original_signal: Dict[str, Any]
    priority: SignalPriority
    priority_score: float
    criteria_met: List[str]
    criteria_failed: List[str]
    recommended_action: str
    bypass_allowed: bool
    position_size_multiplier: float


class EmergencySignalPrioritizer:
    """
    Sistema de priorização de sinais durante emergência.
    Classifica sinais por múltiplos critérios para execução otimizada.
    """

    def __init__(self):
        """Inicializa o priorizador de sinais."""
        # Thresholds para classificação
        self.critical_rsi_low = config.get('emergency.critical_signals.max_rsi', 20)
        self.critical_rsi_high = 80
        self.critical_score = config.get('emergency.critical_signals.min_score', 65)
        self.critical_volume = config.get('emergency.critical_signals.min_volume_ratio', 1.5)

        # Pesos para cálculo de priority_score
        self.weights = {
            'rsi_extreme': 30,      # RSI muito baixo ou alto
            'score_high': 25,       # Score acima do threshold
            'volume_high': 20,      # Volume acima da média
            'trend_aligned': 15,    # Tendência de curto prazo alinhada
            'momentum_strong': 10   # Momentum forte (MACD)
        }

        # Limites de sinais por categoria
        self.max_critical_signals = 3
        self.max_high_signals = 5

        system_logger.info(
            f"EmergencySignalPrioritizer V4.2 inicializado: "
            f"RSI critical=<{self.critical_rsi_low} or >{self.critical_rsi_high}, "
            f"Score critical>={self.critical_score}"
        )

    def prioritize_signals(
        self,
        signals: List[Dict[str, Any]],
        emergency_mode: bool = False
    ) -> List[PrioritizedSignal]:
        """
        Prioriza lista de sinais baseado em múltiplos critérios.

        Args:
            signals: Lista de sinais do scanner
            emergency_mode: Se modo emergência está ativo

        Returns:
            Lista de sinais priorizados ordenados por prioridade
        """
        prioritized = []

        for signal in signals:
            priority_data = self._calculate_priority(signal, emergency_mode)
            prioritized.append(priority_data)

        # Ordena por priority_score (maior primeiro)
        prioritized.sort(key=lambda x: x.priority_score, reverse=True)

        # Limita número de sinais CRITICAL e HIGH
        critical_count = 0
        high_count = 0
        filtered = []

        for p in prioritized:
            if p.priority == SignalPriority.CRITICAL:
                if critical_count < self.max_critical_signals:
                    filtered.append(p)
                    critical_count += 1
                else:
                    # Rebaixa para HIGH
                    p.priority = SignalPriority.HIGH
                    p.bypass_allowed = False
                    filtered.append(p)
            elif p.priority == SignalPriority.HIGH:
                if high_count < self.max_high_signals:
                    filtered.append(p)
                    high_count += 1
                else:
                    p.priority = SignalPriority.MEDIUM
                    filtered.append(p)
            else:
                filtered.append(p)

        # Log resumo
        system_logger.info(
            f"Sinais priorizados: {len(filtered)} total, "
            f"CRITICAL={critical_count}, HIGH={high_count}, "
            f"Emergency={emergency_mode}"
        )

        return filtered

    def _calculate_priority(
        self,
        signal: Dict[str, Any],
        emergency_mode: bool
    ) -> PrioritizedSignal:
        """
        Calcula prioridade de um sinal individual.

        Args:
            signal: Dados do sinal
            emergency_mode: Se modo emergência está ativo

        Returns:
            Sinal com dados de priorização
        """
        symbol = signal.get('symbol', 'UNKNOWN')
        score = signal.get('total_score', signal.get('score', 0))
        rsi = signal.get('rsi', 50)
        volume_ratio = signal.get('volume_ratio', 1.0)
        signal_type = signal.get('signal', '')

        criteria_met = []
        criteria_failed = []
        priority_score = 0.0

        # 1. RSI extremo (maior peso)
        rsi_oversold = rsi <= self.critical_rsi_low
        rsi_overbought = rsi >= self.critical_rsi_high

        if rsi_oversold:
            priority_score += self.weights['rsi_extreme']
            criteria_met.append(f"RSI_OVERSOLD ({rsi:.1f} <= {self.critical_rsi_low})")
        elif rsi_overbought:
            # RSI alto pode ser sinal de venda, não é tão forte para compra
            priority_score += self.weights['rsi_extreme'] * 0.3
            criteria_met.append(f"RSI_OVERBOUGHT ({rsi:.1f} >= {self.critical_rsi_high})")
        else:
            criteria_failed.append(f"RSI_NEUTRAL ({rsi:.1f})")

        # 2. Score alto
        if score >= self.critical_score:
            priority_score += self.weights['score_high']
            criteria_met.append(f"SCORE_HIGH ({score:.1f} >= {self.critical_score})")
        elif score >= self.critical_score * 0.8:  # 80% do crítico
            priority_score += self.weights['score_high'] * 0.5
            criteria_met.append(f"SCORE_MEDIUM ({score:.1f})")
        else:
            criteria_failed.append(f"SCORE_LOW ({score:.1f})")

        # 3. Volume acima da média
        if volume_ratio >= self.critical_volume:
            priority_score += self.weights['volume_high']
            criteria_met.append(f"VOLUME_HIGH ({volume_ratio:.2f}x)")
        elif volume_ratio >= 1.0:
            priority_score += self.weights['volume_high'] * 0.5
            criteria_met.append(f"VOLUME_NORMAL ({volume_ratio:.2f}x)")
        else:
            criteria_failed.append(f"VOLUME_LOW ({volume_ratio:.2f}x)")

        # 4. Tendência alinhada
        ema_12 = signal.get('ema_12', 0)
        ema_26 = signal.get('ema_26', 0)
        if ema_12 > 0 and ema_26 > 0:
            if ema_12 > ema_26:  # Tendência de alta
                priority_score += self.weights['trend_aligned']
                criteria_met.append("TREND_BULLISH")
            else:
                criteria_failed.append("TREND_BEARISH")

        # 5. Momentum (MACD)
        macd = signal.get('macd', 0)
        macd_signal = signal.get('macd_signal', 0)
        if macd > macd_signal:
            priority_score += self.weights['momentum_strong']
            criteria_met.append("MOMENTUM_POSITIVE")
        else:
            criteria_failed.append("MOMENTUM_NEGATIVE")

        # Determina prioridade baseado no score e critérios
        priority, bypass_allowed, pos_multiplier = self._determine_priority(
            priority_score, criteria_met, rsi_oversold, emergency_mode
        )

        # Ação recomendada
        if 'BUY' in signal_type.upper():
            recommended_action = 'BUY'
        elif 'SELL' in signal_type.upper():
            recommended_action = 'SELL'
        else:
            recommended_action = 'HOLD'

        return PrioritizedSignal(
            symbol=symbol,
            original_signal=signal,
            priority=priority,
            priority_score=priority_score,
            criteria_met=criteria_met,
            criteria_failed=criteria_failed,
            recommended_action=recommended_action,
            bypass_allowed=bypass_allowed,
            position_size_multiplier=pos_multiplier
        )

    def _determine_priority(
        self,
        score: float,
        criteria_met: List[str],
        rsi_oversold: bool,
        emergency_mode: bool
    ) -> Tuple[SignalPriority, bool, float]:
        """
        Determina prioridade final baseado no score.

        Returns:
            Tuple (prioridade, bypass_permitido, multiplicador_posição)
        """
        max_score = sum(self.weights.values())
        score_pct = (score / max_score) * 100 if max_score > 0 else 0

        # CRITICAL: Score > 80% E (RSI oversold OU emergency mode)
        if score_pct >= 80 and (rsi_oversold or emergency_mode):
            return SignalPriority.CRITICAL, True, 0.7

        # CRITICAL: RSI oversold + score alto
        if rsi_oversold and score_pct >= 60:
            return SignalPriority.CRITICAL, emergency_mode, 0.7

        # HIGH: Score > 70%
        if score_pct >= 70:
            return SignalPriority.HIGH, False, 0.85

        # MEDIUM: Score > 50%
        if score_pct >= 50:
            return SignalPriority.MEDIUM, False, 1.0

        # LOW: Score > 30%
        if score_pct >= 30:
            return SignalPriority.LOW, False, 1.0

        # SKIP: Score muito baixo
        return SignalPriority.SKIP, False, 0.0

    def get_critical_signals(
        self,
        signals: List[Dict[str, Any]],
        emergency_mode: bool = False
    ) -> List[PrioritizedSignal]:
        """
        Retorna apenas sinais CRITICAL que merecem bypass de limites.

        Args:
            signals: Lista de sinais
            emergency_mode: Se modo emergência está ativo

        Returns:
            Lista de sinais CRITICAL
        """
        prioritized = self.prioritize_signals(signals, emergency_mode)
        return [p for p in prioritized if p.priority == SignalPriority.CRITICAL]

    def get_top_signals(
        self,
        signals: List[Dict[str, Any]],
        n: int = 5,
        emergency_mode: bool = False
    ) -> List[PrioritizedSignal]:
        """
        Retorna os N melhores sinais.

        Args:
            signals: Lista de sinais
            n: Número de sinais a retornar
            emergency_mode: Se modo emergência está ativo

        Returns:
            Lista dos N melhores sinais
        """
        prioritized = self.prioritize_signals(signals, emergency_mode)
        return prioritized[:n]

    def should_bypass_limits(
        self,
        signal: Dict[str, Any],
        emergency_mode: bool
    ) -> Tuple[bool, str]:
        """
        Verifica se um sinal deve ter bypass de limites.

        Args:
            signal: Dados do sinal
            emergency_mode: Se modo emergência está ativo

        Returns:
            Tuple (deve_bypass, motivo)
        """
        if not emergency_mode:
            return False, "not_in_emergency_mode"

        prioritized = self._calculate_priority(signal, emergency_mode)

        if prioritized.bypass_allowed:
            return True, f"critical_signal (score={prioritized.priority_score:.1f})"

        return False, f"priority={prioritized.priority.value}"

    def log_prioritization(self, prioritized: List[PrioritizedSignal]):
        """
        Loga resultado da priorização para debug.

        Args:
            prioritized: Lista de sinais priorizados
        """
        system_logger.info("=" * 60)
        system_logger.info("PRIORIZAÇÃO DE SINAIS")
        system_logger.info("=" * 60)

        for i, p in enumerate(prioritized[:10], 1):
            criteria_str = ", ".join(p.criteria_met[:3])
            system_logger.info(
                f"{i}. {p.symbol}: {p.priority.value.upper()} "
                f"(score={p.priority_score:.1f}) | {criteria_str}"
            )

        system_logger.info("=" * 60)


# Instância global
emergency_signal_prioritizer = EmergencySignalPrioritizer()
