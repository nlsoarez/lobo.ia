"""
Emergency Trade Manager - Sistema de Trades Emergenciais V4.2
Permite override de limites durante modo emergência com critérios rigorosos.

Features:
- Override de limites durante emergência
- Critérios especiais para trades emergenciais
- Controle de posição reduzida para segurança
- Logging detalhado de decisões
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from config_loader import config
from system_logger import system_logger


class TradeDecision(Enum):
    """Tipos de decisão de trade."""
    APPROVED = "approved"
    APPROVED_EMERGENCY = "approved_emergency"
    APPROVED_CRITICAL = "approved_critical"
    REJECTED_LIMIT = "rejected_limit"
    REJECTED_CRITERIA = "rejected_criteria"
    REJECTED_RISK = "rejected_risk"


@dataclass
class EmergencyTradeCriteria:
    """Critérios para trades em modo emergência."""
    min_score: float = 65.0
    max_rsi: float = 25.0
    min_rsi: float = 75.0  # Para oversold (RSI muito baixo é bom para compra)
    min_volume_ratio: float = 1.5
    position_size_percent: float = 70.0  # % do tamanho normal
    max_emergency_trades: int = 5  # Trades adicionais permitidos


@dataclass
class TradeDecisionLog:
    """Log estruturado de decisão de trade."""
    timestamp: datetime
    symbol: str
    signal_score: float
    signal_rsi: float
    volume_ratio: float
    recommended_action: str
    actual_action: str
    decision: TradeDecision
    reasons: List[str]
    limitations: List[str]
    emergency_status: bool
    override_used: bool
    position_size_adjustment: float


class EmergencyTradeManager:
    """
    Gerencia trades durante modo emergência.
    Permite override controlado de limites com critérios rigorosos.
    """

    def __init__(self):
        """Inicializa o gerenciador de trades emergenciais."""
        # Configurações de limites
        self.max_regular_trades = config.get('crypto.max_trades_daily', 20)
        self.max_emergency_trades = config.get('emergency.max_trades', 35)
        self.critical_trade_allowance = config.get('emergency.critical_trade_allowance', 5)

        # Critérios para trades emergenciais
        self.emergency_criteria = EmergencyTradeCriteria(
            min_score=config.get('emergency.critical_signals.min_score', 65.0),
            max_rsi=config.get('emergency.critical_signals.max_rsi', 25.0),
            min_volume_ratio=config.get('emergency.critical_signals.min_volume_ratio', 1.5),
            position_size_percent=config.get('emergency.position_size_multiplier', 0.7) * 100,
            max_emergency_trades=config.get('emergency.critical_signals.allowed_override_trades', 5)
        )

        # Contadores
        self.trades_today = 0
        self.emergency_trades_today = 0
        self.critical_overrides_today = 0
        self.last_reset_date = datetime.now().date()

        # Histórico de decisões
        self.decision_history: List[TradeDecisionLog] = []

        system_logger.info(
            f"EmergencyTradeManager V4.2 inicializado: "
            f"Regular={self.max_regular_trades}, Emergency={self.max_emergency_trades}, "
            f"Critical={self.critical_trade_allowance}"
        )

    def _reset_daily_counters(self):
        """Reseta contadores diários se necessário."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            system_logger.info(
                f"Resetando contadores diários: trades={self.trades_today}, "
                f"emergency={self.emergency_trades_today}, critical={self.critical_overrides_today}"
            )
            self.trades_today = 0
            self.emergency_trades_today = 0
            self.critical_overrides_today = 0
            self.last_reset_date = today
            self.decision_history = []

    def can_trade(
        self,
        signal: Dict[str, Any],
        emergency_mode: bool = False,
        current_positions: int = 0,
        max_positions: int = 8
    ) -> tuple[bool, TradeDecision, List[str]]:
        """
        Verifica se pode executar um trade.

        Args:
            signal: Dados do sinal (score, rsi, volume_ratio, etc.)
            emergency_mode: Se modo emergência está ativo
            current_positions: Número de posições abertas
            max_positions: Máximo de posições permitidas

        Returns:
            Tuple (pode_operar, decisão, motivos)
        """
        self._reset_daily_counters()

        reasons = []
        limitations = []

        symbol = signal.get('symbol', 'UNKNOWN')
        score = signal.get('total_score', signal.get('score', 0))
        rsi = signal.get('rsi', 50)
        volume_ratio = signal.get('volume_ratio', 1.0)

        # Verifica limite de posições
        if current_positions >= max_positions:
            limitations.append(f"max_positions_reached ({current_positions}/{max_positions})")
            return False, TradeDecision.REJECTED_LIMIT, limitations

        # CASO 1: Dentro do limite regular
        if self.trades_today < self.max_regular_trades:
            reasons.append(f"within_regular_limit ({self.trades_today}/{self.max_regular_trades})")
            return True, TradeDecision.APPROVED, reasons

        # CASO 2: Modo emergência ativo - verificar critérios especiais
        if emergency_mode:
            can_emergency, decision, emergency_reasons = self._check_emergency_criteria(
                signal, score, rsi, volume_ratio
            )
            if can_emergency:
                return True, decision, emergency_reasons

            limitations.extend(emergency_reasons)

        # CASO 3: Limite atingido sem override
        limitations.append(f"max_trades_reached ({self.trades_today}/{self.max_regular_trades})")
        limitations.append("emergency_criteria_not_met" if emergency_mode else "not_in_emergency_mode")

        return False, TradeDecision.REJECTED_LIMIT, limitations

    def _check_emergency_criteria(
        self,
        signal: Dict[str, Any],
        score: float,
        rsi: float,
        volume_ratio: float
    ) -> tuple[bool, TradeDecision, List[str]]:
        """
        Verifica se o sinal atende critérios para trade emergencial.

        Returns:
            Tuple (aprovado, decisão, motivos)
        """
        reasons = []

        # Verifica limite de trades emergenciais
        total_emergency = self.emergency_trades_today + self.critical_overrides_today
        if total_emergency >= (self.max_emergency_trades - self.max_regular_trades):
            reasons.append(f"emergency_limit_reached ({total_emergency})")
            return False, TradeDecision.REJECTED_LIMIT, reasons

        # Critérios para aprovação emergencial
        criteria_met = 0
        criteria_details = []

        # 1. Score mínimo
        if score >= self.emergency_criteria.min_score:
            criteria_met += 1
            criteria_details.append(f"score_ok ({score:.1f} >= {self.emergency_criteria.min_score})")
        else:
            criteria_details.append(f"score_low ({score:.1f} < {self.emergency_criteria.min_score})")

        # 2. RSI extremo (oversold = bom para compra)
        rsi_oversold = rsi <= self.emergency_criteria.max_rsi
        rsi_overbought = rsi >= self.emergency_criteria.min_rsi

        if rsi_oversold:
            criteria_met += 2  # RSI muito baixo é forte indicador
            criteria_details.append(f"rsi_oversold ({rsi:.1f} <= {self.emergency_criteria.max_rsi})")
        elif rsi_overbought:
            criteria_details.append(f"rsi_overbought ({rsi:.1f} >= {self.emergency_criteria.min_rsi})")
        else:
            criteria_details.append(f"rsi_neutral ({rsi:.1f})")

        # 3. Volume ratio
        if volume_ratio >= self.emergency_criteria.min_volume_ratio:
            criteria_met += 1
            criteria_details.append(f"volume_ok ({volume_ratio:.2f} >= {self.emergency_criteria.min_volume_ratio})")
        else:
            criteria_details.append(f"volume_low ({volume_ratio:.2f} < {self.emergency_criteria.min_volume_ratio})")

        # Decisão baseada em critérios
        # Precisa de pelo menos 2 critérios (ou RSI oversold que vale 2)
        if criteria_met >= 2:
            # CRITICAL: RSI muito baixo + score alto
            if rsi_oversold and score >= self.emergency_criteria.min_score:
                if self.critical_overrides_today < self.critical_trade_allowance:
                    reasons = criteria_details + [f"critical_override ({self.critical_overrides_today + 1}/{self.critical_trade_allowance})"]
                    return True, TradeDecision.APPROVED_CRITICAL, reasons

            # EMERGENCY: Critérios normais atendidos
            reasons = criteria_details + ["emergency_criteria_met"]
            return True, TradeDecision.APPROVED_EMERGENCY, reasons

        reasons = criteria_details + ["insufficient_criteria"]
        return False, TradeDecision.REJECTED_CRITERIA, reasons

    def register_trade(
        self,
        signal: Dict[str, Any],
        decision: TradeDecision,
        emergency_mode: bool
    ):
        """
        Registra um trade executado.

        Args:
            signal: Dados do sinal
            decision: Tipo de decisão
            emergency_mode: Se estava em modo emergência
        """
        self._reset_daily_counters()

        self.trades_today += 1

        if decision == TradeDecision.APPROVED_EMERGENCY:
            self.emergency_trades_today += 1
        elif decision == TradeDecision.APPROVED_CRITICAL:
            self.critical_overrides_today += 1

        # Log da decisão
        log_entry = TradeDecisionLog(
            timestamp=datetime.now(),
            symbol=signal.get('symbol', 'UNKNOWN'),
            signal_score=signal.get('total_score', signal.get('score', 0)),
            signal_rsi=signal.get('rsi', 50),
            volume_ratio=signal.get('volume_ratio', 1.0),
            recommended_action='BUY',
            actual_action='EXECUTED',
            decision=decision,
            reasons=[decision.value],
            limitations=[],
            emergency_status=emergency_mode,
            override_used=decision in [TradeDecision.APPROVED_EMERGENCY, TradeDecision.APPROVED_CRITICAL],
            position_size_adjustment=self.get_position_size_multiplier(decision)
        )
        self.decision_history.append(log_entry)

        system_logger.info(
            f"Trade registrado: {signal.get('symbol')} | "
            f"Decision={decision.value} | Emergency={emergency_mode} | "
            f"Total hoje={self.trades_today}"
        )

    def get_position_size_multiplier(self, decision: TradeDecision) -> float:
        """
        Retorna multiplicador de tamanho de posição baseado na decisão.

        Args:
            decision: Tipo de decisão

        Returns:
            Multiplicador (0.0 - 1.0)
        """
        if decision == TradeDecision.APPROVED:
            return 1.0
        elif decision == TradeDecision.APPROVED_EMERGENCY:
            return self.emergency_criteria.position_size_percent / 100
        elif decision == TradeDecision.APPROVED_CRITICAL:
            return (self.emergency_criteria.position_size_percent / 100) * 0.8  # Ainda mais conservador
        return 0.0

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status atual do gerenciador.

        Returns:
            Dicionário com status completo
        """
        self._reset_daily_counters()

        return {
            'trades_today': self.trades_today,
            'emergency_trades_today': self.emergency_trades_today,
            'critical_overrides_today': self.critical_overrides_today,
            'max_regular_trades': self.max_regular_trades,
            'max_emergency_trades': self.max_emergency_trades,
            'critical_allowance': self.critical_trade_allowance,
            'remaining_regular': max(0, self.max_regular_trades - self.trades_today),
            'remaining_emergency': max(0, self.max_emergency_trades - self.trades_today),
            'remaining_critical': max(0, self.critical_trade_allowance - self.critical_overrides_today),
            'criteria': {
                'min_score': self.emergency_criteria.min_score,
                'max_rsi': self.emergency_criteria.max_rsi,
                'min_volume_ratio': self.emergency_criteria.min_volume_ratio,
                'position_size_percent': self.emergency_criteria.position_size_percent
            },
            'last_reset': self.last_reset_date.isoformat(),
            'decision_count': len(self.decision_history)
        }

    def log_trading_decision(
        self,
        signal: Dict[str, Any],
        action: str,
        reasons: List[str],
        limitations: List[str],
        emergency_mode: bool
    ) -> Dict[str, Any]:
        """
        Cria log estruturado de decisão de trading.

        Args:
            signal: Dados do sinal
            action: Ação recomendada
            reasons: Motivos da decisão
            limitations: Limitações encontradas
            emergency_mode: Status do modo emergência

        Returns:
            Log estruturado
        """
        can_trade, decision, decision_reasons = self.can_trade(
            signal, emergency_mode
        )

        log = {
            'timestamp': datetime.now().isoformat(),
            'signal': {
                'symbol': signal.get('symbol', 'UNKNOWN'),
                'score': signal.get('total_score', signal.get('score', 0)),
                'rsi': signal.get('rsi', 50),
                'volume_ratio': signal.get('volume_ratio', 1.0),
                'signal_type': signal.get('signal', 'UNKNOWN')
            },
            'recommended_action': action,
            'actual_action': 'EXECUTE' if can_trade else 'SKIP',
            'decision': decision.value,
            'reasons': reasons + decision_reasons,
            'limitations': limitations,
            'suggested_override': can_trade and decision != TradeDecision.APPROVED,
            'emergency_status': {
                'active': emergency_mode,
                'trades_today': self.trades_today,
                'emergency_trades': self.emergency_trades_today,
                'critical_overrides': self.critical_overrides_today
            }
        }

        # Log detalhado
        system_logger.debug(f"Trading Decision: {log}")

        return log


# Instância global
emergency_trade_manager = EmergencyTradeManager()
