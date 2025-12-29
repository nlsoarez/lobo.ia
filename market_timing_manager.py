"""
V4.0 Phase 2: Market Timing Manager
Otimiza timing de entradas baseado em sessões de mercado e volatilidade.
Gerencia horários ideais de trading para crypto 24/7.
"""

import pytz
from datetime import datetime, time, timedelta
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum

from system_logger import system_logger


class MarketSession(Enum):
    """Sessões de mercado global."""
    ASIA = "asia"
    EUROPE = "europe"
    US = "us"
    OVERLAP_ASIA_EUROPE = "asia_europe_overlap"
    OVERLAP_EUROPE_US = "europe_us_overlap"


@dataclass
class SessionInfo:
    """Informações sobre sessão de mercado."""
    session: MarketSession
    name: str
    is_active: bool
    volatility_multiplier: float
    liquidity_level: str  # LOW, MEDIUM, HIGH
    recommended_action: str  # AGGRESSIVE, NORMAL, CONSERVATIVE, AVOID


@dataclass
class TimingAnalysis:
    """Resultado da análise de timing."""
    is_optimal: bool
    current_session: MarketSession
    exposure_multiplier: float
    tp_multiplier: float
    sl_multiplier: float
    confidence: str  # LOW, MEDIUM, HIGH
    reason: str
    next_optimal_in_hours: float


class MarketTimingManager:
    """
    V4.0 Phase 2: Gerenciador de timing de mercado.
    Otimiza entradas baseado em sessões e volatilidade.
    """

    def __init__(self):
        """Inicializa o gerenciador de timing."""
        # Timezone Brasil
        self.brazil_tz = pytz.timezone('America/Sao_Paulo')
        self.utc_tz = pytz.UTC

        # Sessões de mercado (horário UTC)
        self.sessions = {
            MarketSession.ASIA: {
                'start_utc': 0,   # 00:00 UTC
                'end_utc': 8,     # 08:00 UTC
                'weight': 0.7,    # Menos volátil para crypto
                'liquidity': 'MEDIUM',
                'volatility': 0.8
            },
            MarketSession.EUROPE: {
                'start_utc': 8,   # 08:00 UTC
                'end_utc': 16,    # 16:00 UTC
                'weight': 1.0,    # Boa atividade
                'liquidity': 'HIGH',
                'volatility': 1.0
            },
            MarketSession.US: {
                'start_utc': 13,  # 13:00 UTC (9am EST)
                'end_utc': 21,    # 21:00 UTC (5pm EST)
                'weight': 1.2,    # Mais volátil
                'liquidity': 'HIGH',
                'volatility': 1.2
            }
        }

        # Horários ideais de entrada (horário Brasil)
        self.optimal_hours_brazil = [
            (6, 9),    # 06:00-09:00 - Abertura Europa
            (10, 12),  # 10:00-12:00 - Overlap Europa/US
            (14, 16),  # 14:00-16:00 - Mercado US ativo
        ]

        # Horários a evitar (horário Brasil)
        self.avoid_hours_brazil = [
            (2, 5),    # 02:00-05:00 - Baixa liquidez
            (23, 24),  # 23:00-00:00 - Transição
        ]

        # Multiplicadores de sessão
        self.session_multipliers = {
            MarketSession.ASIA: {'tp': 0.9, 'sl': 0.9, 'exposure': 0.8},
            MarketSession.EUROPE: {'tp': 1.0, 'sl': 1.0, 'exposure': 1.0},
            MarketSession.US: {'tp': 1.15, 'sl': 1.1, 'exposure': 1.1},
            MarketSession.OVERLAP_ASIA_EUROPE: {'tp': 1.0, 'sl': 0.95, 'exposure': 1.0},
            MarketSession.OVERLAP_EUROPE_US: {'tp': 1.1, 'sl': 1.05, 'exposure': 1.15}
        }

        system_logger.info("MarketTimingManager V4.0 inicializado")

    def get_brazil_time(self) -> datetime:
        """Retorna horário atual no Brasil."""
        return datetime.now(self.brazil_tz)

    def get_utc_time(self) -> datetime:
        """Retorna horário atual em UTC."""
        return datetime.now(self.utc_tz)

    def get_current_session(self, utc_time: Optional[datetime] = None) -> MarketSession:
        """
        Determina sessão atual baseada no horário UTC.
        Considera overlaps entre sessões.
        """
        if utc_time is None:
            utc_time = self.get_utc_time()

        hour = utc_time.hour

        # Verifica overlaps primeiro (maior atividade)
        # Overlap Ásia/Europa: 08:00-09:00 UTC
        if 8 <= hour < 9:
            return MarketSession.OVERLAP_ASIA_EUROPE

        # Overlap Europa/US: 13:00-16:00 UTC
        if 13 <= hour < 16:
            return MarketSession.OVERLAP_EUROPE_US

        # Sessões individuais
        if 0 <= hour < 8:
            return MarketSession.ASIA
        elif 8 <= hour < 13:
            return MarketSession.EUROPE
        elif 13 <= hour < 21:
            return MarketSession.US
        else:
            return MarketSession.ASIA  # Após 21:00 UTC

    def get_session_info(self, session: MarketSession) -> SessionInfo:
        """
        Retorna informações detalhadas sobre uma sessão.
        """
        multipliers = self.session_multipliers.get(session, self.session_multipliers[MarketSession.EUROPE])

        session_names = {
            MarketSession.ASIA: "Ásia (Tóquio/Sydney)",
            MarketSession.EUROPE: "Europa (Londres/Frankfurt)",
            MarketSession.US: "EUA (Nova York)",
            MarketSession.OVERLAP_ASIA_EUROPE: "Overlap Ásia/Europa",
            MarketSession.OVERLAP_EUROPE_US: "Overlap Europa/EUA"
        }

        liquidity_levels = {
            MarketSession.ASIA: "MEDIUM",
            MarketSession.EUROPE: "HIGH",
            MarketSession.US: "HIGH",
            MarketSession.OVERLAP_ASIA_EUROPE: "HIGH",
            MarketSession.OVERLAP_EUROPE_US: "VERY_HIGH"
        }

        recommendations = {
            MarketSession.ASIA: "CONSERVATIVE",
            MarketSession.EUROPE: "NORMAL",
            MarketSession.US: "AGGRESSIVE",
            MarketSession.OVERLAP_ASIA_EUROPE: "NORMAL",
            MarketSession.OVERLAP_EUROPE_US: "AGGRESSIVE"
        }

        return SessionInfo(
            session=session,
            name=session_names.get(session, "Unknown"),
            is_active=True,
            volatility_multiplier=multipliers['exposure'],
            liquidity_level=liquidity_levels.get(session, "MEDIUM"),
            recommended_action=recommendations.get(session, "NORMAL")
        )

    def is_optimal_entry_time(self, brazil_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Verifica se é horário ideal para entrada.
        Retorna (is_optimal, reason).
        """
        if brazil_time is None:
            brazil_time = self.get_brazil_time()

        hour = brazil_time.hour

        # Verifica se deve evitar
        for avoid_start, avoid_end in self.avoid_hours_brazil:
            if avoid_start <= hour < avoid_end:
                return False, f"Horário de baixa liquidez ({avoid_start}:00-{avoid_end}:00)"

        # Verifica se é horário ideal
        for optimal_start, optimal_end in self.optimal_hours_brazil:
            if optimal_start <= hour < optimal_end:
                return True, f"Horário ideal ({optimal_start}:00-{optimal_end}:00)"

        return True, "Horário aceitável"  # Fora dos horários ideais mas não nos de evitar

    def calculate_session_multipliers(self, session: Optional[MarketSession] = None) -> Dict[str, float]:
        """
        Calcula multiplicadores de TP/SL baseado na sessão.
        """
        if session is None:
            session = self.get_current_session()

        return self.session_multipliers.get(session, {
            'tp': 1.0,
            'sl': 1.0,
            'exposure': 1.0
        })

    def adjust_parameters_by_session(
        self,
        base_tp: float,
        base_sl: float,
        base_exposure: float,
        session: Optional[MarketSession] = None
    ) -> Dict[str, float]:
        """
        Ajusta parâmetros de trading baseado na sessão.
        """
        multipliers = self.calculate_session_multipliers(session)

        return {
            'take_profit': base_tp * multipliers['tp'],
            'stop_loss': base_sl * multipliers['sl'],
            'exposure': base_exposure * multipliers['exposure'],
            'session': session.value if session else self.get_current_session().value
        }

    def get_time_to_next_optimal(self, brazil_time: Optional[datetime] = None) -> float:
        """
        Calcula tempo até o próximo horário ideal (em horas).
        """
        if brazil_time is None:
            brazil_time = self.get_brazil_time()

        current_hour = brazil_time.hour

        # Encontra próximo horário ideal
        for optimal_start, optimal_end in self.optimal_hours_brazil:
            if current_hour < optimal_start:
                return optimal_start - current_hour
            elif optimal_start <= current_hour < optimal_end:
                return 0  # Já está em horário ideal

        # Se passou de todos, próximo é amanhã às 6h
        hours_until_midnight = 24 - current_hour
        hours_until_morning = self.optimal_hours_brazil[0][0]
        return hours_until_midnight + hours_until_morning

    def analyze_timing(self, brazil_time: Optional[datetime] = None) -> TimingAnalysis:
        """
        Análise completa de timing para decisão de trading.
        """
        if brazil_time is None:
            brazil_time = self.get_brazil_time()

        utc_time = brazil_time.astimezone(self.utc_tz)

        # Sessão atual
        current_session = self.get_current_session(utc_time)
        session_info = self.get_session_info(current_session)

        # Horário ideal
        is_optimal, reason = self.is_optimal_entry_time(brazil_time)

        # Multiplicadores
        multipliers = self.calculate_session_multipliers(current_session)

        # Tempo até próximo ideal
        next_optimal = self.get_time_to_next_optimal(brazil_time)

        # Determina confiança
        if is_optimal and session_info.liquidity_level in ['HIGH', 'VERY_HIGH']:
            confidence = 'HIGH'
        elif is_optimal or session_info.liquidity_level == 'HIGH':
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        return TimingAnalysis(
            is_optimal=is_optimal,
            current_session=current_session,
            exposure_multiplier=multipliers['exposure'],
            tp_multiplier=multipliers['tp'],
            sl_multiplier=multipliers['sl'],
            confidence=confidence,
            reason=reason,
            next_optimal_in_hours=next_optimal
        )

    def should_enter_trade(self, signal_strength: float, brazil_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Decide se deve entrar em trade considerando timing.
        signal_strength: 0-100
        """
        timing = self.analyze_timing(brazil_time)

        # Sinais fortes podem ignorar timing
        if signal_strength >= 85:
            return True, f"Sinal forte ({signal_strength:.0f}) supera timing"

        # Em horário ideal, aceita sinais médios
        if timing.is_optimal:
            if signal_strength >= 60:
                return True, f"Horário ideal + sinal bom ({signal_strength:.0f})"
            else:
                return False, f"Sinal fraco ({signal_strength:.0f}) mesmo em horário ideal"

        # Fora do horário ideal, exige sinal mais forte
        if signal_strength >= 75:
            return True, f"Sinal forte ({signal_strength:.0f}) compensa timing"
        else:
            hours_to_wait = timing.next_optimal_in_hours
            return False, f"Sinal ({signal_strength:.0f}) insuficiente. Próximo ideal em {hours_to_wait:.1f}h"

    def get_volatility_adjustment(self, brazil_time: Optional[datetime] = None) -> float:
        """
        Retorna ajuste de volatilidade esperada para o horário.
        Útil para ajustar ATR-based stops.
        """
        timing = self.analyze_timing(brazil_time)
        session_info = self.get_session_info(timing.current_session)

        return session_info.volatility_multiplier

    def get_dashboard_info(self) -> Dict[str, Any]:
        """
        Retorna informações para dashboard de timing.
        """
        brazil_time = self.get_brazil_time()
        timing = self.analyze_timing(brazil_time)
        session_info = self.get_session_info(timing.current_session)

        return {
            'brazil_time': brazil_time.strftime('%H:%M'),
            'session': session_info.name,
            'is_optimal': timing.is_optimal,
            'confidence': timing.confidence,
            'liquidity': session_info.liquidity_level,
            'recommendation': session_info.recommended_action,
            'exposure_mult': f"{timing.exposure_multiplier:.2f}x",
            'tp_mult': f"{timing.tp_multiplier:.2f}x",
            'sl_mult': f"{timing.sl_multiplier:.2f}x",
            'next_optimal_hours': timing.next_optimal_in_hours,
            'reason': timing.reason
        }

    def log_timing_status(self):
        """
        Loga status atual de timing para debug.
        """
        info = self.get_dashboard_info()

        system_logger.info(f"⏰ TIMING STATUS [{info['brazil_time']}]")
        system_logger.info(f"   Sessão: {info['session']}")
        system_logger.info(f"   Ideal: {'✅' if info['is_optimal'] else '❌'} | Confiança: {info['confidence']}")
        system_logger.info(f"   Liquidez: {info['liquidity']} | Ação: {info['recommendation']}")
        system_logger.info(f"   Multipliers: Exp {info['exposure_mult']} | TP {info['tp_mult']} | SL {info['sl_mult']}")
        if not info['is_optimal']:
            system_logger.info(f"   Próximo ideal em: {info['next_optimal_hours']:.1f}h")
