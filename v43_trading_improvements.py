"""
V4.3 Trading Improvements Module
================================
Correcoes criticas baseadas em analise de logs operacionais:
1. TradeLimitManager - Limite rigoroso de trades diarios
2. AdaptiveFilter - Filtros por regime de mercado
3. DynamicTimeoutManager - Timeout baseado em TP/SL
4. SmartTrailingStop - Trailing stop inteligente
5. StrongOverrideValidator - Validacao rigorosa de STRONG_OVERRIDE
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from system_logger import system_logger


# ============================================================================
# EXCECOES CUSTOMIZADAS
# ============================================================================

class TradeLimitExceededError(Exception):
    """Erro quando limite de trades e excedido."""
    pass


class FilterRejectionError(Exception):
    """Erro quando sinal e rejeitado por filtro."""
    pass


# ============================================================================
# ENUMS E DATACLASSES
# ============================================================================

class MarketRegime(Enum):
    """Regime de mercado."""
    BULL = "bull"
    BEAR = "bear"
    LATERAL = "lateral"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class SignalLevel(Enum):
    """Niveis de sinal."""
    STRONG_OVERRIDE = "STRONG_OVERRIDE"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


@dataclass
class TradeDecisionLog:
    """Log de decisao de trade."""
    timestamp: datetime
    symbol: str
    decision: str  # APPROVED, REJECTED, LIMIT_EXCEEDED
    reason: str
    regime: str
    score: float
    rsi: float
    trades_remaining: int


# ============================================================================
# TRADE LIMIT MANAGER
# ============================================================================

class TradeLimitManager:
    """
    V4.3: Gerenciador rigoroso de limites de trades diarios.

    Problema original: Sistema executou 23 trades quando limite era 20.
    Solucao: Impedir fisicamente trades alem do limite.
    """

    def __init__(self, max_daily_trades: int = 20, emergency_max: int = 35):
        """
        Inicializa o gerenciador de limites.

        Args:
            max_daily_trades: Limite normal de trades/dia
            emergency_max: Limite em modo emergencia
        """
        self.max_trades = max_daily_trades
        self.emergency_max = emergency_max
        self.trades_today = 0
        self.reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.trade_log: List[TradeDecisionLog] = []
        self.emergency_mode = False
        self._lock_trading = False

        system_logger.info(f"TradeLimitManager inicializado: max={max_daily_trades}, emergency={emergency_max}")

    def _check_reset(self):
        """Verifica se deve resetar contadores (novo dia UTC)."""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if today_start > self.reset_time:
            system_logger.info(f"TradeLimitManager: Novo dia detectado, resetando contadores")
            self.trades_today = 0
            self.reset_time = today_start
            self._lock_trading = False
            self.trade_log = []

    def can_trade(self) -> Tuple[bool, str, int]:
        """
        Verifica se pode executar trade.

        Returns:
            (bool pode_operar, str motivo, int trades_restantes)
        """
        self._check_reset()

        # Lock manual de emergencia
        if self._lock_trading:
            return False, "TRADING_BLOQUEADO_MANUALMENTE", 0

        # Determina limite atual
        current_limit = self.emergency_max if self.emergency_mode else self.max_trades
        remaining = current_limit - self.trades_today

        if self.trades_today >= current_limit:
            return False, "LIMITE_DIARIO_ATINGIDO", 0

        return True, "OK", remaining

    def increment_trade(self, symbol: str = "", reason: str = ""):
        """
        Registra novo trade executado.

        Args:
            symbol: Simbolo tradado
            reason: Motivo/nivel do trade
        """
        self._check_reset()
        self.trades_today += 1

        log_entry = TradeDecisionLog(
            timestamp=datetime.now(),
            symbol=symbol,
            decision="EXECUTED",
            reason=reason,
            regime="",
            score=0,
            rsi=0,
            trades_remaining=self.get_remaining_trades()
        )
        self.trade_log.append(log_entry)

        current_limit = self.emergency_max if self.emergency_mode else self.max_trades
        system_logger.info(f"TradeLimitManager: Trade {self.trades_today}/{current_limit} ({symbol})")

        # Alerta quando proximo do limite
        if self.get_remaining_trades() <= 3:
            system_logger.warning(f"⚠️ ATENCAO: Apenas {self.get_remaining_trades()} trades restantes!")

    def enforce_limit(self) -> bool:
        """
        IMPEDE fisicamente trade se limite atingido.

        Returns:
            True se pode prosseguir, False se deve parar

        Raises:
            TradeLimitExceededError se limite excedido
        """
        can_trade, reason, remaining = self.can_trade()

        if not can_trade:
            error_msg = f"{reason}. Maximo: {self.emergency_max if self.emergency_mode else self.max_trades}"
            system_logger.error(f"TradeLimitManager: {error_msg}")
            raise TradeLimitExceededError(error_msg)

        return True

    def get_remaining_trades(self) -> int:
        """Retorna numero de trades restantes."""
        self._check_reset()
        current_limit = self.emergency_max if self.emergency_mode else self.max_trades
        return max(0, current_limit - self.trades_today)

    def set_emergency_mode(self, enabled: bool):
        """Ativa/desativa modo emergencia."""
        self.emergency_mode = enabled
        mode_str = "EMERGENCIA" if enabled else "NORMAL"
        limit = self.emergency_max if enabled else self.max_trades
        system_logger.info(f"TradeLimitManager: Modo {mode_str}, limite={limit}")

    def lock_trading(self, reason: str = ""):
        """Bloqueia trading manualmente."""
        self._lock_trading = True
        system_logger.warning(f"TradeLimitManager: Trading BLOQUEADO - {reason}")

    def unlock_trading(self):
        """Desbloqueia trading."""
        self._lock_trading = False
        system_logger.info("TradeLimitManager: Trading DESBLOQUEADO")

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatisticas do gerenciador."""
        current_limit = self.emergency_max if self.emergency_mode else self.max_trades
        return {
            'trades_today': self.trades_today,
            'max_trades': current_limit,
            'remaining': self.get_remaining_trades(),
            'emergency_mode': self.emergency_mode,
            'locked': self._lock_trading,
            'utilization_pct': (self.trades_today / current_limit * 100) if current_limit > 0 else 0
        }


# ============================================================================
# ADAPTIVE FILTER
# ============================================================================

class AdaptiveFilter:
    """
    V4.3: Filtros adaptativos por regime de mercado.

    Problema original: 60.9% de perdas, comprando em mercado bear.
    Solucao: Filtros mais restritivos em bear market.
    """

    def __init__(self):
        """Inicializa filtros adaptativos."""

        # Configuracoes de filtro por regime
        self.regime_filters = {
            MarketRegime.BULL: {
                "min_score": 40,
                "max_rsi": 70,
                "min_rsi": 30,
                "min_volume_ratio": 1.0,
                "required_indicators": ["trend", "volume"],
                "reject_conditions": []
            },
            MarketRegime.BEAR: {
                "min_score": 50,       # MAIS RESTRITIVO
                "max_rsi": 40,         # APENAS OVERSOLD (era 45)
                "min_rsi": 10,
                "min_volume_ratio": 1.2,
                "required_indicators": ["trend", "volume", "momentum"],
                "reject_conditions": ["not_oversold", "low_volume", "weak_momentum"]
            },
            MarketRegime.LATERAL: {
                "min_score": 45,
                "max_rsi": 65,
                "min_rsi": 35,
                "min_volume_ratio": 1.1,
                "required_indicators": ["trend", "atr"],
                "reject_conditions": []
            },
            MarketRegime.VOLATILE: {
                "min_score": 55,       # Mais restritivo em alta volatilidade
                "max_rsi": 60,
                "min_rsi": 40,
                "min_volume_ratio": 1.3,
                "required_indicators": ["trend", "volume", "atr"],
                "reject_conditions": ["extreme_volatility"]
            },
            MarketRegime.UNKNOWN: {
                "min_score": 50,
                "max_rsi": 65,
                "min_rsi": 35,
                "min_volume_ratio": 1.1,
                "required_indicators": ["trend"],
                "reject_conditions": []
            }
        }

        # Estatisticas de filtro
        self.filter_stats = {
            'total_signals': 0,
            'approved': 0,
            'rejected': 0,
            'rejections_by_reason': {}
        }

        system_logger.info("AdaptiveFilter V4.3 inicializado")

    def detect_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """
        Detecta regime de mercado atual.

        Args:
            market_data: Dados de mercado (ema_trend, volatility, etc)

        Returns:
            MarketRegime detectado
        """
        # Extrai metricas
        ema_trend = market_data.get('ema_trend', 0)
        volatility = market_data.get('volatility', 1.0)
        adx = market_data.get('adx', 25)
        btc_change_24h = market_data.get('btc_change_24h', 0)

        # Alta volatilidade tem prioridade
        if volatility > 2.0:
            return MarketRegime.VOLATILE

        # Tendencia forte
        if adx > 25:
            if ema_trend > 0 or btc_change_24h > 2:
                return MarketRegime.BULL
            elif ema_trend < 0 or btc_change_24h < -2:
                return MarketRegime.BEAR

        # Mercado lateral
        if adx < 20:
            return MarketRegime.LATERAL

        # Fallback baseado em EMA
        if ema_trend > 0.5:
            return MarketRegime.BULL
        elif ema_trend < -0.5:
            return MarketRegime.BEAR

        return MarketRegime.LATERAL

    def apply_regime_filter(
        self,
        signal: Dict[str, Any],
        regime: MarketRegime
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Aplica filtros especificos por regime de mercado.

        Args:
            signal: Sinal a ser filtrado
            regime: Regime de mercado atual

        Returns:
            (bool aprovado, str motivo, dict detalhes)
        """
        self.filter_stats['total_signals'] += 1

        filters = self.regime_filters.get(regime, self.regime_filters[MarketRegime.UNKNOWN])

        symbol = signal.get('symbol', 'UNKNOWN')
        score = signal.get('total_score', 0) or signal.get('phase2_score', 0)
        rsi = signal.get('rsi', 50)
        volume_ratio = signal.get('volume_ratio', 1.0)

        details = {
            'symbol': symbol,
            'regime': regime.value,
            'score': score,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'filters_applied': filters
        }

        # Filtro 1: Score minimo
        if score < filters['min_score']:
            reason = f"Score {score:.0f} < minimo {filters['min_score']} para regime {regime.value}"
            self._log_rejection(reason)
            return False, reason, details

        # Filtro 2: RSI em bear market (CRITICO)
        if regime == MarketRegime.BEAR:
            if rsi > filters['max_rsi']:
                reason = f"RSI {rsi:.1f} > {filters['max_rsi']} - NAO OVERSOLD em mercado BEAR"
                self._log_rejection(reason)
                return False, reason, details
        else:
            # Outros regimes - verificar range normal
            if rsi > filters['max_rsi']:
                reason = f"RSI {rsi:.1f} > {filters['max_rsi']} (overbought)"
                self._log_rejection(reason)
                return False, reason, details
            if rsi < filters['min_rsi']:
                reason = f"RSI {rsi:.1f} < {filters['min_rsi']} (muito oversold)"
                self._log_rejection(reason)
                return False, reason, details

        # Filtro 3: Volume minimo
        if volume_ratio < filters['min_volume_ratio']:
            reason = f"Volume ratio {volume_ratio:.2f} < minimo {filters['min_volume_ratio']}"
            self._log_rejection(reason)
            return False, reason, details

        # Filtro 4: Condicoes de rejeicao especificas
        for condition in filters['reject_conditions']:
            if self._check_reject_condition(condition, signal, regime):
                reason = f"Condicao de rejeicao: {condition}"
                self._log_rejection(reason)
                return False, reason, details

        # Aprovado
        self.filter_stats['approved'] += 1
        return True, f"Aprovado para regime {regime.value}", details

    def _check_reject_condition(
        self,
        condition: str,
        signal: Dict[str, Any],
        regime: MarketRegime
    ) -> bool:
        """Verifica condicao especifica de rejeicao."""
        rsi = signal.get('rsi', 50)
        volume_ratio = signal.get('volume_ratio', 1.0)
        momentum = signal.get('momentum_score', 50)

        if condition == "not_oversold" and rsi > 35:
            return True
        if condition == "low_volume" and volume_ratio < 1.2:
            return True
        if condition == "weak_momentum" and momentum < 40:
            return True
        if condition == "extreme_volatility":
            volatility = signal.get('volatility', 1.0)
            if volatility > 3.0:
                return True

        return False

    def _log_rejection(self, reason: str):
        """Registra rejeicao nas estatisticas."""
        self.filter_stats['rejected'] += 1

        # Agrupa por tipo de rejeicao
        base_reason = reason.split(' - ')[0] if ' - ' in reason else reason[:50]
        if base_reason not in self.filter_stats['rejections_by_reason']:
            self.filter_stats['rejections_by_reason'][base_reason] = 0
        self.filter_stats['rejections_by_reason'][base_reason] += 1

        system_logger.debug(f"AdaptiveFilter: Rejeitado - {reason}")

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatisticas de filtragem."""
        total = self.filter_stats['total_signals']
        approved = self.filter_stats['approved']

        return {
            'total_signals': total,
            'approved': approved,
            'rejected': self.filter_stats['rejected'],
            'approval_rate': (approved / total * 100) if total > 0 else 0,
            'top_rejections': sorted(
                self.filter_stats['rejections_by_reason'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# ============================================================================
# DYNAMIC TIMEOUT MANAGER
# ============================================================================

class DynamicTimeoutManager:
    """
    V4.3: Timeout dinamico baseado em TP/SL.

    Problema original: Posicoes fechadas apos 0.9h com TP de 2-4%.
    Solucao: Timeout proporcional ao target.
    """

    def __init__(self):
        """Inicializa gerenciador de timeout."""

        # Configuracoes base
        self.min_timeout_hours = 0.5    # Minimo 30 min
        self.max_timeout_hours = 4.0    # Maximo 4 horas
        self.base_timeout_ratio = 0.5   # TP% * 0.5 = horas base

        # Ajustes por nivel de sinal
        self.level_multipliers = {
            SignalLevel.STRONG_OVERRIDE: 1.5,  # Mais tempo para sinais fortes
            SignalLevel.STRONG: 1.3,
            SignalLevel.MODERATE: 1.0,
            SignalLevel.WEAK: 0.7
        }

        system_logger.info("DynamicTimeoutManager V4.3 inicializado")

    def calculate_timeout(
        self,
        tp_percent: float,
        sl_percent: float,
        volatility: float = 1.0,
        signal_level: SignalLevel = SignalLevel.MODERATE
    ) -> float:
        """
        Calcula timeout proporcional ao TP/SL.

        Formula: base_timeout = TP% / 2
        Ajustado por volatilidade e nivel de sinal.

        Exemplos:
            TP 1% -> timeout 0.5h
            TP 2% -> timeout 1.0h
            TP 4% -> timeout 2.0h

        Args:
            tp_percent: Take profit em percentual (ex: 2.0 para 2%)
            sl_percent: Stop loss em percentual (ex: 1.0 para 1%)
            volatility: Ratio de volatilidade (1.0 = normal)
            signal_level: Nivel do sinal

        Returns:
            Timeout em horas
        """
        # Timeout base proporcional ao TP
        base_timeout = tp_percent * self.base_timeout_ratio

        # Ajuste por volatilidade (alta vol = mais tempo)
        volatility_adjustment = 1 + (volatility - 1) * 0.3

        # Ajuste por nivel de sinal
        level_mult = self.level_multipliers.get(signal_level, 1.0)

        # Calcula timeout final
        timeout = base_timeout * volatility_adjustment * level_mult

        # Aplica limites
        timeout = max(self.min_timeout_hours, min(self.max_timeout_hours, timeout))

        system_logger.debug(
            f"DynamicTimeout: TP={tp_percent}%, SL={sl_percent}%, "
            f"vol={volatility:.2f}, level={signal_level.value} -> {timeout:.2f}h"
        )

        return timeout

    def should_timeout(
        self,
        position: Dict[str, Any],
        current_price: float = None
    ) -> Tuple[bool, str, float]:
        """
        Verifica se posicao deve ser fechada por timeout.

        Args:
            position: Dados da posicao
            current_price: Preco atual (opcional)

        Returns:
            (bool deve_fechar, str motivo, float tempo_aberto_horas)
        """
        entry_time = position.get('entry_time')
        if not entry_time:
            return False, "Sem entry_time", 0

        # Calcula tempo aberto
        now = datetime.now()
        if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo:
            try:
                age_hours = (now.astimezone(entry_time.tzinfo) - entry_time).total_seconds() / 3600
            except:
                age_hours = 0
        else:
            age_hours = (now - entry_time).total_seconds() / 3600

        # Obtem parametros da posicao
        tp_percent = position.get('take_profit', 2.0) * 100  # Converte para %
        sl_percent = position.get('stop_loss', 1.0) * 100
        volatility = position.get('volatility', 1.0)

        # Determina nivel do sinal
        level_str = position.get('entry_level', 'MODERATE')
        try:
            signal_level = SignalLevel(level_str)
        except:
            signal_level = SignalLevel.MODERATE

        # Calcula timeout dinamico
        timeout_hours = self.calculate_timeout(tp_percent, sl_percent, volatility, signal_level)

        # Verifica se excedeu
        if age_hours >= timeout_hours:
            # Verifica P&L atual antes de fechar
            entry_price = position.get('entry_price', 0)
            if current_price and entry_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price * 100

                # Se esta no lucro (mesmo pequeno), extende timeout em 50%
                if pnl_pct > 0.2:  # > 0.2% de lucro
                    extended_timeout = timeout_hours * 1.5
                    if age_hours < extended_timeout:
                        return False, f"Timeout extendido (lucro +{pnl_pct:.2f}%)", age_hours

            return True, f"Timeout atingido ({age_hours:.2f}h >= {timeout_hours:.2f}h)", age_hours

        return False, f"Dentro do timeout ({age_hours:.2f}h / {timeout_hours:.2f}h)", age_hours

    def get_remaining_time(self, position: Dict[str, Any]) -> float:
        """Retorna tempo restante ate timeout em horas."""
        should_close, reason, age_hours = self.should_timeout(position)

        if should_close:
            return 0

        # Recalcula timeout
        tp_percent = position.get('take_profit', 2.0) * 100
        sl_percent = position.get('stop_loss', 1.0) * 100
        volatility = position.get('volatility', 1.0)
        timeout_hours = self.calculate_timeout(tp_percent, sl_percent, volatility)

        return max(0, timeout_hours - age_hours)


# ============================================================================
# SMART TRAILING STOP
# ============================================================================

class SmartTrailingStop:
    """
    V4.3: Trailing stop automatico inteligente.

    Ativa trailing stop apos atingir lucro minimo,
    ajustando stop loss dinamicamente.
    """

    def __init__(self):
        """Inicializa sistema de trailing stop."""

        # Niveis de ativacao por tipo de sinal
        self.activation_levels = {
            SignalLevel.STRONG_OVERRIDE: 1.0,  # Ativa apos 1% de lucro
            SignalLevel.STRONG: 1.5,           # Ativa apos 1.5%
            SignalLevel.MODERATE: 2.0,         # Ativa apos 2%
            SignalLevel.WEAK: 2.5              # Ativa apos 2.5%
        }

        # Distancia do trailing (abaixo do pico)
        self.trailing_distance = 0.5  # 0.5%

        # Estado das posicoes
        self.trailing_state: Dict[str, Dict] = {}

        system_logger.info("SmartTrailingStop V4.3 inicializado")

    def update_trailing(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Atualiza trailing stop para uma posicao.

        Args:
            symbol: Simbolo da posicao
            position: Dados da posicao
            current_price: Preco atual

        Returns:
            Dict com status do trailing
        """
        entry_price = position.get('entry_price', 0)
        if entry_price <= 0:
            return {'active': False, 'reason': 'entry_price invalido'}

        # Calcula P&L atual
        current_pnl_pct = (current_price - entry_price) / entry_price * 100

        # Determina nivel de ativacao
        level_str = position.get('entry_level', 'MODERATE')
        try:
            signal_level = SignalLevel(level_str)
        except:
            signal_level = SignalLevel.MODERATE

        activation_threshold = self.activation_levels.get(signal_level, 2.0)

        # Inicializa estado se necessario
        if symbol not in self.trailing_state:
            self.trailing_state[symbol] = {
                'active': False,
                'trailing_peak': 0,
                'current_stop': position.get('stop_loss', -0.01) * 100,
                'activation_threshold': activation_threshold
            }

        state = self.trailing_state[symbol]

        # Verifica se deve ativar trailing
        if not state['active'] and current_pnl_pct >= activation_threshold:
            state['active'] = True
            state['trailing_peak'] = current_pnl_pct
            state['current_stop'] = current_pnl_pct - self.trailing_distance

            system_logger.info(
                f"🔒 TRAILING ATIVADO {symbol}: Peak={current_pnl_pct:.2f}%, "
                f"Stop={state['current_stop']:.2f}%"
            )

        # Atualiza trailing se ativo
        if state['active']:
            # Atualiza pico se preco subiu
            if current_pnl_pct > state['trailing_peak']:
                state['trailing_peak'] = current_pnl_pct
                new_stop = current_pnl_pct - self.trailing_distance

                # Apenas move stop para cima (nunca para baixo)
                if new_stop > state['current_stop']:
                    old_stop = state['current_stop']
                    state['current_stop'] = new_stop

                    system_logger.info(
                        f"🔒 TRAILING ATUALIZADO {symbol}: "
                        f"Peak={current_pnl_pct:.2f}%, Stop={old_stop:.2f}% -> {new_stop:.2f}%"
                    )

        return {
            'active': state['active'],
            'trailing_peak': state['trailing_peak'],
            'current_stop_pct': state['current_stop'],
            'current_pnl_pct': current_pnl_pct,
            'activation_threshold': activation_threshold,
            'should_close': state['active'] and current_pnl_pct <= state['current_stop']
        }

    def should_close_trailing(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float
    ) -> Tuple[bool, str, float]:
        """
        Verifica se deve fechar posicao por trailing stop.

        Returns:
            (bool deve_fechar, str motivo, float pnl_pct)
        """
        result = self.update_trailing(symbol, position, current_price)

        if result['should_close']:
            pnl = result['current_pnl_pct']
            peak = result['trailing_peak']
            return True, f"Trailing stop atingido (peak={peak:.2f}%, atual={pnl:.2f}%)", pnl

        return False, "Trailing nao atingido", result['current_pnl_pct']

    def remove_position(self, symbol: str):
        """Remove posicao do tracking de trailing."""
        if symbol in self.trailing_state:
            del self.trailing_state[symbol]

    def get_all_states(self) -> Dict[str, Dict]:
        """Retorna estado de todas as posicoes."""
        return self.trailing_state.copy()


# ============================================================================
# STRONG OVERRIDE VALIDATOR
# ============================================================================

class StrongOverrideValidator:
    """
    V4.3: Validacao rigorosa de sinais STRONG_OVERRIDE.

    Criterios obrigatorios:
    1. Score > 65
    2. RSI em zona extrema (<30 ou >70)
    3. Volume > 1.5x medio
    4. Confirmado por 2+ indicadores
    5. Alinhado com regime de mercado
    """

    def __init__(self):
        """Inicializa validador."""

        self.criteria = {
            'min_score': 65,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_volume_ratio': 1.5,
            'min_confirmed_indicators': 2
        }

        # Estatisticas
        self.stats = {
            'total_validated': 0,
            'approved': 0,
            'downgraded': 0,
            'reasons': {}
        }

        system_logger.info("StrongOverrideValidator V4.3 inicializado")

    def validate(
        self,
        signal: Dict[str, Any],
        market_regime: MarketRegime
    ) -> Tuple[bool, str, SignalLevel]:
        """
        Valida se sinal qualifica como STRONG_OVERRIDE.

        Args:
            signal: Dados do sinal
            market_regime: Regime de mercado atual

        Returns:
            (bool valido, str motivo, SignalLevel nivel_final)
        """
        self.stats['total_validated'] += 1

        score = signal.get('total_score', 0) or signal.get('phase2_score', 0)
        rsi = signal.get('rsi', 50)
        volume_ratio = signal.get('volume_ratio', 1.0)
        confirmed_indicators = signal.get('confirmed_indicators', [])

        # Lista de criterios atendidos
        criteria_met = []
        criteria_failed = []

        # Criterio 1: Score > 65
        if score >= self.criteria['min_score']:
            criteria_met.append(f"score={score:.0f}")
        else:
            criteria_failed.append(f"score={score:.0f} < {self.criteria['min_score']}")

        # Criterio 2: RSI em zona extrema
        rsi_extreme = rsi < self.criteria['rsi_oversold'] or rsi > self.criteria['rsi_overbought']
        if rsi_extreme:
            criteria_met.append(f"rsi={rsi:.1f} (extremo)")
        else:
            criteria_failed.append(f"rsi={rsi:.1f} nao extremo")

        # Criterio 3: Volume > 1.5x
        if volume_ratio >= self.criteria['min_volume_ratio']:
            criteria_met.append(f"volume={volume_ratio:.2f}x")
        else:
            criteria_failed.append(f"volume={volume_ratio:.2f}x < {self.criteria['min_volume_ratio']}x")

        # Criterio 4: Confirmacao por indicadores
        num_confirmed = len(confirmed_indicators) if confirmed_indicators else 0
        if num_confirmed >= self.criteria['min_confirmed_indicators']:
            criteria_met.append(f"confirmado por {num_confirmed} indicadores")
        else:
            criteria_failed.append(f"apenas {num_confirmed} indicadores confirmam")

        # Criterio 5: Alinhamento com regime
        regime_aligned = self._check_regime_alignment(rsi, market_regime)
        if regime_aligned:
            criteria_met.append(f"alinhado com regime {market_regime.value}")
        else:
            criteria_failed.append(f"nao alinhado com regime {market_regime.value}")

        # Avalia resultado
        # Precisa de pelo menos 4 dos 5 criterios
        if len(criteria_met) >= 4:
            self.stats['approved'] += 1
            return True, f"STRONG_OVERRIDE validado: {', '.join(criteria_met)}", SignalLevel.STRONG_OVERRIDE
        else:
            self.stats['downgraded'] += 1

            # Registra motivos de downgrade
            for reason in criteria_failed:
                if reason not in self.stats['reasons']:
                    self.stats['reasons'][reason] = 0
                self.stats['reasons'][reason] += 1

            # Determina novo nivel
            if len(criteria_met) >= 3:
                new_level = SignalLevel.STRONG
            elif len(criteria_met) >= 2:
                new_level = SignalLevel.MODERATE
            else:
                new_level = SignalLevel.WEAK

            system_logger.warning(
                f"⚠️ STRONG_OVERRIDE rebaixado para {new_level.value}: "
                f"{', '.join(criteria_failed)}"
            )

            return False, f"Criterios nao atendidos: {', '.join(criteria_failed)}", new_level

    def _check_regime_alignment(self, rsi: float, regime: MarketRegime) -> bool:
        """Verifica se RSI esta alinhado com regime."""
        if regime == MarketRegime.BEAR:
            # Em bear, queremos comprar apenas oversold
            return rsi < self.criteria['rsi_oversold']
        elif regime == MarketRegime.BULL:
            # Em bull, queremos vender apenas overbought (ou comprar pullbacks)
            return rsi > self.criteria['rsi_overbought'] or rsi < 40
        else:
            # Em lateral/volatil, aceita extremos
            return rsi < self.criteria['rsi_oversold'] or rsi > self.criteria['rsi_overbought']

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatisticas de validacao."""
        total = self.stats['total_validated']
        return {
            'total_validated': total,
            'approved': self.stats['approved'],
            'downgraded': self.stats['downgraded'],
            'approval_rate': (self.stats['approved'] / total * 100) if total > 0 else 0,
            'top_rejection_reasons': sorted(
                self.stats['reasons'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# ============================================================================
# INSTANCIAS GLOBAIS
# ============================================================================

# Instancias singleton para uso global
trade_limit_manager = TradeLimitManager(max_daily_trades=20, emergency_max=35)
adaptive_filter = AdaptiveFilter()
dynamic_timeout_manager = DynamicTimeoutManager()
smart_trailing_stop = SmartTrailingStop()
strong_override_validator = StrongOverrideValidator()


# ============================================================================
# FUNCOES DE INTEGRACAO
# ============================================================================

def pre_trade_validation(
    signal: Dict[str, Any],
    market_data: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V4.3: Validacao completa antes de executar trade.

    Integra todos os validadores:
    1. TradeLimitManager
    2. AdaptiveFilter
    3. StrongOverrideValidator

    Args:
        signal: Sinal a ser validado
        market_data: Dados de mercado

    Returns:
        (bool aprovado, str motivo, dict detalhes)
    """
    details = {'signal': signal.get('symbol', 'UNKNOWN')}

    # 1. Verificar limite de trades
    try:
        trade_limit_manager.enforce_limit()
    except TradeLimitExceededError as e:
        return False, str(e), details

    can_trade, reason, remaining = trade_limit_manager.can_trade()
    if not can_trade:
        return False, reason, details

    details['trades_remaining'] = remaining

    # 2. Detectar regime e aplicar filtros
    regime = adaptive_filter.detect_regime(market_data)
    details['regime'] = regime.value

    approved, filter_reason, filter_details = adaptive_filter.apply_regime_filter(signal, regime)
    if not approved:
        return False, filter_reason, {**details, **filter_details}

    # 3. Validar STRONG_OVERRIDE se aplicavel
    signal_level = signal.get('level', 'MODERATE')
    if signal_level == 'STRONG_OVERRIDE':
        valid, override_reason, final_level = strong_override_validator.validate(signal, regime)

        if not valid:
            # Atualiza nivel do sinal
            signal['level'] = final_level.value
            signal['original_level'] = 'STRONG_OVERRIDE'
            details['level_downgraded'] = True
            details['new_level'] = final_level.value

    details['approved'] = True
    return True, "Validacao completa - OK", details


def post_trade_actions(symbol: str, reason: str = ""):
    """
    V4.3: Acoes apos execucao de trade.

    Args:
        symbol: Simbolo tradado
        reason: Motivo/nivel do trade
    """
    trade_limit_manager.increment_trade(symbol, reason)


def position_monitoring(
    symbol: str,
    position: Dict[str, Any],
    current_price: float
) -> Dict[str, Any]:
    """
    V4.3: Monitoramento de posicao aberta.

    Integra:
    1. DynamicTimeoutManager
    2. SmartTrailingStop

    Args:
        symbol: Simbolo da posicao
        position: Dados da posicao
        current_price: Preco atual

    Returns:
        Dict com recomendacoes de acao
    """
    result = {
        'symbol': symbol,
        'should_close': False,
        'close_reason': None
    }

    # 1. Verificar timeout dinamico
    should_timeout, timeout_reason, age_hours = dynamic_timeout_manager.should_timeout(
        position, current_price
    )
    result['age_hours'] = age_hours
    result['timeout_reason'] = timeout_reason

    if should_timeout:
        result['should_close'] = True
        result['close_reason'] = f"TIMEOUT: {timeout_reason}"
        return result

    # 2. Verificar trailing stop
    trailing_close, trailing_reason, pnl = smart_trailing_stop.should_close_trailing(
        symbol, position, current_price
    )
    result['current_pnl_pct'] = pnl
    result['trailing_status'] = smart_trailing_stop.trailing_state.get(symbol, {})

    if trailing_close:
        result['should_close'] = True
        result['close_reason'] = f"TRAILING: {trailing_reason}"
        return result

    return result


# ============================================================================
# LOG DE STATUS
# ============================================================================

def log_v43_status():
    """Loga status de todos os modulos V4.3."""
    system_logger.info("\n" + "="*60)
    system_logger.info("V4.3 TRADING IMPROVEMENTS - STATUS")
    system_logger.info("="*60)

    # Trade Limit
    limit_stats = trade_limit_manager.get_stats()
    system_logger.info(f"\n📊 TRADE LIMIT MANAGER:")
    system_logger.info(f"   Trades: {limit_stats['trades_today']}/{limit_stats['max_trades']}")
    system_logger.info(f"   Restantes: {limit_stats['remaining']}")
    system_logger.info(f"   Utilizacao: {limit_stats['utilization_pct']:.0f}%")

    # Adaptive Filter
    filter_stats = adaptive_filter.get_stats()
    system_logger.info(f"\n🔍 ADAPTIVE FILTER:")
    system_logger.info(f"   Sinais analisados: {filter_stats['total_signals']}")
    system_logger.info(f"   Taxa aprovacao: {filter_stats['approval_rate']:.1f}%")
    if filter_stats['top_rejections']:
        system_logger.info(f"   Top rejeicoes:")
        for reason, count in filter_stats['top_rejections'][:3]:
            system_logger.info(f"      - {reason}: {count}x")

    # Strong Override
    override_stats = strong_override_validator.get_stats()
    system_logger.info(f"\n⚡ STRONG_OVERRIDE VALIDATOR:")
    system_logger.info(f"   Validados: {override_stats['total_validated']}")
    system_logger.info(f"   Taxa aprovacao: {override_stats['approval_rate']:.1f}%")
    system_logger.info(f"   Rebaixados: {override_stats['downgraded']}")

    # Trailing Stop
    trailing_states = smart_trailing_stop.get_all_states()
    active_trailing = sum(1 for s in trailing_states.values() if s.get('active'))
    system_logger.info(f"\n🔒 SMART TRAILING STOP:")
    system_logger.info(f"   Posicoes rastreadas: {len(trailing_states)}")
    system_logger.info(f"   Trailing ativo: {active_trailing}")

    system_logger.info("\n" + "="*60)
