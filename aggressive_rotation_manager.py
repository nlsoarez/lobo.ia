"""
V4.0 Phase 3: Aggressive Rotation Manager
Gerencia rotações automáticas entre posições ativas e candidatas.
Substitui trades fracos por melhores oportunidades em tempo real.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from system_logger import system_logger


class RotationType(Enum):
    """Tipos de rotação."""
    OPPORTUNITY = "opportunity"      # Melhor oportunidade disponível
    DEFENSIVE = "defensive"          # Posição em perda
    TIMEOUT = "timeout"              # Posição estagnada
    REBALANCE = "rebalance"          # Rebalanceamento de portfólio
    DIVERSIFICATION = "diversification"  # Melhorar diversificação


@dataclass
class RotationResult:
    """Resultado de uma rotação."""
    success: bool
    rotation_type: RotationType
    from_symbol: str
    to_symbol: str
    closed_pnl: float
    new_position_size: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RotationConfig:
    """Configuração do sistema de rotação."""
    # Limites de rotação
    max_rotations_per_hour: int = 3
    max_daily_rotations: int = 10
    min_time_between_rotations_seconds: int = 120
    cooldown_after_rotation_minutes: int = 2

    # Thresholds para rotação
    min_score_improvement: float = 10.0
    min_position_age_minutes: int = 15
    max_position_age_minutes: int = 30

    # Thresholds de performance
    underperforming_threshold: float = 40.0
    timeout_profit_threshold: float = 0.3  # 0.3%

    # Controles de risco
    rotation_loss_limit_percent: float = 0.5
    require_confirmation: bool = True

    # Exit conditions
    take_profit_partial_percent: float = 1.5
    take_profit_full_percent: float = 2.0
    stop_loss_aggressive_percent: float = 0.8


class AggressiveRotationManager:
    """
    V4.0 Phase 3: Gerenciador de rotação agressiva.
    Otimiza portfólio através de rotações inteligentes.
    """

    def __init__(self, config: Optional[RotationConfig] = None):
        """Inicializa o gerenciador de rotação."""
        self.config = config or RotationConfig()

        # Contadores e estado
        self.rotation_count_today = 0
        self.rotation_count_hour = 0
        self.rotation_pnl_today = 0.0
        self.last_rotation_time: Optional[datetime] = None
        self.rotation_history: List[RotationResult] = []
        self.hour_start = datetime.now().replace(minute=0, second=0, microsecond=0)

        # Circuit breakers
        self.rotation_paused = False
        self.pause_reason = ""
        self.consecutive_failed_rotations = 0

        system_logger.info("AggressiveRotationManager V4.0 Phase 3 inicializado")

    def reset_daily_stats(self):
        """Reseta estatísticas diárias."""
        self.rotation_count_today = 0
        self.rotation_pnl_today = 0.0
        self.rotation_history = []
        self.consecutive_failed_rotations = 0
        self.rotation_paused = False
        system_logger.info("Estatísticas de rotação resetadas")

    def reset_hourly_stats(self):
        """Reseta estatísticas por hora."""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        if current_hour > self.hour_start:
            self.rotation_count_hour = 0
            self.hour_start = current_hour

    def check_rotation_limits(self) -> Tuple[bool, str]:
        """
        Verifica se pode executar rotação baseado nos limites.
        Returns: (can_rotate, reason)
        """
        self.reset_hourly_stats()

        # Verifica circuit breakers
        if self.rotation_paused:
            return False, f"Rotação pausada: {self.pause_reason}"

        # Limite por hora
        if self.rotation_count_hour >= self.config.max_rotations_per_hour:
            return False, f"Limite por hora atingido ({self.config.max_rotations_per_hour})"

        # Limite diário
        if self.rotation_count_today >= self.config.max_daily_rotations:
            return False, f"Limite diário atingido ({self.config.max_daily_rotations})"

        # Tempo mínimo entre rotações
        if self.last_rotation_time:
            elapsed = (datetime.now() - self.last_rotation_time).total_seconds()
            if elapsed < self.config.min_time_between_rotations_seconds:
                remaining = self.config.min_time_between_rotations_seconds - elapsed
                return False, f"Cooldown ativo ({remaining:.0f}s restantes)"

        # Limite de perda em rotações
        if self.rotation_pnl_today <= -self.config.rotation_loss_limit_percent:
            self.rotation_paused = True
            self.pause_reason = f"Perda em rotações > {self.config.rotation_loss_limit_percent}%"
            return False, self.pause_reason

        return True, "OK"

    def should_rotate_position(
        self,
        current_position: Dict[str, Any],
        candidate_signal: Dict[str, Any],
        current_price: float
    ) -> Tuple[bool, str, float]:
        """
        Determina se deve rotacionar posição atual por candidata.
        Returns: (should_rotate, reason, improvement_score)
        """
        # Verifica limites primeiro
        can_rotate, limit_reason = self.check_rotation_limits()
        if not can_rotate:
            return False, limit_reason, 0

        # Calcula métricas de comparação
        comparison = self.compare_position_vs_candidate(
            current_position, candidate_signal, current_price
        )

        # Verifica idade mínima da posição
        position_age_minutes = comparison.get('position_age_minutes', 0)
        if position_age_minutes < self.config.min_position_age_minutes:
            return False, f"Posição muito nova ({position_age_minutes:.0f}min)", 0

        # Verifica melhoria mínima de score
        score_improvement = comparison.get('score_delta', 0)
        if score_improvement < self.config.min_score_improvement:
            return False, f"Melhoria insuficiente ({score_improvement:.1f} < {self.config.min_score_improvement})", score_improvement

        # Verifica se posição atual está underperforming
        current_performance = comparison.get('current_performance_score', 50)
        if current_performance >= 70 and score_improvement < 20:
            return False, "Posição atual performando bem", score_improvement

        # Rotação aprovada
        reason = f"Score +{score_improvement:.1f}, Performance atual {current_performance:.0f}"
        return True, reason, score_improvement

    def compare_position_vs_candidate(
        self,
        current_position: Dict[str, Any],
        candidate: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Compara posição atual vs candidata em múltiplas dimensões.
        """
        now = datetime.now()

        # Calcula idade da posição
        entry_time = current_position.get('entry_time', now)
        if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo:
            try:
                position_age = (now.astimezone(entry_time.tzinfo) - entry_time).total_seconds() / 60
            except:
                position_age = 0
        else:
            position_age = (now - entry_time).total_seconds() / 60 if entry_time else 0

        # Calcula P&L atual
        entry_price = current_position.get('entry_price', current_price)
        current_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0

        # Scores
        current_score = current_position.get('entry_score', 50)
        candidate_score = candidate.get('total_score', 0) or candidate.get('phase2_score', 0)

        # Volume ratios
        current_volume = current_position.get('volume_ratio', 1.0)
        candidate_volume = candidate.get('volume_ratio', 1.0)

        # Pattern scores (Phase 2)
        current_pattern = current_position.get('pattern_score', 0)
        candidate_pattern = candidate.get('phase2_pattern', 0)

        # Momentum
        current_momentum = current_position.get('momentum_score', 50)
        candidate_momentum = candidate.get('momentum_score', 50)

        # Calcula score de performance atual (0-100)
        performance_score = self._calculate_current_performance_score(
            current_pnl, position_age, current_momentum
        )

        return {
            'position_age_minutes': position_age,
            'current_pnl_percent': current_pnl * 100,
            'score_delta': candidate_score - current_score,
            'volume_delta': candidate_volume - current_volume,
            'pattern_delta': candidate_pattern - current_pattern,
            'momentum_delta': candidate_momentum - current_momentum,
            'current_performance_score': performance_score,
            'candidate_score': candidate_score,
            'current_score': current_score,
            'improvement_weighted': self._calculate_weighted_improvement(
                candidate_score - current_score,
                candidate_volume - current_volume,
                candidate_pattern - current_pattern,
                candidate_momentum - current_momentum
            )
        }

    def _calculate_current_performance_score(
        self,
        pnl_percent: float,
        age_minutes: float,
        momentum: float
    ) -> float:
        """Calcula score de performance da posição atual (0-100)."""
        score = 50  # Base

        # Score por P&L
        if pnl_percent >= 0.02:  # 2%+
            score += 40
        elif pnl_percent >= 0.01:  # 1%+
            score += 25
        elif pnl_percent >= 0.005:  # 0.5%+
            score += 10
        elif pnl_percent < -0.005:  # -0.5%
            score -= 20
        elif pnl_percent < -0.01:  # -1%
            score -= 35

        # Penalidade por tempo (posições antigas perdem pontos)
        if age_minutes > 30:
            score -= min(20, (age_minutes - 30) * 0.5)

        # Ajuste por momentum
        if momentum > 70:
            score += 10
        elif momentum < 30:
            score -= 10

        return max(0, min(100, score))

    def _calculate_weighted_improvement(
        self,
        score_delta: float,
        volume_delta: float,
        pattern_delta: float,
        momentum_delta: float
    ) -> float:
        """Calcula melhoria ponderada."""
        weights = {
            'score': 0.35,
            'volume': 0.25,
            'pattern': 0.25,
            'momentum': 0.15
        }

        # Normaliza deltas para 0-100
        normalized_score = min(100, max(-100, score_delta * 2))
        normalized_volume = min(100, max(-100, volume_delta * 30))
        normalized_pattern = pattern_delta
        normalized_momentum = momentum_delta

        weighted_sum = (
            normalized_score * weights['score'] +
            normalized_volume * weights['volume'] +
            normalized_pattern * weights['pattern'] +
            normalized_momentum * weights['momentum']
        )

        return weighted_sum

    def execute_rotation(
        self,
        current_position: Dict[str, Any],
        candidate_signal: Dict[str, Any],
        current_price: float,
        close_position_callback,
        open_position_callback
    ) -> RotationResult:
        """
        Executa rotação: fecha posição atual e abre nova.
        """
        symbol_from = current_position.get('symbol', 'UNKNOWN')
        symbol_to = candidate_signal.get('symbol', 'UNKNOWN')

        try:
            system_logger.info(f"\n🔄 EXECUTANDO ROTAÇÃO: {symbol_from} → {symbol_to}")

            # 1. Fecha posição atual
            close_result = close_position_callback(
                symbol_from, current_price, 'ROTATION'
            )

            if not close_result.get('success', False):
                self.consecutive_failed_rotations += 1
                return RotationResult(
                    success=False,
                    rotation_type=RotationType.OPPORTUNITY,
                    from_symbol=symbol_from,
                    to_symbol=symbol_to,
                    closed_pnl=0,
                    new_position_size=0,
                    reason=f"Falha ao fechar posição: {close_result.get('error', 'Unknown')}"
                )

            closed_pnl = close_result.get('pnl', 0)

            # 2. Pequena pausa para evitar conflitos
            time.sleep(self.config.cooldown_after_rotation_minutes)

            # 3. Abre nova posição
            open_result = open_position_callback(candidate_signal)

            if not open_result.get('success', False):
                self.consecutive_failed_rotations += 1
                # Posição foi fechada mas nova não abriu - registra P&L
                self.rotation_pnl_today += closed_pnl
                return RotationResult(
                    success=False,
                    rotation_type=RotationType.OPPORTUNITY,
                    from_symbol=symbol_from,
                    to_symbol=symbol_to,
                    closed_pnl=closed_pnl,
                    new_position_size=0,
                    reason=f"Fechou {symbol_from} mas falhou ao abrir {symbol_to}"
                )

            # 4. Rotação bem sucedida
            new_position_size = open_result.get('position_size', 0)

            result = RotationResult(
                success=True,
                rotation_type=RotationType.OPPORTUNITY,
                from_symbol=symbol_from,
                to_symbol=symbol_to,
                closed_pnl=closed_pnl,
                new_position_size=new_position_size,
                reason=f"Rotação executada: P&L {closed_pnl:+.2f}%"
            )

            # 5. Atualiza contadores
            self.rotation_count_today += 1
            self.rotation_count_hour += 1
            self.rotation_pnl_today += closed_pnl
            self.last_rotation_time = datetime.now()
            self.rotation_history.append(result)
            self.consecutive_failed_rotations = 0

            system_logger.info(f"✅ Rotação concluída: {symbol_from} → {symbol_to}")
            system_logger.info(f"   P&L fechamento: {closed_pnl:+.2f}%")
            system_logger.info(f"   Rotações hoje: {self.rotation_count_today}/{self.config.max_daily_rotations}")

            return result

        except Exception as e:
            self.consecutive_failed_rotations += 1
            system_logger.error(f"Erro na rotação: {e}")

            # Circuit breaker: 3 falhas consecutivas
            if self.consecutive_failed_rotations >= 3:
                self.rotation_paused = True
                self.pause_reason = "3 rotações consecutivas falharam"

            return RotationResult(
                success=False,
                rotation_type=RotationType.OPPORTUNITY,
                from_symbol=symbol_from,
                to_symbol=symbol_to,
                closed_pnl=0,
                new_position_size=0,
                reason=f"Erro: {str(e)}"
            )

    def find_best_rotation(
        self,
        positions: Dict[str, Dict],
        candidates: List[Dict],
        price_map: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Encontra a melhor rotação possível.
        Compara todas as posições com todos os candidatos.
        """
        can_rotate, reason = self.check_rotation_limits()
        if not can_rotate:
            return None

        best_rotation = None
        best_improvement = 0

        for symbol, position in positions.items():
            current_price = price_map.get(symbol, 0)
            if current_price <= 0:
                continue

            for candidate in candidates[:5]:  # Top 5 candidatos
                # Não rotacionar para o mesmo símbolo
                if candidate.get('symbol') == symbol:
                    continue

                # Não rotacionar para símbolo já em portfólio
                if candidate.get('symbol') in positions:
                    continue

                should_rotate, reason, improvement = self.should_rotate_position(
                    position, candidate, current_price
                )

                if should_rotate and improvement > best_improvement:
                    best_improvement = improvement
                    best_rotation = {
                        'from_position': position,
                        'from_symbol': symbol,
                        'to_candidate': candidate,
                        'to_symbol': candidate.get('symbol'),
                        'improvement': improvement,
                        'reason': reason,
                        'current_price': current_price
                    }

        return best_rotation

    def get_rotation_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de rotação."""
        successful = sum(1 for r in self.rotation_history if r.success)
        total = len(self.rotation_history)

        return {
            'rotations_today': self.rotation_count_today,
            'rotations_this_hour': self.rotation_count_hour,
            'rotation_pnl_today': self.rotation_pnl_today,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'is_paused': self.rotation_paused,
            'pause_reason': self.pause_reason,
            'consecutive_failures': self.consecutive_failed_rotations,
            'last_rotation': self.last_rotation_time.isoformat() if self.last_rotation_time else None
        }

    def log_rotation_status(self):
        """Loga status atual do sistema de rotação."""
        stats = self.get_rotation_stats()

        system_logger.info(f"\n🔄 STATUS ROTAÇÃO:")
        system_logger.info(f"   Rotações hoje: {stats['rotations_today']}/{self.config.max_daily_rotations}")
        system_logger.info(f"   P&L rotações: {stats['rotation_pnl_today']:+.2f}%")
        system_logger.info(f"   Taxa sucesso: {stats['success_rate']:.0f}%")

        if stats['is_paused']:
            system_logger.warning(f"   ⚠️ PAUSADO: {stats['pause_reason']}")
