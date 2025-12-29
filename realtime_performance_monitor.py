"""
V4.0 Phase 3: Realtime Performance Monitor
Monitora performance de posições em tempo real para decisões de rotação.
Detecta posições underperforming e oportunidades de saída.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from system_logger import system_logger


@dataclass
class PositionMetrics:
    """Métricas de performance de uma posição."""
    symbol: str
    entry_time: datetime
    entry_price: float
    current_price: float
    high_watermark: float
    low_watermark: float
    pnl_percent: float
    max_pnl_percent: float
    max_drawdown_percent: float
    momentum_trend: str  # bullish, bearish, neutral
    volume_trend: str  # increasing, decreasing, stable
    performance_score: float
    time_in_position_minutes: float
    recommendation: str  # HOLD, EXIT, URGENT_EXIT


@dataclass
class PerformanceSnapshot:
    """Snapshot de performance do portfólio."""
    timestamp: datetime
    total_pnl_percent: float
    winning_positions: int
    losing_positions: int
    avg_pnl: float
    best_performer: str
    worst_performer: str
    portfolio_heat: float  # 0-100 (risco agregado)


class RealtimePerformanceMonitor:
    """
    V4.0 Phase 3: Monitor de performance em tempo real.
    Rastreia e analisa performance de posições para rotação.
    """

    def __init__(self, history_size: int = 100):
        """Inicializa o monitor de performance."""
        # Métricas por posição
        self.position_metrics: Dict[str, PositionMetrics] = {}

        # Histórico de preços (para calcular momentum)
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.history_size = history_size

        # Snapshots do portfólio
        self.portfolio_snapshots: deque = deque(maxlen=100)

        # Thresholds
        self.underperforming_threshold = 40
        self.exit_pnl_threshold = -0.8  # -0.8%
        self.stale_position_minutes = 30
        self.stale_profit_threshold = 0.3  # 0.3%

        system_logger.info("RealtimePerformanceMonitor V4.0 Phase 3 inicializado")

    def update_position(
        self,
        symbol: str,
        position: Dict[str, Any],
        current_price: float,
        current_volume: float = 0
    ) -> PositionMetrics:
        """
        Atualiza métricas de uma posição.
        """
        now = datetime.now()

        # Inicializa históricos se necessário
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.history_size)
            self.volume_history[symbol] = deque(maxlen=self.history_size)

        # Adiciona ao histórico
        self.price_history[symbol].append((now, current_price))
        if current_volume > 0:
            self.volume_history[symbol].append((now, current_volume))

        # Dados da posição
        entry_time = position.get('entry_time', now)
        entry_price = position.get('entry_price', current_price)
        max_price = position.get('max_price', entry_price)

        # Calcula métricas
        pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        max_pnl = ((max_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

        # High/Low watermarks
        if symbol in self.position_metrics:
            prev_metrics = self.position_metrics[symbol]
            high_watermark = max(prev_metrics.high_watermark, current_price)
            low_watermark = min(prev_metrics.low_watermark, current_price)
        else:
            high_watermark = max(entry_price, current_price)
            low_watermark = min(entry_price, current_price)

        # Drawdown do high watermark
        max_drawdown = ((high_watermark - current_price) / high_watermark * 100) if high_watermark > 0 else 0

        # Tempo na posição
        try:
            if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo:
                time_in_position = (now.astimezone(entry_time.tzinfo) - entry_time).total_seconds() / 60
            else:
                time_in_position = (now - entry_time).total_seconds() / 60
        except:
            time_in_position = 0

        # Calcula tendências
        momentum_trend = self._calculate_momentum_trend(symbol)
        volume_trend = self._calculate_volume_trend(symbol)

        # Performance score
        performance_score = self._calculate_performance_score(
            pnl_percent, max_pnl, max_drawdown, time_in_position, momentum_trend
        )

        # Recomendação
        recommendation = self._get_recommendation(
            pnl_percent, time_in_position, performance_score, momentum_trend
        )

        metrics = PositionMetrics(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            current_price=current_price,
            high_watermark=high_watermark,
            low_watermark=low_watermark,
            pnl_percent=pnl_percent,
            max_pnl_percent=max_pnl,
            max_drawdown_percent=max_drawdown,
            momentum_trend=momentum_trend,
            volume_trend=volume_trend,
            performance_score=performance_score,
            time_in_position_minutes=time_in_position,
            recommendation=recommendation
        )

        self.position_metrics[symbol] = metrics

        return metrics

    def _calculate_momentum_trend(self, symbol: str) -> str:
        """Calcula tendência de momentum baseado no histórico de preços."""
        if symbol not in self.price_history:
            return 'neutral'

        prices = self.price_history[symbol]
        if len(prices) < 5:
            return 'neutral'

        # Pega últimos 10 preços
        recent_prices = [p[1] for p in list(prices)[-10:]]

        if len(recent_prices) < 3:
            return 'neutral'

        # Calcula mudança percentual
        first_half_avg = np.mean(recent_prices[:len(recent_prices)//2])
        second_half_avg = np.mean(recent_prices[len(recent_prices)//2:])

        if first_half_avg <= 0:
            return 'neutral'

        change = (second_half_avg - first_half_avg) / first_half_avg

        if change > 0.005:  # +0.5%
            return 'strong_bullish' if change > 0.01 else 'bullish'
        elif change < -0.005:  # -0.5%
            return 'strong_bearish' if change < -0.01 else 'bearish'
        else:
            return 'neutral'

    def _calculate_volume_trend(self, symbol: str) -> str:
        """Calcula tendência de volume."""
        if symbol not in self.volume_history:
            return 'stable'

        volumes = self.volume_history[symbol]
        if len(volumes) < 5:
            return 'stable'

        recent_volumes = [v[1] for v in list(volumes)[-10:]]

        if len(recent_volumes) < 3:
            return 'stable'

        first_half_avg = np.mean(recent_volumes[:len(recent_volumes)//2])
        second_half_avg = np.mean(recent_volumes[len(recent_volumes)//2:])

        if first_half_avg <= 0:
            return 'stable'

        change = (second_half_avg - first_half_avg) / first_half_avg

        if change > 0.2:
            return 'increasing'
        elif change < -0.2:
            return 'decreasing'
        else:
            return 'stable'

    def _calculate_performance_score(
        self,
        pnl_percent: float,
        max_pnl: float,
        max_drawdown: float,
        time_minutes: float,
        momentum: str
    ) -> float:
        """Calcula score de performance (0-100)."""
        score = 50  # Base

        # 1. Score por P&L atual (0-30 pontos)
        if pnl_percent >= 2.0:
            score += 30
        elif pnl_percent >= 1.0:
            score += 22
        elif pnl_percent >= 0.5:
            score += 15
        elif pnl_percent >= 0:
            score += 8
        elif pnl_percent >= -0.5:
            score -= 5
        elif pnl_percent >= -1.0:
            score -= 15
        else:
            score -= 30

        # 2. Score por profit per minute (-10 a +10 pontos)
        if time_minutes > 0:
            profit_per_minute = pnl_percent / time_minutes
            if profit_per_minute > 0.05:
                score += 10
            elif profit_per_minute > 0.02:
                score += 5
            elif profit_per_minute < -0.02:
                score -= 5
            elif profit_per_minute < -0.05:
                score -= 10

        # 3. Score por momentum (-15 a +15 pontos)
        momentum_scores = {
            'strong_bullish': 15,
            'bullish': 8,
            'neutral': 0,
            'bearish': -8,
            'strong_bearish': -15
        }
        score += momentum_scores.get(momentum, 0)

        # 4. Penalidade por drawdown (-15 pontos max)
        if max_drawdown > 2.0:
            score -= 15
        elif max_drawdown > 1.0:
            score -= 10
        elif max_drawdown > 0.5:
            score -= 5

        # 5. Penalidade por tempo estagnado (-10 pontos max)
        if time_minutes > 30 and pnl_percent < 0.3:
            score -= min(10, (time_minutes - 30) * 0.3)

        return max(0, min(100, score))

    def _get_recommendation(
        self,
        pnl_percent: float,
        time_minutes: float,
        score: float,
        momentum: str
    ) -> str:
        """Determina recomendação baseado nas métricas."""
        # Saída urgente
        if pnl_percent <= self.exit_pnl_threshold:
            return 'URGENT_EXIT'

        if score < 20:
            return 'URGENT_EXIT'

        # Posição estagnada
        if time_minutes > self.stale_position_minutes and pnl_percent < self.stale_profit_threshold:
            return 'EXIT'

        # Performance fraca
        if score < self.underperforming_threshold:
            return 'EXIT'

        # Momentum bearish com lucro pequeno
        if momentum in ['bearish', 'strong_bearish'] and 0 < pnl_percent < 0.5:
            return 'EXIT'

        return 'HOLD'

    def get_underperforming_positions(
        self,
        threshold: Optional[float] = None
    ) -> List[PositionMetrics]:
        """Retorna posições com performance abaixo do threshold."""
        if threshold is None:
            threshold = self.underperforming_threshold

        underperforming = [
            metrics for metrics in self.position_metrics.values()
            if metrics.performance_score < threshold
        ]

        # Ordena por score (pior primeiro)
        underperforming.sort(key=lambda x: x.performance_score)

        return underperforming

    def get_exit_candidates(self) -> List[PositionMetrics]:
        """Retorna posições candidatas para saída."""
        candidates = [
            metrics for metrics in self.position_metrics.values()
            if metrics.recommendation in ['EXIT', 'URGENT_EXIT']
        ]

        # Ordena por urgência (URGENT_EXIT primeiro) e score
        def sort_key(m):
            urgency = 0 if m.recommendation == 'URGENT_EXIT' else 1
            return (urgency, m.performance_score)

        candidates.sort(key=sort_key)

        return candidates

    def take_portfolio_snapshot(
        self,
        positions: Dict[str, Dict],
        price_map: Dict[str, float]
    ) -> PerformanceSnapshot:
        """Tira snapshot do portfólio atual."""
        now = datetime.now()

        if not positions:
            snapshot = PerformanceSnapshot(
                timestamp=now,
                total_pnl_percent=0,
                winning_positions=0,
                losing_positions=0,
                avg_pnl=0,
                best_performer='N/A',
                worst_performer='N/A',
                portfolio_heat=0
            )
            self.portfolio_snapshots.append(snapshot)
            return snapshot

        pnls = []
        best_pnl = -float('inf')
        worst_pnl = float('inf')
        best_symbol = ''
        worst_symbol = ''

        for symbol, position in positions.items():
            current_price = price_map.get(symbol, 0)
            entry_price = position.get('entry_price', current_price)

            if entry_price > 0 and current_price > 0:
                pnl = ((current_price - entry_price) / entry_price * 100)
                pnls.append(pnl)

                if pnl > best_pnl:
                    best_pnl = pnl
                    best_symbol = symbol

                if pnl < worst_pnl:
                    worst_pnl = pnl
                    worst_symbol = symbol

        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls) if pnls else 0
        winning = sum(1 for p in pnls if p > 0)
        losing = sum(1 for p in pnls if p < 0)

        # Portfolio heat (risco agregado)
        heat = self._calculate_portfolio_heat(positions, price_map)

        snapshot = PerformanceSnapshot(
            timestamp=now,
            total_pnl_percent=total_pnl,
            winning_positions=winning,
            losing_positions=losing,
            avg_pnl=avg_pnl,
            best_performer=f"{best_symbol} ({best_pnl:+.2f}%)" if best_symbol else 'N/A',
            worst_performer=f"{worst_symbol} ({worst_pnl:+.2f}%)" if worst_symbol else 'N/A',
            portfolio_heat=heat
        )

        self.portfolio_snapshots.append(snapshot)

        return snapshot

    def _calculate_portfolio_heat(
        self,
        positions: Dict[str, Dict],
        price_map: Dict[str, float]
    ) -> float:
        """
        Calcula 'calor' do portfólio (0-100).
        Representa risco agregado baseado em posições perdendo e drawdown.
        """
        if not positions:
            return 0

        heat = 0
        count = 0

        for symbol, position in positions.items():
            current_price = price_map.get(symbol, 0)
            entry_price = position.get('entry_price', current_price)
            max_price = position.get('max_price', entry_price)

            if entry_price <= 0:
                continue

            pnl = ((current_price - entry_price) / entry_price * 100)
            drawdown = ((max_price - current_price) / max_price * 100) if max_price > 0 else 0

            # Heat por posição
            pos_heat = 0

            # Perda aumenta heat
            if pnl < 0:
                pos_heat += min(50, abs(pnl) * 20)

            # Drawdown aumenta heat
            pos_heat += min(30, drawdown * 10)

            heat += pos_heat
            count += 1

        # Normaliza para 0-100
        if count > 0:
            heat = heat / count
            # Ajusta para múltiplas posições em risco
            losing_count = sum(1 for s, p in positions.items()
                              if price_map.get(s, 0) < p.get('entry_price', 0))
            if losing_count > 1:
                heat *= (1 + losing_count * 0.1)

        return min(100, heat)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Retorna resumo de performance."""
        if not self.position_metrics:
            return {
                'positions_tracked': 0,
                'avg_performance_score': 0,
                'underperforming_count': 0,
                'exit_candidates': 0
            }

        scores = [m.performance_score for m in self.position_metrics.values()]
        underperforming = self.get_underperforming_positions()
        exit_candidates = self.get_exit_candidates()

        return {
            'positions_tracked': len(self.position_metrics),
            'avg_performance_score': np.mean(scores),
            'min_performance_score': min(scores),
            'max_performance_score': max(scores),
            'underperforming_count': len(underperforming),
            'exit_candidates': len(exit_candidates),
            'latest_snapshot': self.portfolio_snapshots[-1] if self.portfolio_snapshots else None
        }

    def remove_position(self, symbol: str):
        """Remove posição do monitoramento."""
        if symbol in self.position_metrics:
            del self.position_metrics[symbol]
        if symbol in self.price_history:
            del self.price_history[symbol]
        if symbol in self.volume_history:
            del self.volume_history[symbol]

    def log_performance_status(self, positions: Dict[str, Dict], price_map: Dict[str, float]):
        """Loga status de performance."""
        # Atualiza todas as posições
        for symbol, position in positions.items():
            current_price = price_map.get(symbol, 0)
            if current_price > 0:
                self.update_position(symbol, position, current_price)

        summary = self.get_performance_summary()

        system_logger.info(f"\n📈 PERFORMANCE MONITOR:")
        system_logger.info(f"   Posições monitoradas: {summary['positions_tracked']}")
        system_logger.info(f"   Score médio: {summary['avg_performance_score']:.0f}")
        system_logger.info(f"   Underperforming: {summary['underperforming_count']}")
        system_logger.info(f"   Candidatos saída: {summary['exit_candidates']}")

        # Lista posições com recomendação de saída
        exit_candidates = self.get_exit_candidates()
        if exit_candidates:
            system_logger.info("   ⚠️ Saídas recomendadas:")
            for m in exit_candidates[:3]:
                emoji = "🔴" if m.recommendation == 'URGENT_EXIT' else "🟡"
                system_logger.info(
                    f"      {emoji} {m.symbol}: Score={m.performance_score:.0f} "
                    f"P&L={m.pnl_percent:+.2f}% [{m.recommendation}]"
                )
