"""
V4.0 Phase 3: Dynamic Allocation Manager
Gerencia alocação dinâmica de capital baseado em volatilidade e oportunidades.
Implementa Kelly Criterion otimizado para position sizing.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from system_logger import system_logger


@dataclass
class AllocationResult:
    """Resultado de cálculo de alocação."""
    symbol: str
    base_allocation: float
    adjusted_allocation: float
    kelly_fraction: float
    volatility_adjustment: float
    score_adjustment: float
    final_position_size: float
    reason: str


@dataclass
class ReallocationPlan:
    """Plano de realocação de capital."""
    increase_allocations: List[Dict]
    decrease_allocations: List[Dict]
    new_positions: List[Dict]
    freed_capital: float
    total_reallocation: float


class DynamicAllocationManager:
    """
    V4.0 Phase 3: Gerenciador de alocação dinâmica.
    Otimiza alocação de capital usando Kelly Criterion.
    """

    def __init__(self, total_capital: float = 1000.0):
        """Inicializa o gerenciador de alocação."""
        self.total_capital = total_capital
        self.available_capital = total_capital
        self.allocations: Dict[str, float] = {}

        # Limites de alocação
        self.max_single_position_pct = 0.25  # 25% máximo por posição
        self.min_single_position_pct = 0.05  # 5% mínimo por posição
        self.max_total_exposure = 0.95  # 95% máximo exposição total
        self.reserve_capital_pct = 0.05  # 5% reserva mínima

        # Kelly parameters
        self.kelly_fraction = 0.5  # Usar 50% do Kelly (fractional)
        self.base_win_probability = 0.55  # Probabilidade base de ganho
        self.base_payoff_ratio = 2.0  # Ratio ganho/perda base (2:1)

        # Ajustes por volatilidade
        self.high_volatility_reduction = 0.7  # Reduz 30% em alta vol
        self.low_volatility_boost = 1.15  # Aumenta 15% em baixa vol

        system_logger.info(f"DynamicAllocationManager V4.0 inicializado com ${total_capital:.2f}")

    def update_capital(self, new_capital: float, positions_value: float = 0):
        """Atualiza capital total e disponível."""
        self.total_capital = new_capital + positions_value
        self.available_capital = new_capital
        system_logger.debug(f"Capital atualizado: Total=${self.total_capital:.2f}, Disponível=${self.available_capital:.2f}")

    def calculate_kelly_fraction(
        self,
        win_probability: float,
        payoff_ratio: float
    ) -> float:
        """
        Calcula fração de Kelly: f* = (p*b - q) / b
        Onde:
        p = probabilidade de ganho
        q = 1 - p
        b = payoff ratio (ganho/perda)
        """
        if payoff_ratio <= 0 or win_probability <= 0:
            return 0

        q = 1 - win_probability
        kelly = (win_probability * payoff_ratio - q) / payoff_ratio

        # Limita Kelly entre 0% e 25%
        kelly = max(0, min(kelly, 0.25))

        # Aplica fração de segurança
        return kelly * self.kelly_fraction

    def estimate_win_probability(self, signal_score: float) -> float:
        """
        Estima probabilidade de ganho baseado no score do sinal.
        Score 0-100 → Probabilidade 0.40-0.75
        """
        # Mapeia score para probabilidade
        # Score 50 = 55% (base)
        # Score 80+ = 70%
        # Score 30- = 45%
        base = self.base_win_probability

        if signal_score >= 80:
            return min(0.75, base + 0.15)
        elif signal_score >= 70:
            return min(0.70, base + 0.10)
        elif signal_score >= 60:
            return min(0.65, base + 0.05)
        elif signal_score >= 50:
            return base
        elif signal_score >= 40:
            return max(0.45, base - 0.05)
        else:
            return max(0.40, base - 0.10)

    def estimate_payoff_ratio(
        self,
        take_profit: float,
        stop_loss: float,
        volatility_factor: float = 1.0
    ) -> float:
        """
        Estima payoff ratio baseado em TP/SL.
        Ajustado por volatilidade do ativo.
        """
        if stop_loss <= 0:
            return self.base_payoff_ratio

        # Ratio básico
        ratio = take_profit / stop_loss

        # Ajuste por volatilidade (alta vol reduz ratio efetivo)
        adjusted_ratio = ratio / volatility_factor

        return max(0.5, min(5.0, adjusted_ratio))

    def calculate_position_size(
        self,
        symbol: str,
        signal_score: float,
        volatility: float = 1.0,
        take_profit: float = 0.02,
        stop_loss: float = 0.01
    ) -> AllocationResult:
        """
        Calcula tamanho da posição usando Kelly otimizado.
        """
        # Estima probabilidade e payoff
        win_prob = self.estimate_win_probability(signal_score)
        payoff_ratio = self.estimate_payoff_ratio(take_profit, stop_loss, volatility)

        # Calcula fração de Kelly
        kelly = self.calculate_kelly_fraction(win_prob, payoff_ratio)

        # Alocação base (percentual do capital)
        base_allocation = kelly

        # Ajustes
        # 1. Ajuste por score (sinais muito fortes ganham mais)
        score_adjustment = 1.0
        if signal_score >= 80:
            score_adjustment = 1.2
        elif signal_score >= 70:
            score_adjustment = 1.1
        elif signal_score < 50:
            score_adjustment = 0.8

        # 2. Ajuste por volatilidade
        vol_adjustment = 1.0
        if volatility > 1.5:  # Alta volatilidade
            vol_adjustment = self.high_volatility_reduction
        elif volatility < 0.7:  # Baixa volatilidade
            vol_adjustment = self.low_volatility_boost

        # Alocação ajustada
        adjusted_allocation = base_allocation * score_adjustment * vol_adjustment

        # Aplica limites
        adjusted_allocation = max(self.min_single_position_pct, adjusted_allocation)
        adjusted_allocation = min(self.max_single_position_pct, adjusted_allocation)

        # Verifica capital disponível
        max_possible = min(
            self.available_capital / self.total_capital,
            self.max_single_position_pct
        )
        final_allocation = min(adjusted_allocation, max_possible)

        # Calcula tamanho em dólares
        position_size = self.total_capital * final_allocation

        return AllocationResult(
            symbol=symbol,
            base_allocation=base_allocation,
            adjusted_allocation=adjusted_allocation,
            kelly_fraction=kelly,
            volatility_adjustment=vol_adjustment,
            score_adjustment=score_adjustment,
            final_position_size=position_size,
            reason=f"Kelly={kelly:.3f}, WinP={win_prob:.2f}, PayoffR={payoff_ratio:.2f}"
        )

    def calculate_reallocation_plan(
        self,
        current_positions: Dict[str, Dict],
        new_signals: List[Dict],
        price_map: Dict[str, float]
    ) -> ReallocationPlan:
        """
        Calcula plano de realocação de capital.
        """
        increase_allocations = []
        decrease_allocations = []
        new_positions = []
        freed_capital = 0.0

        # 1. Analisa posições atuais
        for symbol, position in current_positions.items():
            current_price = price_map.get(symbol, 0)
            if current_price <= 0:
                continue

            entry_price = position.get('entry_price', current_price)
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

            position_value = position.get('trade_value', 0)
            performance_score = self._calculate_position_performance(position, pnl_pct)

            if performance_score >= 80:
                # Performance excelente → aumentar alocação
                increase_amount = position_value * 0.2  # +20%
                if self.available_capital >= increase_amount:
                    increase_allocations.append({
                        'symbol': symbol,
                        'current_value': position_value,
                        'increase_amount': increase_amount,
                        'reason': f'Performance excelente ({performance_score:.0f})',
                        'new_total': position_value + increase_amount
                    })

            elif performance_score < 30:
                # Performance ruim → reduzir alocação
                decrease_amount = position_value * 0.5  # -50%
                decrease_allocations.append({
                    'symbol': symbol,
                    'current_value': position_value,
                    'decrease_amount': decrease_amount,
                    'reason': f'Performance fraca ({performance_score:.0f})',
                    'new_total': position_value - decrease_amount
                })
                freed_capital += decrease_amount

        # 2. Aloca para novos sinais
        capital_for_new = self.available_capital + freed_capital

        for signal in new_signals[:3]:  # Top 3
            if capital_for_new <= self.total_capital * 0.05:  # Mínimo 5%
                break

            symbol = signal.get('symbol', '')
            if symbol in current_positions:
                continue

            score = signal.get('phase2_score', 0) or signal.get('total_score', 0)
            volatility = signal.get('volatility', 1.0)

            allocation = self.calculate_position_size(
                symbol, score, volatility,
                take_profit=signal.get('phase2_tp', 0.02),
                stop_loss=signal.get('phase2_sl', 0.01)
            )

            if allocation.final_position_size > 0:
                new_positions.append({
                    'symbol': symbol,
                    'allocation': allocation.final_position_size,
                    'score': score,
                    'kelly_fraction': allocation.kelly_fraction,
                    'reason': allocation.reason
                })
                capital_for_new -= allocation.final_position_size

        total_reallocation = (
            sum(i['increase_amount'] for i in increase_allocations) +
            sum(p['allocation'] for p in new_positions)
        )

        return ReallocationPlan(
            increase_allocations=increase_allocations,
            decrease_allocations=decrease_allocations,
            new_positions=new_positions,
            freed_capital=freed_capital,
            total_reallocation=total_reallocation
        )

    def _calculate_position_performance(self, position: Dict, pnl_pct: float) -> float:
        """Calcula score de performance de uma posição (0-100)."""
        score = 50  # Base

        # P&L
        if pnl_pct >= 2.0:
            score += 40
        elif pnl_pct >= 1.0:
            score += 25
        elif pnl_pct >= 0.5:
            score += 15
        elif pnl_pct >= 0:
            score += 5
        elif pnl_pct >= -0.5:
            score -= 10
        elif pnl_pct >= -1.0:
            score -= 25
        else:
            score -= 40

        # Momentum (se disponível)
        momentum = position.get('momentum_score', 50)
        if momentum > 70:
            score += 10
        elif momentum < 30:
            score -= 10

        return max(0, min(100, score))

    def get_optimal_exposure(self, market_volatility: float = 1.0) -> float:
        """
        Calcula exposição ótima baseado em volatilidade do mercado.
        """
        base_exposure = 0.80  # 80% base

        if market_volatility > 2.0:  # Alta volatilidade
            return base_exposure * 0.6  # Reduz para 48%
        elif market_volatility > 1.5:
            return base_exposure * 0.75  # Reduz para 60%
        elif market_volatility < 0.5:  # Baixa volatilidade
            return min(0.95, base_exposure * 1.15)  # Aumenta para 92%
        else:
            return base_exposure

    def should_rebalance(
        self,
        current_positions: Dict[str, Dict],
        target_allocations: Dict[str, float],
        threshold: float = 0.05  # 5% de desvio
    ) -> bool:
        """
        Verifica se deve rebalancear portfólio.
        """
        for symbol, target in target_allocations.items():
            if symbol not in current_positions:
                continue

            current = current_positions[symbol].get('trade_value', 0) / self.total_capital
            deviation = abs(current - target)

            if deviation > threshold:
                return True

        return False

    def get_allocation_stats(self, positions: Dict[str, Dict]) -> Dict[str, Any]:
        """Retorna estatísticas de alocação atual."""
        total_allocated = sum(p.get('trade_value', 0) for p in positions.values())
        exposure_pct = (total_allocated / self.total_capital * 100) if self.total_capital > 0 else 0

        allocations_by_symbol = {
            symbol: (p.get('trade_value', 0) / self.total_capital * 100)
            for symbol, p in positions.items()
        }

        return {
            'total_capital': self.total_capital,
            'available_capital': self.available_capital,
            'total_allocated': total_allocated,
            'exposure_percent': exposure_pct,
            'positions_count': len(positions),
            'allocations': allocations_by_symbol,
            'max_allocation': max(allocations_by_symbol.values()) if allocations_by_symbol else 0,
            'avg_allocation': np.mean(list(allocations_by_symbol.values())) if allocations_by_symbol else 0
        }

    def log_allocation_status(self, positions: Dict[str, Dict]):
        """Loga status de alocação."""
        stats = self.get_allocation_stats(positions)

        system_logger.info(f"\n💰 ALOCAÇÃO DE CAPITAL:")
        system_logger.info(f"   Capital Total: ${stats['total_capital']:.2f}")
        system_logger.info(f"   Disponível: ${stats['available_capital']:.2f}")
        system_logger.info(f"   Alocado: ${stats['total_allocated']:.2f} ({stats['exposure_percent']:.1f}%)")
        system_logger.info(f"   Posições: {stats['positions_count']}")

        if stats['allocations']:
            system_logger.info(f"   Max alocação: {stats['max_allocation']:.1f}%")
            system_logger.info(f"   Média alocação: {stats['avg_allocation']:.1f}%")
