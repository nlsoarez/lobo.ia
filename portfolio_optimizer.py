"""
Otimizador de portfólio para rotatividade de posições.
V4.1 - Gerencia entrada/saída de posições com critérios inteligentes.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum

from config_loader import config
from system_logger import system_logger


class PositionAction(Enum):
    """Ações possíveis para uma posição."""
    HOLD = "hold"
    CLOSE = "close"
    PARTIAL_CLOSE = "partial_close"
    ADD = "add"


class PositionScore:
    """
    Score de uma posição para decisão de rotatividade.
    """

    def __init__(
        self,
        symbol: str,
        current_pnl_pct: float,
        holding_time_hours: float,
        signal_strength: float = 0.5,
        volatility: float = 0.0
    ):
        self.symbol = symbol
        self.current_pnl_pct = current_pnl_pct
        self.holding_time_hours = holding_time_hours
        self.signal_strength = signal_strength
        self.volatility = volatility

        # Calcula score total
        self.total_score = self._calculate_score()

    def _calculate_score(self) -> float:
        """
        Calcula score da posição (maior = melhor para manter).

        Critérios:
        - P&L positivo aumenta score
        - Tempo longo diminui score
        - Sinal forte aumenta score
        """
        score = 50.0  # Base neutra

        # P&L (peso: 40%)
        if self.current_pnl_pct > 0:
            score += min(30, self.current_pnl_pct * 5)  # +5 por % de lucro, max +30
        else:
            score += max(-30, self.current_pnl_pct * 3)  # -3 por % de perda, max -30

        # Tempo de holding (peso: 30%)
        # Posições muito antigas perdem pontos
        if self.holding_time_hours > 48:
            score -= min(20, (self.holding_time_hours - 48) / 24 * 5)  # -5 por dia após 48h

        # Força do sinal atual (peso: 30%)
        score += (self.signal_strength - 0.5) * 40  # -20 a +20

        return max(0, min(100, score))

    def __repr__(self):
        return f"PositionScore({self.symbol}: {self.total_score:.1f})"


class PortfolioOptimizer:
    """
    Otimiza rotatividade de posições no portfólio.

    Funcionalidades:
    - Identifica posições para fechar (estagnadas/perdedoras)
    - Ranking de posições por performance
    - Decisões de take-profit parcial
    - Identificação de oportunidades de rotação
    """

    def __init__(self):
        """Inicializa o otimizador."""
        optimizer_config = config.get('optimizer', {})

        # Configurações de rotatividade
        self.max_holding_hours = optimizer_config.get('max_holding_hours', 72)  # 3 dias
        self.stagnation_threshold = optimizer_config.get('stagnation_threshold', 0.5)  # 0.5%
        self.stagnation_hours = optimizer_config.get('stagnation_hours', 24)  # 24h

        # Take profit parcial
        self.partial_tp_pct = optimizer_config.get('partial_tp_pct', 3.0)  # 3% lucro
        self.partial_tp_close_pct = optimizer_config.get('partial_tp_close', 50)  # Fecha 50%

        # Perda máxima antes de forçar saída
        self.max_loss_pct = optimizer_config.get('max_loss_pct', 5.0)  # 5% perda máxima

        system_logger.info(
            f"PortfolioOptimizer V4.1: max_hold={self.max_holding_hours}h, "
            f"stagnation={self.stagnation_threshold}% em {self.stagnation_hours}h"
        )

    def analyze_positions(
        self,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float],
        signal_strengths: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Analisa todas as posições e retorna recomendações.

        Args:
            positions: Dicionário de posições abertas.
            current_prices: Preços atuais {symbol: price}.
            signal_strengths: Força dos sinais atuais {symbol: strength}.

        Returns:
            Lista de recomendações ordenadas por prioridade.
        """
        if not positions:
            return []

        signal_strengths = signal_strengths or {}
        recommendations = []

        for symbol, position in positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            entry_price = position['avg_price']
            quantity = position['quantity']
            entry_time = position['entry_time']

            # Calcula métricas
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            holding_hours = (datetime.now() - entry_time).total_seconds() / 3600
            signal_strength = signal_strengths.get(symbol, 0.5)

            # Cria score
            score = PositionScore(
                symbol=symbol,
                current_pnl_pct=pnl_pct,
                holding_time_hours=holding_hours,
                signal_strength=signal_strength
            )

            # Determina ação recomendada
            action, reason = self._determine_action(
                pnl_pct, holding_hours, signal_strength, score.total_score
            )

            recommendations.append({
                'symbol': symbol,
                'action': action.value,
                'reason': reason,
                'score': score.total_score,
                'pnl_pct': pnl_pct,
                'holding_hours': holding_hours,
                'signal_strength': signal_strength,
                'current_price': current_price,
                'entry_price': entry_price,
                'quantity': quantity,
                'priority': self._calculate_priority(action, pnl_pct, holding_hours)
            })

        # Ordena por prioridade (maior = mais urgente)
        recommendations.sort(key=lambda x: x['priority'], reverse=True)

        return recommendations

    def _determine_action(
        self,
        pnl_pct: float,
        holding_hours: float,
        signal_strength: float,
        score: float
    ) -> Tuple[PositionAction, str]:
        """
        Determina ação recomendada para posição.

        Returns:
            Tupla (ação, motivo).
        """
        # 1. Perda máxima atingida
        if pnl_pct <= -self.max_loss_pct:
            return PositionAction.CLOSE, f"Perda máxima atingida ({pnl_pct:.2f}%)"

        # 2. Tempo máximo de holding
        if holding_hours >= self.max_holding_hours:
            if pnl_pct <= 0:
                return PositionAction.CLOSE, f"Tempo máximo + prejuízo ({holding_hours:.0f}h)"
            else:
                return PositionAction.CLOSE, f"Tempo máximo atingido ({holding_hours:.0f}h)"

        # 3. Take profit parcial
        if pnl_pct >= self.partial_tp_pct:
            return PositionAction.PARTIAL_CLOSE, f"Take profit parcial ({pnl_pct:.2f}%)"

        # 4. Estagnação (pouca movimentação por muito tempo)
        if holding_hours >= self.stagnation_hours:
            if abs(pnl_pct) <= self.stagnation_threshold:
                return PositionAction.CLOSE, f"Posição estagnada ({pnl_pct:.2f}% em {holding_hours:.0f}h)"

        # 5. Sinal fraco + prejuízo
        if signal_strength < 0.3 and pnl_pct < 0:
            return PositionAction.CLOSE, f"Sinal fraco + prejuízo (sinal: {signal_strength:.2f})"

        # 6. Score muito baixo
        if score < 30:
            return PositionAction.CLOSE, f"Score baixo ({score:.1f})"

        # Default: manter
        return PositionAction.HOLD, "Manter posição"

    def _calculate_priority(
        self,
        action: PositionAction,
        pnl_pct: float,
        holding_hours: float
    ) -> float:
        """Calcula prioridade da ação (maior = mais urgente)."""
        priority = 0

        if action == PositionAction.CLOSE:
            priority = 100
            # Perdas grandes são mais urgentes
            if pnl_pct < 0:
                priority += abs(pnl_pct) * 5
        elif action == PositionAction.PARTIAL_CLOSE:
            priority = 50
            # Lucros grandes são mais urgentes para realizar
            priority += pnl_pct * 2
        elif action == PositionAction.HOLD:
            priority = 0

        return priority

    def get_positions_to_close(
        self,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float],
        max_to_close: int = None
    ) -> List[Dict]:
        """
        Retorna posições que devem ser fechadas.

        Args:
            positions: Posições abertas.
            current_prices: Preços atuais.
            max_to_close: Número máximo de posições a fechar.

        Returns:
            Lista de posições a fechar.
        """
        recommendations = self.analyze_positions(positions, current_prices)

        to_close = [
            r for r in recommendations
            if r['action'] in ['close', 'partial_close']
        ]

        if max_to_close:
            to_close = to_close[:max_to_close]

        return to_close

    def optimize_positions(
        self,
        current_positions: Dict[str, Dict],
        new_signals: List[Dict],
        current_prices: Dict[str, float],
        max_positions: int = 8
    ) -> Dict[str, Any]:
        """
        Otimiza portfólio considerando posições atuais e novos sinais.

        Args:
            current_positions: Posições abertas.
            new_signals: Novos sinais de compra [{symbol, strength, price}, ...].
            current_prices: Preços atuais.
            max_positions: Número máximo de posições.

        Returns:
            Dicionário com ações recomendadas.
        """
        # Analisa posições existentes
        position_analysis = self.analyze_positions(current_positions, current_prices)

        # Identifica posições a fechar
        to_close = [p for p in position_analysis if p['action'] == 'close']
        to_partial_close = [p for p in position_analysis if p['action'] == 'partial_close']
        to_hold = [p for p in position_analysis if p['action'] == 'hold']

        # Calcula espaço para novas posições
        positions_after_close = len(current_positions) - len(to_close)
        available_slots = max(0, max_positions - positions_after_close)

        # Filtra novos sinais (remove símbolos já em carteira)
        existing_symbols = set(current_positions.keys())
        new_candidates = [
            s for s in new_signals
            if s['symbol'] not in existing_symbols
        ]

        # Ordena novos sinais por força
        new_candidates.sort(key=lambda x: x.get('strength', 0), reverse=True)

        # Seleciona melhores candidatos
        to_open = new_candidates[:available_slots]

        # Log resumo
        system_logger.info(
            f"📊 Otimização: {len(to_close)} fechar, {len(to_partial_close)} parcial, "
            f"{len(to_hold)} manter, {len(to_open)} abrir"
        )

        return {
            'to_close': to_close,
            'to_partial_close': to_partial_close,
            'to_hold': to_hold,
            'to_open': to_open,
            'summary': {
                'current_positions': len(current_positions),
                'positions_to_close': len(to_close),
                'positions_to_open': len(to_open),
                'final_positions': positions_after_close + len(to_open)
            }
        }

    def rank_positions(
        self,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Rankeia posições da melhor para a pior.

        Args:
            positions: Posições abertas.
            current_prices: Preços atuais.

        Returns:
            Lista ordenada de posições com scores.
        """
        analysis = self.analyze_positions(positions, current_prices)

        # Ordena por score (maior = melhor)
        analysis.sort(key=lambda x: x['score'], reverse=True)

        return analysis

    def get_rotation_summary(
        self,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> str:
        """
        Retorna resumo da rotatividade em texto.
        """
        analysis = self.analyze_positions(positions, current_prices)

        if not analysis:
            return "Nenhuma posição para analisar."

        lines = [
            "=== Análise de Rotatividade ===",
            f"Total de posições: {len(analysis)}",
            ""
        ]

        # Agrupa por ação
        by_action = {}
        for item in analysis:
            action = item['action']
            if action not in by_action:
                by_action[action] = []
            by_action[action].append(item)

        for action, items in by_action.items():
            lines.append(f"{action.upper()}: {len(items)}")
            for item in items:
                lines.append(
                    f"  {item['symbol']}: {item['pnl_pct']:+.2f}% | "
                    f"{item['holding_hours']:.0f}h | Score: {item['score']:.1f} | "
                    f"{item['reason']}"
                )
            lines.append("")

        return "\n".join(lines)


# Instância global
portfolio_optimizer = PortfolioOptimizer()


if __name__ == "__main__":
    # Teste do otimizador
    from datetime import datetime, timedelta

    optimizer = PortfolioOptimizer()

    # Posições de exemplo
    positions = {
        'PETR4.SA': {
            'quantity': 100,
            'avg_price': 35.00,
            'entry_time': datetime.now() - timedelta(hours=50),
            'total_cost': 3500
        },
        'VALE3.SA': {
            'quantity': 50,
            'avg_price': 68.00,
            'entry_time': datetime.now() - timedelta(hours=10),
            'total_cost': 3400
        },
        'ITUB4.SA': {
            'quantity': 80,
            'avg_price': 30.00,
            'entry_time': datetime.now() - timedelta(hours=80),
            'total_cost': 2400
        }
    }

    # Preços atuais
    prices = {
        'PETR4.SA': 34.50,  # -1.4%
        'VALE3.SA': 70.00,  # +2.9%
        'ITUB4.SA': 29.80   # -0.7%
    }

    # Analisa
    print(optimizer.get_rotation_summary(positions, prices))
