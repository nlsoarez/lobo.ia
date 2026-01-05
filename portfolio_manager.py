"""
Gerenciador de portfólio com controle de risco e posições.
V4.2 - Sistema de Trading de Criptomoedas (USD)
"""

from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from config_loader import config
from system_logger import system_logger


class PortfolioManager:
    """
    Gerencia capital, posições abertas e controle de risco para criptomoedas.
    Implementa stop-loss, take-profit e validações de exposição.

    V4.2 - Crypto Only (USD):
    - Exposição baseada em percentual do capital total
    - Quantidade fracionária para criptomoedas
    - Exposição por posição: 5-20% (configurável)
    - Exposição total máxima: 80% (configurável)
    - Ajuste dinâmico baseado na força do sinal
    """

    def __init__(self, initial_capital: Optional[float] = None):
        """
        Inicializa o gerenciador de portfólio.

        Args:
            initial_capital: Capital inicial em USD. Se None, usa config.yaml.
        """
        # Carrega configurações
        crypto_config = config.get_section('crypto')
        risk_config = config.get_section('risk')

        if initial_capital is None:
            initial_capital = crypto_config.get('capital', 1000.0)

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital

        # V4.2: Configurações de exposição para crypto
        # Exposição base por trade (% do capital TOTAL)
        self.exposure_per_trade = crypto_config.get('exposure', 0.10)  # 10% por trade

        # Limites de exposição por posição
        self.min_exposure_per_trade = crypto_config.get('min_exposure', 0.05)  # Mínimo 5%
        self.max_exposure_per_trade = crypto_config.get('max_exposure_per_trade', 0.20)  # Máximo 20% por posição

        # V4.2: Exposição total máxima 80%
        self.max_total_exposure = crypto_config.get('max_total_exposure', 0.80)

        # V4.2: Número máximo de posições simultâneas
        self.max_positions = crypto_config.get('max_positions', 8)

        # V4.2: Valor mínimo por trade em USD
        self.min_trade_value = crypto_config.get('min_trade_value', 50.0)  # $50 mínimo

        # Configurações de risco
        self.stop_loss_pct = risk_config.get('stop_loss', 0.02)  # 2%
        self.take_profit_pct = risk_config.get('take_profit', 0.05)  # 5%
        self.max_drawdown = risk_config.get('max_drawdown', 0.10)  # 10%

        # Posições abertas: {symbol: {quantity, avg_price, entry_time, stop_loss, take_profit}}
        self.positions: Dict[str, Dict] = {}

        # Histórico de trades fechados
        self.trade_history: List[Dict] = []

        system_logger.info(
            f"Portfolio V4.2 Crypto inicializado: ${self.current_capital:.2f} USD | "
            f"Exposição: {self.exposure_per_trade*100:.1f}% por trade | "
            f"Máx total: {self.max_total_exposure*100:.1f}% | "
            f"Máx posições: {self.max_positions}"
        )

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        signal_strength: float = 0.5
    ) -> int:
        """
        Calcula tamanho da posição baseado no capital, exposição e força do sinal.

        V4.1 Melhorias:
        - Garante quantidade mínima >= 1 quando há capital
        - Ajusta exposição baseado na força do sinal
        - Verifica limite de posições simultâneas
        - Logs detalhados para debugging

        Args:
            symbol: Símbolo do ativo.
            price: Preço atual do ativo.
            signal_strength: Força do sinal (0-1), afeta tamanho da posição.

        Returns:
            Quantidade de ações a comprar (>= 1 ou 0 se impossível).
        """
        # Verifica se já tem posição aberta
        if symbol in self.positions:
            system_logger.debug(f"Já existe posição aberta para {symbol}")
            return 0

        # V4.1: Verifica limite de posições
        if len(self.positions) >= self.max_positions:
            system_logger.warning(
                f"Limite de posições atingido: {len(self.positions)}/{self.max_positions}"
            )
            return 0

        # V4.1: Calcula exposição ajustada pela força do sinal
        # signal_strength 0.5 = 100% da exposição base
        # signal_strength 1.0 = 150% da exposição base
        # signal_strength 0.3 = 80% da exposição base
        signal_multiplier = 0.5 + (signal_strength * 1.0)  # 0.5 a 1.5
        adjusted_exposure = self.exposure_per_trade * signal_multiplier

        # Limita à exposição máxima por posição
        adjusted_exposure = min(adjusted_exposure, self.max_exposure_per_trade)
        adjusted_exposure = max(adjusted_exposure, self.min_exposure_per_trade)

        # Calcula valor a investir (% do capital TOTAL)
        investment_amount = self.current_capital * adjusted_exposure

        # V4.1: Ajusta se exceder capital disponível
        if investment_amount > self.available_capital:
            # Tenta usar o que está disponível (se for suficiente)
            investment_amount = self.available_capital
            system_logger.debug(
                f"Ajustando investimento para capital disponível: R$ {investment_amount:.2f}"
            )

        # V4.1: Verifica valor mínimo de trade
        if investment_amount < self.min_trade_value:
            system_logger.warning(
                f"Capital insuficiente para trade mínimo: R$ {investment_amount:.2f} "
                f"< R$ {self.min_trade_value:.2f}"
            )
            return 0

        # Calcula quantidade de ações
        quantity = int(investment_amount / price)

        # V4.1: Garante quantidade mínima de 1 ação
        if quantity < 1 and price <= self.available_capital:
            quantity = 1
            system_logger.debug(
                f"Ajustando para quantidade mínima: 1 ação de {symbol}"
            )

        if quantity < 1:
            system_logger.warning(
                f"Preço muito alto para capital disponível: {symbol} @ R$ {price:.2f}"
            )
            return 0

        # Valor real da compra
        actual_cost = quantity * price

        # V4.1: Verifica se ainda cabe no capital disponível
        if actual_cost > self.available_capital:
            quantity = int(self.available_capital / price)
            if quantity < 1:
                system_logger.warning(
                    f"Capital disponível insuficiente: R$ {self.available_capital:.2f} "
                    f"para {symbol} @ R$ {price:.2f}"
                )
                return 0
            actual_cost = quantity * price

        # Verifica exposição total
        current_exposure = self._calculate_total_exposure()
        new_total_exposure = current_exposure + actual_cost
        max_exposure_value = self.current_capital * self.max_total_exposure

        if new_total_exposure > max_exposure_value:
            # V4.1: Tenta reduzir quantidade para caber na exposição
            available_for_new = max_exposure_value - current_exposure
            if available_for_new >= price:
                quantity = int(available_for_new / price)
                actual_cost = quantity * price
                system_logger.info(
                    f"Reduzindo posição para caber na exposição máxima: {quantity} ações"
                )
            else:
                system_logger.warning(
                    f"Exposição máxima atingida: {current_exposure:.2f}/{max_exposure_value:.2f} "
                    f"({current_exposure/self.current_capital*100:.1f}%)"
                )
                return 0

        # V4.1: Log detalhado da decisão
        exposure_pct = (actual_cost / self.current_capital) * 100
        total_exposure_pct = (new_total_exposure / self.current_capital) * 100

        system_logger.info(
            f"📊 Posição calculada: {quantity} x {symbol} @ R$ {price:.2f} = R$ {actual_cost:.2f} | "
            f"Exposição: {exposure_pct:.1f}% | Total: {total_exposure_pct:.1f}% | "
            f"Sinal: {signal_strength:.2f}"
        )

        return quantity

    def get_allocation_status(self) -> Dict:
        """
        Retorna status detalhado da alocação de capital.

        Returns:
            Dicionário com métricas de alocação.
        """
        total_exposure = self._calculate_total_exposure()
        max_exposure = self.current_capital * self.max_total_exposure

        return {
            'current_capital': self.current_capital,
            'available_capital': self.available_capital,
            'total_exposure': total_exposure,
            'total_exposure_pct': (total_exposure / self.current_capital) * 100,
            'max_exposure': max_exposure,
            'max_exposure_pct': self.max_total_exposure * 100,
            'remaining_capacity': max_exposure - total_exposure,
            'open_positions': len(self.positions),
            'max_positions': self.max_positions,
            'can_open_new': len(self.positions) < self.max_positions and total_exposure < max_exposure
        }

    def open_position(
        self,
        symbol: str,
        quantity: int,
        price: float
    ) -> bool:
        """
        Abre uma nova posição.

        Args:
            symbol: Símbolo do ativo.
            quantity: Quantidade de ações.
            price: Preço de entrada.

        Returns:
            True se posição foi aberta com sucesso.
        """
        if quantity <= 0:
            system_logger.error(f"Quantidade inválida: {quantity}")
            return False

        if symbol in self.positions:
            system_logger.warning(f"Posição já existe para {symbol}")
            return False

        # Calcula custos
        total_cost = quantity * price

        if total_cost > self.available_capital:
            system_logger.error(
                f"Capital insuficiente para abrir posição: "
                f"R$ {total_cost:.2f} > R$ {self.available_capital:.2f}"
            )
            return False

        # Calcula stop-loss e take-profit
        stop_loss = price * (1 - self.stop_loss_pct)
        take_profit = price * (1 + self.take_profit_pct)

        # Registra posição
        self.positions[symbol] = {
            'quantity': quantity,
            'avg_price': price,
            'entry_time': datetime.now(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'total_cost': total_cost
        }

        # Atualiza capital disponível
        self.available_capital -= total_cost

        system_logger.info(
            f"✅ Posição ABERTA: {quantity} {symbol} @ R$ {price:.2f} "
            f"(SL: {stop_loss:.2f}, TP: {take_profit:.2f})"
        )

        return True

    def close_position(
        self,
        symbol: str,
        price: float,
        reason: str = "manual"
    ) -> Optional[Dict]:
        """
        Fecha uma posição existente.

        Args:
            symbol: Símbolo do ativo.
            price: Preço de saída.
            reason: Motivo do fechamento (manual, stop_loss, take_profit).

        Returns:
            Dicionário com resultado do trade ou None se não existir posição.
        """
        if symbol not in self.positions:
            system_logger.warning(f"Nenhuma posição aberta para {symbol}")
            return None

        position = self.positions[symbol]

        # Calcula resultado
        quantity = position['quantity']
        entry_price = position['avg_price']
        entry_cost = position['total_cost']
        exit_value = quantity * price

        profit = exit_value - entry_cost
        profit_pct = (profit / entry_cost) * 100

        # Atualiza capital
        self.available_capital += exit_value
        self.current_capital += profit

        # Cria registro do trade
        trade_result = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': price,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'profit': profit,
            'profit_pct': profit_pct,
            'reason': reason
        }

        # Adiciona ao histórico
        self.trade_history.append(trade_result)

        # Remove posição
        del self.positions[symbol]

        emoji = "🟢" if profit >= 0 else "🔴"
        system_logger.info(
            f"{emoji} Posição FECHADA: {quantity} {symbol} @ R$ {price:.2f} | "
            f"Lucro: R$ {profit:.2f} ({profit_pct:+.2f}%) | "
            f"Motivo: {reason}"
        )

        return trade_result

    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Verifica se stop-loss ou take-profit foi atingido.

        Args:
            symbol: Símbolo do ativo.
            current_price: Preço atual.

        Returns:
            "stop_loss", "take_profit" ou None.
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Verifica stop-loss
        if current_price <= position['stop_loss']:
            system_logger.warning(
                f"🛑 STOP LOSS atingido: {symbol} @ R$ {current_price:.2f} "
                f"(SL: R$ {position['stop_loss']:.2f})"
            )
            return "stop_loss"

        # Verifica take-profit
        if current_price >= position['take_profit']:
            system_logger.info(
                f"🎯 TAKE PROFIT atingido: {symbol} @ R$ {current_price:.2f} "
                f"(TP: R$ {position['take_profit']:.2f})"
            )
            return "take_profit"

        return None

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Retorna informações de uma posição.

        Args:
            symbol: Símbolo do ativo.

        Returns:
            Dicionário com dados da posição ou None.
        """
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """
        Verifica se existe posição aberta para o símbolo.

        Args:
            symbol: Símbolo do ativo.

        Returns:
            True se existe posição.
        """
        return symbol in self.positions

    def get_all_positions(self) -> Dict[str, Dict]:
        """
        Retorna todas as posições abertas.

        Returns:
            Dicionário com todas as posições.
        """
        return self.positions.copy()

    def _calculate_total_exposure(self) -> float:
        """
        Calcula exposição total atual.

        Returns:
            Valor total investido em posições abertas.
        """
        return sum(pos['total_cost'] for pos in self.positions.values())

    def get_performance_stats(self) -> Dict:
        """
        Calcula estatísticas de performance do portfólio.

        Returns:
            Dicionário com métricas de performance.
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'profit_factor': 0,
                'current_capital': self.current_capital,
                'available_capital': self.available_capital,
                'open_positions': len(self.positions)
            }

        wins = [t for t in self.trade_history if t['profit'] > 0]
        losses = [t for t in self.trade_history if t['profit'] < 0]

        total_wins_value = sum(t['profit'] for t in wins)
        total_losses_value = abs(sum(t['profit'] for t in losses))

        profit_factor = (
            total_wins_value / total_losses_value
            if total_losses_value > 0 else float('inf')
        )

        return {
            'total_trades': len(self.trade_history),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(self.trade_history)) * 100,
            'total_profit': self.current_capital - self.initial_capital,
            'avg_profit': sum(t['profit'] for t in self.trade_history) / len(self.trade_history),
            'max_profit': max((t['profit'] for t in self.trade_history), default=0),
            'max_loss': min((t['profit'] for t in self.trade_history), default=0),
            'profit_factor': profit_factor,
            'current_capital': self.current_capital,
            'available_capital': self.available_capital,
            'open_positions': len(self.positions)
        }

    def is_drawdown_exceeded(self) -> bool:
        """
        Verifica se o drawdown máximo foi excedido.

        Returns:
            True se drawdown foi excedido.
        """
        current_drawdown = (self.initial_capital - self.current_capital) / self.initial_capital

        if current_drawdown > self.max_drawdown:
            system_logger.critical(
                f"⚠️ DRAWDOWN MÁXIMO EXCEDIDO: {current_drawdown*100:.2f}% "
                f"(Máx: {self.max_drawdown*100:.2f}%)"
            )
            return True

        return False
