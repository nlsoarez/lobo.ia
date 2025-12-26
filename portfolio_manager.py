"""
Gerenciador de portfólio com controle de risco e posições.
"""

from typing import Dict, Optional, List, Tuple
from datetime import datetime
from config_loader import config
from system_logger import system_logger


class PortfolioManager:
    """
    Gerencia capital, posições abertas e controle de risco.
    Implementa stop-loss, take-profit e validações de exposição.
    """

    def __init__(self, initial_capital: Optional[float] = None):
        """
        Inicializa o gerenciador de portfólio.

        Args:
            initial_capital: Capital inicial em R$. Se None, usa config.yaml.
        """
        # Carrega configurações
        trading_config = config.get_section('trading')
        risk_config = config.get_section('risk')

        if initial_capital is None:
            initial_capital = trading_config.get('capital', 10000.0)

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital

        # Configurações de exposição
        self.exposure_per_trade = trading_config.get('exposure', 0.03)  # 3%
        self.max_total_exposure = trading_config.get('max_total_exposure', 0.20)  # 20%

        # Configurações de risco
        self.stop_loss_pct = risk_config.get('stop_loss', 0.02)  # 2%
        self.take_profit_pct = risk_config.get('take_profit', 0.05)  # 5%
        self.max_drawdown = risk_config.get('max_drawdown', 0.10)  # 10%

        # Posições abertas: {symbol: {quantity, avg_price, entry_time, stop_loss, take_profit}}
        self.positions: Dict[str, Dict] = {}

        # Histórico de trades fechados
        self.trade_history: List[Dict] = []

        system_logger.info(
            f"Portfolio inicializado: R$ {self.current_capital:.2f} "
            f"(Exposição: {self.exposure_per_trade*100:.1f}% por trade, "
            f"Máx total: {self.max_total_exposure*100:.1f}%)"
        )

    def calculate_position_size(self, symbol: str, price: float) -> int:
        """
        Calcula tamanho da posição baseado no capital e exposição.

        Args:
            symbol: Símbolo do ativo.
            price: Preço atual do ativo.

        Returns:
            Quantidade de ações a comprar.

        Raises:
            ValueError: Se não houver capital suficiente ou exposição excedida.
        """
        # Verifica se já tem posição aberta
        if symbol in self.positions:
            system_logger.warning(f"Já existe posição aberta para {symbol}")
            return 0

        # Calcula valor a investir (% do capital)
        investment_amount = self.current_capital * self.exposure_per_trade

        # Verifica se há capital disponível
        if investment_amount > self.available_capital:
            system_logger.warning(
                f"Capital insuficiente: Necessário R$ {investment_amount:.2f}, "
                f"Disponível R$ {self.available_capital:.2f}"
            )
            return 0

        # Calcula quantidade de ações
        quantity = int(investment_amount / price)

        # Valor real da compra
        actual_cost = quantity * price

        # Verifica exposição total
        total_exposure = self._calculate_total_exposure() + actual_cost
        max_exposure_value = self.current_capital * self.max_total_exposure

        if total_exposure > max_exposure_value:
            system_logger.warning(
                f"Exposição máxima excedida: {total_exposure:.2f} > {max_exposure_value:.2f}"
            )
            return 0

        system_logger.info(
            f"Posição calculada: {quantity} ações de {symbol} @ R$ {price:.2f} "
            f"(Total: R$ {actual_cost:.2f})"
        )

        return quantity

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
