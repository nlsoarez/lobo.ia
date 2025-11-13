"""
Executor de trades com simulação de slippage e taxas.
"""

import time
from typing import Dict, Optional, Any
from datetime import datetime

from config_loader import config
from system_logger import system_logger


class TradeExecutor:
    """
    Executa ordens de compra/venda.
    Suporta modo simulação (paper trading) e execução real (futuro).
    """

    def __init__(self):
        """Inicializa o executor de trades."""
        execution_config = config.get_section('execution')

        self.mode = execution_config.get('mode', 'simulation')
        self.simulate_slippage = execution_config.get('simulate_slippage', 0.001)  # 0.1%
        self.simulate_fees = execution_config.get('simulate_fees', 0.0005)  # 0.05%
        self.execution_delay = execution_config.get('execution_delay', 1)  # segundos

        self.orders = []

        system_logger.info(
            f"TradeExecutor inicializado (Modo: {self.mode}, "
            f"Slippage: {self.simulate_slippage*100:.2f}%, "
            f"Taxas: {self.simulate_fees*100:.2f}%)"
        )

    def execute_order(
        self,
        symbol: str,
        signal: str,
        price: float,
        quantity: int
    ) -> Optional[Dict[str, Any]]:
        """
        Executa uma ordem de compra ou venda.

        Args:
            symbol: Símbolo do ativo.
            signal: Tipo de ordem ("BUY" ou "SELL").
            price: Preço de referência.
            quantity: Quantidade de ações.

        Returns:
            Dicionário com resultado da execução ou None em caso de falha.
        """
        if quantity <= 0:
            system_logger.error(f"Quantidade inválida: {quantity}")
            return None

        if signal not in ['BUY', 'SELL']:
            system_logger.error(f"Sinal inválido: {signal}")
            return None

        system_logger.info(
            f"🔧 Executando ordem: {signal} {quantity} {symbol} @ R$ {price:.2f}"
        )

        if self.mode == 'simulation':
            return self._execute_simulated(symbol, signal, price, quantity)
        elif self.mode == 'paper':
            return self._execute_paper_trading(symbol, signal, price, quantity)
        elif self.mode == 'live':
            return self._execute_live(symbol, signal, price, quantity)
        else:
            system_logger.error(f"Modo de execução inválido: {self.mode}")
            return None

    def _execute_simulated(
        self,
        symbol: str,
        signal: str,
        price: float,
        quantity: int
    ) -> Dict[str, Any]:
        """
        Executa ordem em modo simulação (instantâneo, com slippage e fees).

        Args:
            symbol: Símbolo do ativo.
            signal: Tipo de ordem.
            price: Preço de referência.
            quantity: Quantidade.

        Returns:
            Resultado da execução.
        """
        # Simula delay de execução
        if self.execution_delay > 0:
            time.sleep(self.execution_delay)

        # Aplica slippage (desfavorável ao trader)
        if signal == 'BUY':
            # Compra: preço aumenta
            executed_price = price * (1 + self.simulate_slippage)
        else:
            # Venda: preço diminui
            executed_price = price * (1 - self.simulate_slippage)

        # Calcula taxas
        trade_value = executed_price * quantity
        fees = trade_value * self.simulate_fees

        order_result = {
            'symbol': symbol,
            'action': signal,
            'requested_price': price,
            'executed_price': executed_price,
            'quantity': quantity,
            'trade_value': trade_value,
            'fees': fees,
            'total_cost': trade_value + fees if signal == 'BUY' else trade_value - fees,
            'slippage': executed_price - price if signal == 'BUY' else price - executed_price,
            'execution_time': datetime.now(),
            'status': 'FILLED',
            'mode': 'simulation'
        }

        self.orders.append(order_result)

        system_logger.info(
            f"✅ Ordem SIMULADA executada: {signal} {quantity} {symbol} @ R$ {executed_price:.2f} "
            f"(Slippage: R$ {order_result['slippage']:.4f}, Taxas: R$ {fees:.2f})"
        )

        return order_result

    def _execute_paper_trading(
        self,
        symbol: str,
        signal: str,
        price: float,
        quantity: int
    ) -> Dict[str, Any]:
        """
        Executa ordem em modo paper trading (sem dinheiro real, mas preços reais).

        Args:
            symbol: Símbolo do ativo.
            signal: Tipo de ordem.
            price: Preço de referência.
            quantity: Quantidade.

        Returns:
            Resultado da execução.
        """
        # Similar à simulação, mas poderia buscar preço real da API
        system_logger.info("Paper trading mode - executando como simulação")
        return self._execute_simulated(symbol, signal, price, quantity)

    def _execute_live(
        self,
        symbol: str,
        signal: str,
        price: float,
        quantity: int
    ) -> Optional[Dict[str, Any]]:
        """
        Executa ordem em modo LIVE (dinheiro real).
        ATENÇÃO: Requer integração com broker.

        Args:
            symbol: Símbolo do ativo.
            signal: Tipo de ordem.
            price: Preço de referência.
            quantity: Quantidade.

        Returns:
            Resultado da execução ou None.
        """
        system_logger.critical(
            "⚠️ MODO LIVE não implementado! "
            "Requer integração com broker (ex: XP, Clear, Rico, etc.)"
        )

        # TODO: Implementar integração com broker
        # Exemplo de fluxo:
        # 1. Conectar com API do broker
        # 2. Validar saldo disponível
        # 3. Enviar ordem de mercado/limite
        # 4. Aguardar confirmação
        # 5. Retornar resultado

        return None

    def get_order_history(self, symbol: Optional[str] = None) -> list:
        """
        Retorna histórico de ordens executadas.

        Args:
            symbol: Filtrar por símbolo (opcional).

        Returns:
            Lista de ordens executadas.
        """
        if symbol:
            return [o for o in self.orders if o['symbol'] == symbol]
        return self.orders.copy()

    def get_total_fees(self) -> float:
        """
        Calcula total de taxas pagas.

        Returns:
            Total de taxas.
        """
        return sum(order.get('fees', 0) for order in self.orders)

    def get_total_slippage(self) -> float:
        """
        Calcula total de slippage acumulado.

        Returns:
            Total de slippage.
        """
        return sum(abs(order.get('slippage', 0)) for order in self.orders)
