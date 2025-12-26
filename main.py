"""
Modulo principal do Lobo IA - Orquestracao de trading.
Integra scanner de mercado, analise de sinais, gestao de portfolio e execucao.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import json

from config_loader import config
from system_logger import system_logger
from logger import Logger
from data_collector import DataCollector
from signal_analyzer import SignalAnalyzer
from portfolio_manager import PortfolioManager
from trade_executor import TradeExecutor
from market_scanner import MarketScanner


class LoboTrader:
    """
    Sistema principal de trading do Lobo IA.
    Orquestra todo o fluxo: scanner -> coleta dados -> analisa sinais -> executa trades.
    """

    def __init__(self):
        """Inicializa todos os componentes do sistema."""
        system_logger.info("=" * 60)
        system_logger.info("LOBO IA - Sistema de Trading Autonomo")
        system_logger.info("=" * 60)

        # Carrega configuracoes
        self.trading_config = config.get_section('trading')
        self.data_config = config.get_section('data')
        self.execution_config = config.get_section('execution')
        self.scanner_config = config.get('scanner', {})

        # Inicializa componentes
        self.db_logger = Logger()
        self.portfolio = PortfolioManager()
        self.executor = TradeExecutor()

        # Scanner de mercado (opcional)
        self.use_scanner = self.scanner_config.get('enabled', True)
        self.scanner = MarketScanner() if self.use_scanner else None

        # Lista de simbolos a monitorar
        self.symbols = self.trading_config.get('symbols', ['PETR4.SA'])
        self.dynamic_symbols = []

        system_logger.info(f"Scanner de mercado: {'ATIVO' if self.use_scanner else 'DESATIVADO'}")
        system_logger.info(f"Capital inicial: R$ {self.portfolio.initial_capital:.2f}")

    def update_symbols_from_scanner(self) -> List[str]:
        """
        Atualiza lista de simbolos usando o scanner de mercado.

        Returns:
            Lista de simbolos selecionados pelo scanner.
        """
        if not self.scanner:
            return self.symbols

        try:
            system_logger.info("\nEscaneando mercado B3 para melhores oportunidades...")

            # Obtem top simbolos do scanner
            top_n = self.scanner_config.get('top_stocks', 15)
            opportunities = self.scanner.scan_market()

            # Filtra por sinais de compra
            buy_signals = [op for op in opportunities if 'BUY' in op.get('signal', '')]

            if buy_signals:
                self.dynamic_symbols = [op['symbol'] for op in buy_signals[:top_n]]
                system_logger.info(f"Scanner encontrou {len(buy_signals)} oportunidades de compra")
                system_logger.info(f"Top {len(self.dynamic_symbols)} selecionadas: {', '.join(self.dynamic_symbols)}")
            else:
                # Se nao houver sinais de compra, usa os com melhor score
                self.dynamic_symbols = [op['symbol'] for op in opportunities[:top_n]]
                system_logger.info(f"Nenhum sinal de compra forte. Usando top {len(self.dynamic_symbols)} por score")

            return self.dynamic_symbols

        except Exception as e:
            system_logger.error(f"Erro no scanner: {e}")
            return self.symbols

    def get_active_symbols(self) -> List[str]:
        """
        Retorna lista de simbolos ativos para monitoramento.

        Returns:
            Lista de simbolos.
        """
        if self.use_scanner and self.dynamic_symbols:
            return self.dynamic_symbols
        return self.symbols

    def analisar_e_executar(self, symbol: str, scanner_data: Dict[str, Any] = None) -> bool:
        """
        Analisa um simbolo e executa trade se houver sinal.

        Args:
            symbol: Simbolo a analisar.
            scanner_data: Dados pre-analisados do scanner (opcional).

        Returns:
            True se um trade foi executado.
        """
        try:
            system_logger.info(f"\nAnalisando {symbol}...")

            # 1. Verifica se ja tem posicao aberta
            if self.portfolio.has_position(symbol):
                self._check_exit_conditions(symbol)
                return False

            # 2. Coleta dados de mercado
            collector = DataCollector(
                symbol=symbol,
                period=self.data_config.get('period', '5d'),
                interval=self.data_config.get('interval', '5m')
            )

            data = collector.get_data(use_cache=True)

            if data.empty:
                system_logger.debug(f"Sem dados para {symbol}")
                return False

            system_logger.debug(f"Dados coletados: {len(data)} candles")

            # 3. Analisa sinais tecnicos
            analyzer = SignalAnalyzer(data, symbol=symbol)
            signal = analyzer.generate_signal()

            # Se temos dados do scanner, usa para validar
            if scanner_data and signal:
                scanner_signal = scanner_data.get('signal', '')
                if 'BUY' in scanner_signal and signal['action'] == 'BUY':
                    signal['strength'] = min(1.0, signal.get('strength', 0.5) + 0.2)
                    system_logger.info(f"Sinal validado pelo scanner (forca aumentada)")

            if signal is None:
                system_logger.debug(f"Sem sinal de trade para {symbol}")
                return False

            # 4. Calcula tamanho da posicao
            quantity = self.portfolio.calculate_position_size(
                symbol=symbol,
                price=signal['price']
            )

            if quantity <= 0:
                system_logger.warning(f"Quantidade invalida calculada: {quantity}")
                return False

            # Adiciona quantidade ao sinal
            signal['quantity'] = quantity

            # 5. Executa trade
            success = self._executar_trade(signal)

            return success

        except Exception as e:
            system_logger.error(
                f"Erro ao analisar {symbol}: {str(e)}",
                exc_info=True
            )
            return False

    def _executar_trade(self, signal: Dict[str, Any]) -> bool:
        """
        Executa um trade baseado no sinal.

        Args:
            signal: Sinal de trading.

        Returns:
            True se trade foi executado com sucesso.
        """
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        quantity = signal['quantity']

        system_logger.info(
            f"\nEXECUTANDO TRADE: {action} {quantity} {symbol} @ R$ {price:.2f}"
        )

        try:
            # Executa ordem (simulada ou real)
            order_result = self.executor.execute_order(
                symbol=symbol,
                signal=action,
                price=price,
                quantity=quantity
            )

            if not order_result:
                system_logger.error("Falha ao executar ordem")
                return False

            # Atualiza portfolio
            if action == "BUY":
                success = self.portfolio.open_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=price
                )

                if not success:
                    system_logger.error("Falha ao abrir posicao no portfolio")
                    return False

            # Loga no banco de dados
            indicators_str = json.dumps(signal.get('indicators', {}), ensure_ascii=False)

            self.db_logger.log_trade({
                'symbol': symbol,
                'date': datetime.now(),
                'action': action,
                'price': price,
                'quantity': quantity,
                'profit': 0,
                'indicators': indicators_str,
                'notes': f'Trade executado: {action} {quantity} @ {price:.2f}'
            })

            system_logger.info(f"Trade executado e registrado com sucesso!")
            return True

        except Exception as e:
            system_logger.error(f"Erro ao executar trade: {str(e)}", exc_info=True)
            return False

    def _check_exit_conditions(self, symbol: str):
        """
        Verifica condicoes de saida para posicoes abertas.

        Args:
            symbol: Simbolo da posicao.
        """
        try:
            # Coleta dados atualizados
            collector = DataCollector(
                symbol=symbol,
                period=self.data_config.get('period', '5d'),
                interval=self.data_config.get('interval', '5m')
            )

            data = collector.get_data(use_cache=True)

            if data.empty:
                return

            # Pega preco atual
            current_price = float(data['close'].iloc[-1])

            # Verifica stop-loss / take-profit
            reason = self.portfolio.check_stop_loss_take_profit(symbol, current_price)

            if reason:
                # Fecha posicao
                result = self.portfolio.close_position(
                    symbol=symbol,
                    price=current_price,
                    reason=reason
                )

                if result:
                    # Executa ordem de venda
                    position = result
                    self.executor.execute_order(
                        symbol=symbol,
                        signal="SELL",
                        price=current_price,
                        quantity=position['quantity']
                    )

                    # Loga no banco
                    self.db_logger.log_trade({
                        'symbol': symbol,
                        'date': datetime.now(),
                        'action': 'SELL',
                        'price': current_price,
                        'quantity': position['quantity'],
                        'profit': position['profit'],
                        'indicators': f"reason:{reason}",
                        'notes': f"Fechado por {reason}: Lucro R$ {position['profit']:.2f}"
                    })

        except Exception as e:
            system_logger.error(
                f"Erro ao verificar condicoes de saida para {symbol}: {str(e)}",
                exc_info=True
            )

    def run_iteration(self):
        """
        Executa uma iteracao completa do sistema.
        Escaneia mercado e analisa melhores oportunidades.
        """
        system_logger.info("\n" + "=" * 60)
        system_logger.info(f"Nova iteracao - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        system_logger.info("=" * 60)

        # Verifica drawdown
        if self.portfolio.is_drawdown_exceeded():
            system_logger.critical("SISTEMA PAUSADO: Drawdown maximo excedido!")
            return

        # Atualiza simbolos usando scanner (se habilitado)
        scanner_results = {}
        if self.use_scanner:
            opportunities = self.scanner.scan_market()
            self.dynamic_symbols = [op['symbol'] for op in opportunities if 'BUY' in op.get('signal', '')][:15]

            # Cria mapa de resultados do scanner para referencia
            scanner_results = {op['symbol']: op for op in opportunities}

            if self.dynamic_symbols:
                system_logger.info(f"\nScanner selecionou {len(self.dynamic_symbols)} acoes com sinal de compra:")
                for sym in self.dynamic_symbols[:5]:
                    op = scanner_results.get(sym, {})
                    system_logger.info(
                        f"  - {sym}: Score {op.get('total_score', 0):.1f}, "
                        f"RSI {op.get('rsi', 0):.1f}, Sinal: {op.get('signal', 'N/A')}"
                    )
                if len(self.dynamic_symbols) > 5:
                    system_logger.info(f"  ... e mais {len(self.dynamic_symbols) - 5} acoes")

        # Obtem simbolos ativos
        active_symbols = self.get_active_symbols()

        if not active_symbols:
            system_logger.info("Nenhum simbolo ativo para analise")
            self._print_summary()
            return

        # Analisa cada simbolo
        trades_executed = 0
        for symbol in active_symbols:
            scanner_data = scanner_results.get(symbol)
            if self.analisar_e_executar(symbol, scanner_data):
                trades_executed += 1

        # Mostra resumo
        self._print_summary()

        system_logger.info(f"\nIteracao concluida. Trades executados: {trades_executed}")

    def _print_summary(self):
        """Imprime resumo do portfolio."""
        stats = self.portfolio.get_performance_stats()
        positions = self.portfolio.get_all_positions()

        system_logger.info("\n" + "-" * 60)
        system_logger.info("RESUMO DO PORTFOLIO")
        system_logger.info("-" * 60)
        system_logger.info(f"Capital atual: R$ {stats['current_capital']:.2f}")
        system_logger.info(f"Capital disponivel: R$ {stats['available_capital']:.2f}")
        system_logger.info(f"Lucro total: R$ {stats['total_profit']:.2f}")
        system_logger.info(f"Total de trades: {stats['total_trades']}")
        system_logger.info(f"Win rate: {stats['win_rate']:.1f}%")
        system_logger.info(f"Posicoes abertas: {stats['open_positions']}")

        if positions:
            system_logger.info("\nPosicoes abertas:")
            for symbol, pos in positions.items():
                system_logger.info(
                    f"  - {symbol}: {pos['quantity']} @ R$ {pos['avg_price']:.2f} "
                    f"(SL: {pos['stop_loss']:.2f}, TP: {pos['take_profit']:.2f})"
                )

        system_logger.info("-" * 60)

    def print_market_report(self, top_n: int = 20):
        """
        Imprime relatorio completo do mercado.

        Args:
            top_n: Numero de acoes a mostrar.
        """
        if not self.scanner:
            system_logger.warning("Scanner nao esta habilitado")
            return

        self.scanner.print_report(top_n)


def main():
    """Funcao principal - ponto de entrada do sistema."""
    try:
        # Cria instancia do trader
        trader = LoboTrader()

        # Imprime relatorio do mercado (opcional)
        if config.get('scanner.print_report', False):
            trader.print_market_report()

        # Executa uma iteracao
        trader.run_iteration()

    except KeyboardInterrupt:
        system_logger.info("\n\nSistema interrompido pelo usuario")

    except Exception as e:
        system_logger.critical(f"Erro fatal no sistema: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
