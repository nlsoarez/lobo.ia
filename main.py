"""
Módulo principal do Lobo IA - Orquestração de trading.
Integra coleta de dados, análise de sinais, gestão de portfólio e execução.
"""

from datetime import datetime
from typing import Optional, Dict, Any
import json

from config_loader import config
from system_logger import system_logger
from logger import Logger
from data_collector import DataCollector
from signal_analyzer import SignalAnalyzer
from portfolio_manager import PortfolioManager
from trade_executor import TradeExecutor


class LoboTrader:
    """
    Sistema principal de trading do Lobo IA.
    Orquestra todo o fluxo: coleta dados -> analisa sinais -> executa trades.
    """

    def __init__(self):
        """Inicializa todos os componentes do sistema."""
        system_logger.info("=" * 60)
        system_logger.info("🐺 LOBO IA - Sistema de Trading Autônomo")
        system_logger.info("=" * 60)

        # Carrega configurações
        self.trading_config = config.get_section('trading')
        self.data_config = config.get_section('data')
        self.execution_config = config.get_section('execution')

        # Inicializa componentes
        self.db_logger = Logger()
        self.portfolio = PortfolioManager()
        self.executor = TradeExecutor()

        # Lista de símbolos a monitorar
        self.symbols = self.trading_config.get('symbols', ['PETR4.SA'])

        system_logger.info(f"Símbolos monitorados: {', '.join(self.symbols)}")
        system_logger.info(f"Capital inicial: R$ {self.portfolio.initial_capital:.2f}")

    def analisar_e_executar(self, symbol: str) -> bool:
        """
        Analisa um símbolo e executa trade se houver sinal.

        Args:
            symbol: Símbolo a analisar.

        Returns:
            True se um trade foi executado.
        """
        try:
            system_logger.info(f"\n📊 Analisando {symbol}...")

            # 1. Verifica se já tem posição aberta
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
            system_logger.debug(f"Dados coletados: {len(data)} candles")

            # 3. Analisa sinais técnicos
            analyzer = SignalAnalyzer(data, symbol=symbol)
            signal = analyzer.generate_signal()

            if signal is None:
                system_logger.debug(f"Sem sinal de trade para {symbol}")
                return False

            # 4. Calcula tamanho da posição
            quantity = self.portfolio.calculate_position_size(
                symbol=symbol,
                price=signal['price']
            )

            if quantity <= 0:
                system_logger.warning(f"Quantidade inválida calculada: {quantity}")
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
            f"\n🎯 EXECUTANDO TRADE: {action} {quantity} {symbol} @ R$ {price:.2f}"
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

            # Atualiza portfólio
            if action == "BUY":
                success = self.portfolio.open_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=price
                )

                if not success:
                    system_logger.error("Falha ao abrir posição no portfólio")
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

            system_logger.info(f"✅ Trade executado e registrado com sucesso!")
            return True

        except Exception as e:
            system_logger.error(f"Erro ao executar trade: {str(e)}", exc_info=True)
            return False

    def _check_exit_conditions(self, symbol: str):
        """
        Verifica condições de saída para posições abertas.

        Args:
            symbol: Símbolo da posição.
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

            # Pega preço atual
            current_price = float(data['close'].iloc[-1])

            # Verifica stop-loss / take-profit
            reason = self.portfolio.check_stop_loss_take_profit(symbol, current_price)

            if reason:
                # Fecha posição
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
                f"Erro ao verificar condições de saída para {symbol}: {str(e)}",
                exc_info=True
            )

    def run_iteration(self):
        """
        Executa uma iteração completa do sistema.
        Analisa todos os símbolos configurados.
        """
        system_logger.info("\n" + "=" * 60)
        system_logger.info(f"🔄 Nova iteração - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        system_logger.info("=" * 60)

        # Verifica drawdown
        if self.portfolio.is_drawdown_exceeded():
            system_logger.critical("🛑 SISTEMA PAUSADO: Drawdown máximo excedido!")
            return

        # Analisa cada símbolo
        trades_executed = 0
        for symbol in self.symbols:
            if self.analisar_e_executar(symbol):
                trades_executed += 1

        # Mostra resumo
        self._print_summary()

        system_logger.info(f"\n✅ Iteração concluída. Trades executados: {trades_executed}")

    def _print_summary(self):
        """Imprime resumo do portfólio."""
        stats = self.portfolio.get_performance_stats()
        positions = self.portfolio.get_all_positions()

        system_logger.info("\n" + "-" * 60)
        system_logger.info("📈 RESUMO DO PORTFÓLIO")
        system_logger.info("-" * 60)
        system_logger.info(f"Capital atual: R$ {stats['current_capital']:.2f}")
        system_logger.info(f"Capital disponível: R$ {stats['available_capital']:.2f}")
        system_logger.info(f"Lucro total: R$ {stats['total_profit']:.2f}")
        system_logger.info(f"Total de trades: {stats['total_trades']}")
        system_logger.info(f"Win rate: {stats['win_rate']:.1f}%")
        system_logger.info(f"Posições abertas: {stats['open_positions']}")

        if positions:
            system_logger.info("\n📊 Posições abertas:")
            for symbol, pos in positions.items():
                system_logger.info(
                    f"  - {symbol}: {pos['quantity']} @ R$ {pos['avg_price']:.2f} "
                    f"(SL: {pos['stop_loss']:.2f}, TP: {pos['take_profit']:.2f})"
                )

        system_logger.info("-" * 60)


def main():
    """Função principal - ponto de entrada do sistema."""
    try:
        # Cria instância do trader
        trader = LoboTrader()

        # Executa uma iteração
        trader.run_iteration()

    except KeyboardInterrupt:
        system_logger.info("\n\n⚠️ Sistema interrompido pelo usuário")

    except Exception as e:
        system_logger.critical(f"Erro fatal no sistema: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
