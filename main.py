"""
Modulo principal do Lobo IA - Sistema de Trading de Criptomoedas.
Versao 4.2 - Crypto Only (24/7)

Integra scanner de mercado crypto, analise de sinais, gestao de portfolio e execucao.

V4.1 - Integração com módulos avançados:
- ParallelFetcher: Coleta paralela de dados
- SmartCache: Cache hierárquico multi-nível
- SignalValidator: Validação multi-camada de sinais
- HealthMonitor: Monitoramento de saúde do sistema
- PortfolioOptimizer: Otimização de rotatividade de posições
"""

import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

from config_loader import config
from system_logger import system_logger
from logger import Logger

# V4.1: Importa módulos avançados
try:
    from parallel_fetcher import ParallelFetcher, parallel_fetcher
    from smart_cache import SmartCache, smart_cache
    from signal_validator import SignalValidator, signal_validator
    from health_monitor import HealthMonitor, health_monitor
    from portfolio_optimizer import PortfolioOptimizer, portfolio_optimizer
    HAS_V41_MODULES = True
    system_logger.info("V4.1 módulos carregados: ParallelFetcher, SmartCache, SignalValidator, HealthMonitor, PortfolioOptimizer")
except ImportError as e:
    HAS_V41_MODULES = False
    system_logger.warning(f"V4.1 módulos não disponíveis: {e}")

# Importa crypto scanner
try:
    from crypto_scanner import CryptoScanner
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    system_logger.warning("CryptoScanner não disponível")


class CryptoTrader:
    """
    Sistema principal de trading de criptomoedas do Lobo IA.
    Orquestra todo o fluxo: scanner -> coleta dados -> analisa sinais -> executa trades.
    Opera 24/7 sem restrições de horário.
    """

    def __init__(self):
        """Inicializa todos os componentes do sistema."""
        system_logger.info("=" * 60)
        system_logger.info("LOBO IA V4.2 - Sistema de Trading de Criptomoedas 24/7")
        system_logger.info("=" * 60)

        # Carrega configuracoes
        self.crypto_config = config.get_section('crypto')
        self.data_config = config.get_section('data')
        self.execution_config = config.get_section('execution')
        self.scanner_config = config.get('scanner', {})

        # Inicializa componentes
        self.db_logger = Logger()

        # Scanner de mercado crypto
        if not HAS_CRYPTO:
            raise RuntimeError("CryptoScanner é obrigatório para este sistema!")
        self.scanner = CryptoScanner()

        # Lista de criptomoedas a monitorar
        self.symbols = self.crypto_config.get('symbols', ['BTC-USD', 'ETH-USD'])
        self.dynamic_symbols = []

        # Capital e exposição
        self.capital = self.crypto_config.get('capital', 1000.0)
        self.exposure = self.crypto_config.get('exposure', 0.10)
        self.max_positions = self.crypto_config.get('max_positions', 8)

        # V4.1: Inicializa módulos avançados
        self.has_v41 = HAS_V41_MODULES
        if self.has_v41:
            # Usa instâncias globais dos módulos
            self.parallel_fetcher = parallel_fetcher
            self.smart_cache = smart_cache
            self.signal_validator = signal_validator
            self.health_monitor = health_monitor
            self.portfolio_optimizer = portfolio_optimizer

            # Configura fetcher paralelo
            self.parallel_fetcher.period = self.data_config.get('period', '30d')
            self.parallel_fetcher.interval = self.data_config.get('interval', '1h')

            system_logger.info("📦 Módulos V4.1 ATIVOS:")
            system_logger.info("   - ParallelFetcher (5 workers, 10s timeout)")
            system_logger.info("   - SmartCache (memória + disco)")
            system_logger.info("   - SignalValidator (5 camadas)")
            system_logger.info("   - HealthMonitor (latência + taxa de sucesso)")
            system_logger.info("   - PortfolioOptimizer (rotatividade)")

        system_logger.info(f"Scanner de mercado crypto: ATIVO")
        system_logger.info(f"Capital inicial: ${self.capital:.2f} USD")
        system_logger.info(f"Exposição: {self.exposure*100:.0f}% por trade")
        system_logger.info(f"Máximo de posições: {self.max_positions}")

    def update_symbols_from_scanner(self) -> List[str]:
        """
        Atualiza lista de simbolos usando o scanner de mercado crypto.

        Returns:
            Lista de simbolos selecionados pelo scanner.
        """
        try:
            system_logger.info("\nEscaneando mercado crypto para melhores oportunidades...")

            # Obtem top simbolos do scanner
            top_n = self.scanner_config.get('top_coins', 20)
            opportunities = self.scanner.scan_crypto_market()

            # Filtra por sinais de compra
            buy_signals = [op for op in opportunities if 'BUY' in op.get('signal', '')]

            if buy_signals:
                self.dynamic_symbols = [op['symbol'] for op in buy_signals[:top_n]]
                system_logger.info(f"Scanner encontrou {len(buy_signals)} oportunidades de compra")
                system_logger.info(f"Top {len(self.dynamic_symbols)} selecionadas: {', '.join(self.dynamic_symbols[:5])}")
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
        if self.dynamic_symbols:
            return self.dynamic_symbols
        return self.symbols

    def run_iteration(self):
        """
        Executa uma iteracao completa do sistema.
        Escaneia mercado crypto e analisa melhores oportunidades.

        V4.1 Melhorias:
        - Integração com PortfolioOptimizer para rotatividade
        - Relatório de saúde do sistema via HealthMonitor
        """
        iteration_start = time.time()

        system_logger.info("\n" + "=" * 60)
        system_logger.info(f"Nova iteracao Crypto V4.2 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        system_logger.info("=" * 60)

        # Atualiza simbolos usando scanner
        opportunities = self.scanner.scan_crypto_market()

        if opportunities:
            # Filtra sinais de compra
            buy_signals = [op for op in opportunities if 'BUY' in op.get('signal', '')]
            sell_signals = [op for op in opportunities if 'SELL' in op.get('signal', '')]

            system_logger.info(f"\n📊 Resultados: {len(opportunities)} criptos analisadas")
            system_logger.info(f"   🟢 Sinais de COMPRA: {len(buy_signals)}")
            system_logger.info(f"   🔴 Sinais de VENDA: {len(sell_signals)}")

            if buy_signals:
                system_logger.info("\n🔥 TOP OPORTUNIDADES DE COMPRA:")
                for i, crypto in enumerate(buy_signals[:5], 1):
                    system_logger.info(
                        f"   {i}. {crypto['symbol']}: Score {crypto.get('total_score', 0):.1f} | "
                        f"RSI {crypto.get('rsi', 0):.1f} | ${crypto.get('price', 0):.2f} | {crypto['signal']}"
                    )

        # Mostra resumo
        self._print_summary()

        # V4.1: Relatório de saúde
        iteration_time = time.time() - iteration_start
        if self.has_v41:
            health = self.health_monitor.check_system_health()
            system_logger.info(
                f"\n📈 Health: Status={health['status'].upper()}, "
                f"Latência média={health['data_sources']['avg_latency_ms']:.0f}ms, "
                f"Sucesso={health['data_sources']['success_rate']:.1f}%"
            )

        system_logger.info(f"\n✅ Iteração concluída em {iteration_time:.1f}s")

    def _print_summary(self):
        """Imprime resumo do sistema."""
        system_logger.info("\n" + "-" * 60)
        system_logger.info("RESUMO DO SISTEMA CRYPTO")
        system_logger.info("-" * 60)
        system_logger.info(f"Capital: ${self.capital:.2f} USD")
        system_logger.info(f"Modo: 24/7 Crypto Trading")
        system_logger.info(f"Exposição: {self.exposure*100:.0f}% por trade")
        system_logger.info("-" * 60)


def main():
    """Funcao principal - ponto de entrada do sistema."""
    try:
        # Cria instancia do trader
        trader = CryptoTrader()

        # Executa uma iteracao
        trader.run_iteration()

    except KeyboardInterrupt:
        system_logger.info("\n\nSistema interrompido pelo usuario")

    except Exception as e:
        system_logger.critical(f"Erro fatal no sistema: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
