"""
Inicializador do Lobo IA - Executa o sistema em loop contínuo.
Verifica horário de mercado e executa análises periodicamente.
Otimizado para Railway e ambientes cloud.
"""

import os
import time
import signal
import sys
from datetime import datetime, time as dtime
from typing import Optional

from config_loader import config
from system_logger import system_logger
from main import LoboTrader
from health_server import start_health_server
from b3_calendar import is_holiday, is_weekend, is_trading_day, get_next_trading_day


class MarketScheduler:
    """
    Gerencia agendamento de execução baseado em horários de mercado.
    Verifica se mercado está aberto considerando dias úteis, feriados e horários da B3.
    """

    def __init__(self):
        """Inicializa o scheduler com configurações de mercado."""
        market_config = config.get_section('market')

        self.open_hour = market_config.get('open_hour', 10)
        self.close_hour = market_config.get('close_hour', 18)
        self.trading_days = market_config.get('trading_days', [0, 1, 2, 3, 4])  # Seg-Sex
        self.check_interval = market_config.get('check_interval', 60)  # segundos
        self.check_holidays = market_config.get('check_holidays', True)

        system_logger.info(
            f"Scheduler configurado: {self.open_hour}h-{self.close_hour}h, "
            f"Dias: {self.trading_days}, Verificar feriados: {self.check_holidays}"
        )

    def is_market_open(self) -> bool:
        """
        Verifica se o mercado B3 está aberto no momento.

        Returns:
            True se mercado está aberto.
        """
        now = datetime.now()

        # Verifica fim de semana
        if is_weekend(now):
            return False

        # Verifica feriados da B3
        if self.check_holidays and is_holiday(now):
            return False

        # Verifica dia da semana (0=segunda, 6=domingo)
        if now.weekday() not in self.trading_days:
            return False

        # Verifica horário
        current_hour = now.hour

        # Mercado abre às open_hour e fecha às close_hour
        if not (self.open_hour <= current_hour < self.close_hour):
            return False

        return True

    def get_market_status(self) -> dict:
        """
        Retorna status detalhado do mercado.

        Returns:
            Dicionário com informações do mercado.
        """
        now = datetime.now()
        return {
            'is_open': self.is_market_open(),
            'is_weekend': is_weekend(now),
            'is_holiday': is_holiday(now),
            'is_trading_day': is_trading_day(now),
            'current_time': now.strftime('%H:%M:%S'),
            'current_date': now.strftime('%Y-%m-%d'),
            'next_trading_day': get_next_trading_day(now).strftime('%Y-%m-%d'),
        }

    def time_until_market_open(self) -> Optional[int]:
        """
        Calcula tempo em segundos até abertura do mercado.

        Returns:
            Segundos até abertura ou None se mercado já está aberto.
        """
        if self.is_market_open():
            return None

        now = datetime.now()
        from datetime import timedelta

        # Encontra proximo dia de pregao
        next_trading = get_next_trading_day(now)

        # Calcula proximo horario de abertura
        next_open = datetime.combine(next_trading, dtime(self.open_hour, 0, 0))

        # Se hoje e dia de pregao mas ainda nao abriu
        if is_trading_day(now) and now.hour < self.open_hour:
            next_open = now.replace(hour=self.open_hour, minute=0, second=0, microsecond=0)

        seconds_until = int((next_open - now).total_seconds())
        return max(0, seconds_until)


class LoboSystem:
    """
    Sistema principal que gerencia execução contínua do Lobo IA.
    Otimizado para Railway com health check integrado.
    """

    def __init__(self):
        """Inicializa o sistema."""
        self.scheduler = MarketScheduler()
        self.trader: Optional[LoboTrader] = None
        self.running = True
        self.health_server = None

        # Configura handlers para sinais de sistema
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Inicia health server para Railway
        if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('ENABLE_HEALTH_SERVER'):
            try:
                self.health_server = start_health_server()
                system_logger.info(f"Health server iniciado na porta {self.health_server.port}")
            except Exception as e:
                system_logger.warning(f"Não foi possível iniciar health server: {e}")

        system_logger.info("Sistema Lobo IA iniciado")

    def _signal_handler(self, signum, frame):
        """
        Handler para sinais de interrupção (Ctrl+C, kill, etc).

        Args:
            signum: Número do sinal.
            frame: Frame da pilha.
        """
        system_logger.info(f"\n⚠️ Sinal recebido: {signum}. Encerrando sistema...")
        self.running = False

    def run(self):
        """
        Loop principal do sistema.
        Executa análises quando mercado está aberto.
        """
        system_logger.info("🚀 Iniciando loop principal...")

        try:
            while self.running:
                # Verifica se mercado está aberto
                if self.scheduler.is_market_open():
                    self._execute_iteration()
                else:
                    self._wait_for_market()

                # Aguarda intervalo antes da próxima verificação
                if self.running:
                    system_logger.debug(
                        f"Aguardando {self.scheduler.check_interval}s até próxima verificação..."
                    )
                    time.sleep(self.scheduler.check_interval)

        except Exception as e:
            system_logger.critical(
                f"Erro fatal no loop principal: {str(e)}",
                exc_info=True
            )
            raise

        finally:
            self._shutdown()

    def _execute_iteration(self):
        """Executa uma iteração do sistema de trading."""
        try:
            # Cria instância do trader se não existir
            if self.trader is None:
                self.trader = LoboTrader()

            # Executa análise e trades
            self.trader.run_iteration()

        except Exception as e:
            system_logger.error(
                f"Erro durante execução: {str(e)}",
                exc_info=True
            )

    def _wait_for_market(self):
        """Aguarda abertura do mercado."""
        status = self.scheduler.get_market_status()
        seconds_until = self.scheduler.time_until_market_open()

        if seconds_until and seconds_until > 0:
            hours = seconds_until // 3600
            minutes = (seconds_until % 3600) // 60

            reason = "Mercado fechado"
            if status['is_weekend']:
                reason = "Fim de semana"
            elif status['is_holiday']:
                reason = "Feriado B3"

            system_logger.info(
                f"{reason}. Proximo pregao: {status['next_trading_day']} "
                f"(em {hours}h {minutes}min)"
            )

    def _shutdown(self):
        """Encerra sistema graciosamente."""
        system_logger.info("\n" + "=" * 60)
        system_logger.info("Encerrando sistema Lobo IA")

        # Para health server se estiver rodando
        if self.health_server:
            try:
                self.health_server.stop()
                system_logger.info("Health server encerrado")
            except Exception as e:
                system_logger.warning(f"Erro ao encerrar health server: {e}")

        # Mostra estatísticas finais se trader foi inicializado
        if self.trader:
            try:
                stats = self.trader.portfolio.get_performance_stats()
                system_logger.info("\nESTATISTICAS FINAIS:")
                system_logger.info(f"  Capital final: R$ {stats['current_capital']:.2f}")
                system_logger.info(f"  Lucro/Prejuizo: R$ {stats['total_profit']:.2f}")
                system_logger.info(f"  Total de trades: {stats['total_trades']}")
                system_logger.info(f"  Win rate: {stats['win_rate']:.1f}%")
            except Exception as e:
                system_logger.error(f"Erro ao obter estatísticas finais: {e}")

        system_logger.info("=" * 60)
        system_logger.info("Sistema encerrado com sucesso")


def main():
    """Função principal - ponto de entrada."""
    try:
        # Banner
        print("\n" + "=" * 60)
        print("🐺  LOBO IA - Sistema de Trading Autônomo")
        print("    Trading inteligente para B3")
        print("=" * 60 + "\n")

        # Cria e inicia sistema
        system = LoboSystem()
        system.run()

    except KeyboardInterrupt:
        system_logger.info("\n⚠️ Interrompido pelo usuário")
        sys.exit(0)

    except Exception as e:
        system_logger.critical(f"Erro fatal: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
