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
from logger import Logger

# Importa crypto scanner (opcional)
try:
    from crypto_scanner import CryptoScanner
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    system_logger.warning("Crypto scanner nao disponivel")

# Importa analisador de noticias (opcional)
try:
    from news_analyzer import NewsAnalyzer
    HAS_NEWS = True
except ImportError:
    HAS_NEWS = False
    system_logger.warning("News analyzer nao disponivel")


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

        # Crypto trading 24/7
        self.crypto_enabled = HAS_CRYPTO and config.get('crypto.enabled', True)
        self.crypto_scanner = CryptoScanner() if self.crypto_enabled else None
        self.crypto_interval = config.get('crypto.check_interval', 300)  # 5 minutos
        self.last_crypto_run = None

        # Analisador de noticias
        self.news_enabled = HAS_NEWS
        self.news_analyzer = NewsAnalyzer() if self.news_enabled else None

        # Gerenciamento de posições crypto
        self.crypto_positions = {}  # {symbol: {quantity, entry_price, entry_time}}
        self.crypto_capital = config.get('crypto.capital_usd', 1000.0)
        self.crypto_exposure = config.get('crypto.exposure', 0.05)  # 5% por trade
        self.crypto_stop_loss = config.get('risk.stop_loss', 0.02)
        self.crypto_take_profit = config.get('risk.take_profit', 0.05)
        self.db_logger = Logger()

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
        if self.crypto_enabled:
            system_logger.info("🪙 Crypto trading 24/7 ATIVADO")
        if self.news_enabled:
            system_logger.info("📰 Analise de noticias ATIVADA")

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
        Crypto opera 24/7.
        """
        system_logger.info("🚀 Iniciando loop principal...")

        try:
            while self.running:
                # Verifica se mercado B3 está aberto
                if self.scheduler.is_market_open():
                    self._execute_iteration()
                else:
                    # B3 fechada - executa crypto se habilitado
                    if self.crypto_enabled:
                        self._execute_crypto_iteration()
                    else:
                        self._wait_for_market()

                # Aguarda intervalo antes da próxima verificação
                if self.running:
                    interval = self.scheduler.check_interval
                    system_logger.debug(f"Aguardando {interval}s até próxima verificação...")
                    time.sleep(interval)

        except Exception as e:
            system_logger.critical(
                f"Erro fatal no loop principal: {str(e)}",
                exc_info=True
            )
            raise

        finally:
            self._shutdown()

    def _execute_iteration(self):
        """Executa uma iteração do sistema de trading B3."""
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

    def _execute_crypto_iteration(self):
        """Executa uma iteração de análise e trading de criptomoedas (24/7)."""
        try:
            now = datetime.now()

            # Verifica intervalo minimo entre execuções
            if self.last_crypto_run:
                elapsed = (now - self.last_crypto_run).total_seconds()
                if elapsed < self.crypto_interval:
                    system_logger.debug(f"Crypto: aguardando {self.crypto_interval - elapsed:.0f}s")
                    return

            system_logger.info("\n" + "=" * 60)
            system_logger.info("🪙 CRYPTO SCANNER - Mercado 24/7")
            system_logger.info("=" * 60)

            # Executa scan de criptomoedas
            results = self.crypto_scanner.scan_crypto_market()

            if results:
                # Cria mapa de preços atuais
                price_map = {r['symbol']: r.get('price', 0) for r in results}

                # 1. Verifica posições abertas (stop-loss / take-profit)
                self._check_crypto_positions(price_map)

                # Mostra top oportunidades
                buy_signals = [r for r in results if 'BUY' in r.get('signal', '')]
                sell_signals = [r for r in results if 'SELL' in r.get('signal', '')]

                system_logger.info(f"\n📊 Resultados: {len(results)} criptos analisadas")
                system_logger.info(f"   🟢 Sinais de COMPRA: {len(buy_signals)}")
                system_logger.info(f"   🔴 Sinais de VENDA: {len(sell_signals)}")
                system_logger.info(f"   💼 Posições abertas: {len(self.crypto_positions)}")
                system_logger.info(f"   💰 Capital disponível: ${self.crypto_capital:.2f}")
                system_logger.info(f"   📰 Analise de noticias: {'ATIVADA' if self.news_enabled else 'DESATIVADA'}")

                if buy_signals:
                    system_logger.info("\n🔥 TOP OPORTUNIDADES DE COMPRA:")
                    for i, crypto in enumerate(buy_signals[:5], 1):
                        system_logger.info(
                            f"   {i}. {crypto['symbol']}: Score {crypto['total_score']:.1f} | "
                            f"RSI {crypto.get('rsi', 0):.1f} | ${crypto.get('price', 0):.2f} | {crypto['signal']}"
                        )

                # 2. Executa trades baseado em sinais
                self._execute_crypto_trades(buy_signals, sell_signals, price_map)

            self.last_crypto_run = now

        except Exception as e:
            system_logger.error(f"Erro no crypto scanner: {str(e)}", exc_info=True)

    def _check_crypto_positions(self, price_map: dict):
        """Verifica stop-loss e take-profit das posições crypto abertas."""
        positions_to_close = []

        for symbol, position in self.crypto_positions.items():
            current_price = price_map.get(symbol, 0)
            if current_price <= 0:
                continue

            entry_price = position['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price

            # Stop-loss
            if pnl_pct <= -self.crypto_stop_loss:
                positions_to_close.append((symbol, 'STOP_LOSS', current_price, pnl_pct))
            # Take-profit
            elif pnl_pct >= self.crypto_take_profit:
                positions_to_close.append((symbol, 'TAKE_PROFIT', current_price, pnl_pct))

        # Fecha posições
        for symbol, reason, price, pnl_pct in positions_to_close:
            self._close_crypto_position(symbol, price, reason)

    def _analyze_news_for_crypto(self, symbol: str, signal_data: dict) -> dict:
        """
        Analisa noticias para um ativo e ajusta score.

        Args:
            symbol: Simbolo do ativo
            signal_data: Dados do sinal técnico

        Returns:
            Dicionário com dados do sinal (possivelmente ajustado)
        """
        if not self.news_enabled or not self.news_analyzer:
            return signal_data

        try:
            # Busca e analisa notícias
            news_result = self.news_analyzer.get_news_sentiment(symbol, max_news=5)

            if news_result['news_count'] == 0:
                return signal_data

            # Ajusta score baseado em notícias
            news_score = news_result['overall_score']
            original_score = signal_data.get('total_score', 50)

            # Notícias podem ajustar score em até 15%
            adjustment = news_score * 15  # -15% a +15%
            new_score = original_score + adjustment
            new_score = max(0, min(100, new_score))

            # Atualiza dados do sinal
            signal_data['total_score'] = new_score
            signal_data['news_sentiment'] = news_result['overall_sentiment']
            signal_data['news_score'] = news_score
            signal_data['news_count'] = news_result['news_count']

            if abs(adjustment) > 5:
                emoji = "📈" if adjustment > 0 else "📉"
                system_logger.info(
                    f"   {emoji} Noticias {symbol}: {news_result['overall_sentiment']} "
                    f"(score ajustado: {original_score:.1f} -> {new_score:.1f})"
                )

            return signal_data

        except Exception as e:
            system_logger.debug(f"Erro ao analisar noticias para {symbol}: {e}")
            return signal_data

    def _execute_crypto_trades(self, buy_signals: list, sell_signals: list, price_map: dict):
        """Executa trades de crypto baseado em sinais e noticias."""
        mode = config.get('execution.mode', 'simulation')

        # Limita número de posições abertas
        max_positions = 3
        if len(self.crypto_positions) >= max_positions:
            system_logger.info(f"\n⚠️ Máximo de {max_positions} posições abertas")
            return

        # Analisa noticias para os sinais de compra
        if self.news_enabled:
            system_logger.info("\n📰 Analisando noticias...")
            for crypto in buy_signals[:3]:
                self._analyze_news_for_crypto(crypto['symbol'], crypto)

            # Reordena por score atualizado (considerando noticias)
            buy_signals.sort(key=lambda x: x.get('total_score', 0), reverse=True)

        # Executa compras (top 1 sinal mais forte que não temos posição)
        for crypto in buy_signals[:1]:
            symbol = crypto['symbol']

            # Já tem posição?
            if symbol in self.crypto_positions:
                continue

            price = crypto.get('price', 0)
            if price <= 0:
                continue

            # Verifica se notícias são muito negativas (bloqueia compra)
            news_sentiment = crypto.get('news_sentiment', 'neutral')
            news_score = crypto.get('news_score', 0)

            if news_sentiment == 'negative' and news_score < -0.4:
                system_logger.info(f"\n⚠️ Compra de {symbol} bloqueada por noticias negativas")
                continue

            # Calcula quantidade baseado na exposição
            trade_value = self.crypto_capital * self.crypto_exposure
            quantity = trade_value / price

            if trade_value > self.crypto_capital:
                system_logger.info(f"\n⚠️ Capital insuficiente para {symbol}")
                continue

            # Executa compra
            self._open_crypto_position(symbol, quantity, price, crypto)

    def _open_crypto_position(self, symbol: str, quantity: float, price: float, signal_data: dict):
        """Abre uma posição de crypto."""
        now = datetime.now()

        # Converte numpy types para Python nativos
        price = float(price)
        quantity = float(quantity)
        trade_value = quantity * price

        # Deduz do capital
        self.crypto_capital -= trade_value

        # Registra posição
        self.crypto_positions[symbol] = {
            'quantity': quantity,
            'entry_price': price,
            'entry_time': now,
            'trade_value': trade_value,
            'stop_loss': price * (1 - self.crypto_stop_loss),
            'take_profit': price * (1 + self.crypto_take_profit),
        }

        system_logger.info(f"\n🟢 COMPRA EXECUTADA: {symbol}")
        system_logger.info(f"   Quantidade: {quantity:.6f}")
        system_logger.info(f"   Preço: ${price:.2f}")
        system_logger.info(f"   Valor: ${trade_value:.2f}")
        system_logger.info(f"   Stop-Loss: ${price * (1 - self.crypto_stop_loss):.2f}")
        system_logger.info(f"   Take-Profit: ${price * (1 + self.crypto_take_profit):.2f}")

        # Loga no banco de dados
        rsi = float(signal_data.get('rsi', 0))
        score = float(signal_data.get('total_score', 0))
        news_sentiment = signal_data.get('news_sentiment', 'N/A')
        news_score = signal_data.get('news_score', 0)

        self.db_logger.log_trade({
            'symbol': symbol,
            'date': now,
            'action': 'BUY',
            'price': price,
            'quantity': quantity,
            'profit': 0.0,
            'indicators': f"RSI:{rsi:.1f}, Score:{score:.1f}, News:{news_sentiment}({news_score:+.2f})",
            'notes': f"Crypto BUY - {signal_data.get('signal', 'N/A')}"
        })

    def _close_crypto_position(self, symbol: str, current_price: float, reason: str):
        """Fecha uma posição de crypto."""
        if symbol not in self.crypto_positions:
            return

        position = self.crypto_positions[symbol]
        now = datetime.now()

        # Converte numpy types para Python nativos
        current_price = float(current_price)
        entry_price = float(position['entry_price'])
        quantity = float(position['quantity'])
        entry_value = float(position['trade_value'])
        exit_value = quantity * current_price
        profit = float(exit_value - entry_value)
        pnl_pct = float((current_price - entry_price) / entry_price * 100)

        # Retorna capital + lucro/prejuízo
        self.crypto_capital += exit_value

        # Remove posição
        del self.crypto_positions[symbol]

        emoji = "🟢" if profit >= 0 else "🔴"
        system_logger.info(f"\n{emoji} VENDA EXECUTADA: {symbol} ({reason})")
        system_logger.info(f"   Quantidade: {quantity:.6f}")
        system_logger.info(f"   Preço entrada: ${entry_price:.2f}")
        system_logger.info(f"   Preço saída: ${current_price:.2f}")
        system_logger.info(f"   Lucro/Prejuízo: ${profit:.2f} ({pnl_pct:+.2f}%)")

        # Loga no banco de dados
        self.db_logger.log_trade({
            'symbol': symbol,
            'date': now,
            'action': 'SELL',
            'price': current_price,
            'quantity': quantity,
            'profit': profit,
            'indicators': f"PnL:{pnl_pct:.2f}%",
            'notes': f"Crypto SELL - {reason} - Lucro: ${profit:.2f}"
        })

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
