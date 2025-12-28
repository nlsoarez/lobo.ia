"""
Inicializador do Lobo IA - Executa o sistema em loop contínuo.
Verifica horário de mercado e executa análises periodicamente.
Otimizado para Railway e ambientes cloud.
"""

import os
import time
import signal
import sys
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

# Configura timezone do Brasil (UTC-3)
try:
    import pytz
    BRAZIL_TZ = pytz.timezone('America/Sao_Paulo')
except ImportError:
    # Fallback se pytz nao estiver disponivel
    BRAZIL_TZ = timezone(timedelta(hours=-3))


def get_brazil_time() -> datetime:
    """Retorna datetime no horario de Brasilia."""
    try:
        return datetime.now(BRAZIL_TZ)
    except:
        return datetime.now(timezone(timedelta(hours=-3)))

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
        self.crypto_positions = {}  # {symbol: {quantity, entry_price, entry_time, max_price}}
        self.crypto_capital = config.get('crypto.capital_usd', 1000.0)
        self.crypto_initial_capital = self.crypto_capital  # Para calcular drawdown
        self.crypto_exposure = config.get('crypto.exposure', 0.05)  # 5% por trade
        self.crypto_stop_loss = config.get('risk.stop_loss', 0.02)
        self.crypto_take_profit = config.get('risk.take_profit', 0.05)
        self.db_logger = Logger()

        # =====================================================
        # SISTEMA DE TRADING V3.0 - Filtros Inteligentes
        # =====================================================

        # Trailing stop config
        self.trailing_stop_activation = config.get('risk.trailing_stop_activation', 0.03)
        self.trailing_stop_distance = config.get('risk.trailing_stop_distance', 0.015)

        # Max drawdown enforcement
        self.max_drawdown = config.get('risk.max_drawdown', 0.10)
        self.trading_paused = False

        # Cooldown ADAPTATIVO (win=2h, loss=4h)
        self.cooldown_after_win = 2 * 60 * 60  # 2 horas
        self.cooldown_after_loss = 4 * 60 * 60  # 4 horas
        self.last_trade_time = {}  # {symbol: datetime}
        self.last_trade_result = {}  # {symbol: 'win' ou 'loss'}

        # Max positions
        self.max_positions = 2

        # =====================================================
        # V3.1: SISTEMA DE TRADING INTELIGENTE
        # =====================================================

        # V3.1: SISTEMA DE PONTUAÇÃO DE FILTROS (RELAXADO)
        # Total: 100 pontos, mínimo para entrada: 50 pontos (era 60)
        self.filter_threshold = 50  # Reduzido para permitir mais entradas
        self.filter_weights = {
            'macro_trend': 30,      # EMA50 > EMA200
            'volume': 25,           # Volume ratio
            'news': 20,             # Sentimento de notícias
            'momentum': 15,         # MACD > 0
            'volatility': 10,       # ATR adequado (1-3%)
        }

        # V3.1: NÍVEIS DE SINAL com parâmetros RELAXADOS
        self.signal_levels = {
            'STRONG': {
                'min_score': 60,            # Era 65
                'min_filter_points': 55,    # Era 70 - GRANDE REDUÇÃO
                'max_rsi': 45,              # Era 40
                'exposure': 0.05,           # 5%
                'take_profit': 0.05,        # 5%
                'stop_loss': 0.02,          # 2%
                'trailing_activation': 0.03,
            },
            'MODERATE': {
                'min_score': 50,            # Era 55
                'min_filter_points': 45,    # Era 60 - GRANDE REDUÇÃO
                'max_rsi': 50,              # Era 48
                'exposure': 0.035,          # 3.5%
                'take_profit': 0.04,        # 4%
                'stop_loss': 0.015,         # 1.5%
                'trailing_activation': 0.025,
            },
            'RECOVERY': {
                'min_score': 45,            # Era 50
                'min_filter_points': 35,    # Era 45 - GRANDE REDUÇÃO
                'max_rsi': 40,              # Era 35
                'exposure': 0.02,           # 2%
                'take_profit': 0.03,        # 3%
                'stop_loss': 0.01,          # 1%
                'trailing_activation': None,
            },
        }

        # V3.1: DETECÇÃO DE REGIME DE MERCADO
        self.current_regime = 'LATERAL'  # BULL, LATERAL, BEAR
        self.regime_adx_threshold = 25   # ADX > 25 = tendência forte

        # V3.1: POSITION TIMEOUT - Fecha posições estagnadas
        self.position_timeout_hours = 6  # Fecha após 6 horas sem movimento
        self.stale_position_min_pnl = -0.01  # Só fecha se PnL > -1%

        # V3.1: POSITION ROTATION - Permite rotação para sinais fortes
        self.enable_rotation = True
        self.rotation_min_score = 65  # Score mínimo para rotação
        self.rotation_min_pnl = -0.005  # PnL mínimo da posição a fechar (-0.5%)

        # V3.1: STRONG_BUY OVERRIDE - Relaxa filtros para sinais muito fortes
        self.strong_buy_override = {
            'enabled': True,
            'min_score': 65,
            'max_rsi': 38,
            'volume_override': 1.0,  # Aceita volume 1.0x (era 1.5x)
            'ignore_trend': True,    # Ignora trend filter se RSI < 32
        }

        # V3.1: TRACKING DE ÚLTIMA ENTRADA
        self.last_entry_time = None
        self.hours_without_entry = 0

        # Estatísticas de rejeição (para logging)
        self.rejection_stats = {
            'total_analyzed': 0,
            'total_rejected': 0,
            'reasons': {}
        }

        # RESET desativado - sistema normal
        FORCE_RESET = False  # Reset já executado

        if FORCE_RESET or os.environ.get('RESET_POSITIONS', '').lower() == 'true':
            system_logger.warning("🔄 RESET FORÇADO - Limpando todas as posicoes e trades...")
            if self.db_logger.clear_all_positions():
                system_logger.info("✅ Posicoes e trades de crypto limpos com sucesso!")
                system_logger.info("✅ Capital resetado para $1000.00")
            else:
                system_logger.error("❌ Erro ao limpar posicoes")
        else:
            # Carrega posicoes abertas do banco (sobrevive a restarts)
            self._load_positions_from_db()

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

    def _load_positions_from_db(self):
        """Carrega posicoes abertas do banco de dados (sobrevive a restarts)."""
        try:
            loaded = self.db_logger.load_positions()

            # Se nao encontrou na tabela crypto_positions, tenta recuperar dos trades
            if not loaded:
                system_logger.info("📂 Tabela crypto_positions vazia, tentando recuperar dos trades...")
                loaded = self.db_logger.recover_open_positions_from_trades()

                # Se encontrou posicoes, salva na tabela crypto_positions para proximo restart
                if loaded:
                    system_logger.info(f"📂 Recuperadas {len(loaded)} posicoes dos trades historicos")
                    for symbol, pos in loaded.items():
                        self.db_logger.save_position(symbol, pos)

            if loaded:
                self.crypto_positions = loaded
                # Recalcula capital disponivel
                total_em_posicoes = sum(p['trade_value'] for p in loaded.values())
                self.crypto_capital = self.crypto_initial_capital - total_em_posicoes

                system_logger.info(f"📂 {len(loaded)} posicoes abertas carregadas:")
                for symbol, pos in loaded.items():
                    system_logger.info(
                        f"   - {symbol}: {pos['quantity']:.6f} @ ${pos['entry_price']:.2f} "
                        f"(valor: ${pos['trade_value']:.2f})"
                    )
                system_logger.info(f"   Capital disponivel: ${self.crypto_capital:.2f}")
            else:
                system_logger.info("📂 Nenhuma posicao aberta encontrada")

        except Exception as e:
            system_logger.warning(f"Erro ao carregar posicoes: {e}")

    def _save_position_to_db(self, symbol: str):
        """Salva uma posicao no banco de dados."""
        if symbol in self.crypto_positions:
            try:
                self.db_logger.save_position(symbol, self.crypto_positions[symbol])
            except Exception as e:
                system_logger.warning(f"Erro ao salvar posicao {symbol}: {e}")

    def _delete_position_from_db(self, symbol: str):
        """Remove uma posicao do banco de dados."""
        try:
            self.db_logger.delete_position(symbol)
        except Exception as e:
            system_logger.warning(f"Erro ao deletar posicao {symbol}: {e}")

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

    def _check_max_drawdown(self):
        """
        Verifica se o drawdown maximo foi excedido.
        Pausa trading se perdeu mais de 10% do capital inicial.
        """
        if self.crypto_initial_capital <= 0:
            return

        # Calcula capital atual (disponivel + em posicoes)
        total_capital = self.crypto_capital
        for symbol, position in self.crypto_positions.items():
            total_capital += position.get('trade_value', 0)

        # Calcula drawdown
        drawdown = (self.crypto_initial_capital - total_capital) / self.crypto_initial_capital

        if drawdown >= self.max_drawdown:
            if not self.trading_paused:
                system_logger.warning(
                    f"⚠️ MAX DRAWDOWN ATINGIDO! Drawdown: {drawdown*100:.1f}% "
                    f"(limite: {self.max_drawdown*100:.1f}%)"
                )
                system_logger.warning("Trading PAUSADO ate drawdown reduzir")
                self.trading_paused = True
        else:
            if self.trading_paused and drawdown < (self.max_drawdown * 0.8):
                # Retoma trading se drawdown reduziu para 80% do limite
                system_logger.info(f"✅ Drawdown reduziu para {drawdown*100:.1f}%. Trading RETOMADO.")
                self.trading_paused = False

    def _log_positions_dashboard(self, price_map: dict):
        """
        V3.1: Dashboard de diagnóstico mostrando estado detalhado das posições.
        """
        if not self.crypto_positions:
            return

        now = get_brazil_time()
        system_logger.info("\n" + "=" * 60)
        system_logger.info("📊 V3.1 DASHBOARD DE POSIÇÕES")
        system_logger.info("=" * 60)

        # Calcula tempo desde última entrada
        if self.last_entry_time:
            hours_since_entry = (now - self.last_entry_time).total_seconds() / 3600
            self.hours_without_entry = hours_since_entry
            system_logger.info(f"⏰ Última entrada: {hours_since_entry:.1f}h atrás")
        else:
            system_logger.info("⏰ Última entrada: N/A")

        system_logger.info(f"💰 Capital disponível: ${self.crypto_capital:.2f}")
        system_logger.info(f"📈 Regime de mercado: {self.current_regime}")
        system_logger.info("")

        for i, (symbol, position) in enumerate(self.crypto_positions.items(), 1):
            current_price = price_map.get(symbol, 0)
            entry_price = position['entry_price']
            entry_time = position.get('entry_time', now)
            max_price = position.get('max_price', entry_price)

            # Calcula métricas
            if current_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price * 100
                pnl_usd = position['trade_value'] * (pnl_pct / 100)
            else:
                pnl_pct = 0
                pnl_usd = 0

            # Calcula tempo aberto
            try:
                if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo:
                    open_hours = (now - entry_time).total_seconds() / 3600
                else:
                    open_hours = (datetime.now() - entry_time).total_seconds() / 3600
            except:
                open_hours = 0

            # Verifica trailing status
            max_pnl_pct = (max_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            trailing_activation = position.get('trailing_activation')
            trailing_active = trailing_activation and (max_pnl_pct / 100) >= trailing_activation

            # Status e recomendação
            stop_loss_pct = position.get('stop_loss_pct', self.crypto_stop_loss) * 100
            take_profit_pct = position.get('take_profit_pct', self.crypto_take_profit) * 100

            if pnl_pct >= take_profit_pct * 0.8:
                status = "🎯 PRÓXIMO DO TP"
            elif pnl_pct <= -stop_loss_pct * 0.8:
                status = "⚠️ PRÓXIMO DO SL"
            elif open_hours > self.position_timeout_hours:
                status = "⏰ STALE (TIMEOUT)"
            elif trailing_active:
                status = "🔒 TRAILING ATIVO"
            else:
                status = "📊 MONITORANDO"

            emoji = "🟢" if pnl_pct >= 0 else "🔴"

            system_logger.info(f"{i}. {symbol}")
            system_logger.info(f"   {emoji} P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
            system_logger.info(f"   ⏱️  Aberta há: {open_hours:.1f}h")
            system_logger.info(f"   📈 Entrada: ${entry_price:.2f} | Atual: ${current_price:.2f}")
            system_logger.info(f"   🎯 TP: {take_profit_pct:.1f}% | SL: {stop_loss_pct:.1f}%")
            system_logger.info(f"   📊 Max P&L: {max_pnl_pct:.2f}% | Trailing: {'✅' if trailing_active else '❌'}")
            system_logger.info(f"   📋 Status: {status}")
            system_logger.info("")

        system_logger.info("=" * 60)

    def _check_stale_positions(self, price_map: dict) -> list:
        """
        V3.1: Detecta e fecha posições estagnadas (timeout).
        Retorna lista de posições fechadas.
        """
        now = get_brazil_time()
        closed_positions = []

        for symbol, position in list(self.crypto_positions.items()):
            current_price = price_map.get(symbol, 0)
            if current_price <= 0:
                continue

            entry_price = position['entry_price']
            entry_time = position.get('entry_time', now)
            pnl_pct = (current_price - entry_price) / entry_price

            # Calcula tempo aberto
            try:
                if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo:
                    open_hours = (now - entry_time).total_seconds() / 3600
                else:
                    open_hours = (datetime.now() - entry_time).total_seconds() / 3600
            except:
                open_hours = 0

            # Verifica timeout
            if open_hours >= self.position_timeout_hours:
                # Só fecha se P&L > limite mínimo
                if pnl_pct >= self.stale_position_min_pnl:
                    system_logger.warning(
                        f"⏰ TIMEOUT: {symbol} aberta há {open_hours:.1f}h "
                        f"com P&L {pnl_pct*100:+.2f}% - Fechando..."
                    )
                    self._close_crypto_position(symbol, current_price, f'TIMEOUT_{open_hours:.0f}H')
                    closed_positions.append(symbol)
                else:
                    system_logger.info(
                        f"⏰ {symbol}: Timeout mas P&L {pnl_pct*100:.2f}% < {self.stale_position_min_pnl*100:.1f}% "
                        f"- Mantendo posição"
                    )

        return closed_positions

    def _check_crypto_positions(self, price_map: dict):
        """
        V3.1: Verifica stop-loss, take-profit, TRAILING STOP e TIMEOUT.
        Inclui dashboard diagnóstico detalhado.
        """
        # V3.1: Log dashboard de posições PRIMEIRO
        self._log_positions_dashboard(price_map)

        # V3.1: Verifica posições estagnadas (timeout)
        self._check_stale_positions(price_map)

        positions_to_close = []

        for symbol, position in self.crypto_positions.items():
            current_price = price_map.get(symbol, 0)
            if current_price <= 0:
                continue

            entry_price = position['entry_price']
            max_price = position.get('max_price', entry_price)
            pnl_pct = (current_price - entry_price) / entry_price

            # V3.1: Parâmetros específicos da posição
            stop_loss_pct = position.get('stop_loss_pct', self.crypto_stop_loss)
            take_profit_pct = position.get('take_profit_pct', self.crypto_take_profit)
            trailing_activation = position.get('trailing_activation', self.trailing_stop_activation)
            signal_level = position.get('signal_level', 'N/A')

            # Atualiza preço máximo atingido (para trailing stop)
            if current_price > max_price:
                self.crypto_positions[symbol]['max_price'] = current_price
                max_price = current_price
                # Persiste atualização no banco
                self._save_position_to_db(symbol)

            # Calcula lucro máximo atingido
            max_pnl_pct = (max_price - entry_price) / entry_price

            # V3.1: TRAILING STOP - Usa activation threshold do nível de sinal
            if trailing_activation and max_pnl_pct >= trailing_activation:
                # Trailing stop: fecha se preço cair X% do máximo
                trailing_stop_price = max_price * (1 - self.trailing_stop_distance)

                if current_price <= trailing_stop_price:
                    positions_to_close.append((symbol, f'TRAILING_STOP ({signal_level})', current_price, pnl_pct))
                    system_logger.info(
                        f"   🎯 Trailing Stop V3.1 ativado em {symbol} [{signal_level}]: "
                        f"Max ${max_price:.2f} -> Stop ${trailing_stop_price:.2f}"
                    )
                    continue

            # V3.1: Stop-loss baseado no nível
            if pnl_pct <= -stop_loss_pct:
                positions_to_close.append((symbol, f'STOP_LOSS ({signal_level})', current_price, pnl_pct))
            # V3.1: Take-profit baseado no nível
            elif pnl_pct >= take_profit_pct:
                positions_to_close.append((symbol, f'TAKE_PROFIT ({signal_level})', current_price, pnl_pct))

        # Fecha posições
        for symbol, reason, price, pnl_pct in positions_to_close:
            self._close_crypto_position(symbol, price, reason)

        # Verifica max drawdown após processar posições
        self._check_max_drawdown()

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

    # =========================================================
    # V3.0: MÉTODOS DE ANÁLISE INTELIGENTE
    # =========================================================

    def _detect_market_regime(self, signal_data: dict) -> str:
        """
        V3.0: Detecta regime de mercado baseado em EMAs e força da tendência.

        Returns:
            'BULL', 'LATERAL' ou 'BEAR'
        """
        ema_12 = signal_data.get('ema_12', 0)
        ema_26 = signal_data.get('ema_26', 0)
        ema_50 = signal_data.get('ema_50', 0)
        ema_200 = signal_data.get('ema_200', 0)
        atr_pct = signal_data.get('atr_pct', 2)

        # Se não tiver EMA200, usa lógica simplificada
        if ema_200 == 0:
            if ema_12 > ema_26 > ema_50:
                return 'BULL'
            elif ema_12 < ema_26 < ema_50:
                return 'BEAR'
            return 'LATERAL'

        # Lógica completa com EMA200
        # BULL: EMA12 > EMA26 > EMA50 > EMA200
        if ema_12 > ema_26 > ema_50 and ema_50 > ema_200:
            return 'BULL'

        # BEAR: EMA12 < EMA26 < EMA50 < EMA200
        if ema_12 < ema_26 < ema_50 and ema_50 < ema_200:
            return 'BEAR'

        # LATERAL: EMAs cruzadas ou próximas
        return 'LATERAL'

    def _calculate_filter_score(self, signal_data: dict) -> dict:
        """
        V3.0: Calcula pontuação dos filtros (0-100 pontos).
        Substitui a lógica AND por sistema de pontos.

        Returns:
            Dict com pontuação total e breakdown por filtro
        """
        points = 0
        breakdown = {}

        # 1. MACRO TREND (30 pontos)
        ema_50 = signal_data.get('ema_50', 0)
        ema_200 = signal_data.get('ema_200', 0)
        if ema_200 > 0 and ema_50 > ema_200:
            breakdown['macro_trend'] = 30
            points += 30
        elif ema_200 == 0:
            # Sem EMA200, dá pontuação parcial se tendência curta positiva
            ema_12 = signal_data.get('ema_12', 0)
            ema_26 = signal_data.get('ema_26', 0)
            if ema_12 > ema_26:
                breakdown['macro_trend'] = 15
                points += 15
            else:
                breakdown['macro_trend'] = 0
        else:
            breakdown['macro_trend'] = 0

        # 2. VOLUME (25 pontos) - Escalonado
        vol_ratio = signal_data.get('volume_ratio', 0)
        if vol_ratio >= 1.5:
            breakdown['volume'] = 25
            points += 25
        elif vol_ratio >= 1.2:
            breakdown['volume'] = 20
            points += 20
        elif vol_ratio >= 1.0:
            breakdown['volume'] = 15
            points += 15
        elif vol_ratio >= 0.8:
            breakdown['volume'] = 10
            points += 10
        else:
            breakdown['volume'] = 0

        # 3. NOTÍCIAS (20 pontos)
        news_score = signal_data.get('news_score', 0)
        news_sentiment = signal_data.get('news_sentiment', 'neutral')
        if news_score > 0 or news_sentiment == 'positive':
            breakdown['news'] = 20
            points += 20
        elif news_score >= -0.2:
            breakdown['news'] = 15
            points += 15
        elif news_score >= -0.4:
            breakdown['news'] = 10
            points += 10
        else:
            breakdown['news'] = 0

        # 4. MOMENTUM (15 pontos) - MACD
        macd = signal_data.get('macd', 0)
        macd_hist = signal_data.get('macd_hist', 0)
        if macd > 0 and macd_hist > 0:
            breakdown['momentum'] = 15
            points += 15
        elif macd_hist > 0:
            breakdown['momentum'] = 10
            points += 10
        elif macd > signal_data.get('macd_signal', 0):
            breakdown['momentum'] = 5
            points += 5
        else:
            breakdown['momentum'] = 0

        # 5. VOLATILIDADE (10 pontos) - ATR entre 1-3%
        atr_pct = signal_data.get('atr_pct', 0)
        if 1 <= atr_pct <= 3:
            breakdown['volatility'] = 10
            points += 10
        elif 0.5 <= atr_pct <= 5:
            breakdown['volatility'] = 5
            points += 5
        else:
            breakdown['volatility'] = 0

        return {
            'total_points': points,
            'breakdown': breakdown,
            'passed': points >= self.filter_threshold
        }

    def _determine_signal_level(self, signal_data: dict, filter_result: dict) -> dict:
        """
        V3.0: Determina o nível do sinal (STRONG, MODERATE, RECOVERY).

        Returns:
            Dict com nível, parâmetros e motivo se rejeitado
        """
        total_score = signal_data.get('total_score', 0)
        rsi = signal_data.get('rsi', 50)
        filter_points = filter_result['total_points']
        regime = self._detect_market_regime(signal_data)

        # Atualiza regime atual
        self.current_regime = regime

        # BEAR MARKET: Reduz exposição ou bloqueia
        if regime == 'BEAR':
            # Em bear market, só permite RECOVERY com RSI muito baixo
            if rsi < 30 and filter_points >= 40:
                level = self.signal_levels['RECOVERY'].copy()
                level['exposure'] *= 0.5  # Reduz exposição pela metade
                return {
                    'level': 'RECOVERY_BEAR',
                    'params': level,
                    'regime': regime,
                    'approved': True
                }
            return {
                'level': None,
                'reason': f'Bear market (regime={regime}), RSI={rsi:.1f} não sobrevendido',
                'regime': regime,
                'approved': False
            }

        # Tenta STRONG primeiro
        strong = self.signal_levels['STRONG']
        if (total_score >= strong['min_score'] and
            rsi < strong['max_rsi'] and
            filter_points >= strong['min_filter_points']):
            return {
                'level': 'STRONG',
                'params': strong,
                'regime': regime,
                'approved': True
            }

        # Tenta MODERATE
        moderate = self.signal_levels['MODERATE']
        if (total_score >= moderate['min_score'] and
            rsi < moderate['max_rsi'] and
            filter_points >= moderate['min_filter_points']):
            return {
                'level': 'MODERATE',
                'params': moderate,
                'regime': regime,
                'approved': True
            }

        # Tenta RECOVERY (para mercados laterais)
        recovery = self.signal_levels['RECOVERY']
        if (regime == 'LATERAL' and
            total_score >= recovery['min_score'] and
            rsi < recovery['max_rsi'] and
            filter_points >= recovery['min_filter_points']):
            return {
                'level': 'RECOVERY',
                'params': recovery,
                'regime': regime,
                'approved': True
            }

        # Nenhum nível atendido
        reason_parts = []
        if total_score < moderate['min_score']:
            reason_parts.append(f"score={total_score:.1f}<{moderate['min_score']}")
        if rsi >= moderate['max_rsi']:
            reason_parts.append(f"RSI={rsi:.1f}>={moderate['max_rsi']}")
        if filter_points < moderate['min_filter_points']:
            reason_parts.append(f"filtros={filter_points}<{moderate['min_filter_points']}")

        return {
            'level': None,
            'reason': ', '.join(reason_parts) if reason_parts else 'Critérios não atendidos',
            'regime': regime,
            'approved': False
        }

    def _check_adaptive_cooldown(self, symbol: str) -> tuple:
        """
        V3.0: Cooldown adaptativo - 2h após win, 4h após loss.

        Returns:
            (pode_operar: bool, motivo: str)
        """
        if symbol not in self.last_trade_time:
            return True, None

        last_trade = self.last_trade_time[symbol]
        last_result = self.last_trade_result.get(symbol, 'loss')

        # Define cooldown baseado no resultado
        cooldown = self.cooldown_after_win if last_result == 'win' else self.cooldown_after_loss

        elapsed = (get_brazil_time().replace(tzinfo=None) - last_trade.replace(tzinfo=None)
                   if hasattr(last_trade, 'tzinfo') and last_trade.tzinfo
                   else (datetime.now() - last_trade)).total_seconds()

        if elapsed < cooldown:
            remaining = (cooldown - elapsed) / 3600
            return False, f"cooldown {remaining:.1f}h (após {'win' if last_result == 'win' else 'loss'})"

        return True, None

    def _log_rejection(self, symbol: str, reason: str, signal_data: dict, filter_result: dict):
        """
        V3.0: Logging detalhado de rejeições para análise.
        """
        self.rejection_stats['total_rejected'] += 1
        if reason not in self.rejection_stats['reasons']:
            self.rejection_stats['reasons'][reason] = 0
        self.rejection_stats['reasons'][reason] += 1

        # Log detalhado
        system_logger.info(
            f"   ❌ {symbol} | Score:{signal_data.get('total_score', 0):.1f} | "
            f"Filtros:{filter_result['total_points']}pts | "
            f"RSI:{signal_data.get('rsi', 0):.1f} | "
            f"Vol:{signal_data.get('volume_ratio', 0):.1f}x | "
            f"Regime:{self.current_regime} | "
            f"Motivo: {reason}"
        )

    def _check_strong_buy_override(self, crypto: dict, filter_result: dict) -> dict:
        """
        V3.1: Verifica se sinal qualifica para STRONG_BUY override.
        Relaxa filtros para sinais muito fortes.
        """
        if not self.strong_buy_override.get('enabled', False):
            return {'override': False}

        score = crypto.get('total_score', 0)
        rsi = crypto.get('rsi', 50)
        signal = crypto.get('signal', '')

        # Critérios para override
        min_score = self.strong_buy_override['min_score']
        max_rsi = self.strong_buy_override['max_rsi']

        if score >= min_score and rsi <= max_rsi and 'STRONG' in signal.upper():
            return {
                'override': True,
                'reason': f'STRONG_BUY_OVERRIDE (Score:{score:.1f}, RSI:{rsi:.1f})',
                'params': {
                    'exposure': 0.04,  # 4% - entre STRONG e MODERATE
                    'take_profit': 0.045,
                    'stop_loss': 0.018,
                    'trailing_activation': 0.028,
                }
            }

        return {'override': False}

    def _try_position_rotation(self, new_signal: dict, price_map: dict) -> bool:
        """
        V3.1: Tenta rotação de posição para abrir espaço para sinal forte.
        Fecha a pior posição se o novo sinal for significativamente melhor.
        """
        if not self.enable_rotation:
            return False

        new_score = new_signal.get('total_score', 0)
        if new_score < self.rotation_min_score:
            return False

        # Encontra a pior posição
        worst_symbol = None
        worst_pnl = float('inf')

        for symbol, position in self.crypto_positions.items():
            current_price = price_map.get(symbol, 0)
            if current_price <= 0:
                continue

            entry_price = position['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct < worst_pnl:
                worst_pnl = pnl_pct
                worst_symbol = symbol

        # Verifica se a pior posição pode ser fechada
        if worst_symbol and worst_pnl >= self.rotation_min_pnl:
            current_price = price_map.get(worst_symbol, 0)
            system_logger.warning(
                f"🔄 ROTAÇÃO: Fechando {worst_symbol} (P&L: {worst_pnl*100:+.2f}%) "
                f"para abrir {new_signal['symbol']} (Score: {new_score:.1f})"
            )
            self._close_crypto_position(worst_symbol, current_price, 'ROTATION')
            return True

        return False

    def _execute_crypto_trades(self, buy_signals: list, sell_signals: list, price_map: dict):
        """
        V3.1: Executa trades de crypto com SISTEMA INTELIGENTE:
        - Filter scoring (0-100 pontos, mínimo 50)
        - Signal levels (STRONG/MODERATE/RECOVERY) RELAXADOS
        - STRONG_BUY override para sinais muito fortes
        - Position rotation para abrir espaço
        - Position timeout para posições estagnadas
        - Logging diagnóstico completo
        """
        mode = config.get('execution.mode', 'simulation')

        # V3.1: Verifica se trading está pausado por drawdown
        if self.trading_paused:
            system_logger.warning("\n⚠️ Trading PAUSADO - Max Drawdown atingido")
            return

        # V3.1: NÃO RETORNA MAIS se posições cheias - continua analisando para logs
        positions_full = len(self.crypto_positions) >= self.max_positions

        if positions_full:
            system_logger.info(f"\n⚠️ {self.max_positions}/{self.max_positions} posições abertas - Analisando para rotação...")

        # Analisa notícias para os sinais de compra
        if self.news_enabled:
            system_logger.info("\n📰 Analisando notícias...")
            for crypto in buy_signals[:5]:
                self._analyze_news_for_crypto(crypto['symbol'], crypto)

            # Reordena por score atualizado (considerando notícias)
            buy_signals.sort(key=lambda x: x.get('total_score', 0), reverse=True)

        # V3.1: Estatísticas de análise
        self.rejection_stats['total_analyzed'] += len(buy_signals)

        system_logger.info(f"\n🔍 V3.1 ANÁLISE DE SINAIS ({len(buy_signals)} candidatos)")
        system_logger.info(f"   Regime atual: {self.current_regime}")
        system_logger.info(f"   Posições: {len(self.crypto_positions)}/{self.max_positions}")
        system_logger.info(f"   Rotação habilitada: {'✅' if self.enable_rotation else '❌'}")

        # V3.1: Log header para diagnóstico
        system_logger.info("\n📋 ANÁLISE DETALHADA DE CADA SINAL:")
        system_logger.info("-" * 60)

        # V3.1: Avalia cada sinal com sistema de pontuação
        qualified_signals = []
        for crypto in buy_signals[:10]:  # Analisa top 10
            symbol = crypto['symbol']
            score = crypto.get('total_score', 0)
            rsi = crypto.get('rsi', 50)
            signal_type = crypto.get('signal', 'N/A')

            # V3.1: Calcula pontuação dos filtros SEMPRE (para log)
            filter_result = self._calculate_filter_score(crypto)
            breakdown = filter_result.get('breakdown', {})

            # V3.1: Log detalhado de cada sinal
            system_logger.info(f"\n   📌 {symbol} | {signal_type}")
            system_logger.info(f"      Score: {score:.1f} | RSI: {rsi:.1f} | Vol: {crypto.get('volume_ratio', 0):.2f}x")
            system_logger.info(f"      Filtros ({filter_result['total_points']}pts): "
                             f"Trend:{breakdown.get('macro_trend', 0)} "
                             f"Vol:{breakdown.get('volume', 0)} "
                             f"News:{breakdown.get('news', 0)} "
                             f"Mom:{breakdown.get('momentum', 0)} "
                             f"ATR:{breakdown.get('volatility', 0)}")

            # Já tem posição?
            if symbol in self.crypto_positions:
                system_logger.info(f"      ⏭️  SKIP: Já tem posição aberta")
                continue

            # V3.1: Cooldown ADAPTATIVO
            can_trade, cooldown_reason = self._check_adaptive_cooldown(symbol)
            if not can_trade:
                system_logger.info(f"      ❌ REJECT: {cooldown_reason}")
                self.rejection_stats['total_rejected'] += 1
                if 'cooldown' not in self.rejection_stats['reasons']:
                    self.rejection_stats['reasons']['cooldown'] = 0
                self.rejection_stats['reasons']['cooldown'] += 1
                continue

            # V3.1: Verifica STRONG_BUY override PRIMEIRO
            override = self._check_strong_buy_override(crypto, filter_result)
            if override['override']:
                system_logger.info(f"      🔥 STRONG_BUY OVERRIDE ATIVADO!")
                crypto['_signal_level'] = 'STRONG_OVERRIDE'
                crypto['_level_params'] = override['params']
                crypto['_filter_score'] = filter_result['total_points']
                crypto['_regime'] = self.current_regime
                qualified_signals.append(crypto)
                system_logger.info(f"      ✅ APROVADO: {override['reason']}")
                continue

            # V3.1: Determina nível do sinal (normal)
            signal_level = self._determine_signal_level(crypto, filter_result)

            if not signal_level['approved']:
                reason = signal_level.get('reason', 'Não qualificado')
                system_logger.info(f"      ❌ REJECT: {reason}")
                self.rejection_stats['total_rejected'] += 1
                reason_key = reason.split(',')[0] if ',' in reason else reason
                if reason_key not in self.rejection_stats['reasons']:
                    self.rejection_stats['reasons'][reason_key] = 0
                self.rejection_stats['reasons'][reason_key] += 1
                continue

            # V3.1: Sinal aprovado!
            crypto['_signal_level'] = signal_level['level']
            crypto['_level_params'] = signal_level['params']
            crypto['_filter_score'] = filter_result['total_points']
            crypto['_regime'] = signal_level['regime']

            qualified_signals.append(crypto)

            system_logger.info(f"      ✅ APROVADO: Nível {signal_level['level']}")

        system_logger.info("\n" + "-" * 60)

        # V3.1: Log de resumo
        if qualified_signals:
            system_logger.info(f"\n📊 RESUMO: {len(qualified_signals)} sinais qualificados de {len(buy_signals)}")
            for q in qualified_signals[:3]:
                system_logger.info(f"   🎯 {q['symbol']}: Score {q.get('total_score', 0):.1f} | "
                                 f"Nível: {q.get('_signal_level', 'N/A')}")
        else:
            system_logger.info(f"\n📊 RESUMO: Nenhum sinal qualificado de {len(buy_signals)} analisados")
            # Mostra estatísticas de rejeição
            if self.rejection_stats['reasons']:
                top_reasons = sorted(self.rejection_stats['reasons'].items(),
                                    key=lambda x: x[1], reverse=True)[:5]
                system_logger.info(f"   📉 Top motivos de rejeição:")
                for reason, count in top_reasons:
                    system_logger.info(f"      • {reason}: {count}x")

        # V3.1: Se posições cheias, tenta rotação
        if positions_full and qualified_signals:
            best_signal = qualified_signals[0]
            if self._try_position_rotation(best_signal, price_map):
                positions_full = False  # Agora tem espaço

        # V3.1: Se ainda cheio, não executa
        if positions_full:
            system_logger.info(f"\n⚠️ Posições cheias e rotação não disponível")
            return

        if not qualified_signals:
            return

        # V3.1: Executa compra para o melhor sinal qualificado
        for crypto in qualified_signals[:1]:
            symbol = crypto['symbol']
            price = crypto.get('price', 0)

            if price <= 0:
                continue

            # V3.1: Usa parâmetros do nível de sinal
            level_params = crypto.get('_level_params', self.signal_levels['MODERATE'])
            exposure = level_params['exposure']

            # Calcula quantidade baseado na exposição do nível
            trade_value = self.crypto_capital * exposure
            quantity = trade_value / price

            if trade_value > self.crypto_capital:
                system_logger.info(f"\n⚠️ Capital insuficiente para {symbol}")
                continue

            # V3.1: Executa compra com parâmetros do nível
            self._open_crypto_position(symbol, quantity, price, crypto)

    def _open_crypto_position(self, symbol: str, quantity: float, price: float, signal_data: dict):
        """
        V3.1: Abre uma posição de crypto com parâmetros do nível de sinal.
        Cada nível (STRONG/MODERATE/RECOVERY) tem TP/SL/Trailing diferentes.
        Rastreia última entrada para dashboard.
        """
        now = get_brazil_time()

        # V3.1: Atualiza tracking de última entrada
        self.last_entry_time = now
        self.hours_without_entry = 0

        # Converte numpy types para Python nativos
        price = float(price)
        quantity = float(quantity)
        trade_value = quantity * price

        # V3.0: Pega parâmetros do nível de sinal (ou usa defaults)
        level_params = signal_data.get('_level_params', self.signal_levels['MODERATE'])
        signal_level = signal_data.get('_signal_level', 'MODERATE')
        filter_score = signal_data.get('_filter_score', 0)
        regime = signal_data.get('_regime', 'LATERAL')

        # V3.0: Parâmetros de TP/SL baseados no nível
        stop_loss_pct = level_params.get('stop_loss', self.crypto_stop_loss)
        take_profit_pct = level_params.get('take_profit', self.crypto_take_profit)
        trailing_activation = level_params.get('trailing_activation', self.trailing_stop_activation)

        # Deduz do capital
        self.crypto_capital -= trade_value

        # Registra posição (inclui max_price para trailing stop)
        self.crypto_positions[symbol] = {
            'quantity': quantity,
            'entry_price': price,
            'entry_time': now,
            'trade_value': trade_value,
            'stop_loss': price * (1 - stop_loss_pct),
            'take_profit': price * (1 + take_profit_pct),
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'trailing_activation': trailing_activation,
            'max_price': price,  # Para trailing stop
            'signal_level': signal_level,
            'filter_score': filter_score,
            'regime': regime,
        }

        # Persiste posição no banco (sobrevive a restarts)
        self._save_position_to_db(symbol)

        # Registra cooldown (será atualizado com win/loss no close)
        self.last_trade_time[symbol] = now

        system_logger.info(f"\n🟢 COMPRA V3.0 EXECUTADA: {symbol}")
        system_logger.info(f"   Nível: {signal_level} | Filtros: {filter_score}pts | Regime: {regime}")
        system_logger.info(f"   Quantidade: {quantity:.6f}")
        system_logger.info(f"   Preço: ${price:.2f}")
        system_logger.info(f"   Valor: ${trade_value:.2f}")
        system_logger.info(f"   Stop-Loss: ${price * (1 - stop_loss_pct):.2f} ({stop_loss_pct*100:.1f}%)")
        system_logger.info(f"   Take-Profit: ${price * (1 + take_profit_pct):.2f} ({take_profit_pct*100:.1f}%)")
        if trailing_activation:
            system_logger.info(f"   Trailing Activation: {trailing_activation*100:.1f}%")

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
            'indicators': f"RSI:{rsi:.1f}, Score:{score:.1f}, Level:{signal_level}, Filtros:{filter_score}pts",
            'notes': f"V3.0 {signal_level} BUY - {regime} regime - News:{news_sentiment}({news_score:+.2f})"
        })

    def _close_crypto_position(self, symbol: str, current_price: float, reason: str):
        """
        V3.0: Fecha uma posição de crypto e registra resultado para cooldown adaptativo.
        Win = cooldown 2h, Loss = cooldown 4h.
        """
        if symbol not in self.crypto_positions:
            return

        position = self.crypto_positions[symbol]
        now = get_brazil_time()

        # Converte numpy types para Python nativos
        current_price = float(current_price)
        entry_price = float(position['entry_price'])
        quantity = float(position['quantity'])
        entry_value = float(position['trade_value'])
        exit_value = quantity * current_price
        profit = float(exit_value - entry_value)
        pnl_pct = float((current_price - entry_price) / entry_price * 100)

        # V3.0: Dados extras da posição
        signal_level = position.get('signal_level', 'N/A')
        filter_score = position.get('filter_score', 0)
        regime = position.get('regime', 'N/A')

        # Retorna capital + lucro/prejuízo
        self.crypto_capital += exit_value

        # Remove posição da memória
        del self.crypto_positions[symbol]

        # Remove posição do banco (persistência)
        self._delete_position_from_db(symbol)

        # V3.0: Atualiza cooldown adaptativo baseado no resultado
        is_win = profit >= 0
        self.last_trade_time[symbol] = now
        self.last_trade_result[symbol] = 'win' if is_win else 'loss'

        # V3.0: Calcula próximo cooldown
        next_cooldown = self.cooldown_after_win if is_win else self.cooldown_after_loss
        next_cooldown_hours = next_cooldown / 3600

        emoji = "🟢" if is_win else "🔴"
        system_logger.info(f"\n{emoji} VENDA V3.0 EXECUTADA: {symbol} ({reason})")
        system_logger.info(f"   Nível entrada: {signal_level} | Filtros: {filter_score}pts | Regime: {regime}")
        system_logger.info(f"   Quantidade: {quantity:.6f}")
        system_logger.info(f"   Preço entrada: ${entry_price:.2f}")
        system_logger.info(f"   Preço saída: ${current_price:.2f}")
        system_logger.info(f"   Lucro/Prejuízo: ${profit:.2f} ({pnl_pct:+.2f}%)")
        system_logger.info(f"   Próximo cooldown: {next_cooldown_hours:.1f}h ({'win' if is_win else 'loss'})")

        # Loga no banco de dados
        self.db_logger.log_trade({
            'symbol': symbol,
            'date': now,
            'action': 'SELL',
            'price': current_price,
            'quantity': quantity,
            'profit': profit,
            'indicators': f"PnL:{pnl_pct:.2f}%, Level:{signal_level}, Filtros:{filter_score}pts",
            'notes': f"V3.0 SELL ({reason}) - {'WIN' if is_win else 'LOSS'} - ${profit:.2f} - Cooldown:{next_cooldown_hours:.1f}h"
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
