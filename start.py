"""
Inicializador do Lobo IA - Sistema de Trading de Criptomoedas 24/7.
Executa análises e trading de forma contínua.
Otimizado para Railway e ambientes cloud.

Versão 4.2 - Crypto Only
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
from health_server import start_health_server
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

# V4.0 Phase 2: Importa módulos de seleção avançada
try:
    from asset_ranking_system import AssetRankingSystem
    from technical_pattern_scanner import TechnicalPatternScanner
    from volume_analyzer import VolumeAnalyzer
    from market_timing_manager import MarketTimingManager
    from breakout_detector import BreakoutDetector
    HAS_PHASE2 = True
    system_logger.info("V4.0 Phase 2 módulos carregados")
except ImportError as e:
    HAS_PHASE2 = False
    system_logger.warning(f"Phase 2 módulos não disponíveis: {e}")

# V4.0 Phase 3: Importa módulos de rotação agressiva
try:
    from aggressive_rotation_manager import AggressiveRotationManager
    from intraday_ranking_system import IntradayRankingSystem
    from dynamic_allocation_manager import DynamicAllocationManager
    from realtime_performance_monitor import RealtimePerformanceMonitor
    from rotation_decision_engine import RotationDecisionEngine
    HAS_PHASE3 = True
    system_logger.info("V4.0 Phase 3 módulos carregados")
except ImportError as e:
    HAS_PHASE3 = False
    system_logger.warning(f"Phase 3 módulos não disponíveis: {e}")

# V4.0 Phase 4: Importa módulos de auto-otimização
try:
    from auto_optimized_trading_system import AutoOptimizedTradingSystem
    from auto_optimization_engine import AutoOptimizationEngine
    from market_regime_detector import MarketRegimeDetector
    from reinforcement_learning_agent import TradingRLAgent
    from meta_learning_system import MetaLearningSystem
    from multi_objective_optimizer import MultiObjectiveOptimizer
    HAS_PHASE4 = True
    system_logger.info("V4.0 Phase 4 módulos carregados")
except ImportError as e:
    HAS_PHASE4 = False
    system_logger.warning(f"Phase 4 módulos não disponíveis: {e}")

# V4.1: Importa módulos avançados de robustez
try:
    from parallel_fetcher import ParallelFetcher, parallel_fetcher
    from smart_cache import SmartCache, smart_cache
    from signal_validator import SignalValidator, signal_validator
    from health_monitor import HealthMonitor, health_monitor
    from portfolio_optimizer import PortfolioOptimizer, portfolio_optimizer
    from data_collector import symbol_blacklist
    HAS_V41 = True
    system_logger.info("V4.1 módulos carregados: ParallelFetcher, SmartCache, SignalValidator, HealthMonitor, PortfolioOptimizer")
except ImportError as e:
    HAS_V41 = False
    system_logger.warning(f"V4.1 módulos não disponíveis: {e}")

# V4.2: Importa módulos de emergência e alertas
try:
    from emergency_trade_manager import EmergencyTradeManager, emergency_trade_manager
    from smart_health_metrics import SmartHealthMetrics, smart_health_metrics
    from emergency_signal_prioritizer import EmergencySignalPrioritizer, emergency_signal_prioritizer
    from smart_alert_system import SmartAlertSystem, smart_alert_system
    HAS_V42 = True
    system_logger.info("V4.2 módulos carregados: EmergencyTradeManager, SmartHealthMetrics, SignalPrioritizer, AlertSystem")
except ImportError as e:
    HAS_V42 = False
    system_logger.warning(f"V4.2 módulos não disponíveis: {e}")


class CryptoScheduler:
    """
    Scheduler simples para trading de criptomoedas 24/7.
    Opera continuamente sem restrições de horário.
    """

    def __init__(self):
        """Inicializa o scheduler para crypto 24/7."""
        self.check_interval = config.get('crypto.check_interval', 60)  # segundos
        system_logger.info("Scheduler configurado para Crypto 24/7")

    def is_market_open(self) -> bool:
        """Mercado crypto está sempre aberto."""
        return True

    def get_market_status(self) -> dict:
        """Retorna status do mercado crypto."""
        now = datetime.now()
        return {
            'is_open': True,
            'market': 'CRYPTO',
            'mode': '24/7',
            'current_time': now.strftime('%H:%M:%S'),
            'current_date': now.strftime('%Y-%m-%d'),
        }


class LoboSystem:
    """
    Sistema principal de trading de criptomoedas 24/7.
    Otimizado para Railway com health check integrado.
    """

    def __init__(self):
        """Inicializa o sistema."""
        self.scheduler = CryptoScheduler()
        self.running = True
        self.health_server = None

        # Crypto trading 24/7
        self.crypto_enabled = HAS_CRYPTO and config.get('crypto.enabled', True)
        if not self.crypto_enabled:
            raise RuntimeError("Crypto scanner é obrigatório para este sistema!")
        self.crypto_scanner = CryptoScanner()
        self.crypto_interval = config.get('crypto.check_interval', 300)  # 5 minutos
        self.last_crypto_run = None

        # Analisador de noticias
        self.news_enabled = HAS_NEWS
        self.news_analyzer = NewsAnalyzer() if self.news_enabled else None

        # V4.0 Phase 2: Sistema de seleção avançada
        self.phase2_enabled = HAS_PHASE2
        if self.phase2_enabled:
            self.ranking_system = AssetRankingSystem()
            self.pattern_scanner = TechnicalPatternScanner()
            self.volume_analyzer = VolumeAnalyzer()
            self.timing_manager = MarketTimingManager()
            self.breakout_detector = BreakoutDetector()
            system_logger.info("V4.0 Phase 2: Sistema de seleção avançada ATIVO")
        else:
            self.ranking_system = None
            self.pattern_scanner = None
            self.volume_analyzer = None
            self.timing_manager = None
            self.breakout_detector = None

        # Gerenciamento de posições crypto (inicializa ANTES das fases que usam)
        self.crypto_positions = {}  # {symbol: {quantity, entry_price, entry_time, max_price}}
        self.crypto_capital = config.get('crypto.capital_usd', 1000.0)
        self.crypto_initial_capital = self.crypto_capital  # Para calcular drawdown

        # V4.0 Phase 3: Sistema de rotação agressiva
        self.phase3_enabled = HAS_PHASE3
        if self.phase3_enabled:
            self.rotation_manager = AggressiveRotationManager()
            self.intraday_ranking = IntradayRankingSystem()
            self.allocation_manager = DynamicAllocationManager(total_capital=self.crypto_initial_capital)
            self.performance_monitor = RealtimePerformanceMonitor()
            self.decision_engine = RotationDecisionEngine()
            system_logger.info("V4.0 Phase 3: Sistema de rotação agressiva ATIVO")
        else:
            self.rotation_manager = None
            self.intraday_ranking = None
            self.allocation_manager = None
            self.performance_monitor = None
            self.decision_engine = None

        # V4.0 Phase 4: Sistema auto-otimizado
        self.phase4_enabled = HAS_PHASE4
        if self.phase4_enabled:
            self.auto_optimizer = AutoOptimizedTradingSystem(
                initial_capital=self.crypto_initial_capital
            )
            self.last_optimization_cycle = None
            self.optimization_interval_hours = 6
            system_logger.info("V4.0 Phase 4: Sistema auto-otimizado ATIVO")
        else:
            self.auto_optimizer = None
            self.last_optimization_cycle = None

        # V4.1: Sistema de robustez avançada
        self.v41_enabled = HAS_V41
        if self.v41_enabled:
            # Usa instâncias globais dos módulos
            self.parallel_fetcher_v41 = parallel_fetcher
            self.smart_cache_v41 = smart_cache
            self.signal_validator_v41 = signal_validator
            self.health_monitor_v41 = health_monitor
            self.portfolio_optimizer_v41 = portfolio_optimizer
            self.symbol_blacklist = symbol_blacklist

            # Log status inicial do blacklist
            blacklisted = symbol_blacklist.get_blacklisted_symbols()
            if blacklisted:
                system_logger.info(f"🚫 Blacklist V4.1: {len(blacklisted)} símbolos bloqueados")

            system_logger.info("V4.1: Sistema de robustez ATIVO (Blacklist, Cache, Health, Optimizer)")
        else:
            self.parallel_fetcher_v41 = None
            self.smart_cache_v41 = None
            self.signal_validator_v41 = None
            self.health_monitor_v41 = None
            self.portfolio_optimizer_v41 = None
            self.symbol_blacklist = None

        # V4.2: Sistema de emergência e alertas
        self.v42_enabled = HAS_V42
        if self.v42_enabled:
            # Usa instâncias globais dos módulos V4.2
            self.emergency_trade_manager = emergency_trade_manager
            self.smart_health_metrics = smart_health_metrics
            self.emergency_signal_prioritizer = emergency_signal_prioritizer
            self.smart_alert_system = smart_alert_system

            system_logger.info("V4.2: Sistema de emergência ATIVO (TradeManager, HealthMetrics, SignalPrioritizer, AlertSystem)")
        else:
            self.emergency_trade_manager = None
            self.smart_health_metrics = None
            self.emergency_signal_prioritizer = None
            self.smart_alert_system = None

        self.crypto_exposure = config.get('crypto.exposure', 0.05)  # 5% por trade
        self.crypto_stop_loss = config.get('risk.stop_loss', 0.02)
        self.crypto_take_profit = config.get('risk.take_profit', 0.05)
        self.db_logger = Logger()

        # =====================================================
        # V4.0: SISTEMA AGRESSIVO - META 5% DIÁRIO
        # =====================================================
        # AVISO: Trading de alto risco. Nunca arrisque capital
        # que não pode perder. 5% diário é meta agressiva.
        # =====================================================

        # V4.0: TRAILING STOP AGRESSIVO
        self.trailing_stop_activation = 0.01   # Ativa em +1% (era 3%)
        self.trailing_stop_distance = 0.005    # Distância 0.5% (era 1.5%)

        # V4.0: MAX DRAWDOWN com limites diários
        self.max_drawdown = 0.05               # 5% máximo (era 10%)
        self.trading_paused = False

        # V4.0: COOLDOWN REDUZIDO DRASTICAMENTE
        self.cooldown_after_win = 10 * 60      # 10 minutos (era 2h)
        self.cooldown_after_loss = 30 * 60     # 30 minutos (era 4h)
        self.last_trade_time = {}
        self.last_trade_result = {}

        # V4.0: MAX POSITIONS AUMENTADO
        self.max_positions = 5                 # Aumentado de 2 para 5

        # =====================================================
        # V4.0: PARÂMETROS FASE 1 - FUNDAMENTOS AGRESSIVOS
        # =====================================================

        # V4.0: SISTEMA DE PONTUAÇÃO RELAXADO
        self.filter_threshold = 25             # Reduzido de 45 para 25
        self.filter_weights = {
            'macro_trend': 40,     # Aumentado (era 30)
            'volume': 20,          # Reduzido (era 25)
            'news': 10,            # Reduzido (era 20)
            'momentum': 25,        # Aumentado (era 15)
            'volatility': 5,       # Reduzido (era 10)
        }

        # V4.0: NÍVEIS DE SINAL AGRESSIVOS
        self.signal_levels = {
            'STRONG': {
                'min_score': 40,            # Era 55
                'min_filter_points': 30,    # Era 45
                'max_rsi': 65,              # Era 48 - MUITO MAIS FLEXÍVEL
                'exposure': 0.20,           # 20% por trade (era 5%)
                'take_profit': 0.02,        # 2% TP (era 5%)
                'stop_loss': 0.01,          # 1% SL (era 2%)
                'trailing_activation': 0.01,
            },
            'MODERATE': {
                'min_score': 35,            # Era 45
                'min_filter_points': 25,    # Era 35
                'max_rsi': 70,              # Era 52
                'exposure': 0.15,           # 15% por trade (era 3.5%)
                'take_profit': 0.02,        # 2% TP (era 4%)
                'stop_loss': 0.01,          # 1% SL (era 1.5%)
                'trailing_activation': 0.01,
            },
            'RECOVERY': {
                'min_score': 30,            # Era 40
                'min_filter_points': 20,    # Era 30
                'max_rsi': 35,              # Sobrevendido extremo
                'exposure': 0.10,           # 10% por trade (era 2%)
                'take_profit': 0.015,       # 1.5% TP (era 3%)
                'stop_loss': 0.008,         # 0.8% SL (era 1%)
                'trailing_activation': 0.008,
            },
        }

        # V4.0: DETECÇÃO DE REGIME
        self.current_regime = 'LATERAL'
        self.regime_adx_threshold = 20         # Mais sensível (era 25)

        # V4.0: POSITION TIMEOUT MUITO MAIS CURTO
        self.position_timeout_hours = 0.75     # 45 minutos (era 4h)
        self.stale_position_min_pnl = -0.01    # -1% para fechar (era -2%)

        # V4.0: ROTATION AGRESSIVA
        self.enable_rotation = True
        self.rotation_min_score = 40           # Era 60
        self.rotation_min_pnl = -0.005         # -0.5% (era -1%)

        # V4.0: STRONG_BUY OVERRIDE MUITO RELAXADO
        self.strong_buy_override = {
            'enabled': True,
            'min_score': 40,            # Era 60
            'max_rsi': 65,              # Era 42
            'volume_override': 0.5,     # Aceita volume 0.5x
            'ignore_trend': True,
        }

        # V4.0: MODO EMERGÊNCIA MAIS RÁPIDO
        self.emergency_mode = {
            'enabled': True,
            'trigger_hours': 1,             # 1h sem entradas (era 3h)
            'active': False,
            'max_positions_override': 7,    # Até 7 posições
            'filter_relaxation': 0.6,       # Relaxa 40% (era 20%)
            'duration_hours': 4,            # 4h de duração
            'activated_at': None,
        }

        # V4.0: TRACKING DE ÚLTIMA ENTRADA
        self.last_entry_time = None
        self.hours_without_entry = 0

        # =====================================================
        # V4.1: LIMITES DIÁRIOS DE SEGURANÇA COM CIRCUIT BREAKERS GRADUAIS
        # =====================================================
        self.daily_limits = {
            'target_profit': 0.05,      # Meta: +5% diário
            'max_profit': 0.10,         # Parar em +10%
            'max_loss': -0.03,          # Parar em -3%
            'max_trades': 20,           # Máximo 20 trades/dia
            'consecutive_losses_pause': 6,  # Pausa total apenas após 6 perdas
        }

        # V4.1: CIRCUIT BREAKERS GRADUAIS
        self.circuit_breakers = {
            'levels': [
                {'losses': 2, 'action': 'reduce_25', 'multiplier': 0.75},   # 2 perdas → -25% exposição
                {'losses': 3, 'action': 'reduce_50', 'multiplier': 0.50},   # 3 perdas → -50% exposição
                {'losses': 4, 'action': 'pause_5min', 'pause_minutes': 5},  # 4 perdas → pausa 5min
                {'losses': 5, 'action': 'pause_15min', 'pause_minutes': 15},# 5 perdas → pausa 15min
                {'losses': 6, 'action': 'stop', 'stop': True},              # 6 perdas → para
            ],
            'current_level': 0,
            'exposure_multiplier': 1.0,
            'pause_until': None,
            'recovery_wins': 2,  # Wins necessários para resetar
        }

        # V4.1: AJUSTES PARA MERCADO LATERAL
        self.market_regime = {
            'current': 'unknown',  # bullish, bearish, lateral
            'lateral_adjustments': {
                'min_signal_strength': 0.7,    # Sinal mais forte necessário
                'reduce_position_size': 0.5,    # -50% tamanho
                'increase_tp': 1.5,             # +50% take profit
                'decrease_sl': 0.75,            # -25% stop loss
                'max_positions': 2,             # Máximo 2 posições
            },
            'detection_enabled': True,
        }

        # V4.0: TRACKING DIÁRIO
        self.daily_stats = {
            'start_capital': self.crypto_initial_capital,
            'current_profit_pct': 0.0,
            'trades_today': 0,
            'wins_today': 0,
            'losses_today': 0,
            'consecutive_losses': 0,
            'day_started': get_brazil_time().date(),
            'target_reached': False,
            'limit_reached': False,
        }

        # V4.0: HORÁRIO IDEAL DE OPERAÇÃO (06:00-14:00)
        self.optimal_trading_hours = {
            'enabled': True,
            'start_hour': 6,
            'end_hour': 14,
            'reduce_exposure_outside': 0.5,  # 50% exposição fora do horário
        }

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
        Loop principal do sistema de crypto trading 24/7.
        Executa análises continuamente sem restrições de horário.
        """
        system_logger.info("🚀 Iniciando loop principal - Crypto Trading 24/7...")

        try:
            while self.running:
                # Executa crypto trading 24/7
                self._execute_crypto_iteration()

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

                    # V4.0 Phase 2: Aplica seleção avançada
                    buy_signals = self._apply_phase2_selection(buy_signals, price_map)

                # 2. Executa trades baseado em sinais
                self._execute_crypto_trades(buy_signals, sell_signals, price_map)

                # V4.0 Phase 4: Executa ciclo de auto-otimização
                if self.phase4_enabled:
                    # Prepara dados de mercado para Phase 4
                    market_data = [{'close': r.get('price', 0), 'volume': 0,
                                   'score': r.get('total_score', 50)}
                                  for r in results[:50]]

                    # Busca trades recentes do banco
                    recent_trades = []
                    if hasattr(self, 'db_logger') and self.db_logger:
                        try:
                            recent_trades = self.db_logger.get_recent_trades(days=7) or []
                        except:
                            pass

                    self._run_phase4_optimization_cycle(market_data, recent_trades)

                # V4.1: Relatório de saúde do sistema
                if self.v41_enabled:
                    health = self.health_monitor_v41.check_system_health()
                    system_logger.info(
                        f"\n📈 V4.1 Health: {health['status'].upper()} | "
                        f"Latência={health['data_sources']['avg_latency_ms']:.0f}ms | "
                        f"Sucesso={health['data_sources']['success_rate']:.1f}%"
                    )

                    # Log blacklist atualizado
                    if self.symbol_blacklist:
                        blacklisted = self.symbol_blacklist.get_blacklisted_symbols()
                        if blacklisted:
                            system_logger.info(f"   🚫 Blacklist: {len(blacklisted)} símbolos ({', '.join(blacklisted[:5])}{'...' if len(blacklisted) > 5 else ''})")

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

    def _check_daily_reset(self):
        """V4.0: Reseta estatísticas diárias à meia-noite."""
        today = get_brazil_time().date()
        if self.daily_stats['day_started'] != today:
            old_profit = self.daily_stats['current_profit_pct']
            system_logger.info(f"\n🌅 NOVO DIA DE TRADING - Resetando estatísticas")
            system_logger.info(f"   Resultado ontem: {old_profit*100:+.2f}%")

            self.daily_stats = {
                'start_capital': self.crypto_capital + sum(
                    p.get('trade_value', 0) for p in self.crypto_positions.values()
                ),
                'current_profit_pct': 0.0,
                'trades_today': 0,
                'wins_today': 0,
                'losses_today': 0,
                'consecutive_losses': 0,
                'day_started': today,
                'target_reached': False,
                'limit_reached': False,
            }

    def _update_daily_stats(self, profit: float, is_win: bool):
        """V4.0: Atualiza estatísticas diárias após cada trade."""
        self.daily_stats['trades_today'] += 1

        if is_win:
            self.daily_stats['wins_today'] += 1
            self.daily_stats['consecutive_losses'] = 0
        else:
            self.daily_stats['losses_today'] += 1
            self.daily_stats['consecutive_losses'] += 1

        # Recalcula lucro do dia
        current_capital = self.crypto_capital + sum(
            p.get('trade_value', 0) for p in self.crypto_positions.values()
        )
        start_capital = self.daily_stats['start_capital']
        if start_capital > 0:
            self.daily_stats['current_profit_pct'] = (current_capital - start_capital) / start_capital

        # Verifica se atingiu meta ou limite
        pct = self.daily_stats['current_profit_pct']
        if pct >= self.daily_limits['target_profit']:
            self.daily_stats['target_reached'] = True
            system_logger.info(f"\n🎯 META DIÁRIA ATINGIDA: {pct*100:+.2f}%!")

        if pct >= self.daily_limits['max_profit']:
            self.daily_stats['limit_reached'] = True
            system_logger.warning(f"\n🛑 LUCRO MÁXIMO ATINGIDO: {pct*100:+.2f}% - Parando operações")

        if pct <= self.daily_limits['max_loss']:
            self.daily_stats['limit_reached'] = True
            system_logger.warning(f"\n🛑 PERDA MÁXIMA ATINGIDA: {pct*100:+.2f}% - Parando operações")

    def _check_circuit_breakers(self) -> tuple:
        """
        V4.1: Verifica circuit breakers graduais.
        Returns: (can_trade, exposure_multiplier, message)
        """
        losses = self.daily_stats.get('consecutive_losses', 0)
        now = get_brazil_time()

        # Verifica se está em pausa
        pause_until = self.circuit_breakers.get('pause_until')
        if pause_until and now < pause_until:
            remaining = (pause_until - now).total_seconds() / 60
            return False, 0, f"Em pausa por circuit breaker ({remaining:.0f}min restantes)"

        # Reseta se pausa expirou
        if pause_until and now >= pause_until:
            self.circuit_breakers['pause_until'] = None
            system_logger.info("✅ Pausa de circuit breaker expirada - Retomando operações")

        # Encontra o nível de circuit breaker atual
        current_level = None
        for level in self.circuit_breakers['levels']:
            if losses >= level['losses']:
                current_level = level

        if current_level is None:
            self.circuit_breakers['exposure_multiplier'] = 1.0
            return True, 1.0, "Operação normal"

        # Aplica ação do circuit breaker
        action = current_level['action']

        if 'reduce' in action:
            multiplier = current_level.get('multiplier', 0.5)
            self.circuit_breakers['exposure_multiplier'] = multiplier
            return True, multiplier, f"CB ativo: {action} (exposição {multiplier*100:.0f}%)"

        elif 'pause' in action:
            pause_minutes = current_level.get('pause_minutes', 5)
            self.circuit_breakers['pause_until'] = now + timedelta(minutes=pause_minutes)
            system_logger.warning(f"🚨 CIRCUIT BREAKER: {losses} perdas → Pausa de {pause_minutes}min")
            return False, 0, f"CB: Pausa de {pause_minutes}min ativada"

        elif current_level.get('stop'):
            system_logger.warning(f"🛑 CIRCUIT BREAKER: {losses} perdas → Operações suspensas")
            return False, 0, "CB: Operações suspensas"

        return True, 1.0, "Operação normal"

    def _on_trade_result(self, is_win: bool):
        """
        V4.1: Atualiza estatísticas após resultado de trade.
        Gerencia circuit breakers e recuperação.
        """
        if is_win:
            self.daily_stats['wins_today'] += 1
            self.daily_stats['consecutive_losses'] = 0
            # Reseta circuit breaker após wins consecutivos
            self.circuit_breakers['exposure_multiplier'] = 1.0
            self.circuit_breakers['pause_until'] = None
        else:
            self.daily_stats['losses_today'] += 1
            self.daily_stats['consecutive_losses'] += 1

    def _check_daily_limits(self) -> bool:
        """
        V4.1: Verifica se pode continuar operando baseado nos limites diários.
        Inclui circuit breakers graduais.
        Returns: True se pode operar, False se deve parar.
        """
        self._check_daily_reset()

        # Verifica limite de lucro/perda
        if self.daily_stats['limit_reached']:
            return False

        # Verifica máximo de trades
        if self.daily_stats['trades_today'] >= self.daily_limits['max_trades']:
            system_logger.info(f"⚠️ Máximo de trades diários atingido ({self.daily_limits['max_trades']})")
            return False

        # V4.1: Verifica circuit breakers graduais
        can_trade, exposure_mult, message = self._check_circuit_breakers()
        if not can_trade:
            system_logger.warning(f"⚠️ Circuit breaker: {message}")
            return False

        # Atualiza multiplicador de exposição
        self.circuit_breakers['exposure_multiplier'] = exposure_mult

        return True

    def _is_optimal_trading_hour(self) -> tuple:
        """
        V4.0: Verifica se está no horário ideal de operação.
        Returns: (is_optimal, exposure_multiplier)
        """
        if not self.optimal_trading_hours.get('enabled', False):
            return True, 1.0

        now = get_brazil_time()
        hour = now.hour

        start = self.optimal_trading_hours['start_hour']
        end = self.optimal_trading_hours['end_hour']

        if start <= hour < end:
            return True, 1.0
        else:
            return False, self.optimal_trading_hours['reduce_exposure_outside']

    def _check_emergency_mode(self):
        """
        V4.0: Verifica e ativa modo de emergência.
        CORRIGIDO: Não ativa na inicialização (bug 999h).
        Considera múltiplos fatores: tempo, performance, posições.
        """
        if not self.emergency_mode.get('enabled', False):
            return

        now = get_brazil_time()

        # V4.0 FIX: Não ativa emergência na primeira execução
        # Só ativa após pelo menos uma iteração completa
        if not hasattr(self, '_first_iteration_complete'):
            self._first_iteration_complete = False

        if not self._first_iteration_complete:
            # Marca primeira iteração como completa após 5 minutos de execução
            if hasattr(self, '_system_start_time'):
                elapsed = (now - self._system_start_time).total_seconds() / 60
                if elapsed >= 5:
                    self._first_iteration_complete = True
            else:
                self._system_start_time = now
            return  # Não verifica emergência ainda

        # Calcula horas desde última entrada
        if self.last_entry_time:
            try:
                if hasattr(self.last_entry_time, 'tzinfo') and self.last_entry_time.tzinfo:
                    hours_since = (now - self.last_entry_time).total_seconds() / 3600
                else:
                    hours_since = (datetime.now() - self.last_entry_time).total_seconds() / 3600
            except:
                hours_since = 0
            self.hours_without_entry = hours_since
        else:
            # V4.0 FIX: Se nunca entrou, usa tempo desde início do sistema
            if hasattr(self, '_system_start_time'):
                hours_since = (now - self._system_start_time).total_seconds() / 3600
            else:
                hours_since = 0  # Não ativa emergência se não souber quando começou
            self.hours_without_entry = hours_since

        # V4.0 FIX: Critérios múltiplos para ativar emergência
        should_activate = False
        activation_reasons = []

        # Critério 1: Tempo sem entrada (apenas se > trigger_hours E não temos posições)
        if hours_since >= self.emergency_mode['trigger_hours']:
            if len(self.crypto_positions) == 0:
                activation_reasons.append(f"{hours_since:.1f}h sem entradas")
                should_activate = True

        # Critério 2: Performance ruim do dia
        daily_pnl = self.daily_stats.get('current_profit_pct', 0)
        if daily_pnl < -0.02 and self.daily_stats.get('trades_today', 0) >= 2:
            activation_reasons.append(f"PnL diário: {daily_pnl*100:.1f}%")
            should_activate = True

        # Critério 3: Perdas consecutivas
        if self.daily_stats.get('consecutive_losses', 0) >= 3:
            activation_reasons.append(f"{self.daily_stats['consecutive_losses']} perdas consecutivas")
            should_activate = True

        # Verifica se deve ativar modo emergência
        if should_activate and not self.emergency_mode['active']:
            self.emergency_mode['active'] = True
            self.emergency_mode['activated_at'] = now
            system_logger.warning(f"\n🚨 MODO EMERGÊNCIA ATIVADO!")
            for reason in activation_reasons:
                system_logger.warning(f"   Motivo: {reason}")
            system_logger.warning(f"   Max positions: {self.max_positions} → {self.emergency_mode['max_positions_override']}")
            system_logger.warning(f"   Filtros relaxados em {(1-self.emergency_mode['filter_relaxation'])*100:.0f}%")
            system_logger.warning(f"   Duração: {self.emergency_mode['duration_hours']}h")

        # Verifica se deve desativar modo emergência
        if self.emergency_mode['active'] and self.emergency_mode['activated_at']:
            try:
                activated_at = self.emergency_mode['activated_at']
                if hasattr(activated_at, 'tzinfo') and activated_at.tzinfo:
                    hours_active = (now - activated_at).total_seconds() / 3600
                else:
                    hours_active = (datetime.now() - activated_at).total_seconds() / 3600

                # Desativa após duração OU se fez entrada com sucesso
                if hours_active >= self.emergency_mode['duration_hours']:
                    self.emergency_mode['active'] = False
                    self.emergency_mode['activated_at'] = None
                    system_logger.info(f"\n✅ Modo emergência DESATIVADO após {hours_active:.1f}h")
                elif self.last_entry_time and self.last_entry_time > activated_at:
                    # Desativa se fez entrada durante emergência
                    self.emergency_mode['active'] = False
                    self.emergency_mode['activated_at'] = None
                    system_logger.info(f"\n✅ Modo emergência DESATIVADO (entrada realizada)")
            except:
                pass

    def _get_effective_max_positions(self) -> int:
        """V3.2: Retorna max positions considerando modo emergência."""
        if self.emergency_mode.get('active', False):
            return self.emergency_mode['max_positions_override']
        return self.max_positions

    def _get_effective_filter_threshold(self) -> float:
        """V3.2: Retorna filter threshold considerando modo emergência."""
        if self.emergency_mode.get('active', False):
            return self.filter_threshold * self.emergency_mode['filter_relaxation']
        return self.filter_threshold

    def _apply_phase2_selection(self, buy_signals: list, price_map: dict) -> list:
        """
        V4.0 Phase 2: Aplica seleção avançada de ativos.
        Usa padrões técnicos, volume, timing e breakouts.
        Retorna sinais reordenados e enriquecidos.
        """
        if not self.phase2_enabled or not self.ranking_system:
            return buy_signals

        try:
            # Verifica timing primeiro
            timing = self.timing_manager.analyze_timing()
            system_logger.info(f"\n📊 V4.0 PHASE 2: SELEÇÃO AVANÇADA")
            system_logger.info(f"   Sessão: {timing.current_session.value}")
            system_logger.info(f"   Horário ideal: {'✅' if timing.is_optimal else '❌'}")
            system_logger.info(f"   Confiança: {timing.confidence}")

            enhanced_signals = []

            for crypto in buy_signals[:20]:  # Analisa top 20
                symbol = crypto['symbol']
                price = crypto.get('price', price_map.get(symbol, 0))

                # Obtém dados históricos do scanner (se disponível)
                # Nota: Em produção, buscar df do cache do crypto_scanner
                df = self._get_crypto_dataframe(symbol)

                if df is None or len(df) < 20:
                    # Sem dados suficientes, mantém score original
                    crypto['phase2_score'] = crypto.get('total_score', 50)
                    crypto['phase2_applied'] = False
                    enhanced_signals.append(crypto)
                    continue

                # Calcula score completo Phase 2
                asset_score = self.ranking_system.calculate_comprehensive_score(
                    symbol, crypto, df
                )

                # Enriquece sinal com dados Phase 2
                crypto['phase2_score'] = asset_score.total_score
                crypto['phase2_technical'] = asset_score.technical_score
                crypto['phase2_volume'] = asset_score.volume_score
                crypto['phase2_pattern'] = asset_score.pattern_score
                crypto['phase2_timing'] = asset_score.timing_score
                crypto['phase2_breakout'] = asset_score.breakout_score
                crypto['phase2_recommendation'] = asset_score.recommendation
                crypto['phase2_confidence'] = asset_score.confidence
                crypto['phase2_risk'] = asset_score.risk_level
                crypto['phase2_rr'] = asset_score.risk_reward
                crypto['phase2_applied'] = True

                # Ajusta parâmetros se recomendado
                if asset_score.recommendation in ['STRONG_BUY', 'BUY']:
                    params = self.ranking_system.get_adjusted_parameters(asset_score)
                    crypto['phase2_exposure'] = params['exposure']
                    crypto['phase2_tp'] = params['take_profit']
                    crypto['phase2_sl'] = params['stop_loss']

                enhanced_signals.append(crypto)

            # Reordena por phase2_score
            enhanced_signals.sort(key=lambda x: x.get('phase2_score', 0), reverse=True)

            # Log top 5 com Phase 2
            system_logger.info(f"\n🏆 TOP 5 APÓS PHASE 2:")
            for i, sig in enumerate(enhanced_signals[:5], 1):
                p2 = sig.get('phase2_applied', False)
                if p2:
                    system_logger.info(
                        f"   {i}. {sig['symbol']}: P2={sig['phase2_score']:.0f} "
                        f"(T:{sig.get('phase2_technical', 0):.0f} V:{sig.get('phase2_volume', 0):.0f} "
                        f"P:{sig.get('phase2_pattern', 0):.0f}) | {sig.get('phase2_recommendation', 'N/A')}"
                    )
                else:
                    system_logger.info(
                        f"   {i}. {sig['symbol']}: Score={sig.get('total_score', 0):.0f} (P2 não aplicado)"
                    )

            return enhanced_signals

        except Exception as e:
            system_logger.warning(f"Erro Phase 2: {e}. Usando seleção padrão.")
            return buy_signals

    def _get_crypto_dataframe(self, symbol: str):
        """
        V4.0 Phase 2: Obtém DataFrame de dados históricos para um símbolo.
        Usa cache do crypto_scanner se disponível.
        """
        try:
            if self.crypto_scanner and hasattr(self.crypto_scanner, '_cache'):
                cache = self.crypto_scanner._cache
                if symbol in cache:
                    return cache[symbol].get('df')

            # Fallback: busca dados frescos
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='7d', interval='1h')
            return df if len(df) > 0 else None

        except Exception as e:
            system_logger.debug(f"Erro obtendo df para {symbol}: {e}")
            return None

    def _apply_phase3_rotation(self, buy_signals: list, price_map: dict) -> bool:
        """
        V4.0 Phase 3: Aplica sistema de rotação agressiva.
        Analisa posições atuais e candidatos para rotação otimizada.
        Returns: True se rotação foi executada.
        """
        if not self.phase3_enabled or not self.rotation_manager:
            return False

        try:
            # Atualiza capital no allocation manager
            if self.allocation_manager:
                positions_value = sum(p.get('trade_value', 0) for p in self.crypto_positions.values())
                self.allocation_manager.update_capital(self.crypto_capital, positions_value)

            # Atualiza métricas de performance de cada posição
            if self.performance_monitor and self.crypto_positions:
                for symbol, position in self.crypto_positions.items():
                    current_price = price_map.get(symbol, 0)
                    if current_price > 0:
                        self.performance_monitor.update_position(
                            symbol, position, current_price,
                            current_volume=0  # Volume seria atualizado se disponível
                        )

                # Log posições com baixa performance
                underperforming = self.performance_monitor.get_underperforming_positions(threshold=30)
                if underperforming:
                    system_logger.info(f"\n⚠️ PHASE 3: {len(underperforming)} posições com baixa performance:")
                    for pos in underperforming[:3]:
                        system_logger.info(f"   - {pos.symbol}: Score={pos.performance_score:.0f}, "
                                          f"P&L={pos.pnl_percent:.2f}%, Rec={pos.recommendation}")

            # Verifica se pode rotacionar
            if not self.rotation_manager.can_rotate():
                system_logger.debug("Phase 3: Rotação não disponível (limite ou cooldown)")
                return False

            # Ranking das posições atuais
            if self.intraday_ranking and self.crypto_positions:
                ranked_positions = self.intraday_ranking.rank_current_positions(
                    self.crypto_positions, price_map
                )

                # Ranking dos candidatos
                ranked_candidates = self.intraday_ranking.rank_candidate_signals(
                    buy_signals, self.crypto_positions, max_candidates=10
                )

                if ranked_positions and ranked_candidates:
                    # Encontra oportunidades de rotação
                    rotation_opportunities = self.intraday_ranking.get_rotation_candidates(
                        self.crypto_positions, buy_signals, price_map,
                        min_improvement=15.0
                    )

                    if rotation_opportunities:
                        system_logger.info(f"\n🔄 PHASE 3 ROTATION: {len(rotation_opportunities)} oportunidades")

                        # Usa decision engine para melhor decisão
                        if self.decision_engine:
                            best_rotation = rotation_opportunities[0]
                            exit_symbol = best_rotation['exit_symbol']
                            candidate = best_rotation['candidate']

                            position = self.crypto_positions.get(exit_symbol)
                            current_price = price_map.get(exit_symbol, 0)

                            if position and current_price > 0:
                                decision = self.decision_engine.evaluate_rotation_scenario(
                                    position, candidate, current_price
                                )

                                self.decision_engine.log_decision(decision)

                                if decision.should_rotate:
                                    # Executa rotação
                                    system_logger.info(
                                        f"🔄 EXECUTANDO ROTAÇÃO: {exit_symbol} → {candidate['symbol']}"
                                    )

                                    # Fecha posição atual
                                    self._close_crypto_position(
                                        exit_symbol, current_price,
                                        f"PHASE3_ROTATION ({decision.scenario.value})"
                                    )

                                    # Registra no rotation manager
                                    self.rotation_manager.rotation_count_today += 1
                                    self.rotation_manager.rotation_count_hour += 1
                                    self.rotation_manager.last_rotation_time = datetime.now()

                                    return True  # Rotação executada

            return False

        except Exception as e:
            system_logger.warning(f"Erro Phase 3 rotation: {e}")
            return False

    def _get_phase3_allocation(self, signal: dict) -> float:
        """
        V4.0 Phase 3: Calcula alocação usando Kelly Criterion.
        Returns: Percentual do capital a alocar (0.0 - 0.25).
        """
        if not self.phase3_enabled or not self.allocation_manager:
            # Fallback para alocação padrão
            return signal.get('phase2_exposure', 0.15)

        try:
            score = signal.get('phase2_score', 0) or signal.get('total_score', 50)
            volatility = signal.get('volatility', 1.0)
            take_profit = signal.get('phase2_tp', 0.02)
            stop_loss = signal.get('phase2_sl', 0.01)

            allocation = self.allocation_manager.calculate_position_size(
                signal['symbol'], score, volatility, take_profit, stop_loss
            )

            system_logger.info(
                f"   📊 Kelly Allocation: {allocation.final_position_size:.2f} "
                f"({allocation.kelly_fraction*100:.1f}% Kelly, {allocation.reason})"
            )

            return allocation.adjusted_allocation

        except Exception as e:
            system_logger.debug(f"Erro Kelly allocation: {e}")
            return 0.15

    def _run_phase4_optimization_cycle(self, market_data: list, recent_trades: list):
        """
        V4.0 Phase 4: Executa ciclo de auto-otimização.
        Detecta regime, otimiza parâmetros e aplica meta-aprendizado.
        """
        if not self.phase4_enabled or not self.auto_optimizer:
            return

        try:
            now = datetime.now()

            # Verifica intervalo mínimo entre ciclos
            if self.last_optimization_cycle:
                hours_since = (now - self.last_optimization_cycle).total_seconds() / 3600
                if hours_since < self.optimization_interval_hours:
                    return

            system_logger.info(f"\n🤖 PHASE 4: Iniciando ciclo de auto-otimização...")

            # Executa ciclo de otimização
            cycle_result = self.auto_optimizer.run_optimization_cycle(
                market_data=market_data,
                recent_trades=recent_trades
            )

            self.last_optimization_cycle = now

            # Loga resultado
            if cycle_result['actions_taken']:
                system_logger.info(f"   Ações executadas: {', '.join(cycle_result['actions_taken'])}")

            if cycle_result['new_regime']:
                system_logger.info(f"   Regime detectado: {cycle_result['new_regime']}")

            if cycle_result['strategy_changed']:
                system_logger.info(f"   ✅ Estratégia atualizada")

            # Atualiza estratégia atual se otimizada
            if cycle_result['strategy_changed'] and self.auto_optimizer.current_strategy:
                self._apply_phase4_strategy(self.auto_optimizer.current_strategy)

        except Exception as e:
            system_logger.warning(f"Erro no ciclo Phase 4: {e}")

    def _apply_phase4_strategy(self, strategy: dict):
        """V4.0 Phase 4: Aplica estratégia otimizada."""
        try:
            # Atualiza parâmetros de entrada
            if 'entry_params' in strategy:
                entry = strategy['entry_params']
                if 'score_threshold' in entry:
                    self.filter_threshold = entry['score_threshold']

            # Atualiza parâmetros de saída
            if 'exit_params' in strategy:
                exit_p = strategy['exit_params']
                if 'take_profit' in exit_p:
                    self.crypto_take_profit = exit_p['take_profit'] / 100
                if 'stop_loss' in exit_p:
                    self.crypto_stop_loss = exit_p['stop_loss'] / 100
                if 'timeout_minutes' in exit_p:
                    self.position_timeout_hours = exit_p['timeout_minutes'] / 60

            # Atualiza parâmetros de risco
            if 'risk_params' in strategy:
                risk = strategy['risk_params']
                if 'position_size_percent' in risk:
                    self.crypto_exposure = risk['position_size_percent']

            system_logger.info(f"   📊 Estratégia Phase 4 aplicada:")
            system_logger.info(f"      Threshold: {self.filter_threshold}")
            system_logger.info(f"      TP/SL: {self.crypto_take_profit*100:.1f}%/{self.crypto_stop_loss*100:.1f}%")
            system_logger.info(f"      Exposure: {self.crypto_exposure*100:.1f}%")

        except Exception as e:
            system_logger.warning(f"Erro aplicando estratégia Phase 4: {e}")

    def _record_trade_for_learning(self, symbol: str, result: dict):
        """V4.0 Phase 4: Registra trade para aprendizado."""
        if not self.phase4_enabled or not self.auto_optimizer:
            return

        try:
            trade_data = {
                'symbol': symbol,
                'result': 'success' if result.get('profit', 0) > 0 else 'failure',
                'profit': result.get('profit', 0),
                'profit_pct': result.get('profit_pct', 0),
                'entry_rsi': result.get('rsi', 50),
                'volume_ratio': result.get('volume_ratio', 1.0),
                'regime': self.current_regime,
                'hold_time_minutes': result.get('hold_time_minutes', 30),
                'timestamp': datetime.now()
            }

            self.auto_optimizer.record_trade_result(trade_data)

        except Exception as e:
            system_logger.debug(f"Erro registrando trade para learning: {e}")

    def _log_positions_dashboard(self, price_map: dict):
        """
        V4.0: Dashboard agressivo com métricas de 5% diário.
        """
        now = get_brazil_time()

        # V4.0: Verifica limites e emergência
        self._check_emergency_mode()

        system_logger.info("\n" + "=" * 60)
        system_logger.info("🐺 V4.0 LOBO AGRESSIVO - META 5% DIÁRIO")
        system_logger.info("=" * 60)

        # V4.0: Métricas do dia
        daily_pct = self.daily_stats['current_profit_pct'] * 100
        target_pct = self.daily_limits['target_profit'] * 100
        progress = min(100, (daily_pct / target_pct) * 100) if target_pct > 0 else 0

        if daily_pct >= 0:
            emoji = "🟢" if daily_pct >= target_pct else "📈"
        else:
            emoji = "🔴"

        system_logger.info(f"{emoji} P&L HOJE: {daily_pct:+.2f}% / Meta: {target_pct:.1f}% ({progress:.0f}%)")
        system_logger.info(f"   Trades: {self.daily_stats['trades_today']}/{self.daily_limits['max_trades']} | "
                          f"Wins: {self.daily_stats['wins_today']} | Losses: {self.daily_stats['losses_today']}")

        if self.daily_stats['consecutive_losses'] > 0:
            system_logger.info(f"   ⚠️ Perdas consecutivas: {self.daily_stats['consecutive_losses']}")

        if self.daily_stats['target_reached']:
            system_logger.info(f"   🎯 META ATINGIDA!")

        # V4.0: Horário ideal
        is_optimal, exposure_mult = self._is_optimal_trading_hour()
        hour_status = "✅ HORÁRIO IDEAL" if is_optimal else f"⚡ Fora do horário (exposição {exposure_mult*100:.0f}%)"
        system_logger.info(f"   {hour_status}")

        # V4.0: Status do modo emergência
        if self.emergency_mode.get('active', False):
            system_logger.info("🚨 MODO EMERGÊNCIA: ATIVO")
            if self.emergency_mode['activated_at']:
                try:
                    activated_at = self.emergency_mode['activated_at']
                    if hasattr(activated_at, 'tzinfo') and activated_at.tzinfo:
                        hours_active = (now - activated_at).total_seconds() / 3600
                    else:
                        hours_active = (datetime.now() - activated_at).total_seconds() / 3600
                    remaining = self.emergency_mode['duration_hours'] - hours_active
                    system_logger.info(f"   Tempo restante: {remaining:.1f}h")
                except:
                    pass
        else:
            hours_until_emergency = max(0, self.emergency_mode['trigger_hours'] - self.hours_without_entry)
            system_logger.info(f"⚡ Modo emergência em: {hours_until_emergency:.1f}h")

        # V3.2: Mostra max positions efetivo
        effective_max = self._get_effective_max_positions()
        system_logger.info(f"📍 Max posições: {effective_max} {'(EMERGENCY)' if effective_max > self.max_positions else ''}")

        if not self.crypto_positions:
            system_logger.info("\n📂 Nenhuma posição aberta")
            system_logger.info("=" * 60)
            return

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
            if current_price > 0 and entry_price > 0:
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
        V4.1: Detecta e fecha posições estagnadas (timeout).
        Inclui fechamento forçado para ativos na blacklist.
        Retorna lista de posições fechadas.
        """
        now = get_brazil_time()
        closed_positions = []

        # V4.1: Importa blacklist para verificação
        try:
            from crypto_scanner import CRYPTO_BLACKLIST
        except ImportError:
            CRYPTO_BLACKLIST = set()

        for symbol, position in list(self.crypto_positions.items()):
            current_price = price_map.get(symbol, 0)
            entry_price = position['entry_price']
            entry_time = position.get('entry_time', now)

            # Calcula tempo aberto
            try:
                if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo:
                    open_hours = (now - entry_time).total_seconds() / 3600
                else:
                    open_hours = (datetime.now() - entry_time).total_seconds() / 3600
            except:
                open_hours = 0

            # V4.1 FIX: Fecha forçadamente ativos na blacklist (sem dados)
            if symbol in CRYPTO_BLACKLIST:
                system_logger.warning(
                    f"🚫 BLACKLIST: {symbol} está na blacklist - Fechando forçadamente!"
                )
                # Usa preço de entrada como fallback se não houver preço atual
                close_price = current_price if current_price > 0 else entry_price
                self._close_crypto_position(symbol, close_price, 'BLACKLIST_FORCE_CLOSE')
                closed_positions.append(symbol)
                continue

            # V4.1 FIX: Timeout forçado após 2h (independente do P&L)
            if open_hours >= 2.0:
                system_logger.warning(
                    f"🚨 TIMEOUT FORÇADO: {symbol} aberta há {open_hours:.1f}h - Fechando!"
                )
                close_price = current_price if current_price > 0 else entry_price
                self._close_crypto_position(symbol, close_price, f'FORCED_TIMEOUT_{open_hours:.0f}H')
                closed_positions.append(symbol)
                continue

            # Se não há preço atual ou entry_price, pula para próxima iteração
            if current_price <= 0:
                system_logger.warning(f"⚠️ {symbol}: Sem preço atual disponível")
                continue

            if entry_price <= 0:
                system_logger.warning(f"⚠️ {symbol}: Entry price inválido ({entry_price})")
                continue

            pnl_pct = (current_price - entry_price) / entry_price

            # Verifica timeout normal
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
                        f"- Mantendo posição (máx 2h)"
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
            if entry_price <= 0:
                system_logger.warning(f"⚠️ {symbol}: Entry price inválido ({entry_price}), pulando...")
                continue

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
        V3.2: STRONG_BUY override AGRESSIVO.
        Critérios relaxados:
        - Score >= 60 (era 65)
        - RSI <= 42 (era 38)
        - Aceita qualquer sinal com 'BUY' (não precisa ser STRONG)
        """
        if not self.strong_buy_override.get('enabled', False):
            return {'override': False}

        score = crypto.get('total_score', 0)
        rsi = crypto.get('rsi', 50)
        signal = crypto.get('signal', '')

        # V3.2: Critérios RELAXADOS para override
        min_score = self.strong_buy_override['min_score']  # 60
        max_rsi = self.strong_buy_override['max_rsi']      # 42

        # V3.2: Em modo emergência, relaxa ainda mais
        if self.emergency_mode.get('active', False):
            min_score = min_score * 0.9  # 54
            max_rsi = max_rsi * 1.1      # 46

        # V3.2: Aceita STRONG_BUY ou apenas BUY com score alto
        is_buy_signal = 'BUY' in signal.upper()
        is_strong = 'STRONG' in signal.upper()

        # Override se: STRONG_BUY com critérios normais OU BUY com RSI muito baixo
        override_approved = False
        override_reason = ""

        if is_strong and score >= min_score and rsi <= max_rsi:
            override_approved = True
            override_reason = f'STRONG_BUY (Score:{score:.1f}≥{min_score:.0f}, RSI:{rsi:.1f}≤{max_rsi:.0f})'

        elif is_buy_signal and rsi <= 35 and score >= (min_score * 0.9):
            # V3.2: BUY normal com RSI muito baixo também qualifica
            override_approved = True
            override_reason = f'OVERSOLD_BUY (Score:{score:.1f}, RSI:{rsi:.1f}≤35)'

        elif self.emergency_mode.get('active', False) and is_buy_signal and score >= 55:
            # V3.2: Em modo emergência, qualquer BUY com score razoável
            override_approved = True
            override_reason = f'EMERGENCY_BUY (Score:{score:.1f}≥55, Emergency Mode)'

        if override_approved:
            return {
                'override': True,
                'reason': override_reason,
                'params': {
                    'exposure': 0.035,  # 3.5% - exposição reduzida para overrides
                    'take_profit': 0.04,
                    'stop_loss': 0.015,
                    'trailing_activation': 0.025,
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
            if entry_price <= 0:
                continue

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
        V4.0: Sistema AGRESSIVO para meta de 5% diário:
        - Filter threshold 25pts (relaxado de 45)
        - Take Profit 2%, Stop Loss 1%
        - 5 posições simultâneas
        - Position timeout 45 minutos
        - Cooldown: 10min win / 30min loss
        - Limites diários: +10% max / -3% max
        - Horário ideal: 06:00-14:00
        """
        mode = config.get('execution.mode', 'simulation')

        # V4.0: Reset diário à meia-noite
        self._check_daily_reset()

        # V3.2: Verifica se trading está pausado por drawdown
        if self.trading_paused:
            system_logger.warning("\n⚠️ Trading PAUSADO - Max Drawdown atingido")
            return

        # V4.0: Verifica limites diários
        if not self._check_daily_limits():
            return

        # V3.2: Usa max positions EFETIVO (considera modo emergência)
        effective_max_positions = self._get_effective_max_positions()
        positions_full = len(self.crypto_positions) >= effective_max_positions

        if positions_full:
            system_logger.info(f"\n⚠️ {len(self.crypto_positions)}/{effective_max_positions} posições abertas - Analisando para rotação...")
        else:
            system_logger.info(f"\n✅ SLOTS DISPONÍVEIS: {effective_max_positions - len(self.crypto_positions)} de {effective_max_positions}")

        # Analisa notícias para os sinais de compra
        if self.news_enabled:
            system_logger.info("\n📰 Analisando notícias...")
            for crypto in buy_signals[:5]:
                self._analyze_news_for_crypto(crypto['symbol'], crypto)

            # Reordena por score atualizado (considerando notícias)
            buy_signals.sort(key=lambda x: x.get('total_score', 0), reverse=True)

        # V3.2: Estatísticas de análise
        self.rejection_stats['total_analyzed'] += len(buy_signals)

        # V3.2: Calcula threshold efetivo (considera modo emergência)
        effective_threshold = self._get_effective_filter_threshold()

        system_logger.info(f"\n🔍 V3.2 ANÁLISE DE SINAIS ({len(buy_signals)} candidatos)")
        system_logger.info(f"   Regime atual: {self.current_regime}")
        system_logger.info(f"   Posições: {len(self.crypto_positions)}/{effective_max_positions}")
        system_logger.info(f"   Rotação habilitada: {'✅' if self.enable_rotation else '❌'}")
        system_logger.info(f"   Modo emergência: {'🚨 ATIVO' if self.emergency_mode.get('active', False) else '⚡ Standby'}")
        system_logger.info(f"   Filter threshold: {effective_threshold:.0f}pts {'(RELAXADO)' if effective_threshold < self.filter_threshold else ''}")

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

        # V4.0 Phase 3: Tenta rotação inteligente PRIMEIRO
        if positions_full and qualified_signals and self.phase3_enabled:
            if self._apply_phase3_rotation(qualified_signals, price_map):
                positions_full = False  # Rotação abriu espaço
                system_logger.info("✅ Phase 3 rotation executada com sucesso")

        # V3.1: Se posições cheias, tenta rotação legacy
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

            # V4.0 Phase 3: Usa Kelly allocation se disponível
            if self.phase3_enabled:
                exposure = self._get_phase3_allocation(crypto)
            else:
                # V3.1: Usa parâmetros do nível de sinal
                level_params = crypto.get('_level_params', self.signal_levels['MODERATE'])
                exposure = level_params['exposure']

            # Calcula quantidade baseado na exposição
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
        pnl_pct = float((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0

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

        # V4.0: Atualiza estatísticas diárias
        self._update_daily_stats(profit, is_win)

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


def reset_database():
    """Reseta todos os dados do banco para nova medição."""
    print("\n" + "=" * 60)
    print("🔄 RESET COMPLETO - Lobo IA")
    print("=" * 60)

    try:
        logger = Logger()
        cursor = logger.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM trades")
        trades_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM crypto_positions")
        positions_count = cursor.fetchone()[0]

        print(f"\n📊 Dados encontrados:")
        print(f"   - Trades: {trades_count}")
        print(f"   - Posições: {positions_count}")

        print("\n🗑️ Limpando banco de dados...")
        cursor.execute("DELETE FROM crypto_positions")
        cursor.execute("DELETE FROM trades")
        logger.conn.commit()

        print("✅ Banco de dados resetado!")
        print("   - 0 posições abertas")
        print("   - 0 trades históricos")
        print("   - $1000.00 capital inicial")
        print("=" * 60 + "\n")

        logger.close()
        return True

    except Exception as e:
        print(f"❌ Erro no reset: {e}")
        return False


def main():
    """Função principal - ponto de entrada."""
    try:
        # Verifica se deve resetar dados
        if os.environ.get('RESET_DATA', '').lower() in ('1', 'true', 'yes'):
            system_logger.warning("🔄 RESET_DATA detectado - Limpando banco...")
            reset_database()

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
