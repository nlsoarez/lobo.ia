"""
V4.0 Phase 4: Auto-Optimized Trading System
Sistema integrado que combina todos os módulos de auto-otimização.
Coordena otimização, detecção de regime, RL e meta-aprendizado.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from system_logger import system_logger

# Importa módulos Phase 4
try:
    from auto_optimization_engine import AutoOptimizationEngine
    from market_regime_detector import MarketRegimeDetector, MarketRegime
    from reinforcement_learning_agent import TradingRLAgent, TradingAction
    from multi_objective_optimizer import MultiObjectiveOptimizer, OptimizationObjective
    from meta_learning_system import MetaLearningSystem
    HAS_PHASE4_MODULES = True
except ImportError as e:
    HAS_PHASE4_MODULES = False
    system_logger.warning(f"Phase 4 modules não disponíveis: {e}")


class OptimizationState(Enum):
    """Estado do sistema de otimização."""
    IDLE = "idle"
    OPTIMIZING = "optimizing"
    ADAPTING = "adapting"
    LEARNING = "learning"
    PAUSED = "paused"


@dataclass
class SafetyControl:
    """Controles de segurança para otimização."""
    max_daily_loss_pct: float = 0.04
    max_parameter_drift_pct: float = 0.50
    min_model_confidence: float = 0.60
    max_consecutive_failures: int = 3
    rollback_depth: int = 3
    emergency_params: Optional[Dict] = None


@dataclass
class SystemState:
    """Estado atual do sistema."""
    optimization_state: OptimizationState
    current_regime: Optional[str]
    regime_confidence: float
    last_optimization: Optional[datetime]
    last_adaptation: Optional[datetime]
    daily_pnl: float
    strategy_version: int
    safety_triggered: bool


class AutoOptimizedTradingSystem:
    """
    V4.0 Phase 4: Sistema de trading auto-otimizado.
    Integra todos os componentes de ML e otimização.
    """

    def __init__(self, initial_capital: float = 1000.0):
        """Inicializa o sistema auto-otimizado."""
        self.initial_capital = initial_capital

        # Inicializa componentes Phase 4
        self.optimizer = AutoOptimizationEngine() if HAS_PHASE4_MODULES else None
        self.regime_detector = MarketRegimeDetector() if HAS_PHASE4_MODULES else None
        self.rl_agent = TradingRLAgent(state_size=20, action_size=4) if HAS_PHASE4_MODULES else None
        self.meta_learner = MetaLearningSystem() if HAS_PHASE4_MODULES else None

        # Multi-objective optimizer (configuração lazy)
        self.multi_objective_opt = None

        # Estado atual
        self.state = SystemState(
            optimization_state=OptimizationState.IDLE,
            current_regime=None,
            regime_confidence=0.0,
            last_optimization=None,
            last_adaptation=None,
            daily_pnl=0.0,
            strategy_version=1,
            safety_triggered=False
        )

        # Estratégia atual
        self.current_strategy: Dict[str, Any] = self._get_default_strategy()

        # Histórico de estratégias (para rollback)
        self.strategy_history: deque = deque(maxlen=10)
        self.strategy_history.append({
            'strategy': self.current_strategy.copy(),
            'timestamp': datetime.now(),
            'version': 1
        })

        # Controles de segurança
        self.safety = SafetyControl()
        self.safety.emergency_params = self._get_default_strategy()

        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.optimization_history: deque = deque(maxlen=100)

        # Contadores
        self.consecutive_failures = 0
        self.optimization_count = 0

        # Intervalos de otimização
        self.optimization_interval_hours = 6
        self.adaptation_interval_hours = 1
        self.learning_interval_minutes = 15

        system_logger.info(f"AutoOptimizedTradingSystem V4.0 inicializado")
        system_logger.info(f"   Módulos Phase 4: {'ATIVOS' if HAS_PHASE4_MODULES else 'INDISPONÍVEIS'}")

    def _get_default_strategy(self) -> Dict[str, Any]:
        """Retorna estratégia padrão."""
        return {
            'entry_params': {
                'score_threshold': 40,
                'volume_ratio_min': 2.0,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'pattern_score_min': 70
            },
            'exit_params': {
                'take_profit': 2.0,
                'stop_loss': 1.0,
                'trailing_stop_activation': 1.0,
                'trailing_stop_distance': 0.5,
                'timeout_minutes': 45
            },
            'risk_params': {
                'position_size_percent': 0.15,
                'max_daily_loss': 0.03,
                'kelly_fraction': 0.3
            }
        }

    def run_optimization_cycle(self, market_data: List[Dict],
                              recent_trades: List[Dict]) -> Dict[str, Any]:
        """
        Executa um ciclo completo de otimização.

        Args:
            market_data: Dados de mercado recentes
            recent_trades: Trades recentes para aprendizado

        Returns:
            Resultado do ciclo de otimização
        """
        cycle_result = {
            'timestamp': datetime.now(),
            'actions_taken': [],
            'strategy_changed': False,
            'new_regime': None,
            'optimization_score': None
        }

        # 1. Verifica segurança
        if not self._check_safety():
            cycle_result['actions_taken'].append('safety_check_failed')
            return cycle_result

        # 2. Detecta regime de mercado
        if self.regime_detector and market_data:
            regime_result = self._update_regime(market_data)
            if regime_result:
                cycle_result['new_regime'] = regime_result['regime']
                cycle_result['actions_taken'].append('regime_detected')

        # 3. Verifica se deve otimizar
        if self._should_optimize():
            optimization_result = self._run_optimization(market_data)
            if optimization_result:
                cycle_result['optimization_score'] = optimization_result.get('score')
                cycle_result['strategy_changed'] = True
                cycle_result['actions_taken'].append('optimization_executed')

        # 4. Aplica meta-aprendizado
        if self.meta_learner and recent_trades:
            adaptation_result = self._apply_meta_learning(recent_trades)
            if adaptation_result.get('adaptations_applied', 0) > 0:
                cycle_result['strategy_changed'] = True
                cycle_result['actions_taken'].append('meta_learning_applied')

        # 5. Treina RL Agent
        if self.rl_agent and recent_trades:
            training_result = self._train_rl_agent(recent_trades)
            if training_result:
                cycle_result['actions_taken'].append('rl_training_completed')

        # 6. Atualiza estado
        self._update_state()

        return cycle_result

    def _check_safety(self) -> bool:
        """Verifica controles de segurança."""
        # Verifica perda diária máxima
        if self.state.daily_pnl < -self.safety.max_daily_loss_pct:
            system_logger.warning(f"⚠️ SAFETY: Perda diária excedida ({self.state.daily_pnl*100:.1f}%)")
            self.state.safety_triggered = True
            self.state.optimization_state = OptimizationState.PAUSED
            return False

        # Verifica falhas consecutivas
        if self.consecutive_failures >= self.safety.max_consecutive_failures:
            system_logger.warning(f"⚠️ SAFETY: {self.consecutive_failures} falhas consecutivas")
            self._rollback_strategy()
            self.consecutive_failures = 0
            return False

        # Verifica drift de parâmetros
        if self._check_parameter_drift() > self.safety.max_parameter_drift_pct:
            system_logger.warning("⚠️ SAFETY: Drift excessivo de parâmetros")
            self._rollback_strategy()
            return False

        self.state.safety_triggered = False
        return True

    def _check_parameter_drift(self) -> float:
        """Calcula drift dos parâmetros em relação ao padrão."""
        if not self.safety.emergency_params:
            return 0.0

        total_drift = 0
        param_count = 0

        for category in self.current_strategy:
            if category in self.safety.emergency_params:
                for param in self.current_strategy[category]:
                    if param in self.safety.emergency_params[category]:
                        current = self.current_strategy[category][param]
                        default = self.safety.emergency_params[category][param]
                        if default != 0:
                            drift = abs(current - default) / abs(default)
                            total_drift += drift
                            param_count += 1

        return total_drift / param_count if param_count > 0 else 0

    def _update_regime(self, market_data: List[Dict]) -> Optional[Dict]:
        """Atualiza detecção de regime."""
        if not self.regime_detector or not market_data:
            return None

        features = self.regime_detector.prepare_features(market_data)
        if not features:
            return None

        detection = self.regime_detector.detect_regime(features)

        self.state.current_regime = detection.regime.value
        self.state.regime_confidence = detection.confidence

        # Adapta estratégia ao regime
        regime_strategy = self.regime_detector.get_regime_strategy(detection.regime)
        self._adapt_strategy_to_regime(regime_strategy)

        return {
            'regime': detection.regime.value,
            'confidence': detection.confidence
        }

    def _adapt_strategy_to_regime(self, regime_strategy):
        """Adapta estratégia ao regime de mercado."""
        # Aplica multiplicadores
        if 'exit_params' in self.current_strategy:
            self.current_strategy['exit_params']['take_profit'] *= regime_strategy.take_profit_multiplier
            self.current_strategy['exit_params']['stop_loss'] *= regime_strategy.stop_loss_multiplier

        if 'risk_params' in self.current_strategy:
            self.current_strategy['risk_params']['position_size_percent'] *= regime_strategy.position_sizing

    def _should_optimize(self) -> bool:
        """Verifica se deve executar otimização."""
        if self.state.optimization_state == OptimizationState.PAUSED:
            return False

        if self.state.last_optimization is None:
            return True

        hours_since = (datetime.now() - self.state.last_optimization).total_seconds() / 3600
        return hours_since >= self.optimization_interval_hours

    def _run_optimization(self, market_data: List[Dict]) -> Optional[Dict]:
        """Executa otimização de parâmetros."""
        if not self.optimizer:
            return None

        self.state.optimization_state = OptimizationState.OPTIMIZING

        try:
            # Configura dados históricos
            self.optimizer.historical_data = market_data

            # Executa otimização
            result = self.optimizer.run_optimization(method='bayesian', n_iterations=30)

            if result.best_score > 0:
                # Aplica novos parâmetros
                self._apply_optimized_params(result.best_params)

                self.optimization_count += 1
                self.state.last_optimization = datetime.now()
                self.consecutive_failures = 0

                system_logger.info(f"✅ Otimização #{self.optimization_count} concluída "
                                 f"(score: {result.best_score:.4f})")

                return {
                    'score': result.best_score,
                    'params': result.best_params,
                    'iterations': result.iterations
                }
            else:
                self.consecutive_failures += 1
                return None

        except Exception as e:
            system_logger.error(f"Erro na otimização: {e}")
            self.consecutive_failures += 1
            return None

        finally:
            self.state.optimization_state = OptimizationState.IDLE

    def _apply_optimized_params(self, params: Dict):
        """Aplica parâmetros otimizados à estratégia."""
        # Salva estratégia atual no histórico
        self.strategy_history.append({
            'strategy': self.current_strategy.copy(),
            'timestamp': datetime.now(),
            'version': self.state.strategy_version
        })

        # Aplica novos parâmetros
        for category, param_dict in params.items():
            if category in self.current_strategy:
                for param, value in param_dict.items():
                    if param in self.current_strategy[category]:
                        self.current_strategy[category][param] = value

        self.state.strategy_version += 1

    def _apply_meta_learning(self, recent_trades: List[Dict]) -> Dict:
        """Aplica meta-aprendizado."""
        if not self.meta_learner:
            return {'adaptations_applied': 0}

        self.state.optimization_state = OptimizationState.LEARNING

        try:
            # Aprende de trades recentes
            for trade in recent_trades[-50:]:
                self.meta_learner.learn_from_experience(trade)

            # Calcula performance recente
            recent_performance = self._calculate_recent_performance(recent_trades)

            # Obtém contexto de mercado
            market_context = {
                'market_regime': self.state.current_regime or 'sideways',
                'time_of_day': datetime.now().hour,
                'volatility': recent_performance.get('avg_volatility', 0.02)
            }

            # Sugere adaptações
            adaptations = self.meta_learner.suggest_adaptations(
                recent_performance, market_context
            )

            # Aplica adaptações
            if adaptations:
                self.current_strategy = self.meta_learner.adapt_strategy(
                    self.current_strategy, adaptations
                )

                self.state.last_adaptation = datetime.now()

                return {'adaptations_applied': len([a for a in adaptations if a.applied])}

            return {'adaptations_applied': 0}

        finally:
            self.state.optimization_state = OptimizationState.IDLE

    def _train_rl_agent(self, recent_trades: List[Dict]) -> Optional[Dict]:
        """Treina agente RL com trades recentes."""
        if not self.rl_agent or len(recent_trades) < 20:
            return None

        try:
            training_stats = self.rl_agent.train_on_historical_data(
                recent_trades[-100:], epochs=5
            )

            return training_stats

        except Exception as e:
            system_logger.warning(f"Erro no treinamento RL: {e}")
            return None

    def _calculate_recent_performance(self, trades: List[Dict]) -> Dict:
        """Calcula métricas de performance recente."""
        if not trades:
            return {}

        profits = [t.get('profit', 0) for t in trades]
        wins = sum(1 for p in profits if p > 0)

        return {
            'win_rate': wins / len(trades) if trades else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'total_profit': sum(profits),
            'max_drawdown': min(0, min(profits)) if profits else 0,
            'profit_factor': (sum(p for p in profits if p > 0) /
                            abs(sum(p for p in profits if p < 0)) if any(p < 0 for p in profits) else 2.0),
            'avg_trade_duration_minutes': np.mean([
                t.get('hold_time_minutes', 30) for t in trades
            ]) if trades else 30,
            'avg_volatility': np.mean([t.get('volatility', 0.02) for t in trades]) if trades else 0.02
        }

    def _rollback_strategy(self):
        """Reverte para estratégia anterior."""
        if len(self.strategy_history) >= 2:
            # Pega estratégia anterior
            previous = self.strategy_history[-2]
            self.current_strategy = previous['strategy'].copy()
            self.state.strategy_version = previous['version']

            system_logger.warning(f"🔄 ROLLBACK: Revertendo para versão {previous['version']}")
        else:
            # Usa parâmetros de emergência
            self.current_strategy = self.safety.emergency_params.copy()
            self.state.strategy_version = 0

            system_logger.warning("🔄 ROLLBACK: Usando parâmetros de emergência")

    def _update_state(self):
        """Atualiza estado do sistema."""
        # Atualiza PnL diário
        # (seria calculado com base em trades reais)

        # Registra no histórico
        self.performance_history.append({
            'timestamp': datetime.now(),
            'regime': self.state.current_regime,
            'regime_confidence': self.state.regime_confidence,
            'strategy_version': self.state.strategy_version,
            'daily_pnl': self.state.daily_pnl
        })

    def get_trading_decision(self, market_data: Dict,
                            position_data: Optional[Dict] = None) -> Dict:
        """
        Obtém decisão de trading usando todos os módulos.

        Args:
            market_data: Dados atuais de mercado
            position_data: Dados da posição atual (se houver)

        Returns:
            Decisão de trading com recomendações
        """
        decision = {
            'action': 'HOLD',
            'confidence': 0.5,
            'reasons': [],
            'strategy_params': self.current_strategy.copy(),
            'regime': self.state.current_regime
        }

        # 1. Consulta RL Agent
        if self.rl_agent:
            rl_recommendation = self.rl_agent.get_action_recommendation(
                market_data, position_data
            )
            decision['rl_action'] = rl_recommendation['action']
            decision['rl_confidence'] = rl_recommendation['confidence']

            if rl_recommendation['should_execute']:
                decision['reasons'].append(rl_recommendation['reason'])

                if rl_recommendation['action'] == 'BUY':
                    decision['action'] = 'BUY'
                    decision['confidence'] = rl_recommendation['confidence']
                elif rl_recommendation['action'] in ['SELL', 'CLOSE']:
                    decision['action'] = 'SELL'
                    decision['confidence'] = rl_recommendation['confidence']

        # 2. Ajusta confiança baseado em regime
        if self.state.current_regime:
            regime = self.state.current_regime
            if regime == 'bear' and decision['action'] == 'BUY':
                decision['confidence'] *= 0.7
                decision['reasons'].append("Confiança reduzida: mercado em bear")
            elif regime == 'bull' and decision['action'] == 'BUY':
                decision['confidence'] *= 1.2
                decision['reasons'].append("Confiança aumentada: mercado em bull")

        # 3. Aplica parâmetros da estratégia atual
        decision['position_size'] = self.current_strategy['risk_params']['position_size_percent']
        decision['take_profit'] = self.current_strategy['exit_params']['take_profit']
        decision['stop_loss'] = self.current_strategy['exit_params']['stop_loss']

        return decision

    def record_trade_result(self, trade_data: Dict):
        """Registra resultado de trade para aprendizado."""
        if self.meta_learner:
            self.meta_learner.learn_from_experience(trade_data)

        # Atualiza PnL diário
        self.state.daily_pnl += trade_data.get('profit_pct', 0)

        # Atualiza contadores
        if trade_data.get('profit', 0) < 0:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status completo do sistema."""
        status = {
            'state': self.state.optimization_state.value,
            'strategy_version': self.state.strategy_version,
            'current_regime': self.state.current_regime,
            'regime_confidence': self.state.regime_confidence,
            'daily_pnl': self.state.daily_pnl,
            'safety_triggered': self.state.safety_triggered,
            'last_optimization': self.state.last_optimization.isoformat() if self.state.last_optimization else None,
            'optimization_count': self.optimization_count,
            'consecutive_failures': self.consecutive_failures
        }

        # Adiciona estatísticas dos módulos
        if self.rl_agent:
            status['rl_agent'] = self.rl_agent.get_stats()

        if self.meta_learner:
            status['meta_learning'] = self.meta_learner.get_learning_stats()

        if self.optimizer:
            status['optimizer'] = self.optimizer.get_optimization_summary()

        if self.regime_detector:
            status['regime_stats'] = self.regime_detector.get_regime_statistics()

        return status

    def log_system_status(self):
        """Loga status do sistema."""
        status = self.get_system_status()

        system_logger.info(f"\n🤖 AUTO-OPTIMIZED TRADING SYSTEM V4.0")
        system_logger.info("=" * 60)
        system_logger.info(f"   Estado: {status['state']}")
        system_logger.info(f"   Versão estratégia: v{status['strategy_version']}")
        system_logger.info(f"   Regime: {status['current_regime'] or 'N/A'} "
                         f"({status['regime_confidence']*100:.0f}%)")
        system_logger.info(f"   PnL diário: {status['daily_pnl']*100:+.2f}%")
        system_logger.info(f"   Otimizações: {status['optimization_count']}")
        system_logger.info(f"   Safety: {'⚠️ TRIGGERED' if status['safety_triggered'] else '✅ OK'}")
        system_logger.info("=" * 60)

        # Loga módulos individuais
        if self.regime_detector:
            self.regime_detector.log_regime_status()

        if self.rl_agent:
            self.rl_agent.log_agent_status()

        if self.meta_learner:
            self.meta_learner.log_learning_status()

        if self.optimizer:
            self.optimizer.log_optimization_status()

