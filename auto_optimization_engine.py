"""
V4.0 Phase 4: Auto-Optimization Engine
Motor de otimização automática de parâmetros usando diferentes algoritmos.
Suporta Bayesian, Genetic Algorithm e Random Search.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json

from system_logger import system_logger


@dataclass
class OptimizationResult:
    """Resultado de uma otimização."""
    best_params: Dict[str, Any]
    best_score: float
    iterations: int
    method: str
    convergence_history: List[float]
    timestamp: datetime
    duration_seconds: float


@dataclass
class ParameterBounds:
    """Limites para um parâmetro otimizável."""
    min_val: float
    max_val: float
    step: float = 0.1
    is_integer: bool = False


@dataclass
class OptimizationObjective:
    """Objetivo de otimização."""
    name: str
    target: float
    weight: float
    min_val: float
    max_val: float
    maximize: bool = True


class FastBacktester:
    """Backtester rápido para avaliação de parâmetros."""

    def __init__(self, historical_data: List[Dict], entry_params: Dict,
                 exit_params: Dict, risk_params: Dict):
        self.data = historical_data
        self.entry_params = entry_params
        self.exit_params = exit_params
        self.risk_params = risk_params

    def run(self, days: int = 30) -> Dict[str, Any]:
        """Executa simulação rápida."""
        trades = []
        equity_curve = [1000.0]  # Capital inicial

        # Simula trades baseado nos parâmetros
        score_threshold = self.entry_params.get('score_threshold', 40)
        take_profit = self.exit_params.get('take_profit', 2.0) / 100
        stop_loss = self.exit_params.get('stop_loss', 1.0) / 100
        position_size = self.risk_params.get('position_size_percent', 0.15)

        for i, candle in enumerate(self.data[-days*24:]):  # ~24 candles por dia
            # Simula entrada baseada em score
            simulated_score = candle.get('score', 50) + np.random.normal(0, 10)

            if simulated_score >= score_threshold:
                # Simula resultado do trade
                price_change = np.random.normal(0.001, 0.02)  # Volatilidade típica

                if price_change >= take_profit:
                    profit = take_profit * position_size
                    result = 'win'
                elif price_change <= -stop_loss:
                    profit = -stop_loss * position_size
                    result = 'loss'
                else:
                    profit = price_change * position_size
                    result = 'win' if profit > 0 else 'loss'

                trades.append({
                    'entry_time': datetime.now() - timedelta(hours=len(self.data) - i),
                    'profit_pct': profit,
                    'result': result,
                    'duration_minutes': np.random.randint(10, 60)
                })

                equity_curve.append(equity_curve[-1] * (1 + profit))

        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_equity': equity_curve[-1] if equity_curve else 1000.0,
            'initial_equity': 1000.0
        }


class AutoOptimizationEngine:
    """
    V4.0 Phase 4: Motor de auto-otimização.
    Otimiza parâmetros usando diferentes algoritmos.
    """

    def __init__(self, historical_data: Optional[List[Dict]] = None):
        """Inicializa o motor de otimização."""
        self.historical_data = historical_data or []
        self.best_params: Dict[str, Any] = {}
        self.optimization_history: List[OptimizationResult] = []
        self.convergence_threshold = 0.001
        self.max_iterations_without_improvement = 20

        # Espaço de parâmetros otimizáveis
        self.parameter_space = {
            'entry_params': {
                'score_threshold': ParameterBounds(20, 50, 1, True),
                'volume_ratio_min': ParameterBounds(1.5, 5.0, 0.1),
                'rsi_oversold': ParameterBounds(25, 40, 1, True),
                'rsi_overbought': ParameterBounds(60, 75, 1, True),
                'pattern_score_min': ParameterBounds(60, 90, 1, True),
            },
            'exit_params': {
                'take_profit': ParameterBounds(1.0, 4.0, 0.1),
                'stop_loss': ParameterBounds(0.5, 2.0, 0.1),
                'trailing_stop_activation': ParameterBounds(0.5, 2.0, 0.1),
                'trailing_stop_distance': ParameterBounds(0.2, 1.0, 0.1),
                'timeout_minutes': ParameterBounds(15, 60, 5, True),
            },
            'risk_params': {
                'position_size_percent': ParameterBounds(0.05, 0.25, 0.01),
                'max_daily_loss': ParameterBounds(0.01, 0.05, 0.01),
                'kelly_fraction': ParameterBounds(0.1, 0.5, 0.05),
            }
        }

        # Objetivos de otimização
        self.objectives = [
            OptimizationObjective('daily_return', 0.05, 0.35, 0.02, 0.15),
            OptimizationObjective('max_drawdown', -0.03, 0.25, -0.10, -0.01, maximize=False),
            OptimizationObjective('sharpe_ratio', 2.5, 0.20, 1.0, 5.0),
            OptimizationObjective('win_rate', 0.70, 0.15, 0.50, 0.90),
            OptimizationObjective('profit_factor', 2.5, 0.05, 1.2, 5.0),
        ]

        # Cache de avaliações
        self._evaluation_cache: Dict[str, Dict] = {}

        system_logger.info("AutoOptimizationEngine V4.0 inicializado")

    def run_optimization(self, method: str = 'bayesian',
                        n_iterations: int = 50) -> OptimizationResult:
        """
        Executa otimização usando método especificado.

        Args:
            method: 'bayesian', 'genetic', 'random_search', ou 'gradient'
            n_iterations: Número máximo de iterações
        """
        start_time = datetime.now()

        system_logger.info(f"\n🔧 INICIANDO OTIMIZAÇÃO ({method.upper()})")
        system_logger.info(f"   Iterações: {n_iterations}")

        if method == 'bayesian':
            best_params, best_score, history = self._bayesian_optimization(n_iterations)
        elif method == 'genetic':
            best_params, best_score, history = self._genetic_algorithm(n_iterations)
        elif method == 'random_search':
            best_params, best_score, history = self._random_search(n_iterations)
        else:
            raise ValueError(f"Método desconhecido: {method}")

        duration = (datetime.now() - start_time).total_seconds()

        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            iterations=len(history),
            method=method,
            convergence_history=history,
            timestamp=datetime.now(),
            duration_seconds=duration
        )

        self.best_params = best_params
        self.optimization_history.append(result)

        system_logger.info(f"\n✅ OTIMIZAÇÃO CONCLUÍDA")
        system_logger.info(f"   Melhor score: {best_score:.4f}")
        system_logger.info(f"   Duração: {duration:.1f}s")

        return result

    def _bayesian_optimization(self, n_iterations: int) -> Tuple[Dict, float, List[float]]:
        """
        Otimização Bayesiana usando Gaussian Process surrogate.
        Implementação simplificada sem dependências externas.
        """
        # Inicializa com random sampling
        best_params = self._sample_random_params()
        best_score = self._evaluate_params(best_params)
        history = [best_score]

        # Histórico para construir surrogate
        X_observed = [self._params_to_vector(best_params)]
        y_observed = [best_score]

        no_improvement_count = 0

        for i in range(n_iterations):
            # Amostra candidatos e escolhe o mais promissor
            candidates = [self._sample_random_params() for _ in range(10)]

            # Usa histórico para guiar busca (surrogate simplificado)
            if len(X_observed) > 5:
                candidate_scores = []
                for candidate in candidates:
                    # Estima score baseado em similaridade com pontos anteriores
                    estimated_score = self._estimate_score(
                        candidate, X_observed, y_observed
                    )
                    # Adiciona exploração
                    exploration_bonus = np.random.uniform(0, 0.1)
                    candidate_scores.append(estimated_score + exploration_bonus)

                best_candidate_idx = np.argmax(candidate_scores)
                candidate = candidates[best_candidate_idx]
            else:
                # Fase inicial: exploração pura
                candidate = candidates[0]

            # Avalia candidato
            score = self._evaluate_params(candidate)

            X_observed.append(self._params_to_vector(candidate))
            y_observed.append(score)

            if score > best_score:
                best_score = score
                best_params = candidate
                no_improvement_count = 0
                system_logger.info(f"   Iteração {i+1}: Novo melhor score = {best_score:.4f}")
            else:
                no_improvement_count += 1

            history.append(best_score)

            # Early stopping
            if no_improvement_count >= self.max_iterations_without_improvement:
                system_logger.info(f"   Early stopping após {i+1} iterações")
                break

        return best_params, best_score, history

    def _genetic_algorithm(self, n_iterations: int) -> Tuple[Dict, float, List[float]]:
        """
        Algoritmo genético para otimização.
        """
        population_size = 20
        mutation_rate = 0.1
        crossover_rate = 0.7
        elite_size = 2

        # Inicializa população
        population = [self._sample_random_params() for _ in range(population_size)]
        fitness = [self._evaluate_params(p) for p in population]

        best_idx = np.argmax(fitness)
        best_params = population[best_idx]
        best_score = fitness[best_idx]
        history = [best_score]

        for gen in range(n_iterations):
            # Seleção por torneio
            new_population = []

            # Elitismo: mantém os melhores
            sorted_indices = np.argsort(fitness)[::-1]
            for i in range(elite_size):
                new_population.append(population[sorted_indices[i]])

            # Crossover e mutação
            while len(new_population) < population_size:
                # Seleção por torneio
                parent1 = self._tournament_select(population, fitness)
                parent2 = self._tournament_select(population, fitness)

                # Crossover
                if np.random.random() < crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutação
                if np.random.random() < mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population
            fitness = [self._evaluate_params(p) for p in population]

            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_score:
                best_score = fitness[gen_best_idx]
                best_params = population[gen_best_idx]
                system_logger.info(f"   Geração {gen+1}: Novo melhor score = {best_score:.4f}")

            history.append(best_score)

        return best_params, best_score, history

    def _random_search(self, n_iterations: int) -> Tuple[Dict, float, List[float]]:
        """
        Busca aleatória simples.
        """
        best_params = self._sample_random_params()
        best_score = self._evaluate_params(best_params)
        history = [best_score]

        for i in range(n_iterations):
            candidate = self._sample_random_params()
            score = self._evaluate_params(candidate)

            if score > best_score:
                best_score = score
                best_params = candidate
                system_logger.info(f"   Iteração {i+1}: Novo melhor score = {best_score:.4f}")

            history.append(best_score)

        return best_params, best_score, history

    def _sample_random_params(self) -> Dict[str, Dict[str, float]]:
        """Amostra parâmetros aleatórios dentro dos limites."""
        params = {}

        for category, param_bounds in self.parameter_space.items():
            params[category] = {}
            for param_name, bounds in param_bounds.items():
                if bounds.is_integer:
                    value = np.random.randint(int(bounds.min_val), int(bounds.max_val) + 1)
                else:
                    value = np.random.uniform(bounds.min_val, bounds.max_val)
                    # Arredonda para o step
                    value = round(value / bounds.step) * bounds.step
                params[category][param_name] = value

        return params

    def _params_to_vector(self, params: Dict) -> List[float]:
        """Converte parâmetros para vetor."""
        vector = []
        for category in sorted(params.keys()):
            for param_name in sorted(params[category].keys()):
                vector.append(params[category][param_name])
        return vector

    def _vector_to_params(self, vector: List[float]) -> Dict:
        """Converte vetor para parâmetros."""
        params = {}
        idx = 0
        for category in sorted(self.parameter_space.keys()):
            params[category] = {}
            for param_name in sorted(self.parameter_space[category].keys()):
                params[category][param_name] = vector[idx]
                idx += 1
        return params

    def _estimate_score(self, candidate: Dict, X_observed: List,
                       y_observed: List) -> float:
        """Estima score usando interpolação simples (surrogate)."""
        candidate_vec = np.array(self._params_to_vector(candidate))
        X = np.array(X_observed)
        y = np.array(y_observed)

        # Calcula distâncias
        distances = np.linalg.norm(X - candidate_vec, axis=1)

        # Peso inversamente proporcional à distância
        weights = 1.0 / (distances + 0.001)
        weights /= weights.sum()

        # Score ponderado
        estimated = np.sum(weights * y)

        return estimated

    def _evaluate_params(self, params: Dict, backtest_days: int = 30) -> float:
        """Avalia parâmetros usando backtesting."""
        # Cria chave para cache
        cache_key = json.dumps(params, sort_keys=True)
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]['composite_score']

        # Executa simulação
        backtester = FastBacktester(
            historical_data=self.historical_data,
            entry_params=params.get('entry_params', {}),
            exit_params=params.get('exit_params', {}),
            risk_params=params.get('risk_params', {})
        )

        results = backtester.run(backtest_days)

        # Calcula métricas
        metrics = self._calculate_metrics(results)

        # Calcula score composto
        composite_score = self._calculate_composite_score(metrics)

        # Cache
        self._evaluation_cache[cache_key] = {
            'metrics': metrics,
            'composite_score': composite_score
        }

        return composite_score

    def _calculate_metrics(self, results: Dict) -> Dict[str, float]:
        """Calcula métricas de performance."""
        trades = results['trades']
        equity = results['equity_curve']

        if not trades:
            return {
                'daily_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'profit_factor': 1.0,
                'total_trades': 0
            }

        # Daily return
        total_return = (equity[-1] - equity[0]) / equity[0]
        days = max(1, len(equity) / 24)
        daily_return = total_return / days

        # Max drawdown
        peak = equity[0]
        max_dd = 0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        # Win rate
        wins = sum(1 for t in trades if t['result'] == 'win')
        win_rate = wins / len(trades) if trades else 0

        # Profit factor
        gross_profit = sum(t['profit_pct'] for t in trades if t['profit_pct'] > 0)
        gross_loss = abs(sum(t['profit_pct'] for t in trades if t['profit_pct'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0

        # Sharpe ratio (simplificado)
        returns = [t['profit_pct'] for t in trades]
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 0.001) * np.sqrt(252)
        else:
            sharpe = 0

        return {
            'daily_return': daily_return,
            'max_drawdown': -max_dd,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades)
        }

    def _calculate_composite_score(self, metrics: Dict) -> float:
        """Calcula score composto dos objetivos."""
        total_score = 0

        for objective in self.objectives:
            metric_value = metrics.get(objective.name, 0)

            # Normaliza entre min e max
            normalized = (metric_value - objective.min_val) / \
                        (objective.max_val - objective.min_val + 0.001)
            normalized = max(0, min(1, normalized))

            # Inverte se queremos minimizar
            if not objective.maximize:
                normalized = 1 - normalized

            total_score += normalized * objective.weight

        return total_score

    def _tournament_select(self, population: List, fitness: List,
                          tournament_size: int = 3) -> Dict:
        """Seleção por torneio para algoritmo genético."""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = indices[np.argmax([fitness[i] for i in indices])]
        return population[best_idx].copy()

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover de dois parâmetros."""
        child = {}
        for category in parent1.keys():
            child[category] = {}
            for param in parent1[category].keys():
                # 50% chance de herdar de cada pai
                if np.random.random() < 0.5:
                    child[category][param] = parent1[category][param]
                else:
                    child[category][param] = parent2[category][param]
        return child

    def _mutate(self, params: Dict) -> Dict:
        """Mutação de parâmetros."""
        mutated = {}
        for category, param_dict in params.items():
            mutated[category] = {}
            for param_name, value in param_dict.items():
                bounds = self.parameter_space[category][param_name]

                # 20% chance de mutar cada parâmetro
                if np.random.random() < 0.2:
                    # Mutação gaussiana
                    std = (bounds.max_val - bounds.min_val) * 0.1
                    new_value = value + np.random.normal(0, std)
                    new_value = max(bounds.min_val, min(bounds.max_val, new_value))

                    if bounds.is_integer:
                        new_value = int(round(new_value))
                    else:
                        new_value = round(new_value / bounds.step) * bounds.step

                    mutated[category][param_name] = new_value
                else:
                    mutated[category][param_name] = value

        return mutated

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Retorna resumo das otimizações."""
        if not self.optimization_history:
            return {'total_optimizations': 0}

        recent = self.optimization_history[-1]

        return {
            'total_optimizations': len(self.optimization_history),
            'best_score_ever': max(o.best_score for o in self.optimization_history),
            'latest_score': recent.best_score,
            'latest_method': recent.method,
            'latest_timestamp': recent.timestamp.isoformat(),
            'avg_duration': np.mean([o.duration_seconds for o in self.optimization_history]),
            'best_params': self.best_params
        }

    def log_optimization_status(self):
        """Loga status da otimização."""
        summary = self.get_optimization_summary()

        system_logger.info(f"\n🔧 OTIMIZAÇÃO STATUS:")
        system_logger.info(f"   Total otimizações: {summary['total_optimizations']}")
        if summary['total_optimizations'] > 0:
            system_logger.info(f"   Melhor score: {summary['best_score_ever']:.4f}")
            system_logger.info(f"   Último método: {summary['latest_method']}")
            system_logger.info(f"   Duração média: {summary['avg_duration']:.1f}s")

