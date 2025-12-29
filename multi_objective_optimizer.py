"""
V4.0 Phase 4: Multi-Objective Optimizer
Otimização multi-objetivo usando NSGA-II simplificado.
Implementação sem dependências externas de otimização.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import random

from system_logger import system_logger


@dataclass
class Individual:
    """Indivíduo na população."""
    genes: np.ndarray
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    rank: int = 0
    crowding_distance: float = 0.0

    def dominates(self, other: 'Individual') -> bool:
        """Verifica se este indivíduo domina outro (Pareto)."""
        better_in_one = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:  # Maximizando
                return False
            if self.objectives[i] < other.objectives[i]:
                better_in_one = True
        return better_in_one


@dataclass
class OptimizationObjective:
    """Objetivo de otimização."""
    name: str
    weight: float = 1.0
    minimize: bool = True
    target: Optional[float] = None
    bounds: Tuple[float, float] = (0.0, 1.0)


@dataclass
class ParetoSolution:
    """Solução no frente de Pareto."""
    parameters: Dict[str, float]
    objectives: Dict[str, float]
    rank: int
    crowding_distance: float


class MultiObjectiveOptimizer:
    """
    V4.0 Phase 4: Otimizador multi-objetivo.
    Implementa NSGA-II simplificado para encontrar frente de Pareto.
    """

    def __init__(self, objectives: List[OptimizationObjective]):
        """
        Inicializa o otimizador.

        Args:
            objectives: Lista de objetivos a otimizar
        """
        self.objectives = objectives
        self.n_objectives = len(objectives)

        # Configurações do algoritmo genético
        self.population_size = 50
        self.n_generations = 100
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1
        self.mutation_strength = 0.1

        # Resultados
        self.pareto_front: List[Individual] = []
        self.all_solutions: List[Individual] = []
        self.generation_history: List[Dict] = []

        # Espaço de parâmetros
        self.param_bounds: Dict[str, Tuple[float, float]] = {}
        self.param_names: List[str] = []

        system_logger.info(f"MultiObjectiveOptimizer V4.0 inicializado com {len(objectives)} objetivos")

    def set_parameter_space(self, param_bounds: Dict[str, Tuple[float, float]]):
        """Define espaço de parâmetros para otimização."""
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        system_logger.info(f"Espaço de parâmetros: {len(self.param_names)} dimensões")

    def optimize(self, objective_function: Callable[[Dict], Dict[str, float]],
                n_generations: Optional[int] = None) -> Tuple[List[ParetoSolution], np.ndarray]:
        """
        Executa otimização multi-objetivo.

        Args:
            objective_function: Função que recebe parâmetros e retorna valores dos objetivos
            n_generations: Número de gerações (usa default se None)

        Returns:
            Tuple com lista de soluções Pareto e matriz da frente
        """
        if n_generations:
            self.n_generations = n_generations

        system_logger.info(f"\n🎯 INICIANDO OTIMIZAÇÃO MULTI-OBJETIVO")
        system_logger.info(f"   População: {self.population_size}")
        system_logger.info(f"   Gerações: {self.n_generations}")
        system_logger.info(f"   Objetivos: {[o.name for o in self.objectives]}")

        # Inicializa população
        population = self._initialize_population()

        # Avalia população inicial
        self._evaluate_population(population, objective_function)

        # Loop de gerações
        for gen in range(self.n_generations):
            # Seleção e reprodução
            offspring = self._create_offspring(population)

            # Avalia offspring
            self._evaluate_population(offspring, objective_function)

            # Combina população e offspring
            combined = population + offspring

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined)

            # Seleciona próxima geração
            population = self._select_next_generation(fronts)

            # Registra progresso
            best_objectives = self._get_best_objectives(population)
            self.generation_history.append({
                'generation': gen + 1,
                'best_objectives': best_objectives,
                'front_size': len(fronts[0]) if fronts else 0
            })

            if (gen + 1) % 20 == 0:
                system_logger.info(f"   Geração {gen+1}/{self.n_generations}: "
                                 f"Frente Pareto com {len(fronts[0])} soluções")

        # Extrai frente de Pareto final
        self.pareto_front = [ind for ind in population if ind.rank == 0]
        self.all_solutions = population

        # Converte para formato de retorno
        pareto_solutions = self._individuals_to_solutions(self.pareto_front)
        pareto_matrix = np.array([ind.objectives for ind in self.pareto_front])

        system_logger.info(f"\n✅ OTIMIZAÇÃO CONCLUÍDA")
        system_logger.info(f"   Soluções na frente de Pareto: {len(pareto_solutions)}")

        return pareto_solutions, pareto_matrix

    def _initialize_population(self) -> List[Individual]:
        """Inicializa população aleatória."""
        population = []

        for _ in range(self.population_size):
            genes = np.zeros(len(self.param_names))

            for i, param_name in enumerate(self.param_names):
                low, high = self.param_bounds[param_name]
                genes[i] = np.random.uniform(low, high)

            population.append(Individual(genes=genes))

        return population

    def _evaluate_population(self, population: List[Individual],
                            objective_function: Callable):
        """Avalia todos os indivíduos."""
        for ind in population:
            if len(ind.objectives) == 0:
                # Converte genes para parâmetros
                params = self._genes_to_params(ind.genes)

                # Avalia função objetivo
                obj_values = objective_function(params)

                # Armazena objetivos (negativos se minimizando)
                objectives = []
                for obj in self.objectives:
                    value = obj_values.get(obj.name, 0)
                    if obj.minimize:
                        value = -value  # NSGA-II maximiza
                    objectives.append(value)

                ind.objectives = np.array(objectives)

    def _genes_to_params(self, genes: np.ndarray) -> Dict[str, float]:
        """Converte genes para dicionário de parâmetros."""
        return {name: genes[i] for i, name in enumerate(self.param_names)}

    def _params_to_genes(self, params: Dict[str, float]) -> np.ndarray:
        """Converte parâmetros para genes."""
        return np.array([params[name] for name in self.param_names])

    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Ordena população por dominância de Pareto.
        Retorna lista de frentes (frente 0 é a melhor).
        """
        n = len(population)
        domination_count = [0] * n  # Quantos dominam este indivíduo
        dominated_set = [[] for _ in range(n)]  # Quem este indivíduo domina
        fronts = [[]]

        # Calcula dominância
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i], population[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j], population[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        # Primeira frente: não dominados
        for i in range(n):
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(population[i])

        # Frentes seguintes
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for ind in fronts[current_front]:
                idx = population.index(ind)
                for dominated_idx in dominated_set[idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        population[dominated_idx].rank = current_front + 1
                        next_front.append(population[dominated_idx])

            current_front += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove última frente vazia

    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Verifica se ind1 domina ind2."""
        better_in_one = False
        for i in range(len(ind1.objectives)):
            if ind1.objectives[i] < ind2.objectives[i]:  # Estamos maximizando
                return False
            if ind1.objectives[i] > ind2.objectives[i]:
                better_in_one = True
        return better_in_one

    def _calculate_crowding_distance(self, front: List[Individual]):
        """Calcula crowding distance para uma frente."""
        n = len(front)
        if n <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return

        for ind in front:
            ind.crowding_distance = 0

        # Para cada objetivo
        for m in range(self.n_objectives):
            # Ordena por este objetivo
            front.sort(key=lambda x: x.objectives[m])

            # Extremos têm distância infinita
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calcula para os outros
            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0:
                continue

            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    front[i + 1].objectives[m] - front[i - 1].objectives[m]
                ) / obj_range

    def _select_next_generation(self, fronts: List[List[Individual]]) -> List[Individual]:
        """Seleciona próxima geração usando crowding distance."""
        next_gen = []

        for front in fronts:
            self._calculate_crowding_distance(front)

            if len(next_gen) + len(front) <= self.population_size:
                next_gen.extend(front)
            else:
                # Precisa selecionar parte da frente
                remaining = self.population_size - len(next_gen)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                next_gen.extend(front[:remaining])
                break

        return next_gen

    def _create_offspring(self, population: List[Individual]) -> List[Individual]:
        """Cria offspring por crossover e mutação."""
        offspring = []

        while len(offspring) < self.population_size:
            # Seleção por torneio
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            # Crossover
            if random.random() < self.crossover_prob:
                child1_genes, child2_genes = self._sbx_crossover(
                    parent1.genes, parent2.genes
                )
            else:
                child1_genes = parent1.genes.copy()
                child2_genes = parent2.genes.copy()

            # Mutação
            if random.random() < self.mutation_prob:
                child1_genes = self._polynomial_mutation(child1_genes)
            if random.random() < self.mutation_prob:
                child2_genes = self._polynomial_mutation(child2_genes)

            offspring.append(Individual(genes=child1_genes))
            if len(offspring) < self.population_size:
                offspring.append(Individual(genes=child2_genes))

        return offspring

    def _tournament_select(self, population: List[Individual],
                          tournament_size: int = 2) -> Individual:
        """Seleção por torneio binário."""
        candidates = random.sample(population, tournament_size)

        # Compara por rank, depois crowding distance
        best = candidates[0]
        for candidate in candidates[1:]:
            if candidate.rank < best.rank:
                best = candidate
            elif candidate.rank == best.rank and \
                 candidate.crowding_distance > best.crowding_distance:
                best = candidate

        return best

    def _sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                      eta: float = 15) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        n = len(parent1)
        child1 = np.zeros(n)
        child2 = np.zeros(n)

        for i in range(n):
            if random.random() > 0.5:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:
                u = random.random()
                if u < 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

                # Aplica bounds
                low, high = list(self.param_bounds.values())[i]
                child1[i] = np.clip(child1[i], low, high)
                child2[i] = np.clip(child2[i], low, high)

        return child1, child2

    def _polynomial_mutation(self, genes: np.ndarray, eta: float = 20) -> np.ndarray:
        """Polynomial Mutation."""
        mutated = genes.copy()

        for i in range(len(genes)):
            if random.random() < 1.0 / len(genes):
                low, high = list(self.param_bounds.values())[i]

                delta = (genes[i] - low) / (high - low)
                u = random.random()

                if u < 0.5:
                    delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                mutated[i] = genes[i] + delta_q * (high - low)
                mutated[i] = np.clip(mutated[i], low, high)

        return mutated

    def _get_best_objectives(self, population: List[Individual]) -> Dict[str, float]:
        """Retorna melhores valores de cada objetivo."""
        best = {}
        for i, obj in enumerate(self.objectives):
            values = [ind.objectives[i] for ind in population]
            # Reverte se estávamos minimizando
            if obj.minimize:
                best[obj.name] = -min(values)
            else:
                best[obj.name] = max(values)
        return best

    def _individuals_to_solutions(self, individuals: List[Individual]) -> List[ParetoSolution]:
        """Converte indivíduos para soluções Pareto."""
        solutions = []

        for ind in individuals:
            params = self._genes_to_params(ind.genes)

            objectives = {}
            for i, obj in enumerate(self.objectives):
                value = ind.objectives[i]
                if obj.minimize:
                    value = -value  # Reverte
                objectives[obj.name] = value

            solutions.append(ParetoSolution(
                parameters=params,
                objectives=objectives,
                rank=ind.rank,
                crowding_distance=ind.crowding_distance
            ))

        return solutions

    def select_best_solution(self, weights: Optional[Dict[str, float]] = None) -> ParetoSolution:
        """
        Seleciona melhor solução da frente de Pareto.

        Args:
            weights: Pesos para cada objetivo (usa default se None)
        """
        if not self.pareto_front:
            raise ValueError("Nenhuma solução Pareto disponível. Execute optimize() primeiro.")

        if weights is None:
            weights = {obj.name: obj.weight for obj in self.objectives}

        solutions = self._individuals_to_solutions(self.pareto_front)

        best_score = float('-inf')
        best_solution = solutions[0]

        for solution in solutions:
            score = 0
            for obj_name, weight in weights.items():
                # Normaliza objetivo
                obj = next((o for o in self.objectives if o.name == obj_name), None)
                if obj:
                    value = solution.objectives.get(obj_name, 0)
                    normalized = (value - obj.bounds[0]) / (obj.bounds[1] - obj.bounds[0] + 0.001)
                    score += normalized * weight

            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution

    def analyze_tradeoffs(self) -> Dict[str, Any]:
        """Analisa trade-offs entre objetivos."""
        if not self.pareto_front:
            return {'error': 'Nenhuma solução disponível'}

        pareto_matrix = np.array([ind.objectives for ind in self.pareto_front])

        analysis = {
            'n_solutions': len(self.pareto_front),
            'objective_ranges': {},
            'correlations': {},
            'tradeoffs': []
        }

        # Ranges de cada objetivo
        for i, obj in enumerate(self.objectives):
            values = pareto_matrix[:, i]
            if obj.minimize:
                values = -values
            analysis['objective_ranges'][obj.name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

        # Correlações entre objetivos
        for i in range(self.n_objectives):
            for j in range(i + 1, self.n_objectives):
                obj1 = self.objectives[i].name
                obj2 = self.objectives[j].name

                corr = np.corrcoef(pareto_matrix[:, i], pareto_matrix[:, j])[0, 1]

                analysis['correlations'][f"{obj1}_vs_{obj2}"] = float(corr)

                # Interpreta correlação
                if corr < -0.5:
                    tradeoff = "Forte trade-off (conflitantes)"
                elif corr > 0.5:
                    tradeoff = "Sinergia (complementares)"
                else:
                    tradeoff = "Fracamente relacionados"

                analysis['tradeoffs'].append({
                    'objectives': (obj1, obj2),
                    'correlation': float(corr),
                    'interpretation': tradeoff
                })

        return analysis

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Retorna resumo da otimização."""
        summary = {
            'objectives': [obj.name for obj in self.objectives],
            'n_generations': len(self.generation_history),
            'population_size': self.population_size,
            'pareto_front_size': len(self.pareto_front),
            'total_solutions': len(self.all_solutions)
        }

        if self.pareto_front:
            best = self.select_best_solution()
            summary['best_solution'] = {
                'parameters': best.parameters,
                'objectives': best.objectives
            }

        if self.generation_history:
            summary['final_best_objectives'] = self.generation_history[-1]['best_objectives']

        return summary

    def log_optimization_results(self):
        """Loga resultados da otimização."""
        summary = self.get_optimization_summary()
        analysis = self.analyze_tradeoffs()

        system_logger.info(f"\n🎯 RESULTADOS MULTI-OBJETIVO:")
        system_logger.info(f"   Gerações: {summary['n_generations']}")
        system_logger.info(f"   Soluções Pareto: {summary['pareto_front_size']}")

        if 'best_solution' in summary:
            system_logger.info(f"\n   📊 Melhor solução:")
            for obj, value in summary['best_solution']['objectives'].items():
                system_logger.info(f"      {obj}: {value:.4f}")

        if analysis.get('tradeoffs'):
            system_logger.info(f"\n   🔄 Trade-offs:")
            for tradeoff in analysis['tradeoffs'][:3]:
                system_logger.info(f"      {tradeoff['objectives'][0]} vs {tradeoff['objectives'][1]}: "
                                 f"{tradeoff['interpretation']}")

