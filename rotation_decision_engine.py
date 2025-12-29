"""
V4.0 Phase 3: Rotation Decision Engine
Motor de decisão para avaliação de cenários de rotação.
Integra todos os módulos Phase 3 para decisões otimizadas.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from system_logger import system_logger


class RotationScenario(Enum):
    """Tipos de cenário de rotação."""
    HIGH_CONFIDENCE = "high_confidence"      # Alta confiança - executar
    MODERATE = "moderate"                     # Moderada - avaliar condições
    DEFENSIVE = "defensive"                   # Defensiva - reduzir risco
    OPPORTUNITY = "opportunity"               # Oportunidade - novo ativo forte
    NO_ACTION = "no_action"                   # Sem ação necessária


class RotationPriority(Enum):
    """Prioridade de rotação."""
    CRITICAL = 1    # Executar imediatamente
    HIGH = 2        # Executar no próximo ciclo
    MEDIUM = 3      # Executar quando conveniente
    LOW = 4         # Opcional
    NONE = 5        # Não executar


@dataclass
class RotationDecision:
    """Resultado de decisão de rotação."""
    scenario: RotationScenario
    priority: RotationPriority
    should_rotate: bool
    exit_symbol: Optional[str]
    entry_symbol: Optional[str]
    expected_improvement: float
    confidence_score: float
    reasons: List[str]
    risk_factors: List[str]
    recommended_allocation: float


@dataclass
class ScenarioAnalysis:
    """Análise de cenário para decisão."""
    position_quality: float          # 0-100
    candidate_quality: float         # 0-100
    market_condition: float          # 0-100
    timing_score: float              # 0-100
    risk_level: float                # 0-100 (maior = mais arriscado)
    opportunity_cost: float          # Custo de não rotacionar


class RotationDecisionEngine:
    """
    V4.0 Phase 3: Motor de decisão para rotações.
    Analisa múltiplos fatores para decisões otimizadas.
    """

    def __init__(self):
        """Inicializa o motor de decisão."""
        # Thresholds para decisões
        self.min_improvement_threshold = 15.0      # 15% melhoria mínima
        self.high_confidence_threshold = 25.0      # 25% = alta confiança
        self.defensive_pnl_threshold = -1.0        # -1% aciona defensiva
        self.opportunity_score_threshold = 80      # Score 80+ = oportunidade

        # Pesos para cálculo de decisão
        self.weights = {
            'improvement': 0.35,      # Melhoria esperada
            'candidate_quality': 0.25, # Qualidade do candidato
            'position_quality': 0.20,  # Qualidade posição atual
            'timing': 0.10,            # Timing de mercado
            'risk': 0.10               # Fator de risco
        }

        # Multiplicadores de cenário
        self.scenario_multipliers = {
            RotationScenario.HIGH_CONFIDENCE: 1.2,
            RotationScenario.MODERATE: 1.0,
            RotationScenario.DEFENSIVE: 0.8,
            RotationScenario.OPPORTUNITY: 1.1,
            RotationScenario.NO_ACTION: 0.0
        }

        # Histórico de decisões
        self.decision_history: List[Dict] = []
        self.successful_rotations = 0
        self.failed_rotations = 0

        system_logger.info("RotationDecisionEngine V4.0 inicializado")

    def evaluate_rotation_scenario(
        self,
        current_position: Dict,
        candidate_signal: Dict,
        current_price: float,
        market_metrics: Optional[Dict] = None
    ) -> RotationDecision:
        """
        Avalia cenário de rotação e retorna decisão.
        """
        # Análise de cenário
        analysis = self._analyze_scenario(
            current_position, candidate_signal, current_price, market_metrics
        )

        # Determina tipo de cenário
        scenario = self._determine_scenario(analysis, current_position, candidate_signal)

        # Calcula melhoria esperada
        expected_improvement = self.calculate_expected_improvement(
            current_position, candidate_signal, current_price
        )

        # Calcula confiança
        confidence = self._calculate_confidence(analysis, expected_improvement)

        # Determina prioridade
        priority = self._determine_priority(scenario, confidence, expected_improvement)

        # Decisão final
        should_rotate = self._should_execute_rotation(
            scenario, priority, expected_improvement, confidence
        )

        # Coleta razões e riscos
        reasons, risk_factors = self._collect_reasons_and_risks(
            analysis, scenario, expected_improvement
        )

        # Calcula alocação recomendada
        recommended_allocation = self._calculate_recommended_allocation(
            candidate_signal, confidence, analysis.risk_level
        )

        decision = RotationDecision(
            scenario=scenario,
            priority=priority,
            should_rotate=should_rotate,
            exit_symbol=current_position.get('symbol'),
            entry_symbol=candidate_signal.get('symbol'),
            expected_improvement=expected_improvement,
            confidence_score=confidence,
            reasons=reasons,
            risk_factors=risk_factors,
            recommended_allocation=recommended_allocation
        )

        # Registra decisão
        self._record_decision(decision)

        return decision

    def evaluate_opportunity_rotation(
        self,
        candidate_signal: Dict,
        available_capital: float,
        current_positions: Dict[str, Dict],
        market_metrics: Optional[Dict] = None
    ) -> RotationDecision:
        """
        Avalia oportunidade de nova entrada (sem fechar posição).
        """
        symbol = candidate_signal.get('symbol', '')

        # Verifica se já temos posição
        if symbol in current_positions:
            return RotationDecision(
                scenario=RotationScenario.NO_ACTION,
                priority=RotationPriority.NONE,
                should_rotate=False,
                exit_symbol=None,
                entry_symbol=symbol,
                expected_improvement=0,
                confidence_score=0,
                reasons=["Já possui posição no ativo"],
                risk_factors=[],
                recommended_allocation=0
            )

        # Score do candidato
        candidate_score = candidate_signal.get('phase2_score', 0) or candidate_signal.get('total_score', 0)

        # Análise de oportunidade
        is_strong_opportunity = candidate_score >= self.opportunity_score_threshold

        # Timing
        timing_score = self._get_timing_score(market_metrics)

        # Risco de concentração
        num_positions = len(current_positions)
        concentration_risk = num_positions >= 4  # Já temos muitas posições

        # Calcula confiança
        confidence = min(100, candidate_score * 0.8 + timing_score * 0.2)

        # Determina cenário
        if is_strong_opportunity and not concentration_risk:
            scenario = RotationScenario.OPPORTUNITY
            priority = RotationPriority.HIGH if candidate_score >= 85 else RotationPriority.MEDIUM
            should_enter = available_capital >= 50  # Mínimo $50
        else:
            scenario = RotationScenario.NO_ACTION
            priority = RotationPriority.NONE
            should_enter = False

        # Razões
        reasons = []
        risk_factors = []

        if is_strong_opportunity:
            reasons.append(f"Score alto: {candidate_score:.0f}")
        if timing_score >= 70:
            reasons.append(f"Timing favorável: {timing_score:.0f}")

        if concentration_risk:
            risk_factors.append(f"Muitas posições abertas: {num_positions}")
        if available_capital < 100:
            risk_factors.append(f"Capital limitado: ${available_capital:.2f}")

        # Alocação
        recommended_allocation = 0
        if should_enter:
            # 15-25% do capital disponível baseado em score
            base_pct = 0.15 + (candidate_score - 80) / 100 * 0.10
            recommended_allocation = min(available_capital * 0.25, available_capital * base_pct)

        return RotationDecision(
            scenario=scenario,
            priority=priority,
            should_rotate=should_enter,
            exit_symbol=None,
            entry_symbol=symbol,
            expected_improvement=candidate_score - 50,  # Melhoria vs média
            confidence_score=confidence,
            reasons=reasons,
            risk_factors=risk_factors,
            recommended_allocation=recommended_allocation
        )

    def calculate_expected_improvement(
        self,
        current_position: Dict,
        candidate_signal: Dict,
        current_price: float
    ) -> float:
        """
        Calcula melhoria esperada de rotação em percentual.
        Considera: scores, P&L atual, potencial do candidato.
        """
        # P&L atual da posição
        entry_price = current_position.get('entry_price', current_price)
        current_pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

        # Scores
        position_score = current_position.get('score', 50)
        candidate_score = candidate_signal.get('phase2_score', 0) or candidate_signal.get('total_score', 0)

        # Diferença de score (normalizada)
        score_diff = (candidate_score - position_score) / 100 * 50  # Max 50%

        # Potencial de TP
        candidate_tp = candidate_signal.get('phase2_tp', 0.02) * 100  # Converte para %
        position_remaining_tp = current_position.get('take_profit_pct', 2.0) - current_pnl_pct
        tp_diff = candidate_tp - max(0, position_remaining_tp)

        # Risco relativo
        candidate_sl = candidate_signal.get('phase2_sl', 0.01) * 100
        position_sl = abs(current_position.get('stop_loss_pct', 1.0))
        risk_improvement = position_sl - candidate_sl  # Menos SL = melhor

        # Momentum
        candidate_momentum = candidate_signal.get('momentum_score', 50)
        position_momentum = current_position.get('momentum_score', 50)
        momentum_diff = (candidate_momentum - position_momentum) / 100 * 20  # Max 20%

        # Calcula melhoria total
        improvement = (
            score_diff * 0.4 +
            tp_diff * 0.3 +
            risk_improvement * 0.15 +
            momentum_diff * 0.15
        )

        # Penaliza se posição atual está lucrativa
        if current_pnl_pct > 0.5:
            improvement -= current_pnl_pct * 0.3  # Penaliza por lucro não realizado

        # Bônus se posição atual está negativa
        if current_pnl_pct < -0.5:
            improvement += abs(current_pnl_pct) * 0.2  # Incentiva sair de perdedora

        return improvement

    def find_best_rotation_pair(
        self,
        positions: Dict[str, Dict],
        candidates: List[Dict],
        price_map: Dict[str, float]
    ) -> Optional[RotationDecision]:
        """
        Encontra o melhor par posição-candidato para rotação.
        """
        best_decision: Optional[RotationDecision] = None
        best_score = 0

        for symbol, position in positions.items():
            current_price = price_map.get(symbol, 0)
            if current_price <= 0:
                continue

            for candidate in candidates:
                if candidate.get('symbol') == symbol:
                    continue

                decision = self.evaluate_rotation_scenario(
                    position, candidate, current_price
                )

                if not decision.should_rotate:
                    continue

                # Score combinado
                combined_score = (
                    decision.expected_improvement * 0.5 +
                    decision.confidence_score * 0.3 +
                    (100 - decision.priority.value * 20) * 0.2
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_decision = decision

        return best_decision

    def _analyze_scenario(
        self,
        position: Dict,
        candidate: Dict,
        current_price: float,
        market_metrics: Optional[Dict]
    ) -> ScenarioAnalysis:
        """Analisa cenário completo."""
        # Qualidade da posição atual
        entry_price = position.get('entry_price', current_price)
        pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        position_score = position.get('score', 50)
        position_momentum = position.get('momentum_score', 50)

        position_quality = (
            position_score * 0.4 +
            position_momentum * 0.3 +
            (50 + pnl_pct * 10) * 0.3  # P&L contribui
        )
        position_quality = max(0, min(100, position_quality))

        # Qualidade do candidato
        candidate_score = candidate.get('phase2_score', 0) or candidate.get('total_score', 0)
        candidate_momentum = candidate.get('momentum_score', 50)
        candidate_volume = candidate.get('volume_score', 50)

        candidate_quality = (
            candidate_score * 0.5 +
            candidate_momentum * 0.3 +
            candidate_volume * 0.2
        )

        # Condição de mercado
        market_condition = 60  # Default neutro
        if market_metrics:
            market_condition = market_metrics.get('overall_score', 60)

        # Timing
        timing_score = self._get_timing_score(market_metrics)

        # Risco
        risk_level = self._calculate_risk_level(position, candidate, pnl_pct)

        # Custo de oportunidade
        opportunity_cost = max(0, candidate_quality - position_quality)

        return ScenarioAnalysis(
            position_quality=position_quality,
            candidate_quality=candidate_quality,
            market_condition=market_condition,
            timing_score=timing_score,
            risk_level=risk_level,
            opportunity_cost=opportunity_cost
        )

    def _determine_scenario(
        self,
        analysis: ScenarioAnalysis,
        position: Dict,
        candidate: Dict
    ) -> RotationScenario:
        """Determina tipo de cenário de rotação."""
        # Calcula métricas chave
        quality_diff = analysis.candidate_quality - analysis.position_quality

        # Alta confiança: grande diferença de qualidade + timing bom
        if quality_diff >= 30 and analysis.timing_score >= 60:
            return RotationScenario.HIGH_CONFIDENCE

        # Defensiva: posição ruim ou risco alto
        if analysis.position_quality < 30 or analysis.risk_level > 70:
            return RotationScenario.DEFENSIVE

        # Oportunidade: candidato muito forte
        if analysis.candidate_quality >= 85 and quality_diff >= 20:
            return RotationScenario.OPPORTUNITY

        # Moderada: melhoria razoável
        if quality_diff >= 15 and analysis.timing_score >= 50:
            return RotationScenario.MODERATE

        # Sem ação
        return RotationScenario.NO_ACTION

    def _calculate_confidence(
        self,
        analysis: ScenarioAnalysis,
        expected_improvement: float
    ) -> float:
        """Calcula score de confiança (0-100)."""
        confidence = 50  # Base

        # Contribuição de cada fator
        confidence += (analysis.candidate_quality - 50) * 0.3
        confidence += expected_improvement * 0.4
        confidence += (analysis.timing_score - 50) * 0.2
        confidence -= (analysis.risk_level - 50) * 0.2

        return max(0, min(100, confidence))

    def _determine_priority(
        self,
        scenario: RotationScenario,
        confidence: float,
        expected_improvement: float
    ) -> RotationPriority:
        """Determina prioridade de execução."""
        if scenario == RotationScenario.NO_ACTION:
            return RotationPriority.NONE

        if scenario == RotationScenario.HIGH_CONFIDENCE:
            if confidence >= 80:
                return RotationPriority.CRITICAL
            return RotationPriority.HIGH

        if scenario == RotationScenario.DEFENSIVE:
            return RotationPriority.HIGH  # Defensiva é sempre prioritária

        if scenario == RotationScenario.OPPORTUNITY:
            if confidence >= 70:
                return RotationPriority.HIGH
            return RotationPriority.MEDIUM

        # Moderada
        if confidence >= 70 and expected_improvement >= 25:
            return RotationPriority.MEDIUM
        return RotationPriority.LOW

    def _should_execute_rotation(
        self,
        scenario: RotationScenario,
        priority: RotationPriority,
        expected_improvement: float,
        confidence: float
    ) -> bool:
        """Decide se deve executar rotação."""
        if scenario == RotationScenario.NO_ACTION:
            return False

        if priority == RotationPriority.NONE:
            return False

        # Crítico e Alto: sempre executa
        if priority in [RotationPriority.CRITICAL, RotationPriority.HIGH]:
            return True

        # Médio: precisa de melhoria e confiança
        if priority == RotationPriority.MEDIUM:
            return expected_improvement >= self.min_improvement_threshold and confidence >= 60

        # Baixo: só se muito bom
        if priority == RotationPriority.LOW:
            return expected_improvement >= self.high_confidence_threshold and confidence >= 75

        return False

    def _calculate_risk_level(
        self,
        position: Dict,
        candidate: Dict,
        current_pnl_pct: float
    ) -> float:
        """Calcula nível de risco (0-100)."""
        risk = 30  # Base

        # Risco de P&L
        if current_pnl_pct < -1.0:
            risk += 20  # Posição em perda
        elif current_pnl_pct > 1.5:
            risk += 10  # Pode perder lucro

        # Risco de volatilidade do candidato
        candidate_vol = candidate.get('volatility', 1.0)
        if candidate_vol > 1.5:
            risk += 15
        elif candidate_vol > 2.0:
            risk += 25

        # Risco de timing
        position_age = position.get('age_minutes', 0)
        if position_age < 5:
            risk += 15  # Posição muito nova

        # Risco de spread/liquidez
        candidate_liquidity = candidate.get('liquidity_score', 50)
        if candidate_liquidity < 30:
            risk += 20

        return max(0, min(100, risk))

    def _get_timing_score(self, market_metrics: Optional[Dict]) -> float:
        """Obtém score de timing de mercado."""
        if not market_metrics:
            return 60  # Neutro

        return market_metrics.get('timing_score', 60)

    def _collect_reasons_and_risks(
        self,
        analysis: ScenarioAnalysis,
        scenario: RotationScenario,
        expected_improvement: float
    ) -> Tuple[List[str], List[str]]:
        """Coleta razões e fatores de risco."""
        reasons = []
        risks = []

        # Razões
        if analysis.candidate_quality >= 75:
            reasons.append(f"Candidato de alta qualidade: {analysis.candidate_quality:.0f}")

        if analysis.opportunity_cost >= 20:
            reasons.append(f"Custo de oportunidade alto: {analysis.opportunity_cost:.0f}")

        if expected_improvement >= self.high_confidence_threshold:
            reasons.append(f"Melhoria significativa: {expected_improvement:.1f}%")
        elif expected_improvement >= self.min_improvement_threshold:
            reasons.append(f"Melhoria moderada: {expected_improvement:.1f}%")

        if analysis.timing_score >= 70:
            reasons.append(f"Timing favorável: {analysis.timing_score:.0f}")

        if analysis.position_quality < 40:
            reasons.append(f"Posição atual fraca: {analysis.position_quality:.0f}")

        # Riscos
        if analysis.risk_level >= 70:
            risks.append(f"Alto risco: {analysis.risk_level:.0f}")
        elif analysis.risk_level >= 50:
            risks.append(f"Risco moderado: {analysis.risk_level:.0f}")

        if analysis.timing_score < 40:
            risks.append(f"Timing desfavorável: {analysis.timing_score:.0f}")

        if analysis.market_condition < 40:
            risks.append(f"Mercado desfavorável: {analysis.market_condition:.0f}")

        if scenario == RotationScenario.DEFENSIVE:
            risks.append("Rotação defensiva - priorizar preservação de capital")

        return reasons, risks

    def _calculate_recommended_allocation(
        self,
        candidate: Dict,
        confidence: float,
        risk_level: float
    ) -> float:
        """Calcula alocação recomendada para novo ativo."""
        # Base: 15% do capital
        base_allocation_pct = 0.15

        # Ajuste por confiança
        if confidence >= 80:
            base_allocation_pct *= 1.3  # +30%
        elif confidence >= 70:
            base_allocation_pct *= 1.15  # +15%
        elif confidence < 50:
            base_allocation_pct *= 0.7  # -30%

        # Ajuste por risco
        if risk_level >= 70:
            base_allocation_pct *= 0.6  # -40%
        elif risk_level >= 50:
            base_allocation_pct *= 0.8  # -20%

        # Limites
        return max(0.05, min(0.25, base_allocation_pct))  # 5-25%

    def _record_decision(self, decision: RotationDecision):
        """Registra decisão no histórico."""
        self.decision_history.append({
            'timestamp': datetime.now().isoformat(),
            'scenario': decision.scenario.value,
            'priority': decision.priority.value,
            'should_rotate': decision.should_rotate,
            'exit_symbol': decision.exit_symbol,
            'entry_symbol': decision.entry_symbol,
            'expected_improvement': decision.expected_improvement,
            'confidence': decision.confidence_score
        })

        # Mantém apenas últimas 100 decisões
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]

    def record_rotation_result(self, success: bool, actual_improvement: float):
        """Registra resultado de rotação executada."""
        if success:
            self.successful_rotations += 1
        else:
            self.failed_rotations += 1

        if self.decision_history:
            self.decision_history[-1]['result'] = {
                'success': success,
                'actual_improvement': actual_improvement
            }

    def get_decision_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de decisões."""
        total = self.successful_rotations + self.failed_rotations
        success_rate = (self.successful_rotations / total * 100) if total > 0 else 0

        scenario_counts = {}
        for decision in self.decision_history:
            scenario = decision.get('scenario', 'unknown')
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        return {
            'total_decisions': len(self.decision_history),
            'successful_rotations': self.successful_rotations,
            'failed_rotations': self.failed_rotations,
            'success_rate': success_rate,
            'scenario_distribution': scenario_counts,
            'avg_expected_improvement': (
                sum(d['expected_improvement'] for d in self.decision_history) /
                len(self.decision_history) if self.decision_history else 0
            ),
            'avg_confidence': (
                sum(d['confidence'] for d in self.decision_history) /
                len(self.decision_history) if self.decision_history else 0
            )
        }

    def log_decision(self, decision: RotationDecision):
        """Loga detalhes da decisão."""
        action = "ROTACIONAR" if decision.should_rotate else "MANTER"

        system_logger.info(f"\n🎯 DECISÃO DE ROTAÇÃO: {action}")
        system_logger.info(f"   Cenário: {decision.scenario.value}")
        system_logger.info(f"   Prioridade: {decision.priority.name}")
        system_logger.info(f"   Sair: {decision.exit_symbol or 'N/A'}")
        system_logger.info(f"   Entrar: {decision.entry_symbol or 'N/A'}")
        system_logger.info(f"   Melhoria esperada: {decision.expected_improvement:.1f}%")
        system_logger.info(f"   Confiança: {decision.confidence_score:.1f}")

        if decision.reasons:
            system_logger.info(f"   Razões: {', '.join(decision.reasons[:3])}")
        if decision.risk_factors:
            system_logger.info(f"   Riscos: {', '.join(decision.risk_factors[:3])}")

        if decision.should_rotate:
            system_logger.info(f"   Alocação recomendada: {decision.recommended_allocation*100:.1f}%")

