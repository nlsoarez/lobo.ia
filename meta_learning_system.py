"""
V4.0 Phase 4: Meta-Learning System
Sistema que aprende a aprender - ajusta estratégias de aprendizado.
Implementa padrão de reconhecimento e adaptação contínua.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
import hashlib

from system_logger import system_logger


class AdaptationType(Enum):
    """Tipos de adaptação."""
    INCREASE_SELECTIVITY = "increase_selectivity"
    REDUCE_RISK = "reduce_risk"
    ACCELERATE_ROTATION = "accelerate_rotation"
    WIDEN_STOPS = "widen_stops"
    TIGHTEN_STOPS = "tighten_stops"
    INCREASE_POSITION_SIZE = "increase_position_size"
    REDUCE_POSITION_SIZE = "reduce_position_size"
    AVOID_CONTEXT = "avoid_context"
    PREFER_CONTEXT = "prefer_context"


@dataclass
class PatternSignature:
    """Assinatura de um padrão de trading."""
    key: str
    success_count: int = 0
    failure_count: int = 0
    total_profit: float = 0.0
    avg_hold_time: float = 0.0
    contexts: List[Dict] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def occurrences(self) -> int:
        return self.success_count + self.failure_count


@dataclass
class Adaptation:
    """Adaptação sugerida."""
    type: AdaptationType
    action: str
    reason: str
    confidence: float
    impact: Dict[str, float] = field(default_factory=dict)
    applied: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LearningStrategy:
    """Estratégia de aprendizado."""
    confidence_threshold: float = 0.7
    adaptation_aggressiveness: float = 0.1
    memory_length_days: int = 30
    pattern_decay_rate: float = 0.95
    min_pattern_occurrences: int = 3


class MetaLearningSystem:
    """
    V4.0 Phase 4: Sistema de meta-aprendizado.
    Aprende padrões de sucesso/falha e adapta estratégias.
    """

    def __init__(self):
        """Inicializa o sistema de meta-aprendizado."""
        # Conhecimento meta (padrões aprendidos)
        self.meta_knowledge: Dict[str, PatternSignature] = {}

        # Estratégias de aprendizado
        self.learning_strategy = LearningStrategy()

        # Histórico de adaptações
        self.adaptation_history: deque = deque(maxlen=500)

        # Contextos conhecidos
        self.context_performance: Dict[str, Dict] = defaultdict(lambda: {
            'success_count': 0,
            'failure_count': 0,
            'total_profit': 0.0
        })

        # Métricas de aprendizado
        self.learning_metrics = {
            'patterns_learned': 0,
            'adaptations_applied': 0,
            'successful_adaptations': 0,
            'learning_rate': 0.0
        }

        # Cache de performance recente
        self.recent_performance: deque = deque(maxlen=100)

        system_logger.info("MetaLearningSystem V4.0 inicializado")

    def learn_from_experience(self, experience: Dict):
        """
        Aprende de uma experiência de trading.

        Args:
            experience: Dict com dados do trade (result, profit, context, etc.)
        """
        # Extrai padrão do trade
        pattern_key = self._create_pattern_key(experience)

        # Atualiza conhecimento
        if pattern_key not in self.meta_knowledge:
            self.meta_knowledge[pattern_key] = PatternSignature(key=pattern_key)
            self.learning_metrics['patterns_learned'] += 1

        pattern = self.meta_knowledge[pattern_key]

        # Atualiza estatísticas do padrão
        is_success = experience.get('result', '') == 'success' or experience.get('profit', 0) > 0

        if is_success:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1

        pattern.total_profit += experience.get('profit', 0)
        pattern.last_seen = datetime.now()

        # Armazena contexto
        context = self._extract_context(experience)
        pattern.contexts.append(context)
        if len(pattern.contexts) > 50:
            pattern.contexts = pattern.contexts[-50:]

        # Atualiza performance do contexto
        context_key = self._create_context_key(context)
        if is_success:
            self.context_performance[context_key]['success_count'] += 1
        else:
            self.context_performance[context_key]['failure_count'] += 1
        self.context_performance[context_key]['total_profit'] += experience.get('profit', 0)

        # Adiciona à performance recente
        self.recent_performance.append({
            'timestamp': datetime.now(),
            'success': is_success,
            'profit': experience.get('profit', 0),
            'pattern_key': pattern_key
        })

        # Decay de padrões antigos
        self._decay_old_patterns()

    def _create_pattern_key(self, experience: Dict) -> str:
        """Cria chave única para padrão baseado em features."""
        # Discretiza features principais
        rsi_bucket = int(experience.get('entry_rsi', 50) // 10)
        vol_bucket = int(experience.get('volume_ratio', 1.0))
        pattern_type = experience.get('pattern_type', 'none')[:3]
        trend = 'up' if experience.get('trend_strength', 0) > 0 else 'down'
        regime = experience.get('regime', 'sideways')[:3]

        key = f"rsi{rsi_bucket}_vol{vol_bucket}_{pattern_type}_{trend}_{regime}"
        return key

    def _extract_context(self, experience: Dict) -> Dict:
        """Extrai contexto relevante de uma experiência."""
        return {
            'market_regime': experience.get('regime', 'sideways'),
            'time_of_day': experience.get('entry_hour', 12),
            'day_of_week': experience.get('day_of_week', 0),
            'volatility': experience.get('volatility', 0.02),
            'symbol': experience.get('symbol', 'UNKNOWN'),
            'signal_strength': experience.get('signal_strength', 50)
        }

    def _create_context_key(self, context: Dict) -> str:
        """Cria chave única para contexto."""
        regime = context.get('market_regime', 'sideways')[:3]
        hour_bucket = context.get('time_of_day', 12) // 4  # 4 buckets de 6h
        vol_bucket = 'high' if context.get('volatility', 0.02) > 0.03 else 'low'

        return f"{regime}_{hour_bucket}_{vol_bucket}"

    def _decay_old_patterns(self):
        """Aplica decay em padrões antigos."""
        cutoff = datetime.now() - timedelta(days=self.learning_strategy.memory_length_days)
        decay_rate = self.learning_strategy.pattern_decay_rate

        for pattern in self.meta_knowledge.values():
            if pattern.last_seen < cutoff:
                # Aplica decay
                pattern.success_count = int(pattern.success_count * decay_rate)
                pattern.failure_count = int(pattern.failure_count * decay_rate)
                pattern.total_profit *= decay_rate

    def suggest_adaptations(self, current_performance: Dict,
                           market_context: Dict) -> List[Adaptation]:
        """
        Sugere adaptações baseadas em performance atual.

        Args:
            current_performance: Métricas de performance recente
            market_context: Contexto atual de mercado

        Returns:
            Lista de adaptações sugeridas
        """
        adaptations = []

        # 1. Analisa win rate
        win_rate = current_performance.get('win_rate', 0.5)
        if win_rate < 0.55:
            adaptations.append(Adaptation(
                type=AdaptationType.INCREASE_SELECTIVITY,
                action="Aumentar threshold de entrada em 5 pontos",
                reason=f"Win rate baixo: {win_rate*100:.1f}%",
                confidence=0.8,
                impact={'score_threshold': 5}
            ))

        if win_rate > 0.75:
            adaptations.append(Adaptation(
                type=AdaptationType.INCREASE_POSITION_SIZE,
                action="Aumentar position size em 10%",
                reason=f"Win rate alto: {win_rate*100:.1f}%",
                confidence=0.7,
                impact={'position_size_multiplier': 1.1}
            ))

        # 2. Analisa drawdown
        max_drawdown = current_performance.get('max_drawdown', 0)
        if max_drawdown < -0.04:
            adaptations.append(Adaptation(
                type=AdaptationType.REDUCE_RISK,
                action="Reduzir position size em 20%",
                reason=f"Drawdown excessivo: {max_drawdown*100:.1f}%",
                confidence=0.9,
                impact={'position_size_multiplier': 0.8}
            ))

        # 3. Analisa tempo médio de trades
        avg_duration = current_performance.get('avg_trade_duration_minutes', 30)
        if avg_duration > 50:
            adaptations.append(Adaptation(
                type=AdaptationType.ACCELERATE_ROTATION,
                action="Reduzir timeout para 30 minutos",
                reason=f"Trades muito longos: {avg_duration:.0f}min",
                confidence=0.7,
                impact={'timeout_minutes': 30}
            ))

        # 4. Analisa profit factor
        profit_factor = current_performance.get('profit_factor', 1.0)
        if profit_factor < 1.3:
            adaptations.append(Adaptation(
                type=AdaptationType.TIGHTEN_STOPS,
                action="Reduzir stop loss em 20%",
                reason=f"Profit factor baixo: {profit_factor:.2f}",
                confidence=0.75,
                impact={'stop_loss_multiplier': 0.8}
            ))

        # 5. Verifica contexto específico
        context_key = self._create_context_key(market_context)
        context_stats = self.context_performance.get(context_key, {})

        total_trades = context_stats.get('success_count', 0) + context_stats.get('failure_count', 0)
        if total_trades >= 5:
            context_win_rate = context_stats.get('success_count', 0) / total_trades

            if context_win_rate < 0.4:
                adaptations.append(Adaptation(
                    type=AdaptationType.AVOID_CONTEXT,
                    action=f"Evitar trades no contexto {context_key}",
                    reason=f"Performance histórica ruim: {context_win_rate*100:.0f}% win rate",
                    confidence=0.85,
                    impact={'avoid_context': context_key}
                ))
            elif context_win_rate > 0.7:
                adaptations.append(Adaptation(
                    type=AdaptationType.PREFER_CONTEXT,
                    action=f"Priorizar trades no contexto {context_key}",
                    reason=f"Performance histórica boa: {context_win_rate*100:.0f}% win rate",
                    confidence=0.8,
                    impact={'prefer_context': context_key}
                ))

        # 6. Busca padrões com alta taxa de sucesso
        for pattern in self.meta_knowledge.values():
            if pattern.occurrences >= self.learning_strategy.min_pattern_occurrences:
                if pattern.success_rate > 0.8:
                    adaptations.append(Adaptation(
                        type=AdaptationType.PREFER_CONTEXT,
                        action=f"Priorizar padrão {pattern.key}",
                        reason=f"Alta taxa sucesso: {pattern.success_rate*100:.0f}%",
                        confidence=min(0.9, 0.5 + pattern.occurrences / 20),
                        impact={'prefer_pattern': pattern.key}
                    ))

        return adaptations

    def adapt_strategy(self, current_strategy: Dict,
                      adaptations: List[Adaptation]) -> Dict:
        """
        Aplica adaptações à estratégia atual.

        Args:
            current_strategy: Estratégia atual
            adaptations: Lista de adaptações a aplicar

        Returns:
            Estratégia adaptada
        """
        adapted_strategy = self._deep_copy_dict(current_strategy)
        applied_adaptations = []

        for adaptation in adaptations:
            if adaptation.confidence >= self.learning_strategy.confidence_threshold:
                # Aplica adaptação
                if adaptation.type == AdaptationType.INCREASE_SELECTIVITY:
                    if 'entry_params' in adapted_strategy:
                        adapted_strategy['entry_params']['score_threshold'] = \
                            adapted_strategy['entry_params'].get('score_threshold', 40) + 5

                elif adaptation.type == AdaptationType.REDUCE_RISK:
                    if 'risk_params' in adapted_strategy:
                        current_size = adapted_strategy['risk_params'].get('position_size_percent', 0.15)
                        adapted_strategy['risk_params']['position_size_percent'] = current_size * 0.8

                elif adaptation.type == AdaptationType.INCREASE_POSITION_SIZE:
                    if 'risk_params' in adapted_strategy:
                        current_size = adapted_strategy['risk_params'].get('position_size_percent', 0.15)
                        adapted_strategy['risk_params']['position_size_percent'] = min(0.25, current_size * 1.1)

                elif adaptation.type == AdaptationType.ACCELERATE_ROTATION:
                    if 'exit_params' in adapted_strategy:
                        adapted_strategy['exit_params']['timeout_minutes'] = 30

                elif adaptation.type == AdaptationType.TIGHTEN_STOPS:
                    if 'exit_params' in adapted_strategy:
                        current_sl = adapted_strategy['exit_params'].get('stop_loss', 1.0)
                        adapted_strategy['exit_params']['stop_loss'] = current_sl * 0.8

                elif adaptation.type == AdaptationType.WIDEN_STOPS:
                    if 'exit_params' in adapted_strategy:
                        current_sl = adapted_strategy['exit_params'].get('stop_loss', 1.0)
                        adapted_strategy['exit_params']['stop_loss'] = min(2.0, current_sl * 1.2)

                # Marca como aplicada
                adaptation.applied = True
                applied_adaptations.append(adaptation)

                # Registra no histórico
                self.adaptation_history.append({
                    'timestamp': datetime.now(),
                    'adaptation': adaptation,
                    'old_strategy': current_strategy.copy(),
                    'new_strategy': adapted_strategy.copy()
                })

                self.learning_metrics['adaptations_applied'] += 1

        if applied_adaptations:
            system_logger.info(f"\n🔄 META-LEARNING: {len(applied_adaptations)} adaptações aplicadas")
            for a in applied_adaptations[:3]:
                system_logger.info(f"   - {a.type.value}: {a.action}")

        return adapted_strategy

    def evaluate_adaptation_effectiveness(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Avalia eficácia das adaptações recentes.
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)

        # Filtra adaptações recentes
        recent_adaptations = [
            a for a in self.adaptation_history
            if a['timestamp'] >= cutoff
        ]

        if not recent_adaptations:
            return {'status': 'no_recent_adaptations'}

        # Calcula performance antes e depois
        performance_changes = []

        for i, adaptation_record in enumerate(recent_adaptations):
            # Busca trades antes e depois da adaptação
            adaptation_time = adaptation_record['timestamp']

            trades_before = [
                p for p in self.recent_performance
                if p['timestamp'] < adaptation_time and
                   p['timestamp'] >= adaptation_time - timedelta(hours=6)
            ]

            trades_after = [
                p for p in self.recent_performance
                if p['timestamp'] >= adaptation_time and
                   p['timestamp'] <= adaptation_time + timedelta(hours=6)
            ]

            if trades_before and trades_after:
                win_rate_before = sum(1 for t in trades_before if t['success']) / len(trades_before)
                win_rate_after = sum(1 for t in trades_after if t['success']) / len(trades_after)

                profit_before = sum(t['profit'] for t in trades_before)
                profit_after = sum(t['profit'] for t in trades_after)

                performance_changes.append({
                    'adaptation_type': adaptation_record['adaptation'].type.value,
                    'win_rate_change': win_rate_after - win_rate_before,
                    'profit_change': profit_after - profit_before,
                    'effective': win_rate_after > win_rate_before or profit_after > profit_before
                })

        # Calcula estatísticas
        if performance_changes:
            effective_count = sum(1 for p in performance_changes if p['effective'])
            self.learning_metrics['successful_adaptations'] = effective_count

            return {
                'total_adaptations': len(performance_changes),
                'effective_adaptations': effective_count,
                'effectiveness_rate': effective_count / len(performance_changes),
                'avg_win_rate_change': np.mean([p['win_rate_change'] for p in performance_changes]),
                'avg_profit_change': np.mean([p['profit_change'] for p in performance_changes]),
                'details': performance_changes
            }

        return {'status': 'insufficient_data'}

    def optimize_learning_strategy(self):
        """
        Otimiza estratégia de aprendizado baseado em eficácia.
        """
        effectiveness = self.evaluate_adaptation_effectiveness()

        if effectiveness.get('status') in ['no_recent_adaptations', 'insufficient_data']:
            return

        effectiveness_rate = effectiveness.get('effectiveness_rate', 0.5)

        # Ajusta threshold de confiança
        if effectiveness_rate > 0.7:
            # Adaptações estão funcionando - pode ser mais agressivo
            self.learning_strategy.confidence_threshold *= 0.95
            self.learning_strategy.adaptation_aggressiveness = min(0.3,
                self.learning_strategy.adaptation_aggressiveness * 1.1)
        elif effectiveness_rate < 0.4:
            # Adaptações não funcionam - ser mais conservador
            self.learning_strategy.confidence_threshold = min(0.9,
                self.learning_strategy.confidence_threshold * 1.1)
            self.learning_strategy.adaptation_aggressiveness *= 0.9

        # Atualiza learning rate
        self.learning_metrics['learning_rate'] = effectiveness_rate

        system_logger.info(f"\n📊 META-LEARNING OPTIMIZATION:")
        system_logger.info(f"   Effectiveness rate: {effectiveness_rate*100:.1f}%")
        system_logger.info(f"   New confidence threshold: {self.learning_strategy.confidence_threshold:.2f}")
        system_logger.info(f"   Adaptation aggressiveness: {self.learning_strategy.adaptation_aggressiveness:.2f}")

    def get_pattern_insights(self, top_n: int = 10) -> Dict[str, Any]:
        """Retorna insights sobre padrões aprendidos."""
        if not self.meta_knowledge:
            return {'patterns': []}

        # Ordena padrões por taxa de sucesso
        patterns = list(self.meta_knowledge.values())
        patterns = [p for p in patterns if p.occurrences >= self.learning_strategy.min_pattern_occurrences]
        patterns.sort(key=lambda x: x.success_rate, reverse=True)

        top_patterns = []
        for p in patterns[:top_n]:
            top_patterns.append({
                'key': p.key,
                'success_rate': p.success_rate,
                'occurrences': p.occurrences,
                'total_profit': p.total_profit,
                'last_seen': p.last_seen.isoformat()
            })

        # Padrões a evitar
        worst_patterns = sorted(patterns, key=lambda x: x.success_rate)[:5]
        avoid_patterns = []
        for p in worst_patterns:
            if p.success_rate < 0.4:
                avoid_patterns.append({
                    'key': p.key,
                    'success_rate': p.success_rate,
                    'occurrences': p.occurrences
                })

        return {
            'total_patterns': len(self.meta_knowledge),
            'active_patterns': len(patterns),
            'top_patterns': top_patterns,
            'patterns_to_avoid': avoid_patterns
        }

    def get_learning_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de aprendizado."""
        return {
            'patterns_learned': self.learning_metrics['patterns_learned'],
            'adaptations_applied': self.learning_metrics['adaptations_applied'],
            'successful_adaptations': self.learning_metrics['successful_adaptations'],
            'learning_rate': self.learning_metrics['learning_rate'],
            'adaptation_history_size': len(self.adaptation_history),
            'context_performance_entries': len(self.context_performance),
            'current_strategy': {
                'confidence_threshold': self.learning_strategy.confidence_threshold,
                'adaptation_aggressiveness': self.learning_strategy.adaptation_aggressiveness,
                'memory_length_days': self.learning_strategy.memory_length_days
            }
        }

    def _deep_copy_dict(self, d: Dict) -> Dict:
        """Copia profunda de dicionário."""
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = self._deep_copy_dict(v)
            elif isinstance(v, list):
                result[k] = v.copy()
            else:
                result[k] = v
        return result

    def log_learning_status(self):
        """Loga status do sistema de meta-aprendizado."""
        stats = self.get_learning_stats()
        insights = self.get_pattern_insights(3)

        system_logger.info(f"\n🧠 META-LEARNING STATUS:")
        system_logger.info(f"   Padrões aprendidos: {stats['patterns_learned']}")
        system_logger.info(f"   Adaptações aplicadas: {stats['adaptations_applied']}")
        system_logger.info(f"   Taxa de sucesso: {stats['learning_rate']*100:.1f}%")

        if insights['top_patterns']:
            system_logger.info(f"   Top padrões:")
            for p in insights['top_patterns'][:3]:
                system_logger.info(f"      {p['key']}: {p['success_rate']*100:.0f}% ({p['occurrences']} trades)")

