"""
V4.0 Phase 3: Intraday Ranking System
Ranking em tempo real de posições ativas e candidatas.
Permite comparação dinâmica para decisões de rotação.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from system_logger import system_logger


@dataclass
class RankedPosition:
    """Posição rankeada com métricas."""
    symbol: str
    dynamic_score: float
    momentum_score: float
    volume_trend_score: float
    pattern_progress_score: float
    technical_alignment_score: float
    time_penalty: float
    pnl_percent: float
    age_minutes: float
    recommendation: str  # HOLD, ROTATE, URGENT_ROTATE


@dataclass
class RankedCandidate:
    """Candidato rankeado com métricas."""
    symbol: str
    final_score: float
    base_score: float
    diversification_score: float
    timing_score: float
    volume_score: float
    pattern_score: float
    recommendation: str  # STRONG_ENTRY, ENTRY, WAIT


class IntradayRankingSystem:
    """
    V4.0 Phase 3: Sistema de ranking intraday.
    Rankeia posições e candidatos em tempo real.
    """

    def __init__(self):
        """Inicializa o sistema de ranking."""
        # Pesos para ranking de posições
        self.position_weights = {
            'momentum': 0.30,
            'volume_trend': 0.20,
            'pattern_progress': 0.20,
            'technical_alignment': 0.15,
            'pnl_factor': 0.15
        }

        # Pesos para ranking de candidatos
        self.candidate_weights = {
            'base_score': 0.35,
            'volume': 0.25,
            'diversification': 0.20,
            'timing': 0.20
        }

        # Penalidades e bonus
        self.time_penalty_per_minute = 0.5  # Penalidade por minuto após 20min
        self.time_penalty_start = 20  # Minutos
        self.max_time_penalty = 30  # Penalidade máxima

        system_logger.info("IntradayRankingSystem V4.0 Phase 3 inicializado")

    def rank_current_positions(
        self,
        positions: Dict[str, Dict],
        price_map: Dict[str, float]
    ) -> List[RankedPosition]:
        """
        Rankeia posições ativas baseado em performance em tempo real.
        """
        ranked = []
        now = datetime.now()

        for symbol, position in positions.items():
            current_price = price_map.get(symbol, 0)
            if current_price <= 0:
                continue

            try:
                # Calcula métricas
                entry_price = position.get('entry_price', current_price)
                pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                # Idade da posição
                entry_time = position.get('entry_time', now)
                try:
                    if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo:
                        age_minutes = (now.astimezone(entry_time.tzinfo) - entry_time).total_seconds() / 60
                    else:
                        age_minutes = (now - entry_time).total_seconds() / 60
                except:
                    age_minutes = 0

                # Scores individuais
                momentum_score = self._calculate_momentum_score(position, current_price)
                volume_trend_score = self._calculate_volume_trend_score(position)
                pattern_progress_score = self._calculate_pattern_progress_score(position, pnl_percent)
                technical_alignment_score = self._calculate_technical_alignment_score(position)

                # Penalidade por tempo
                time_penalty = self._calculate_time_penalty(age_minutes)

                # PnL factor (bonus ou penalidade)
                pnl_factor = self._calculate_pnl_factor(pnl_percent)

                # Score dinâmico final
                dynamic_score = (
                    momentum_score * self.position_weights['momentum'] +
                    volume_trend_score * self.position_weights['volume_trend'] +
                    pattern_progress_score * self.position_weights['pattern_progress'] +
                    technical_alignment_score * self.position_weights['technical_alignment'] +
                    pnl_factor * self.position_weights['pnl_factor'] -
                    time_penalty
                )

                dynamic_score = max(0, min(100, dynamic_score))

                # Determina recomendação
                recommendation = self._get_position_recommendation(dynamic_score, pnl_percent, age_minutes)

                ranked.append(RankedPosition(
                    symbol=symbol,
                    dynamic_score=dynamic_score,
                    momentum_score=momentum_score,
                    volume_trend_score=volume_trend_score,
                    pattern_progress_score=pattern_progress_score,
                    technical_alignment_score=technical_alignment_score,
                    time_penalty=time_penalty,
                    pnl_percent=pnl_percent,
                    age_minutes=age_minutes,
                    recommendation=recommendation
                ))

            except Exception as e:
                system_logger.debug(f"Erro rankeando posição {symbol}: {e}")

        # Ordena por score (maior primeiro)
        ranked.sort(key=lambda x: x.dynamic_score, reverse=True)

        return ranked

    def _calculate_momentum_score(self, position: Dict, current_price: float) -> float:
        """Calcula score de momentum (0-100)."""
        entry_price = position.get('entry_price', current_price)
        max_price = position.get('max_price', entry_price)

        if entry_price <= 0:
            return 50

        # Momentum baseado em P&L
        pnl = (current_price - entry_price) / entry_price

        if pnl >= 0.02:  # 2%+
            base_score = 90
        elif pnl >= 0.01:  # 1%+
            base_score = 75
        elif pnl >= 0.005:  # 0.5%+
            base_score = 60
        elif pnl >= 0:
            base_score = 50
        elif pnl >= -0.005:  # -0.5%
            base_score = 40
        elif pnl >= -0.01:  # -1%
            base_score = 25
        else:
            base_score = 10

        # Bonus se perto do máximo
        if max_price > entry_price:
            distance_from_max = (max_price - current_price) / max_price
            if distance_from_max < 0.002:  # Perto do max
                base_score = min(100, base_score + 10)

        return base_score

    def _calculate_volume_trend_score(self, position: Dict) -> float:
        """Calcula score de tendência de volume (0-100)."""
        volume_ratio = position.get('volume_ratio', 1.0)

        if volume_ratio >= 3.0:
            return 90
        elif volume_ratio >= 2.0:
            return 75
        elif volume_ratio >= 1.5:
            return 60
        elif volume_ratio >= 1.0:
            return 50
        else:
            return 30

    def _calculate_pattern_progress_score(self, position: Dict, pnl_percent: float) -> float:
        """Calcula score de progresso do padrão (0-100)."""
        pattern_score = position.get('pattern_score', 0)

        # Se tem padrão detectado
        if pattern_score > 0:
            # Verifica se está progredindo conforme esperado
            target_pnl = position.get('target_pnl', 2.0)
            progress = (pnl_percent / target_pnl) if target_pnl > 0 else 0

            if progress >= 0.8:  # 80%+ do target
                return min(100, pattern_score * 1.2)
            elif progress >= 0.5:  # 50%+ do target
                return pattern_score
            elif progress >= 0:
                return pattern_score * 0.8
            else:  # Negativo
                return pattern_score * 0.5

        return 50  # Score neutro sem padrão

    def _calculate_technical_alignment_score(self, position: Dict) -> float:
        """Calcula score de alinhamento técnico (0-100)."""
        # Baseado nos indicadores da posição
        rsi = position.get('rsi', 50)
        macd_positive = position.get('macd_positive', False)
        trend_up = position.get('trend_up', True)

        score = 50

        # RSI
        if 40 <= rsi <= 60:
            score += 15  # Zona neutra, espaço para crescer
        elif 30 <= rsi <= 40:
            score += 20  # Oversold, bom
        elif rsi < 30:
            score += 10  # Muito oversold
        elif 60 <= rsi <= 70:
            score += 5  # Overbought leve
        else:
            score -= 10  # Muito overbought

        # MACD
        if macd_positive:
            score += 15

        # Tendência
        if trend_up:
            score += 15

        return max(0, min(100, score))

    def _calculate_time_penalty(self, age_minutes: float) -> float:
        """Calcula penalidade por tempo."""
        if age_minutes <= self.time_penalty_start:
            return 0

        penalty = (age_minutes - self.time_penalty_start) * self.time_penalty_per_minute
        return min(penalty, self.max_time_penalty)

    def _calculate_pnl_factor(self, pnl_percent: float) -> float:
        """Calcula fator de P&L (bonus ou penalidade)."""
        if pnl_percent >= 2.0:
            return 100
        elif pnl_percent >= 1.0:
            return 80
        elif pnl_percent >= 0.5:
            return 65
        elif pnl_percent >= 0:
            return 50
        elif pnl_percent >= -0.5:
            return 35
        elif pnl_percent >= -1.0:
            return 20
        else:
            return 0

    def _get_position_recommendation(
        self,
        score: float,
        pnl_percent: float,
        age_minutes: float
    ) -> str:
        """Determina recomendação para posição."""
        # Rotação urgente
        if score < 25 or (pnl_percent < -0.8 and age_minutes > 10):
            return "URGENT_ROTATE"

        # Rotação recomendada
        if score < 40 or (pnl_percent < 0.3 and age_minutes > 25):
            return "ROTATE"

        return "HOLD"

    def rank_candidate_signals(
        self,
        signals: List[Dict],
        current_positions: Dict[str, Dict],
        max_candidates: int = 10
    ) -> List[RankedCandidate]:
        """
        Rankeia sinais candidatos considerando diversificação.
        """
        ranked = []

        for signal in signals[:20]:  # Considera top 20
            symbol = signal.get('symbol', '')

            # Pula se já em portfólio
            if symbol in current_positions:
                continue

            try:
                # Score base (Phase 1 + Phase 2)
                base_score = signal.get('phase2_score', 0) or signal.get('total_score', 0)

                # Score de volume
                volume_ratio = signal.get('volume_ratio', 1.0)
                volume_score = self._score_volume(volume_ratio)

                # Score de diversificação
                diversification_score = self._calculate_diversification_score(
                    signal, current_positions
                )

                # Score de timing
                timing_score = signal.get('phase2_timing', 50)

                # Pattern score
                pattern_score = signal.get('phase2_pattern', 0)

                # Score final ponderado
                final_score = (
                    base_score * self.candidate_weights['base_score'] +
                    volume_score * self.candidate_weights['volume'] +
                    diversification_score * self.candidate_weights['diversification'] +
                    timing_score * self.candidate_weights['timing']
                )

                # Bonus por padrão
                if pattern_score >= 70:
                    final_score *= 1.1

                final_score = min(100, final_score)

                # Determina recomendação
                recommendation = self._get_candidate_recommendation(final_score, base_score, volume_score)

                ranked.append(RankedCandidate(
                    symbol=symbol,
                    final_score=final_score,
                    base_score=base_score,
                    diversification_score=diversification_score,
                    timing_score=timing_score,
                    volume_score=volume_score,
                    pattern_score=pattern_score,
                    recommendation=recommendation
                ))

            except Exception as e:
                system_logger.debug(f"Erro rankeando candidato {symbol}: {e}")

        # Ordena por score final
        ranked.sort(key=lambda x: x.final_score, reverse=True)

        return ranked[:max_candidates]

    def _score_volume(self, volume_ratio: float) -> float:
        """Converte volume ratio para score (0-100)."""
        if volume_ratio >= 5.0:
            return 100
        elif volume_ratio >= 3.0:
            return 85
        elif volume_ratio >= 2.0:
            return 70
        elif volume_ratio >= 1.5:
            return 55
        elif volume_ratio >= 1.0:
            return 40
        else:
            return 20

    def _calculate_diversification_score(
        self,
        candidate: Dict,
        current_positions: Dict
    ) -> float:
        """Calcula score de diversificação (0-100)."""
        if not current_positions:
            return 100  # Máxima diversificação se portfólio vazio

        candidate_symbol = candidate.get('symbol', '')
        candidate_category = self._get_crypto_category(candidate_symbol)

        # Conta posições por categoria
        category_counts = {}
        for symbol in current_positions:
            cat = self._get_crypto_category(symbol)
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Penaliza se já tem muitos da mesma categoria
        same_category_count = category_counts.get(candidate_category, 0)

        if same_category_count == 0:
            return 100  # Nova categoria, máxima diversificação
        elif same_category_count == 1:
            return 70
        elif same_category_count == 2:
            return 40
        else:
            return 20

    def _get_crypto_category(self, symbol: str) -> str:
        """Determina categoria do crypto baseado no símbolo."""
        symbol_upper = symbol.upper()

        major = ['BTC-USD', 'ETH-USD']
        defi = ['LINK', 'AAVE', 'UNI', 'MKR', 'CRV', 'SUSHI', 'COMP', 'SNX']
        gaming = ['AXS', 'SAND', 'MANA', 'ENJ', 'GALA', 'IMX', 'APE']
        layer2 = ['ARB', 'OP', 'MATIC']

        if any(m in symbol_upper for m in major):
            return 'major'
        elif any(d in symbol_upper for d in defi):
            return 'defi'
        elif any(g in symbol_upper for g in gaming):
            return 'gaming'
        elif any(l in symbol_upper for l in layer2):
            return 'layer2'
        else:
            return 'altcoin'

    def _get_candidate_recommendation(
        self,
        final_score: float,
        base_score: float,
        volume_score: float
    ) -> str:
        """Determina recomendação para candidato."""
        if final_score >= 75 and volume_score >= 70:
            return "STRONG_ENTRY"
        elif final_score >= 60:
            return "ENTRY"
        else:
            return "WAIT"

    def get_rotation_candidates(
        self,
        positions: Dict[str, Dict],
        signals: List[Dict],
        price_map: Dict[str, float],
        min_improvement: float = 10
    ) -> List[Dict[str, Any]]:
        """
        Identifica candidatos para rotação.
        Retorna pares (posição atual, candidato) ordenados por potencial.
        """
        # Rankeia posições e candidatos
        ranked_positions = self.rank_current_positions(positions, price_map)
        ranked_candidates = self.rank_candidate_signals(signals, positions)

        rotation_opportunities = []

        # Procura posições fracas vs candidatos fortes
        for pos in ranked_positions:
            if pos.recommendation not in ['ROTATE', 'URGENT_ROTATE']:
                continue

            for cand in ranked_candidates:
                if cand.recommendation not in ['STRONG_ENTRY', 'ENTRY']:
                    continue

                improvement = cand.final_score - pos.dynamic_score

                if improvement >= min_improvement:
                    rotation_opportunities.append({
                        'from_symbol': pos.symbol,
                        'from_score': pos.dynamic_score,
                        'from_pnl': pos.pnl_percent,
                        'from_age': pos.age_minutes,
                        'from_recommendation': pos.recommendation,
                        'to_symbol': cand.symbol,
                        'to_score': cand.final_score,
                        'to_recommendation': cand.recommendation,
                        'improvement': improvement,
                        'priority': 'HIGH' if pos.recommendation == 'URGENT_ROTATE' else 'MEDIUM'
                    })

        # Ordena por improvement
        rotation_opportunities.sort(key=lambda x: x['improvement'], reverse=True)

        return rotation_opportunities

    def log_rankings(
        self,
        positions: Dict[str, Dict],
        signals: List[Dict],
        price_map: Dict[str, float]
    ):
        """Loga rankings para debug."""
        ranked_positions = self.rank_current_positions(positions, price_map)
        ranked_candidates = self.rank_candidate_signals(signals, positions, max_candidates=5)

        system_logger.info("\n📊 RANKING INTRADAY")
        system_logger.info("-" * 50)

        if ranked_positions:
            system_logger.info("POSIÇÕES ATIVAS:")
            for i, pos in enumerate(ranked_positions, 1):
                emoji = "🟢" if pos.recommendation == "HOLD" else "🟡" if pos.recommendation == "ROTATE" else "🔴"
                system_logger.info(
                    f"  {i}. {emoji} {pos.symbol}: Score={pos.dynamic_score:.0f} "
                    f"P&L={pos.pnl_percent:+.2f}% Age={pos.age_minutes:.0f}min [{pos.recommendation}]"
                )
        else:
            system_logger.info("  Nenhuma posição ativa")

        system_logger.info("")
        if ranked_candidates:
            system_logger.info("CANDIDATOS:")
            for i, cand in enumerate(ranked_candidates, 1):
                emoji = "🎯" if cand.recommendation == "STRONG_ENTRY" else "📈"
                system_logger.info(
                    f"  {i}. {emoji} {cand.symbol}: Score={cand.final_score:.0f} "
                    f"Vol={cand.volume_score:.0f} Div={cand.diversification_score:.0f} [{cand.recommendation}]"
                )
        else:
            system_logger.info("  Nenhum candidato qualificado")

        system_logger.info("-" * 50)
