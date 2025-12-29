"""
V4.0 Phase 2: Asset Ranking System
Sistema de ranking para priorizar os melhores ativos para trading.
Integra todos os módulos da Fase 2 para seleção otimizada.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from system_logger import system_logger
from data_utils import get_close_column, normalize_dataframe_columns

# Importa módulos da Fase 2
from technical_pattern_scanner import TechnicalPatternScanner
from volume_analyzer import VolumeAnalyzer
from market_timing_manager import MarketTimingManager
from breakout_detector import BreakoutDetector


@dataclass
class AssetScore:
    """Score completo de um ativo."""
    symbol: str
    total_score: float
    technical_score: float
    volume_score: float
    pattern_score: float
    timing_score: float
    breakout_score: float
    recommendation: str  # STRONG_BUY, BUY, HOLD, AVOID
    confidence: str  # LOW, MEDIUM, HIGH
    risk_level: str  # LOW, MEDIUM, HIGH
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward: float


class AssetRankingSystem:
    """
    V4.0 Phase 2: Sistema de ranking de ativos.
    Integra análises técnica, volume, timing e padrões.
    """

    def __init__(self):
        """Inicializa o sistema de ranking."""
        # Inicializa módulos da Fase 2
        self.pattern_scanner = TechnicalPatternScanner()
        self.volume_analyzer = VolumeAnalyzer()
        self.timing_manager = MarketTimingManager()
        self.breakout_detector = BreakoutDetector()

        # Pesos para score final
        self.weights = {
            'technical': 0.30,     # Indicadores técnicos
            'volume': 0.20,        # Volume e liquidez
            'pattern': 0.20,       # Padrões gráficos
            'timing': 0.15,        # Timing de mercado
            'breakout': 0.15       # Breakouts
        }

        # Thresholds
        self.min_score_strong_buy = 75
        self.min_score_buy = 60
        self.min_score_hold = 40

        system_logger.info("AssetRankingSystem V4.0 inicializado")

    def calculate_technical_score(self, crypto_data: Dict[str, Any]) -> float:
        """
        Calcula score técnico baseado em indicadores.
        Espera dados do crypto_scanner.
        """
        try:
            score = 0

            # RSI Score (0-25 pontos)
            rsi = crypto_data.get('rsi', 50)
            if 30 <= rsi <= 45:
                score += 25  # Sobrevendido (melhor)
            elif 45 < rsi <= 55:
                score += 20  # Neutro
            elif 55 < rsi <= 70:
                score += 10  # Sobrecomprado leve
            elif rsi < 30:
                score += 15  # Muito sobrevendido (risco de bounce)
            else:
                score += 0   # Muito sobrecomprado

            # MACD Score (0-25 pontos)
            macd = crypto_data.get('macd', 0)
            macd_signal = crypto_data.get('macd_signal', 0)
            macd_hist = crypto_data.get('macd_hist', 0)

            if macd > macd_signal and macd_hist > 0:
                score += 25  # Bullish crossover
            elif macd > macd_signal:
                score += 15  # Bullish
            elif macd < macd_signal and macd_hist < 0:
                score += 0   # Bearish crossover
            else:
                score += 10  # Neutro

            # Trend Score (0-25 pontos)
            price = crypto_data.get('price', 0)
            ema_20 = crypto_data.get('ema_20', price)
            ema_50 = crypto_data.get('ema_50', price)

            if price > ema_20 > ema_50:
                score += 25  # Uptrend forte
            elif price > ema_20:
                score += 18  # Uptrend moderado
            elif price < ema_20 < ema_50:
                score += 5   # Downtrend
            else:
                score += 12  # Lateral

            # Signal Strength (0-25 pontos)
            signal = crypto_data.get('signal', '')
            total_score_existing = crypto_data.get('total_score', 50)

            if 'STRONG' in signal.upper() and 'BUY' in signal.upper():
                score += 25
            elif 'BUY' in signal.upper():
                score += 18
            elif 'HOLD' in signal.upper():
                score += 10
            else:
                score += 0

            return min(100, score)

        except Exception as e:
            system_logger.debug(f"Erro calculando technical score: {e}")
            return 50

    def calculate_volume_score(self, symbol: str, df: pd.DataFrame, price: float) -> Tuple[float, Dict]:
        """
        Calcula score de volume usando VolumeAnalyzer.
        """
        try:
            analysis = self.volume_analyzer.get_comprehensive_analysis(symbol, df, price)
            return analysis['combined_score'], analysis
        except Exception as e:
            system_logger.debug(f"Erro calculando volume score para {symbol}: {e}")
            return 50, {}

    def calculate_pattern_score(self, symbol: str, df: pd.DataFrame) -> Tuple[float, List[Dict]]:
        """
        Calcula score de padrões usando TechnicalPatternScanner.
        """
        try:
            patterns = self.pattern_scanner.scan_all_patterns(symbol, df)

            if not patterns:
                return 30, []  # Score base sem padrões

            best_pattern = patterns[0]
            base_score = best_pattern['score']

            # Bonus por múltiplos padrões
            if len(patterns) >= 2:
                base_score = min(100, base_score * 1.1)

            return base_score, patterns

        except Exception as e:
            system_logger.debug(f"Erro calculando pattern score para {symbol}: {e}")
            return 30, []

    def calculate_timing_score(self) -> Tuple[float, Dict]:
        """
        Calcula score de timing usando MarketTimingManager.
        """
        try:
            timing = self.timing_manager.analyze_timing()

            base_score = 50

            # Bonus por horário ideal
            if timing.is_optimal:
                base_score += 30

            # Bonus por sessão de alta liquidez
            if timing.confidence == 'HIGH':
                base_score += 20
            elif timing.confidence == 'MEDIUM':
                base_score += 10

            return min(100, base_score), {
                'is_optimal': timing.is_optimal,
                'session': timing.current_session.value,
                'exposure_mult': timing.exposure_multiplier,
                'confidence': timing.confidence
            }

        except Exception as e:
            system_logger.debug(f"Erro calculando timing score: {e}")
            return 50, {}

    def calculate_breakout_score(self, symbol: str, df: pd.DataFrame) -> Tuple[float, Optional[Dict]]:
        """
        Calcula score de breakout usando BreakoutDetector.
        """
        try:
            breakout = self.breakout_detector.get_best_breakout(symbol, df)

            if breakout is None:
                return 30, None  # Score base sem breakout

            return breakout['strength'], breakout

        except Exception as e:
            system_logger.debug(f"Erro calculando breakout score para {symbol}: {e}")
            return 30, None

    def calculate_comprehensive_score(
        self,
        symbol: str,
        crypto_data: Dict[str, Any],
        df: pd.DataFrame
    ) -> AssetScore:
        """
        Calcula score completo para um ativo.
        """
        try:
            # V4.1 FIX: Usa extração segura da coluna close
            close_col = get_close_column(df)
            default_price = close_col.iloc[-1] if close_col is not None and len(close_col) > 0 else 0
            price = crypto_data.get('price', default_price)

            # Calcula scores individuais
            technical_score = self.calculate_technical_score(crypto_data)
            volume_score, volume_analysis = self.calculate_volume_score(symbol, df, price)
            pattern_score, patterns = self.calculate_pattern_score(symbol, df)
            timing_score, timing_data = self.calculate_timing_score()
            breakout_score, breakout_data = self.calculate_breakout_score(symbol, df)

            # Score ponderado
            total_score = (
                technical_score * self.weights['technical'] +
                volume_score * self.weights['volume'] +
                pattern_score * self.weights['pattern'] +
                timing_score * self.weights['timing'] +
                breakout_score * self.weights['breakout']
            )

            # Determina recomendação
            if total_score >= self.min_score_strong_buy:
                recommendation = 'STRONG_BUY'
            elif total_score >= self.min_score_buy:
                recommendation = 'BUY'
            elif total_score >= self.min_score_hold:
                recommendation = 'HOLD'
            else:
                recommendation = 'AVOID'

            # Determina confiança
            high_scores = sum(1 for s in [technical_score, volume_score, pattern_score]
                            if s >= 70)
            if high_scores >= 3:
                confidence = 'HIGH'
            elif high_scores >= 2:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

            # Determina nível de risco
            if volume_score < 50 or breakout_score < 40:
                risk_level = 'HIGH'
            elif volume_score >= 70 and pattern_score >= 60:
                risk_level = 'LOW'
            else:
                risk_level = 'MEDIUM'

            # Calcula targets
            entry_price = price
            target_price = price * 1.02  # Default 2%
            stop_loss = price * 0.99     # Default 1%

            # Usa targets do padrão se disponível
            if patterns:
                best_pattern = patterns[0]
                if best_pattern.get('target_price', 0) > 0:
                    target_price = best_pattern['target_price']
                if best_pattern.get('stop_loss', 0) > 0:
                    stop_loss = best_pattern['stop_loss']
                if best_pattern.get('entry_price', 0) > 0:
                    entry_price = best_pattern['entry_price']

            # Usa targets do breakout se disponível
            elif breakout_data:
                if breakout_data.get('target', 0) > 0:
                    target_price = breakout_data['target']
                if breakout_data.get('stop_loss', 0) > 0:
                    stop_loss = breakout_data['stop_loss']

            # Calcula R:R
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward = reward / risk if risk > 0 else 0

            return AssetScore(
                symbol=symbol,
                total_score=total_score,
                technical_score=technical_score,
                volume_score=volume_score,
                pattern_score=pattern_score,
                timing_score=timing_score,
                breakout_score=breakout_score,
                recommendation=recommendation,
                confidence=confidence,
                risk_level=risk_level,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward=risk_reward
            )

        except Exception as e:
            system_logger.error(f"Erro calculando score completo para {symbol}: {e}")
            return AssetScore(
                symbol=symbol,
                total_score=0,
                technical_score=0,
                volume_score=0,
                pattern_score=0,
                timing_score=0,
                breakout_score=0,
                recommendation='AVOID',
                confidence='LOW',
                risk_level='HIGH',
                entry_price=0,
                target_price=0,
                stop_loss=0,
                risk_reward=0
            )

    def rank_assets(
        self,
        assets_data: Dict[str, Tuple[Dict, pd.DataFrame]],
        top_n: int = 10
    ) -> List[AssetScore]:
        """
        Rankeia múltiplos ativos e retorna os melhores.

        Args:
            assets_data: Dict com {symbol: (crypto_data, df)}
            top_n: Número de ativos a retornar
        """
        scores = []

        for symbol, (crypto_data, df) in assets_data.items():
            try:
                score = self.calculate_comprehensive_score(symbol, crypto_data, df)
                scores.append(score)
            except Exception as e:
                system_logger.debug(f"Erro rankeando {symbol}: {e}")

        # Ordena por score total
        scores.sort(key=lambda x: x.total_score, reverse=True)

        return scores[:top_n]

    def get_top_opportunities(
        self,
        assets_data: Dict[str, Tuple[Dict, pd.DataFrame]],
        min_score: float = 60,
        max_risk: str = 'MEDIUM'
    ) -> List[AssetScore]:
        """
        Retorna as melhores oportunidades filtradas.
        """
        all_scores = self.rank_assets(assets_data, top_n=50)

        # Filtra por score mínimo
        filtered = [s for s in all_scores if s.total_score >= min_score]

        # Filtra por nível de risco
        risk_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        max_risk_num = risk_levels.get(max_risk, 2)
        filtered = [s for s in filtered if risk_levels.get(s.risk_level, 3) <= max_risk_num]

        return filtered

    def get_selection_summary(self, scores: List[AssetScore]) -> Dict[str, Any]:
        """
        Retorna resumo da seleção de ativos.
        """
        if not scores:
            return {
                'total_analyzed': 0,
                'strong_buy': 0,
                'buy': 0,
                'hold': 0,
                'avoid': 0,
                'average_score': 0,
                'top_3': []
            }

        recommendations = {'STRONG_BUY': 0, 'BUY': 0, 'HOLD': 0, 'AVOID': 0}
        for score in scores:
            recommendations[score.recommendation] += 1

        return {
            'total_analyzed': len(scores),
            'strong_buy': recommendations['STRONG_BUY'],
            'buy': recommendations['BUY'],
            'hold': recommendations['HOLD'],
            'avoid': recommendations['AVOID'],
            'average_score': np.mean([s.total_score for s in scores]),
            'top_3': [
                {
                    'symbol': s.symbol,
                    'score': s.total_score,
                    'recommendation': s.recommendation,
                    'rr': s.risk_reward
                }
                for s in scores[:3]
            ]
        }

    def log_ranking_results(self, scores: List[AssetScore], max_display: int = 10):
        """
        Loga resultados do ranking para debug.
        """
        system_logger.info("\n" + "=" * 60)
        system_logger.info("📊 V4.0 PHASE 2: ASSET RANKING RESULTS")
        system_logger.info("=" * 60)

        summary = self.get_selection_summary(scores)
        system_logger.info(f"Analisados: {summary['total_analyzed']} | "
                          f"STRONG_BUY: {summary['strong_buy']} | "
                          f"BUY: {summary['buy']} | "
                          f"HOLD: {summary['hold']}")
        system_logger.info(f"Score médio: {summary['average_score']:.1f}")

        system_logger.info("\n🏆 TOP ASSETS:")
        system_logger.info("-" * 60)

        for i, score in enumerate(scores[:max_display], 1):
            emoji = "🟢" if score.recommendation in ['STRONG_BUY', 'BUY'] else "🟡" if score.recommendation == 'HOLD' else "🔴"
            system_logger.info(
                f"{i}. {emoji} {score.symbol}: {score.total_score:.1f}pts | "
                f"{score.recommendation} | RR:{score.risk_reward:.2f} | "
                f"Risk:{score.risk_level}"
            )
            system_logger.info(
                f"   Tech:{score.technical_score:.0f} Vol:{score.volume_score:.0f} "
                f"Pat:{score.pattern_score:.0f} Tim:{score.timing_score:.0f} "
                f"Brk:{score.breakout_score:.0f}"
            )

        system_logger.info("=" * 60)

    def get_adjusted_parameters(self, score: AssetScore) -> Dict[str, float]:
        """
        Retorna parâmetros de trade ajustados baseado no score.
        """
        # Obtém multiplicadores de timing
        timing = self.timing_manager.analyze_timing()

        # Base TP/SL da Fase 1
        base_tp = 0.02  # 2%
        base_sl = 0.01  # 1%
        base_exposure = 0.20  # 20%

        # Ajusta por score
        if score.total_score >= 80:
            score_mult = 1.2  # Mais agressivo
        elif score.total_score >= 70:
            score_mult = 1.1
        elif score.total_score >= 60:
            score_mult = 1.0
        else:
            score_mult = 0.8  # Mais conservador

        # Ajusta por risco
        risk_mult = {
            'LOW': 1.1,
            'MEDIUM': 1.0,
            'HIGH': 0.8
        }.get(score.risk_level, 1.0)

        # Calcula parâmetros finais
        final_exposure = base_exposure * score_mult * risk_mult * timing.exposure_multiplier
        final_tp = base_tp * timing.tp_multiplier
        final_sl = base_sl * timing.sl_multiplier

        # Limites de segurança
        final_exposure = min(0.30, max(0.05, final_exposure))  # 5-30%
        final_tp = min(0.05, max(0.01, final_tp))              # 1-5%
        final_sl = min(0.03, max(0.005, final_sl))             # 0.5-3%

        return {
            'exposure': final_exposure,
            'take_profit': final_tp,
            'stop_loss': final_sl,
            'entry_price': score.entry_price,
            'target_price': score.target_price,
            'score_stop_loss': score.stop_loss,
            'risk_reward': score.risk_reward,
            'timing_session': timing.current_session.value,
            'timing_optimal': timing.is_optimal
        }
