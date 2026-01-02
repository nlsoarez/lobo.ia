"""
Validador de sinais multi-camada para confirmação de trades.
V4.1 - Adiciona múltiplas camadas de confirmação para sinais.
"""

from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np

from config_loader import config
from system_logger import system_logger


class SignalConfidence(Enum):
    """Níveis de confiança do sinal."""
    VERY_HIGH = "very_high"      # >= 0.85
    HIGH = "high"                # >= 0.70
    MEDIUM = "medium"            # >= 0.50
    LOW = "low"                  # >= 0.30
    VERY_LOW = "very_low"        # < 0.30

    @classmethod
    def from_score(cls, score: float) -> 'SignalConfidence':
        if score >= 0.85:
            return cls.VERY_HIGH
        elif score >= 0.70:
            return cls.HIGH
        elif score >= 0.50:
            return cls.MEDIUM
        elif score >= 0.30:
            return cls.LOW
        else:
            return cls.VERY_LOW


class ValidationCheck:
    """Resultado de uma verificação individual."""

    def __init__(
        self,
        name: str,
        passed: bool,
        score: float,
        weight: float,
        details: str = ""
    ):
        self.name = name
        self.passed = passed
        self.score = score  # 0-1
        self.weight = weight
        self.details = details

    def __repr__(self):
        status = "✓" if self.passed else "✗"
        return f"{status} {self.name}: {self.score:.2f} ({self.details})"


class SignalValidator:
    """
    Valida sinais de trading com múltiplas camadas de confirmação.

    Camadas de validação:
    1. RSI - Condições de oversold/overbought
    2. Volume - Confirmação de volume anormal
    3. Tendência - Alinhamento com tendência maior
    4. MACD - Momentum e divergências
    5. Suporte/Resistência - Proximidade de níveis-chave
    """

    def __init__(self):
        """Inicializa o validador."""
        validator_config = config.get('validator', {})

        # Pesos das validações
        self.weights = validator_config.get('weights', {
            'rsi': 0.20,
            'volume': 0.20,
            'trend': 0.25,
            'macd': 0.20,
            'support_resistance': 0.15
        })

        # Limiar mínimo de confiança
        self.min_confidence = validator_config.get('min_confidence', 0.50)

        # Configurações de indicadores
        self.rsi_oversold = validator_config.get('rsi_oversold', 30)
        self.rsi_overbought = validator_config.get('rsi_overbought', 70)
        self.volume_threshold = validator_config.get('volume_threshold', 1.5)  # 150% do normal

        system_logger.info(
            f"SignalValidator V4.1 inicializado: min_confidence={self.min_confidence}"
        )

    def validate_signal(
        self,
        symbol: str,
        signal_type: str,  # 'BUY' or 'SELL'
        df: pd.DataFrame,
        additional_data: Dict = None
    ) -> Dict[str, Any]:
        """
        Valida um sinal com múltiplas camadas.

        Args:
            symbol: Símbolo do ativo.
            signal_type: Tipo do sinal ('BUY' ou 'SELL').
            df: DataFrame com dados OHLCV e indicadores.
            additional_data: Dados adicionais (ex: níveis de suporte/resistência).

        Returns:
            Dicionário com resultado da validação.
        """
        if df is None or df.empty or len(df) < 20:
            return {
                'valid': False,
                'confidence': 0.0,
                'confidence_level': SignalConfidence.VERY_LOW.value,
                'checks': [],
                'reason': 'Dados insuficientes para validação'
            }

        additional_data = additional_data or {}
        checks = []
        is_buy = signal_type.upper() == 'BUY'

        # 1. Validação RSI
        checks.append(self._check_rsi(df, is_buy))

        # 2. Validação Volume
        checks.append(self._check_volume(df))

        # 3. Validação Tendência
        checks.append(self._check_trend(df, is_buy))

        # 4. Validação MACD
        checks.append(self._check_macd(df, is_buy))

        # 5. Validação Suporte/Resistência
        if 'support' in additional_data or 'resistance' in additional_data:
            checks.append(self._check_support_resistance(
                df, is_buy,
                additional_data.get('support'),
                additional_data.get('resistance')
            ))

        # Calcula score total ponderado
        total_weight = sum(c.weight for c in checks)
        if total_weight > 0:
            confidence = sum(c.score * c.weight for c in checks) / total_weight
        else:
            confidence = 0.0

        # Determina se é válido
        valid = confidence >= self.min_confidence

        # Conta checks passados
        passed_checks = sum(1 for c in checks if c.passed)

        # Determina motivo se inválido
        reason = ""
        if not valid:
            failed_checks = [c.name for c in checks if not c.passed]
            reason = f"Falha em: {', '.join(failed_checks)}"

        result = {
            'valid': valid,
            'confidence': confidence,
            'confidence_level': SignalConfidence.from_score(confidence).value,
            'passed_checks': passed_checks,
            'total_checks': len(checks),
            'checks': [
                {
                    'name': c.name,
                    'passed': c.passed,
                    'score': c.score,
                    'details': c.details
                }
                for c in checks
            ],
            'reason': reason if not valid else 'Sinal válido'
        }

        # Log do resultado
        log_emoji = "✅" if valid else "❌"
        system_logger.info(
            f"{log_emoji} Validação {symbol} {signal_type}: "
            f"Confiança={confidence:.2f} ({SignalConfidence.from_score(confidence).value}) | "
            f"Checks: {passed_checks}/{len(checks)}"
        )

        return result

    def _check_rsi(self, df: pd.DataFrame, is_buy: bool) -> ValidationCheck:
        """Valida condição do RSI."""
        try:
            # Tenta obter RSI do DataFrame
            rsi = None
            for col in ['rsi', 'RSI', 'rsi_14']:
                if col in df.columns:
                    rsi = df[col].iloc[-1]
                    break

            if rsi is None:
                # Calcula RSI se não disponível
                import ta
                close = df['close'] if 'close' in df.columns else df['Close']
                rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

            if pd.isna(rsi):
                return ValidationCheck("RSI", False, 0.0, self.weights['rsi'], "RSI não disponível")

            if is_buy:
                # Para compra: RSI baixo é bom (oversold)
                if rsi < self.rsi_oversold:
                    score = 1.0
                    passed = True
                    details = f"Oversold ({rsi:.1f})"
                elif rsi < 45:
                    score = 0.7
                    passed = True
                    details = f"Região de compra ({rsi:.1f})"
                elif rsi > self.rsi_overbought:
                    score = 0.1
                    passed = False
                    details = f"Overbought ({rsi:.1f})"
                else:
                    score = 0.5
                    passed = True
                    details = f"Neutro ({rsi:.1f})"
            else:
                # Para venda: RSI alto é bom (overbought)
                if rsi > self.rsi_overbought:
                    score = 1.0
                    passed = True
                    details = f"Overbought ({rsi:.1f})"
                elif rsi > 55:
                    score = 0.7
                    passed = True
                    details = f"Região de venda ({rsi:.1f})"
                elif rsi < self.rsi_oversold:
                    score = 0.1
                    passed = False
                    details = f"Oversold ({rsi:.1f})"
                else:
                    score = 0.5
                    passed = True
                    details = f"Neutro ({rsi:.1f})"

            return ValidationCheck("RSI", passed, score, self.weights['rsi'], details)

        except Exception as e:
            return ValidationCheck("RSI", False, 0.0, self.weights['rsi'], f"Erro: {e}")

    def _check_volume(self, df: pd.DataFrame) -> ValidationCheck:
        """Valida confirmação de volume."""
        try:
            volume = df['volume'] if 'volume' in df.columns else df['Volume']

            current_volume = volume.iloc[-1]
            avg_volume = volume.rolling(window=20).mean().iloc[-1]

            if pd.isna(avg_volume) or avg_volume == 0:
                return ValidationCheck("Volume", False, 0.0, self.weights['volume'], "Volume médio não disponível")

            volume_ratio = current_volume / avg_volume

            if volume_ratio >= 2.0:
                score = 1.0
                passed = True
                details = f"Volume muito alto ({volume_ratio:.1f}x)"
            elif volume_ratio >= self.volume_threshold:
                score = 0.8
                passed = True
                details = f"Volume acima da média ({volume_ratio:.1f}x)"
            elif volume_ratio >= 1.0:
                score = 0.6
                passed = True
                details = f"Volume normal ({volume_ratio:.1f}x)"
            elif volume_ratio >= 0.7:
                score = 0.4
                passed = False
                details = f"Volume abaixo da média ({volume_ratio:.1f}x)"
            else:
                score = 0.2
                passed = False
                details = f"Volume muito baixo ({volume_ratio:.1f}x)"

            return ValidationCheck("Volume", passed, score, self.weights['volume'], details)

        except Exception as e:
            return ValidationCheck("Volume", False, 0.0, self.weights['volume'], f"Erro: {e}")

    def _check_trend(self, df: pd.DataFrame, is_buy: bool) -> ValidationCheck:
        """Valida alinhamento com tendência."""
        try:
            close = df['close'] if 'close' in df.columns else df['Close']
            current_price = close.iloc[-1]

            # Calcula EMAs
            ema_9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
            ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
            ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1] if len(close) >= 50 else ema_21

            if is_buy:
                # Para compra: preço acima das EMAs e EMAs alinhadas para alta
                if current_price > ema_9 > ema_21 > ema_50:
                    score = 1.0
                    passed = True
                    details = "Tendência de alta forte"
                elif current_price > ema_9 > ema_21:
                    score = 0.8
                    passed = True
                    details = "Tendência de alta"
                elif current_price > ema_21:
                    score = 0.6
                    passed = True
                    details = "Acima da EMA21"
                elif current_price > ema_50:
                    score = 0.4
                    passed = False
                    details = "Apenas acima da EMA50"
                else:
                    score = 0.2
                    passed = False
                    details = "Tendência de baixa"
            else:
                # Para venda: inverso
                if current_price < ema_9 < ema_21 < ema_50:
                    score = 1.0
                    passed = True
                    details = "Tendência de baixa forte"
                elif current_price < ema_9 < ema_21:
                    score = 0.8
                    passed = True
                    details = "Tendência de baixa"
                elif current_price < ema_21:
                    score = 0.6
                    passed = True
                    details = "Abaixo da EMA21"
                else:
                    score = 0.3
                    passed = False
                    details = "Tendência de alta"

            return ValidationCheck("Tendência", passed, score, self.weights['trend'], details)

        except Exception as e:
            return ValidationCheck("Tendência", False, 0.0, self.weights['trend'], f"Erro: {e}")

    def _check_macd(self, df: pd.DataFrame, is_buy: bool) -> ValidationCheck:
        """Valida momentum via MACD."""
        try:
            close = df['close'] if 'close' in df.columns else df['Close']

            # Calcula MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal

            macd_value = macd.iloc[-1]
            signal_value = signal.iloc[-1]
            hist_value = histogram.iloc[-1]
            hist_prev = histogram.iloc[-2] if len(histogram) > 1 else 0

            if is_buy:
                # Para compra: MACD cruzando para cima ou histograma positivo crescente
                if macd_value > signal_value and hist_value > hist_prev:
                    score = 1.0
                    passed = True
                    details = "MACD bullish + histograma crescente"
                elif macd_value > signal_value:
                    score = 0.7
                    passed = True
                    details = "MACD bullish"
                elif hist_value > hist_prev and hist_value > 0:
                    score = 0.6
                    passed = True
                    details = "Histograma positivo crescente"
                elif hist_value > hist_prev:
                    score = 0.4
                    passed = False
                    details = "Histograma crescente (ainda negativo)"
                else:
                    score = 0.2
                    passed = False
                    details = "MACD bearish"
            else:
                # Para venda: inverso
                if macd_value < signal_value and hist_value < hist_prev:
                    score = 1.0
                    passed = True
                    details = "MACD bearish + histograma decrescente"
                elif macd_value < signal_value:
                    score = 0.7
                    passed = True
                    details = "MACD bearish"
                elif hist_value < hist_prev and hist_value < 0:
                    score = 0.6
                    passed = True
                    details = "Histograma negativo decrescente"
                else:
                    score = 0.3
                    passed = False
                    details = "MACD bullish"

            return ValidationCheck("MACD", passed, score, self.weights['macd'], details)

        except Exception as e:
            return ValidationCheck("MACD", False, 0.0, self.weights['macd'], f"Erro: {e}")

    def _check_support_resistance(
        self,
        df: pd.DataFrame,
        is_buy: bool,
        support: float = None,
        resistance: float = None
    ) -> ValidationCheck:
        """Valida proximidade de suporte/resistência."""
        try:
            close = df['close'] if 'close' in df.columns else df['Close']
            current_price = close.iloc[-1]

            if is_buy:
                if support is not None:
                    distance_pct = ((current_price - support) / support) * 100
                    if distance_pct <= 2:
                        score = 1.0
                        passed = True
                        details = f"Próximo ao suporte ({distance_pct:.1f}%)"
                    elif distance_pct <= 5:
                        score = 0.7
                        passed = True
                        details = f"Perto do suporte ({distance_pct:.1f}%)"
                    else:
                        score = 0.4
                        passed = False
                        details = f"Longe do suporte ({distance_pct:.1f}%)"
                else:
                    score = 0.5
                    passed = True
                    details = "Suporte não definido"
            else:
                if resistance is not None:
                    distance_pct = ((resistance - current_price) / current_price) * 100
                    if distance_pct <= 2:
                        score = 1.0
                        passed = True
                        details = f"Próximo à resistência ({distance_pct:.1f}%)"
                    elif distance_pct <= 5:
                        score = 0.7
                        passed = True
                        details = f"Perto da resistência ({distance_pct:.1f}%)"
                    else:
                        score = 0.4
                        passed = False
                        details = f"Longe da resistência ({distance_pct:.1f}%)"
                else:
                    score = 0.5
                    passed = True
                    details = "Resistência não definida"

            return ValidationCheck(
                "Suporte/Resistência",
                passed,
                score,
                self.weights['support_resistance'],
                details
            )

        except Exception as e:
            return ValidationCheck(
                "Suporte/Resistência",
                False,
                0.0,
                self.weights['support_resistance'],
                f"Erro: {e}"
            )

    def get_validation_summary(
        self,
        symbol: str,
        signal_type: str,
        df: pd.DataFrame
    ) -> str:
        """
        Retorna resumo da validação em texto.
        """
        result = self.validate_signal(symbol, signal_type, df)

        lines = [
            f"=== Validação de Sinal: {symbol} {signal_type} ===",
            f"Confiança: {result['confidence']:.2f} ({result['confidence_level']})",
            f"Válido: {'Sim' if result['valid'] else 'Não'}",
            f"Checks: {result['passed_checks']}/{result['total_checks']}",
            ""
        ]

        for check in result['checks']:
            status = "✓" if check['passed'] else "✗"
            lines.append(f"  {status} {check['name']}: {check['score']:.2f} - {check['details']}")

        if not result['valid']:
            lines.append(f"\nMotivo: {result['reason']}")

        return "\n".join(lines)


# Instância global
signal_validator = SignalValidator()


if __name__ == "__main__":
    # Teste do validador
    import numpy as np

    # Cria dados de teste
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'close': np.cumsum(np.random.randn(n)) + 100,
        'volume': np.random.randint(100000, 500000, n)
    })
    df['open'] = df['close'] - np.random.rand(n)
    df['high'] = df['close'] + np.random.rand(n)
    df['low'] = df['close'] - np.random.rand(n)

    validator = SignalValidator()
    print(validator.get_validation_summary('TEST.SA', 'BUY', df))
