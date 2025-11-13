"""
Analisador de sinais técnicos com indicadores configuráveis.
"""

import ta
import pandas as pd
from typing import Dict, Optional, Any
from config_loader import config
from system_logger import system_logger


class SignalAnalyzer:
    """
    Analisa dados de mercado e gera sinais de compra/venda.
    Usa indicadores técnicos configuráveis (RSI, EMA, etc.).
    """

    def __init__(self, data: pd.DataFrame, symbol: str = "PETR4.SA"):
        """
        Inicializa o analisador de sinais.

        Args:
            data: DataFrame com dados OHLCV.
            symbol: Símbolo do ativo sendo analisado.

        Raises:
            ValueError: Se os dados forem inválidos ou insuficientes.
        """
        self.symbol = symbol
        self.data = data.copy()  # Cria cópia para não modificar original

        # Carrega configurações de estratégia
        strategy_config = config.get_section('strategy')
        self.indicators_config = strategy_config.get('indicators', {})

        # Parâmetros de indicadores
        self.rsi_period = self.indicators_config.get('rsi_period', 14)
        self.rsi_oversold = self.indicators_config.get('rsi_oversold', 30)
        self.rsi_overbought = self.indicators_config.get('rsi_overbought', 70)
        self.ema_fast = self.indicators_config.get('ema_fast', 9)
        self.ema_slow = self.indicators_config.get('ema_slow', 21)

        # Valida dados
        self._validate_data()

    def _validate_data(self):
        """
        Valida se os dados estão adequados para análise.

        Raises:
            ValueError: Se os dados forem inválidos.
        """
        if self.data.empty:
            raise ValueError("DataFrame vazio fornecido ao SignalAnalyzer")

        # Verifica colunas necessárias
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in self.data.columns]

        if missing:
            raise ValueError(
                f"Colunas faltando: {missing}. "
                f"Disponíveis: {list(self.data.columns)}"
            )

        # Verifica se há dados suficientes para cálculo de indicadores
        min_rows = max(self.rsi_period, self.ema_slow) + 10
        if len(self.data) < min_rows:
            raise ValueError(
                f"Dados insuficientes para cálculo de indicadores. "
                f"Necessário: {min_rows}, Disponível: {len(self.data)}"
            )

    def _add_indicators(self):
        """
        Adiciona indicadores técnicos ao DataFrame.
        Calcula RSI, EMA rápida e EMA lenta.
        """
        try:
            # RSI (Relative Strength Index)
            self.data['rsi'] = ta.momentum.RSIIndicator(
                close=self.data['close'],
                window=self.rsi_period
            ).rsi()

            # EMA rápida
            self.data['ema_fast'] = ta.trend.EMAIndicator(
                close=self.data['close'],
                window=self.ema_fast
            ).ema_indicator()

            # EMA lenta
            self.data['ema_slow'] = ta.trend.EMAIndicator(
                close=self.data['close'],
                window=self.ema_slow
            ).ema_indicator()

            # MACD (Moving Average Convergence Divergence) - adicional
            macd = ta.trend.MACD(close=self.data['close'])
            self.data['macd'] = macd.macd()
            self.data['macd_signal'] = macd.macd_signal()
            self.data['macd_diff'] = macd.macd_diff()

            # Volume médio (para análise de volume)
            self.data['volume_sma'] = self.data['volume'].rolling(window=20).mean()

            system_logger.debug(
                f"Indicadores calculados para {self.symbol}: "
                f"RSI, EMA({self.ema_fast}, {self.ema_slow}), MACD"
            )

        except Exception as e:
            system_logger.error(f"Erro ao calcular indicadores: {e}", exc_info=True)
            raise

    def generate_signal(self) -> Optional[Dict[str, Any]]:
        """
        Gera sinal de trading baseado em indicadores técnicos.

        Estratégia:
        - COMPRA: RSI < oversold, preço > EMA rápida, EMA rápida > EMA lenta, volume acima da média
        - VENDA: RSI > overbought, preço < EMA rápida, EMA rápida < EMA lenta

        Returns:
            Dicionário com sinal (symbol, action, price) ou None se não houver sinal.
        """
        # Adiciona indicadores
        self._add_indicators()

        # Remove linhas com NaN (primeiras linhas após cálculo de indicadores)
        self.data.dropna(inplace=True)

        if self.data.empty:
            system_logger.warning("Dados insuficientes após cálculo de indicadores")
            return None

        # Pega a última linha para análise
        last_row = self.data.iloc[-1]

        # Extrai valores
        rsi = last_row['rsi']
        close = last_row['close']
        ema_fast = last_row['ema_fast']
        ema_slow = last_row['ema_slow']
        volume = last_row['volume']
        volume_sma = last_row['volume_sma']
        macd_diff = last_row['macd_diff']

        system_logger.debug(
            f"Análise {self.symbol}: RSI={rsi:.2f}, Close={close:.2f}, "
            f"EMA_Fast={ema_fast:.2f}, EMA_Slow={ema_slow:.2f}, "
            f"Vol={volume:.0f}, Vol_SMA={volume_sma:.0f}"
        )

        # Lógica de sinal de COMPRA
        buy_conditions = [
            rsi < self.rsi_oversold,              # RSI em zona de sobrevenda
            close > ema_fast,                      # Preço acima da EMA rápida
            ema_fast > ema_slow,                   # Tendência de alta (EMA rápida > EMA lenta)
            volume > volume_sma * 0.8,             # Volume razoável (80% da média)
            macd_diff > 0                          # MACD positivo (momentum de alta)
        ]

        if all(buy_conditions):
            signal = {
                "symbol": self.symbol,
                "action": "BUY",
                "price": float(close),
                "indicators": {
                    "rsi": float(rsi),
                    "ema_fast": float(ema_fast),
                    "ema_slow": float(ema_slow),
                    "macd_diff": float(macd_diff),
                    "volume_ratio": float(volume / volume_sma)
                }
            }

            system_logger.info(
                f"🟢 SINAL DE COMPRA: {self.symbol} @ {close:.2f} "
                f"(RSI: {rsi:.1f}, EMA: {ema_fast:.2f}/{ema_slow:.2f})"
            )

            return signal

        # Lógica de sinal de VENDA
        sell_conditions = [
            rsi > self.rsi_overbought,             # RSI em zona de sobrecompra
            close < ema_fast,                      # Preço abaixo da EMA rápida
            ema_fast < ema_slow,                   # Tendência de baixa (EMA rápida < EMA lenta)
            macd_diff < 0                          # MACD negativo (momentum de baixa)
        ]

        if all(sell_conditions):
            signal = {
                "symbol": self.symbol,
                "action": "SELL",
                "price": float(close),
                "indicators": {
                    "rsi": float(rsi),
                    "ema_fast": float(ema_fast),
                    "ema_slow": float(ema_slow),
                    "macd_diff": float(macd_diff),
                    "volume_ratio": float(volume / volume_sma)
                }
            }

            system_logger.info(
                f"🔴 SINAL DE VENDA: {self.symbol} @ {close:.2f} "
                f"(RSI: {rsi:.1f}, EMA: {ema_fast:.2f}/{ema_slow:.2f})"
            )

            return signal

        # Sem sinal claro
        system_logger.debug(f"Sem sinal claro para {self.symbol} (RSI: {rsi:.1f})")
        return None

    def get_current_indicators(self) -> Dict[str, float]:
        """
        Retorna valores atuais dos indicadores (último candle).

        Returns:
            Dicionário com valores dos indicadores.
        """
        if 'rsi' not in self.data.columns:
            self._add_indicators()
            self.data.dropna(inplace=True)

        if self.data.empty:
            return {}

        last_row = self.data.iloc[-1]

        return {
            'rsi': float(last_row.get('rsi', 0)),
            'ema_fast': float(last_row.get('ema_fast', 0)),
            'ema_slow': float(last_row.get('ema_slow', 0)),
            'macd': float(last_row.get('macd', 0)),
            'macd_signal': float(last_row.get('macd_signal', 0)),
            'macd_diff': float(last_row.get('macd_diff', 0)),
            'close': float(last_row.get('close', 0)),
            'volume': float(last_row.get('volume', 0))
        }
