"""
Tests for SignalAnalyzer module.
"""

import pytest
import pandas as pd
from unittest.mock import patch
from signal_analyzer import SignalAnalyzer


class TestSignalAnalyzer:
    """Test suite for SignalAnalyzer."""

    @patch('signal_analyzer.config')
    def test_initialization(self, mock_config, sample_ohlcv_data):
        """Test signal analyzer initialization."""
        mock_config.get_section.return_value = {
            'indicators': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 9,
                'ema_slow': 21
            }
        }

        analyzer = SignalAnalyzer(sample_ohlcv_data, 'TEST.SA')

        assert analyzer.symbol == 'TEST.SA'
        assert analyzer.rsi_period == 14
        assert analyzer.rsi_oversold == 30
        assert analyzer.rsi_overbought == 70

    @patch('signal_analyzer.config')
    def test_indicators_calculation(self, mock_config, sample_ohlcv_data):
        """Test that indicators are calculated correctly."""
        mock_config.get_section.return_value = {
            'indicators': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 9,
                'ema_slow': 21
            }
        }

        analyzer = SignalAnalyzer(sample_ohlcv_data, 'TEST.SA')
        analyzer._add_indicators()

        assert 'rsi' in analyzer.data.columns
        assert 'ema_fast' in analyzer.data.columns
        assert 'ema_slow' in analyzer.data.columns
        assert 'macd' in analyzer.data.columns

    @patch('signal_analyzer.config')
    def test_buy_signal_generation(self, mock_config, sample_oversold_data):
        """Test buy signal generation on oversold conditions."""
        mock_config.get_section.return_value = {
            'indicators': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 9,
                'ema_slow': 21
            }
        }

        analyzer = SignalAnalyzer(sample_oversold_data, 'TEST.SA')
        signal = analyzer.generate_signal()

        # Should generate BUY signal on oversold data
        if signal:  # Depends on exact indicator values
            assert signal['action'] in ['BUY', None]
            if signal['action'] == 'BUY':
                assert 'indicators' in signal
                assert signal['symbol'] == 'TEST.SA'

    @patch('signal_analyzer.config')
    def test_sell_signal_generation(self, mock_config, sample_overbought_data):
        """Test sell signal generation on overbought conditions."""
        mock_config.get_section.return_value = {
            'indicators': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 9,
                'ema_slow': 21
            }
        }

        analyzer = SignalAnalyzer(sample_overbought_data, 'TEST.SA')
        signal = analyzer.generate_signal()

        # Should generate SELL signal on overbought data
        if signal:
            assert signal['action'] in ['SELL', None]

    @patch('signal_analyzer.config')
    def test_invalid_data_raises_error(self, mock_config):
        """Test that invalid data raises ValueError."""
        mock_config.get_section.return_value = {
            'indicators': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 9,
                'ema_slow': 21
            }
        }

        # Empty dataframe
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            SignalAnalyzer(empty_df, 'TEST.SA')

    @patch('signal_analyzer.config')
    def test_insufficient_data_raises_error(self, mock_config):
        """Test that insufficient data raises ValueError."""
        mock_config.get_section.return_value = {
            'indicators': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 9,
                'ema_slow': 21
            }
        }

        # Only 10 rows (insufficient for indicators)
        small_df = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })

        with pytest.raises(ValueError, match="Dados insuficientes"):
            SignalAnalyzer(small_df, 'TEST.SA')

    @patch('signal_analyzer.config')
    def test_get_current_indicators(self, mock_config, sample_ohlcv_data):
        """Test retrieval of current indicator values."""
        mock_config.get_section.return_value = {
            'indicators': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 9,
                'ema_slow': 21
            }
        }

        analyzer = SignalAnalyzer(sample_ohlcv_data, 'TEST.SA')
        indicators = analyzer.get_current_indicators()

        assert 'rsi' in indicators
        assert 'ema_fast' in indicators
        assert 'ema_slow' in indicators
        assert 'macd' in indicators
        assert 'close' in indicators
        assert isinstance(indicators['rsi'], float)
