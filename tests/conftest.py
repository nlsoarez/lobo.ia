"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_config():
    """Sample configuration for tests."""
    return {
        'trading': {
            'symbols': ['TEST.SA'],
            'capital': 10000.0,
            'exposure': 0.03,
            'max_total_exposure': 0.20
        },
        'strategy': {
            'indicators': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 9,
                'ema_slow': 21
            }
        },
        'risk': {
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'max_drawdown': 0.10
        },
        'execution': {
            'mode': 'simulation',
            'simulate_slippage': 0.001,
            'simulate_fees': 0.0005,
            'execution_delay': 0
        }
    }


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')

    # Generate realistic price movement
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    data = pd.DataFrame({
        'datetime': dates,
        'open': close_prices + np.random.randn(100) * 0.1,
        'high': close_prices + np.abs(np.random.randn(100) * 0.3),
        'low': close_prices - np.abs(np.random.randn(100) * 0.3),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100)
    })

    return data


@pytest.fixture
def sample_oversold_data():
    """Generate data that triggers oversold condition (RSI < 30)."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')

    # Generate declining prices to create oversold condition
    close_prices = 100 - np.linspace(0, 20, 100) + np.random.randn(100) * 0.2

    data = pd.DataFrame({
        'datetime': dates,
        'open': close_prices + 0.1,
        'high': close_prices + 0.3,
        'low': close_prices - 0.3,
        'close': close_prices,
        'volume': np.random.randint(5000, 15000, 100)
    })

    return data


@pytest.fixture
def sample_overbought_data():
    """Generate data that triggers overbought condition (RSI > 70)."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')

    # Generate rising prices to create overbought condition
    close_prices = 100 + np.linspace(0, 20, 100) + np.random.randn(100) * 0.2

    data = pd.DataFrame({
        'datetime': dates,
        'open': close_prices + 0.1,
        'high': close_prices + 0.3,
        'low': close_prices - 0.3,
        'close': close_prices,
        'volume': np.random.randint(5000, 15000, 100)
    })

    return data


@pytest.fixture
def sample_trade():
    """Sample trade dictionary."""
    return {
        'symbol': 'TEST.SA',
        'date': datetime.now(),
        'action': 'BUY',
        'price': 100.0,
        'quantity': 10,
        'profit': 0,
        'indicators': 'RSI:28.5, EMA:99.8',
        'notes': 'Test trade'
    }


@pytest.fixture
def sample_signal():
    """Sample trading signal."""
    return {
        'symbol': 'TEST.SA',
        'action': 'BUY',
        'price': 100.0,
        'indicators': {
            'rsi': 28.5,
            'ema_fast': 99.8,
            'ema_slow': 100.2,
            'macd_diff': 0.5,
            'volume_ratio': 1.2
        }
    }
