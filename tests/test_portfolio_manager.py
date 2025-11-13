"""
Tests for PortfolioManager module.
"""

import pytest
from unittest.mock import patch, MagicMock
from portfolio_manager import PortfolioManager


class TestPortfolioManager:
    """Test suite for PortfolioManager."""

    @patch('portfolio_manager.config')
    def test_initialization(self, mock_config):
        """Test portfolio initialization with default values."""
        mock_config.get_section.side_effect = lambda x: {
            'trading': {'capital': 10000, 'exposure': 0.03, 'max_total_exposure': 0.20},
            'risk': {'stop_loss': 0.02, 'take_profit': 0.05, 'max_drawdown': 0.10}
        }[x]

        portfolio = PortfolioManager()

        assert portfolio.initial_capital == 10000
        assert portfolio.current_capital == 10000
        assert portfolio.available_capital == 10000
        assert portfolio.exposure_per_trade == 0.03
        assert portfolio.stop_loss_pct == 0.02
        assert portfolio.take_profit_pct == 0.05
        assert len(portfolio.positions) == 0

    @patch('portfolio_manager.config')
    def test_calculate_position_size(self, mock_config):
        """Test position size calculation."""
        mock_config.get_section.side_effect = lambda x: {
            'trading': {'capital': 10000, 'exposure': 0.03, 'max_total_exposure': 0.20},
            'risk': {'stop_loss': 0.02, 'take_profit': 0.05, 'max_drawdown': 0.10}
        }[x]

        portfolio = PortfolioManager()
        quantity = portfolio.calculate_position_size('TEST.SA', 100.0)

        # 10000 * 0.03 = 300 / 100 = 3 shares
        assert quantity == 3

    @patch('portfolio_manager.config')
    def test_open_position_success(self, mock_config):
        """Test opening a position successfully."""
        mock_config.get_section.side_effect = lambda x: {
            'trading': {'capital': 10000, 'exposure': 0.03, 'max_total_exposure': 0.20},
            'risk': {'stop_loss': 0.02, 'take_profit': 0.05, 'max_drawdown': 0.10}
        }[x]

        portfolio = PortfolioManager()
        success = portfolio.open_position('TEST.SA', 10, 100.0)

        assert success is True
        assert 'TEST.SA' in portfolio.positions
        assert portfolio.positions['TEST.SA']['quantity'] == 10
        assert portfolio.positions['TEST.SA']['avg_price'] == 100.0
        assert portfolio.available_capital == 9000  # 10000 - 1000

    @patch('portfolio_manager.config')
    def test_open_position_duplicate(self, mock_config):
        """Test that opening duplicate position fails."""
        mock_config.get_section.side_effect = lambda x: {
            'trading': {'capital': 10000, 'exposure': 0.03, 'max_total_exposure': 0.20},
            'risk': {'stop_loss': 0.02, 'take_profit': 0.05, 'max_drawdown': 0.10}
        }[x]

        portfolio = PortfolioManager()
        portfolio.open_position('TEST.SA', 10, 100.0)
        success = portfolio.open_position('TEST.SA', 5, 100.0)

        assert success is False

    @patch('portfolio_manager.config')
    def test_close_position_profit(self, mock_config):
        """Test closing a position with profit."""
        mock_config.get_section.side_effect = lambda x: {
            'trading': {'capital': 10000, 'exposure': 0.03, 'max_total_exposure': 0.20},
            'risk': {'stop_loss': 0.02, 'take_profit': 0.05, 'max_drawdown': 0.10}
        }[x]

        portfolio = PortfolioManager()
        portfolio.open_position('TEST.SA', 10, 100.0)
        result = portfolio.close_position('TEST.SA', 110.0)

        assert result is not None
        assert result['profit'] == 100.0  # (110 - 100) * 10
        assert portfolio.current_capital == 10100.0
        assert 'TEST.SA' not in portfolio.positions

    @patch('portfolio_manager.config')
    def test_close_position_loss(self, mock_config):
        """Test closing a position with loss."""
        mock_config.get_section.side_effect = lambda x: {
            'trading': {'capital': 10000, 'exposure': 0.03, 'max_total_exposure': 0.20},
            'risk': {'stop_loss': 0.02, 'take_profit': 0.05, 'max_drawdown': 0.10}
        }[x]

        portfolio = PortfolioManager()
        portfolio.open_position('TEST.SA', 10, 100.0)
        result = portfolio.close_position('TEST.SA', 90.0)

        assert result is not None
        assert result['profit'] == -100.0  # (90 - 100) * 10
        assert portfolio.current_capital == 9900.0

    @patch('portfolio_manager.config')
    def test_check_stop_loss(self, mock_config):
        """Test stop-loss detection."""
        mock_config.get_section.side_effect = lambda x: {
            'trading': {'capital': 10000, 'exposure': 0.03, 'max_total_exposure': 0.20},
            'risk': {'stop_loss': 0.02, 'take_profit': 0.05, 'max_drawdown': 0.10}
        }[x]

        portfolio = PortfolioManager()
        portfolio.open_position('TEST.SA', 10, 100.0)

        # Stop-loss should be at 98.0 (100 * 0.98)
        result = portfolio.check_stop_loss_take_profit('TEST.SA', 97.0)
        assert result == 'stop_loss'

    @patch('portfolio_manager.config')
    def test_check_take_profit(self, mock_config):
        """Test take-profit detection."""
        mock_config.get_section.side_effect = lambda x: {
            'trading': {'capital': 10000, 'exposure': 0.03, 'max_total_exposure': 0.20},
            'risk': {'stop_loss': 0.02, 'take_profit': 0.05, 'max_drawdown': 0.10}
        }[x]

        portfolio = PortfolioManager()
        portfolio.open_position('TEST.SA', 10, 100.0)

        # Take-profit should be at 105.0 (100 * 1.05)
        result = portfolio.check_stop_loss_take_profit('TEST.SA', 106.0)
        assert result == 'take_profit'

    @patch('portfolio_manager.config')
    def test_performance_stats(self, mock_config):
        """Test performance statistics calculation."""
        mock_config.get_section.side_effect = lambda x: {
            'trading': {'capital': 10000, 'exposure': 0.03, 'max_total_exposure': 0.20},
            'risk': {'stop_loss': 0.02, 'take_profit': 0.05, 'max_drawdown': 0.10}
        }[x]

        portfolio = PortfolioManager()

        # Execute some trades
        portfolio.open_position('TEST1.SA', 10, 100.0)
        portfolio.close_position('TEST1.SA', 110.0)  # +100

        portfolio.open_position('TEST2.SA', 10, 100.0)
        portfolio.close_position('TEST2.SA', 95.0)   # -50

        stats = portfolio.get_performance_stats()

        assert stats['total_trades'] == 2
        assert stats['wins'] == 1
        assert stats['losses'] == 1
        assert stats['win_rate'] == 50.0
        assert stats['total_profit'] == 50.0  # 100 - 50

    @patch('portfolio_manager.config')
    def test_drawdown_check(self, mock_config):
        """Test drawdown maximum check."""
        mock_config.get_section.side_effect = lambda x: {
            'trading': {'capital': 10000, 'exposure': 0.03, 'max_total_exposure': 0.20},
            'risk': {'stop_loss': 0.02, 'take_profit': 0.05, 'max_drawdown': 0.10}
        }[x]

        portfolio = PortfolioManager()

        # Simulate losses to reach drawdown
        portfolio.current_capital = 8900  # 11% loss

        assert portfolio.is_drawdown_exceeded() is True

    @patch('portfolio_manager.config')
    def test_max_exposure_limit(self, mock_config):
        """Test that maximum exposure is respected."""
        mock_config.get_section.side_effect = lambda x: {
            'trading': {'capital': 10000, 'exposure': 0.03, 'max_total_exposure': 0.20},
            'risk': {'stop_loss': 0.02, 'take_profit': 0.05, 'max_drawdown': 0.10}
        }[x]

        portfolio = PortfolioManager()

        # Open positions up to max exposure (20% = 2000)
        portfolio.open_position('TEST1.SA', 10, 100.0)  # 1000
        portfolio.open_position('TEST2.SA', 10, 100.0)  # 1000

        # Try to open another position (should fail due to max exposure)
        quantity = portfolio.calculate_position_size('TEST3.SA', 100.0)
        assert quantity == 0
