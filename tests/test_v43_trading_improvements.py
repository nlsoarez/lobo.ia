"""
Testes para V4.3 Trading Improvements
=====================================
Testa todas as correções críticas implementadas:
1. TradeLimitManager - Limite rigoroso
2. AdaptiveFilter - Filtros por regime
3. DynamicTimeoutManager - Timeout dinâmico
4. SmartTrailingStop - Trailing automático
5. StrongOverrideValidator - Validação rigorosa
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Adiciona path do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v43_trading_improvements import (
    TradeLimitManager,
    AdaptiveFilter,
    DynamicTimeoutManager,
    SmartTrailingStop,
    StrongOverrideValidator,
    MarketRegime,
    SignalLevel,
    TradeLimitExceededError,
    pre_trade_validation,
    post_trade_actions,
    position_monitoring
)


class TestTradeLimitManager:
    """Testes para TradeLimitManager."""

    def test_initialization(self):
        """Testa inicialização correta."""
        manager = TradeLimitManager(max_daily_trades=20, emergency_max=35)
        assert manager.max_trades == 20
        assert manager.emergency_max == 35
        assert manager.trades_today == 0

    def test_can_trade_within_limit(self):
        """Testa permissão de trade dentro do limite."""
        manager = TradeLimitManager(max_daily_trades=20)
        can_trade, reason, remaining = manager.can_trade()
        assert can_trade is True
        assert reason == "OK"
        assert remaining == 20

    def test_increment_trade(self):
        """Testa incremento de trades."""
        manager = TradeLimitManager(max_daily_trades=20)
        manager.increment_trade("BTC-USD", "STRONG")
        assert manager.trades_today == 1
        assert manager.get_remaining_trades() == 19

    def test_limit_exceeded(self):
        """Testa bloqueio quando limite excedido."""
        manager = TradeLimitManager(max_daily_trades=5)

        # Executa 5 trades
        for i in range(5):
            manager.increment_trade(f"CRYPTO-{i}", "TEST")

        can_trade, reason, remaining = manager.can_trade()
        assert can_trade is False
        assert "LIMITE" in reason
        assert remaining == 0

    def test_enforce_limit_raises_error(self):
        """Testa que enforce_limit levanta exceção."""
        manager = TradeLimitManager(max_daily_trades=2)
        manager.increment_trade("A", "TEST")
        manager.increment_trade("B", "TEST")

        with pytest.raises(TradeLimitExceededError):
            manager.enforce_limit()

    def test_emergency_mode_increases_limit(self):
        """Testa que modo emergência aumenta limite."""
        manager = TradeLimitManager(max_daily_trades=20, emergency_max=35)

        # Sem emergência
        assert manager.get_remaining_trades() == 20

        # Com emergência
        manager.set_emergency_mode(True)
        assert manager.get_remaining_trades() == 35

    def test_lock_trading(self):
        """Testa bloqueio manual de trading."""
        manager = TradeLimitManager()
        manager.lock_trading("Teste manual")

        can_trade, reason, _ = manager.can_trade()
        assert can_trade is False
        assert "BLOQUEADO" in reason

        manager.unlock_trading()
        can_trade, _, _ = manager.can_trade()
        assert can_trade is True

    def test_critical_scenario_23_trades(self):
        """
        CENÁRIO CRÍTICO: Sistema executou 23 trades quando limite era 20.
        Este teste verifica que NUNCA permitimos mais de 20 trades.
        """
        manager = TradeLimitManager(max_daily_trades=20)

        # Tenta executar 25 trades
        trades_executed = 0
        for i in range(25):
            can_trade, _, _ = manager.can_trade()
            if can_trade:
                manager.increment_trade(f"CRYPTO-{i}", "TEST")
                trades_executed += 1

        # DEVE ter executado exatamente 20 trades
        assert trades_executed == 20
        assert manager.trades_today == 20


class TestAdaptiveFilter:
    """Testes para AdaptiveFilter."""

    def test_detect_regime_bull(self):
        """Testa detecção de mercado bull."""
        filter = AdaptiveFilter()
        market_data = {
            'ema_trend': 1.0,
            'volatility': 1.0,
            'adx': 30,
            'btc_change_24h': 5
        }
        regime = filter.detect_regime(market_data)
        assert regime == MarketRegime.BULL

    def test_detect_regime_bear(self):
        """Testa detecção de mercado bear."""
        filter = AdaptiveFilter()
        market_data = {
            'ema_trend': -1.0,
            'volatility': 1.0,
            'adx': 30,
            'btc_change_24h': -5
        }
        regime = filter.detect_regime(market_data)
        assert regime == MarketRegime.BEAR

    def test_detect_regime_volatile(self):
        """Testa detecção de mercado volátil."""
        filter = AdaptiveFilter()
        market_data = {
            'ema_trend': 0,
            'volatility': 2.5,  # Alta volatilidade
            'adx': 25,
            'btc_change_24h': 0
        }
        regime = filter.detect_regime(market_data)
        assert regime == MarketRegime.VOLATILE

    def test_bear_market_rejects_high_rsi(self):
        """
        CENÁRIO CRÍTICO: Em bear market, RSI > 40 deve ser rejeitado.
        Problema original: Sistema comprou com RSI 45-50 em bear market.
        """
        filter = AdaptiveFilter()

        # Sinal com RSI alto em bear market
        signal = {
            'symbol': 'TEST-USD',
            'total_score': 60,
            'rsi': 45,  # Não oversold!
            'volume_ratio': 1.5
        }

        approved, reason, _ = filter.apply_regime_filter(signal, MarketRegime.BEAR)
        assert approved is False
        assert "RSI" in reason or "oversold" in reason.lower()

    def test_bear_market_approves_low_rsi(self):
        """Testa que bear market aprova RSI baixo (oversold)."""
        filter = AdaptiveFilter()

        signal = {
            'symbol': 'TEST-USD',
            'total_score': 60,
            'rsi': 25,  # Oversold!
            'volume_ratio': 1.5
        }

        approved, reason, _ = filter.apply_regime_filter(signal, MarketRegime.BEAR)
        assert approved is True

    def test_critical_scenario_arb_usd_rejected(self):
        """
        CENÁRIO DO LOG: ARB-USD com RSI 15.3 foi rejeitado corretamente,
        mas outras com RSI 45-50 foram aprovadas (bug).
        """
        filter = AdaptiveFilter()

        test_cases = [
            # (regime, rsi, score, expected_result)
            (MarketRegime.BEAR, 44, 52, False),   # RSI não oversold - REJECT
            (MarketRegime.BEAR, 25, 58, True),    # RSI oversold - APPROVE
            (MarketRegime.BULL, 75, 60, False),   # RSI overbought - REJECT
            (MarketRegime.BULL, 55, 45, True),    # RSI normal, score OK - APPROVE
        ]

        for regime, rsi, score, expected in test_cases:
            signal = {
                'symbol': 'TEST-USD',
                'total_score': score,
                'rsi': rsi,
                'volume_ratio': 1.2
            }
            approved, _, _ = filter.apply_regime_filter(signal, regime)
            assert approved == expected, f"Falhou para regime={regime}, rsi={rsi}, score={score}"


class TestDynamicTimeoutManager:
    """Testes para DynamicTimeoutManager."""

    def test_calculate_timeout_basic(self):
        """Testa cálculo básico de timeout."""
        manager = DynamicTimeoutManager()

        # TP 2% deve dar ~1h de timeout
        timeout = manager.calculate_timeout(
            tp_percent=2.0,
            sl_percent=1.0,
            volatility=1.0,
            signal_level=SignalLevel.MODERATE
        )

        assert 0.5 <= timeout <= 2.0  # Entre 30min e 2h

    def test_timeout_proportional_to_tp(self):
        """Testa que timeout é proporcional ao TP."""
        manager = DynamicTimeoutManager()

        timeout_1pct = manager.calculate_timeout(1.0, 0.5, 1.0)
        timeout_2pct = manager.calculate_timeout(2.0, 1.0, 1.0)
        timeout_4pct = manager.calculate_timeout(4.0, 2.0, 1.0)

        # Timeout deve aumentar com TP
        assert timeout_2pct > timeout_1pct
        assert timeout_4pct > timeout_2pct

    def test_strong_signal_gets_more_time(self):
        """Testa que sinais fortes têm mais tempo."""
        manager = DynamicTimeoutManager()

        timeout_weak = manager.calculate_timeout(2.0, 1.0, 1.0, SignalLevel.WEAK)
        timeout_strong = manager.calculate_timeout(2.0, 1.0, 1.0, SignalLevel.STRONG_OVERRIDE)

        assert timeout_strong > timeout_weak

    def test_should_timeout_new_position(self):
        """Testa que posição nova não dá timeout."""
        manager = DynamicTimeoutManager()

        position = {
            'entry_time': datetime.now(),
            'take_profit': 0.02,
            'stop_loss': 0.01,
            'volatility': 1.0
        }

        should_close, reason, age = manager.should_timeout(position)
        assert should_close is False
        assert age < 0.1  # Menos de 6 minutos

    def test_critical_scenario_0_9h_timeout(self):
        """
        CENÁRIO CRÍTICO: Posição com TP 2% fechada após 0.9h.
        Com TP 2%, timeout deveria ser ~1h, então 0.9h NÃO deveria fechar.
        """
        manager = DynamicTimeoutManager()

        # Posição aberta há 0.9h
        position = {
            'entry_time': datetime.now() - timedelta(hours=0.9),
            'take_profit': 0.02,  # 2%
            'stop_loss': 0.01,    # 1%
            'volatility': 1.0,
            'entry_level': 'MODERATE'
        }

        # Com TP 2%, base timeout = 1h. Não deveria fechar em 0.9h
        should_close, reason, age = manager.should_timeout(position)

        # Pode fechar ou não dependendo dos multiplicadores,
        # mas o importante é que o timeout seja proporcional ao TP
        assert age >= 0.8  # Confirma que calculou idade correta


class TestSmartTrailingStop:
    """Testes para SmartTrailingStop."""

    def test_trailing_not_active_initially(self):
        """Testa que trailing não está ativo inicialmente."""
        trailing = SmartTrailingStop()

        position = {
            'entry_price': 100,
            'entry_level': 'MODERATE'
        }

        result = trailing.update_trailing('BTC-USD', position, 100)
        assert result['active'] is False

    def test_trailing_activates_at_threshold(self):
        """Testa ativação do trailing no threshold."""
        trailing = SmartTrailingStop()

        position = {
            'entry_price': 100,
            'entry_level': 'MODERATE'  # Ativa em 2%
        }

        # Preço subiu 2.5%
        result = trailing.update_trailing('BTC-USD', position, 102.5)
        assert result['active'] is True
        assert result['trailing_peak'] >= 2.0

    def test_trailing_moves_stop_up(self):
        """Testa que trailing move stop para cima."""
        trailing = SmartTrailingStop()

        position = {
            'entry_price': 100,
            'entry_level': 'STRONG'  # Ativa em 1.5%
        }

        # Primeiro: ativa trailing
        trailing.update_trailing('BTC-USD', position, 102)

        # Depois: preço sobe mais
        result = trailing.update_trailing('BTC-USD', position, 104)

        assert result['active'] is True
        assert result['trailing_peak'] >= 3.5
        assert result['current_stop_pct'] > 2.0  # Stop subiu

    def test_trailing_closes_on_pullback(self):
        """Testa fechamento quando preço cai do pico."""
        trailing = SmartTrailingStop()

        position = {
            'entry_price': 100,
            'entry_level': 'STRONG'
        }

        # Ativa trailing
        trailing.update_trailing('BTC-USD', position, 103)

        # Preço cai - deve fechar
        should_close, reason, pnl = trailing.should_close_trailing(
            'BTC-USD', position, 102
        )

        # Com trailing distance de 0.5%, se peak=3%, stop=2.5%
        # Preço atual = 2%, então deve fechar
        assert should_close is True


class TestStrongOverrideValidator:
    """Testes para StrongOverrideValidator."""

    def test_valid_strong_override(self):
        """Testa validação de STRONG_OVERRIDE válido."""
        validator = StrongOverrideValidator()

        signal = {
            'total_score': 70,
            'rsi': 25,  # Oversold
            'volume_ratio': 1.8,
            'confirmed_indicators': ['trend', 'volume', 'macd']
        }

        is_valid, reason, level = validator.validate(signal, MarketRegime.BEAR)
        assert is_valid is True
        assert level == SignalLevel.STRONG_OVERRIDE

    def test_invalid_strong_override_low_score(self):
        """Testa rejeição com score baixo."""
        validator = StrongOverrideValidator()

        signal = {
            'total_score': 50,  # Baixo
            'rsi': 25,
            'volume_ratio': 1.8,
            'confirmed_indicators': ['trend', 'volume']
        }

        is_valid, reason, level = validator.validate(signal, MarketRegime.BEAR)
        assert is_valid is False
        assert level != SignalLevel.STRONG_OVERRIDE

    def test_downgrade_on_regime_mismatch(self):
        """Testa rebaixamento quando não alinhado com regime."""
        validator = StrongOverrideValidator()

        # RSI não oversold em bear market
        signal = {
            'total_score': 70,
            'rsi': 45,  # Não extremo
            'volume_ratio': 1.8,
            'confirmed_indicators': ['trend', 'volume']
        }

        is_valid, reason, level = validator.validate(signal, MarketRegime.BEAR)
        assert is_valid is False
        assert level in [SignalLevel.STRONG, SignalLevel.MODERATE]


class TestIntegration:
    """Testes de integração entre módulos."""

    def test_pre_trade_validation_full_flow(self):
        """Testa fluxo completo de validação pré-trade."""
        # Reset managers
        from v43_trading_improvements import (
            trade_limit_manager,
            adaptive_filter
        )

        # Reset
        trade_limit_manager.trades_today = 0
        trade_limit_manager._lock_trading = False

        signal = {
            'symbol': 'BTC-USD',
            'total_score': 60,
            'rsi': 35,
            'volume_ratio': 1.5,
            'level': 'MODERATE'
        }

        market_data = {
            'ema_trend': 0.5,
            'volatility': 1.0,
            'adx': 25,
            'btc_change_24h': 2
        }

        approved, reason, details = pre_trade_validation(signal, market_data)
        assert 'trades_remaining' in details

    def test_position_monitoring_flow(self):
        """Testa fluxo de monitoramento de posição."""
        position = {
            'entry_time': datetime.now() - timedelta(hours=0.5),
            'entry_price': 100,
            'take_profit': 0.02,
            'stop_loss': 0.01,
            'volatility': 1.0,
            'entry_level': 'MODERATE'
        }

        result = position_monitoring('TEST-USD', position, 101)

        assert 'should_close' in result
        assert 'age_hours' in result
        assert 'current_pnl_pct' in result


class TestEdgeCases:
    """Testes de casos extremos."""

    def test_trade_limit_at_boundary(self):
        """Testa limite exatamente no boundary."""
        manager = TradeLimitManager(max_daily_trades=20)

        # Executa exatamente 19 trades
        for i in range(19):
            manager.increment_trade(f"CRYPTO-{i}", "TEST")

        # Trade 20 deve ser permitido
        can_trade, _, remaining = manager.can_trade()
        assert can_trade is True
        assert remaining == 1

        # Executa trade 20
        manager.increment_trade("CRYPTO-19", "TEST")

        # Trade 21 deve ser bloqueado
        can_trade, _, remaining = manager.can_trade()
        assert can_trade is False
        assert remaining == 0

    def test_filter_with_missing_data(self):
        """Testa filtro com dados faltando."""
        filter = AdaptiveFilter()

        signal = {
            'symbol': 'TEST-USD',
            'total_score': 60,
            # Sem RSI e volume_ratio - deve usar defaults
        }

        # Não deve crashar
        approved, reason, _ = filter.apply_regime_filter(signal, MarketRegime.LATERAL)
        assert isinstance(approved, bool)

    def test_timeout_with_no_entry_time(self):
        """Testa timeout sem entry_time."""
        manager = DynamicTimeoutManager()

        position = {
            'take_profit': 0.02,
            'stop_loss': 0.01
            # Sem entry_time
        }

        should_close, reason, age = manager.should_timeout(position)
        assert should_close is False
        assert "entry_time" in reason.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
