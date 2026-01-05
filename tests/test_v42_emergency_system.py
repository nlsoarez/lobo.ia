#!/usr/bin/env python3
"""
Script de Validação do Sistema V4.2
Testes de integração para Emergency Trading System.

Execute: python tests/test_v42_emergency_system.py
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergency_trade_manager import EmergencyTradeManager, TradeDecision
from smart_health_metrics import SmartHealthMetrics, HealthStatus
from emergency_signal_prioritizer import EmergencySignalPrioritizer, SignalPriority
from smart_alert_system import SmartAlertSystem, AlertLevel, AlertCategory


class TestResults:
    """Armazena resultados dos testes."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def add_result(self, name: str, passed: bool, details: str = ""):
        self.tests.append({
            'name': name,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 RESULTADOS DOS TESTES V4.2")
        print("=" * 70)

        for test in self.tests:
            status = "✅ PASS" if test['passed'] else "❌ FAIL"
            print(f"{status} | {test['name']}")
            if test['details'] and not test['passed']:
                print(f"        └─ {test['details']}")

        print("\n" + "-" * 70)
        total = self.passed + self.failed
        rate = (self.passed / total * 100) if total > 0 else 0
        print(f"📈 Total: {total} | Passed: {self.passed} | Failed: {self.failed} | Rate: {rate:.1f}%")
        print("=" * 70)

        return self.failed == 0


results = TestResults()


# =============================================================================
# TESTES DO EMERGENCY TRADE MANAGER
# =============================================================================

def test_emergency_override_basic():
    """Testa override básico de limites durante emergência."""
    manager = EmergencyTradeManager()

    # Reset para teste limpo
    manager.trades_today = 21  # Acima do limite normal (20)
    manager.emergency_trades_today = 0
    manager.critical_overrides_today = 0

    # Sinal forte: ARB-USD com RSI=15.3, Score=72
    signal = {
        'symbol': 'ARB-USD',
        'total_score': 72,
        'rsi': 15.3,
        'volume_ratio': 1.8,
        'signal': 'STRONG_BUY'
    }

    # Sem modo emergência - deve rejeitar
    can_trade, decision, reasons = manager.can_trade(signal, emergency_mode=False)

    passed = not can_trade and decision == TradeDecision.REJECTED_LIMIT
    results.add_result(
        "Emergency Override - Rejeita sem modo emergência",
        passed,
        f"can_trade={can_trade}, decision={decision.value}"
    )

    # Com modo emergência - deve aprovar como CRITICAL
    can_trade, decision, reasons = manager.can_trade(signal, emergency_mode=True)

    passed = can_trade and decision == TradeDecision.APPROVED_CRITICAL
    results.add_result(
        "Emergency Override - Aprova com modo emergência",
        passed,
        f"can_trade={can_trade}, decision={decision.value}, reasons={reasons}"
    )


def test_critical_trade_allowance():
    """Testa limite de trades críticos (5 adicionais)."""
    manager = EmergencyTradeManager()

    # Simula já ter usado todos os overrides
    manager.trades_today = 25
    manager.critical_overrides_today = 5  # Limite atingido

    signal = {
        'symbol': 'SOL-USD',
        'total_score': 80,
        'rsi': 12,
        'volume_ratio': 2.0,
        'signal': 'STRONG_BUY'
    }

    can_trade, decision, reasons = manager.can_trade(signal, emergency_mode=True)

    passed = not can_trade  # Deve rejeitar pois limite crítico atingido
    results.add_result(
        "Critical Trade Allowance - Respeita limite de 5",
        passed,
        f"critical_overrides={manager.critical_overrides_today}, decision={decision.value}"
    )


def test_position_size_multiplier():
    """Testa multiplicador de posição para trades emergenciais."""
    manager = EmergencyTradeManager()

    # Teste para cada tipo de decisão
    tests = [
        (TradeDecision.APPROVED, 1.0),
        (TradeDecision.APPROVED_EMERGENCY, 0.7),
        (TradeDecision.APPROVED_CRITICAL, 0.56),  # 0.7 * 0.8
    ]

    all_passed = True
    for decision, expected in tests:
        actual = manager.get_position_size_multiplier(decision)
        if abs(actual - expected) > 0.01:
            all_passed = False
            break

    results.add_result(
        "Position Size Multiplier - Valores corretos",
        all_passed,
        f"APPROVED=1.0, EMERGENCY=0.7, CRITICAL=0.56"
    )


def test_daily_counter_reset():
    """Testa reset automático de contadores diários."""
    manager = EmergencyTradeManager()

    # Simula contadores de ontem
    manager.trades_today = 30
    manager.emergency_trades_today = 10
    manager.last_reset_date = (datetime.now() - timedelta(days=1)).date()

    # Chama método que deve resetar
    manager._reset_daily_counters()

    passed = (
        manager.trades_today == 0 and
        manager.emergency_trades_today == 0 and
        manager.last_reset_date == datetime.now().date()
    )

    results.add_result(
        "Daily Counter Reset - Reset automático",
        passed,
        f"trades={manager.trades_today}, date={manager.last_reset_date}"
    )


# =============================================================================
# TESTES DO SMART HEALTH METRICS
# =============================================================================

def test_success_rate_calculation():
    """Testa cálculo correto de success rate."""
    metrics = SmartHealthMetrics()

    # Simula 15 wins, 6 losses
    for i in range(15):
        metrics.record_trade_result(profit=10.0, symbol=f"WIN-{i}")
    for i in range(6):
        metrics.record_trade_result(profit=-5.0, symbol=f"LOSS-{i}")

    success_rate = metrics.calculate_success_rate()
    expected = (15 / 21) * 100  # 71.43%

    passed = abs(success_rate - expected) < 0.1
    results.add_result(
        "Success Rate - Cálculo correto (15W/6L)",
        passed,
        f"Esperado: {expected:.2f}%, Obtido: {success_rate:.2f}%"
    )


def test_health_status_multidimensional():
    """Testa status de saúde multidimensional (2/3 GREEN = HEALTHY)."""
    metrics = SmartHealthMetrics()

    # Simula condições boas
    metrics.total_trades_executed = 20
    metrics.total_trades_successful = 15  # 75% success

    for _ in range(10):
        metrics.record_api_call(success=True, latency_ms=30)  # Baixa latência

    status, details = metrics.update_health_status()

    passed = status == HealthStatus.HEALTHY
    results.add_result(
        "Health Status - HEALTHY com 2/3 GREEN",
        passed,
        f"Status: {status.value}, Summary: {details['summary']}"
    )


def test_inconsistency_detection():
    """Testa detecção de inconsistências nas métricas."""
    metrics = SmartHealthMetrics()

    # Simula inconsistência: success rate baixo mas será avaliado
    metrics.total_trades_executed = 10
    metrics.total_trades_successful = 2  # 20% success

    # Força latência boa para ter 2/3 GREEN (HEALTHY)
    for _ in range(20):
        metrics.record_api_call(success=True, latency_ms=20)

    status, details = metrics.update_health_status()

    # Verifica se alerta foi gerado
    has_inconsistency = len(metrics.inconsistency_alerts) > 0

    results.add_result(
        "Inconsistency Detection - Alerta gerado",
        has_inconsistency,
        f"Alertas: {len(metrics.inconsistency_alerts)}"
    )


def test_zero_trades_not_inconsistent():
    """Testa que 0 trades não gera falsa inconsistência."""
    metrics = SmartHealthMetrics()

    # Sem trades - success rate é 0 mas não é erro
    metrics.total_trades_executed = 0

    for _ in range(10):
        metrics.record_api_call(success=True, latency_ms=20)

    status, details = metrics.update_health_status()

    # Não deve ter alertas de inconsistência
    passed = len(metrics.inconsistency_alerts) == 0

    results.add_result(
        "Zero Trades - Sem falsa inconsistência",
        passed,
        f"Trades: 0, Alertas: {len(metrics.inconsistency_alerts)}"
    )


# =============================================================================
# TESTES DO SIGNAL PRIORITIZER
# =============================================================================

def test_signal_priority_classification():
    """Testa classificação de prioridade de sinais."""
    prioritizer = EmergencySignalPrioritizer()

    signals = [
        # CRITICAL: RSI muito baixo + score alto
        {'symbol': 'ARB-USD', 'total_score': 72, 'rsi': 15, 'volume_ratio': 1.8, 'signal': 'STRONG_BUY'},
        # HIGH: Score alto
        {'symbol': 'SOL-USD', 'total_score': 65, 'rsi': 45, 'volume_ratio': 1.2, 'signal': 'BUY'},
        # MEDIUM: Score médio
        {'symbol': 'ADA-USD', 'total_score': 50, 'rsi': 50, 'volume_ratio': 1.0, 'signal': 'BUY'},
        # LOW: Score baixo
        {'symbol': 'XRP-USD', 'total_score': 35, 'rsi': 55, 'volume_ratio': 0.8, 'signal': 'HOLD'},
    ]

    prioritized = prioritizer.prioritize_signals(signals, emergency_mode=True)

    # Verifica ordem (maior score primeiro)
    correct_order = (
        prioritized[0].symbol == 'ARB-USD' and
        prioritized[0].priority in [SignalPriority.CRITICAL, SignalPriority.HIGH]
    )

    results.add_result(
        "Signal Priority - Classificação correta",
        correct_order,
        f"Top: {prioritized[0].symbol} ({prioritized[0].priority.value})"
    )


def test_critical_signal_bypass():
    """Testa bypass de limites para sinais críticos."""
    prioritizer = EmergencySignalPrioritizer()

    # Sinal crítico: RSI oversold + score alto
    signal = {
        'symbol': 'ETH-USD',
        'total_score': 75,
        'rsi': 18,
        'volume_ratio': 2.0,
        'signal': 'STRONG_BUY'
    }

    should_bypass, reason = prioritizer.should_bypass_limits(signal, emergency_mode=True)

    passed = should_bypass
    results.add_result(
        "Critical Signal Bypass - Aprovado",
        passed,
        f"bypass={should_bypass}, reason={reason}"
    )


def test_max_critical_signals_limit():
    """Testa limite máximo de sinais CRITICAL (3)."""
    prioritizer = EmergencySignalPrioritizer()

    # 5 sinais todos fortes
    signals = [
        {'symbol': f'CRYPTO-{i}', 'total_score': 80, 'rsi': 15, 'volume_ratio': 2.0, 'signal': 'STRONG_BUY'}
        for i in range(5)
    ]

    prioritized = prioritizer.prioritize_signals(signals, emergency_mode=True)

    critical_count = sum(1 for p in prioritized if p.priority == SignalPriority.CRITICAL)

    passed = critical_count <= 3
    results.add_result(
        "Max Critical Signals - Limite de 3",
        passed,
        f"Critical count: {critical_count}"
    )


# =============================================================================
# TESTES DO SMART ALERT SYSTEM
# =============================================================================

def test_metric_consistency_alert():
    """Testa alerta para métricas inconsistentes."""
    alerts = SmartAlertSystem()

    # Cenário: HEALTHY mas success rate 0% com trades
    generated = alerts.check_metric_consistency(
        health_status="healthy",
        success_rate=0.0,
        api_success_rate=95.0,
        total_trades=10
    )

    passed = len(generated) > 0 and any(
        a.category == AlertCategory.METRIC_INCONSISTENCY for a in generated
    )

    results.add_result(
        "Metric Consistency Alert - Gerado corretamente",
        passed,
        f"Alertas gerados: {len(generated)}"
    )


def test_missed_opportunity_alert():
    """Testa alerta para oportunidades perdidas."""
    alerts = SmartAlertSystem()

    # 5 sinais fortes não executados
    signals = [
        {'symbol': f'CRYPTO-{i}', 'total_score': 75, 'rsi': 30, 'signal': 'BUY'}
        for i in range(5)
    ]

    generated = alerts.check_missed_opportunities(
        signals=signals,
        executed_symbols=[],
        reason="limit_reached"
    )

    passed = len(generated) > 0 and any(
        a.category == AlertCategory.MISSED_OPPORTUNITY for a in generated
    )

    results.add_result(
        "Missed Opportunity Alert - Gerado corretamente",
        passed,
        f"Alertas gerados: {len(generated)}"
    )


def test_rsi_extreme_sustained_alert():
    """Testa alerta para RSI extremo sustentado."""
    alerts = SmartAlertSystem()

    # Simula RSI extremo por vários minutos
    # Força timestamps antigos
    now = datetime.now()
    alerts.rsi_extreme_tracker['BTC-USD'] = [
        now - timedelta(minutes=35),
        now - timedelta(minutes=25),
        now - timedelta(minutes=15),
    ]

    # Adiciona mais um ponto atual
    alert = alerts.check_rsi_extreme(
        symbol='BTC-USD',
        rsi=18,
        threshold_low=20,
        sustained_minutes=30
    )

    passed = alert is not None and alert.category == AlertCategory.RSI_EXTREME
    results.add_result(
        "RSI Extreme Sustained Alert - Gerado após 30min",
        passed,
        f"Alert: {alert.title if alert else 'None'}"
    )


def test_trade_limit_alert():
    """Testa alerta quando limite de trades é problema."""
    alerts = SmartAlertSystem()

    alert = alerts.check_trade_limits(
        trades_today=20,
        max_trades=20,
        emergency_mode=False,
        pending_signals=5
    )

    passed = alert is not None and alert.category == AlertCategory.TRADE_LIMIT
    results.add_result(
        "Trade Limit Alert - Gerado corretamente",
        passed,
        f"Alert: {alert.title if alert else 'None'}"
    )


# =============================================================================
# TESTES DE INTEGRAÇÃO
# =============================================================================

def test_full_emergency_flow():
    """Testa fluxo completo de emergência."""
    trade_manager = EmergencyTradeManager()
    health_metrics = SmartHealthMetrics()
    prioritizer = EmergencySignalPrioritizer()
    alert_system = SmartAlertSystem()

    # Simula cenário: limite atingido, modo emergência ativo
    trade_manager.trades_today = 21

    # Sinal forte aparece
    signal = {
        'symbol': 'ARB-USD',
        'total_score': 72,
        'rsi': 15.3,
        'volume_ratio': 1.8,
        'signal': 'STRONG_BUY'
    }

    # 1. Prioriza sinal
    prioritized = prioritizer.prioritize_signals([signal], emergency_mode=True)
    is_critical = prioritized[0].priority == SignalPriority.CRITICAL

    # 2. Verifica se pode executar
    can_trade, decision, reasons = trade_manager.can_trade(signal, emergency_mode=True)

    # 3. Registra métricas se executado
    if can_trade:
        trade_manager.register_trade(signal, decision, emergency_mode=True)
        health_metrics.record_trade_result(profit=5.0, symbol=signal['symbol'])

    # 4. Verifica status
    status, details = health_metrics.update_health_status()

    passed = can_trade and decision == TradeDecision.APPROVED_CRITICAL
    results.add_result(
        "Full Emergency Flow - Execução completa",
        passed,
        f"can_trade={can_trade}, decision={decision.value}, status={status.value}"
    )


def test_circuit_breaker_simulation():
    """Simula sequência de perdas e circuit breakers."""
    # Nota: Este teste verifica a lógica conceitual
    # A implementação real do circuit breaker está no start.py

    losses = 0
    exposure_multiplier = 1.0
    paused = False
    stopped = False

    # Simula 5 perdas
    circuit_breaker_levels = [
        {'losses': 2, 'action': 'reduce_50', 'multiplier': 0.5},
        {'losses': 3, 'action': 'pause_5min', 'pause_minutes': 5},
        {'losses': 4, 'action': 'pause_15min', 'pause_minutes': 15},
        {'losses': 5, 'action': 'stop_trading', 'stop': True},
    ]

    for i in range(5):
        losses += 1

        for level in circuit_breaker_levels:
            if losses >= level['losses']:
                if 'multiplier' in level:
                    exposure_multiplier = level['multiplier']
                if 'pause_minutes' in level:
                    paused = True
                if level.get('stop'):
                    stopped = True

    passed = stopped and losses == 5
    results.add_result(
        "Circuit Breaker Simulation - Stop após 5 perdas",
        passed,
        f"losses={losses}, stopped={stopped}, multiplier={exposure_multiplier}"
    )


# =============================================================================
# EXECUTAR TESTES
# =============================================================================

def run_all_tests():
    """Executa todos os testes."""
    print("\n" + "=" * 70)
    print("🧪 VALIDAÇÃO DO SISTEMA V4.2 - EMERGENCY TRADING SYSTEM")
    print("=" * 70)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    # Emergency Trade Manager
    print("\n📦 EmergencyTradeManager Tests...")
    test_emergency_override_basic()
    test_critical_trade_allowance()
    test_position_size_multiplier()
    test_daily_counter_reset()

    # Smart Health Metrics
    print("\n📦 SmartHealthMetrics Tests...")
    test_success_rate_calculation()
    test_health_status_multidimensional()
    test_inconsistency_detection()
    test_zero_trades_not_inconsistent()

    # Signal Prioritizer
    print("\n📦 EmergencySignalPrioritizer Tests...")
    test_signal_priority_classification()
    test_critical_signal_bypass()
    test_max_critical_signals_limit()

    # Smart Alert System
    print("\n📦 SmartAlertSystem Tests...")
    test_metric_consistency_alert()
    test_missed_opportunity_alert()
    test_rsi_extreme_sustained_alert()
    test_trade_limit_alert()

    # Integration Tests
    print("\n📦 Integration Tests...")
    test_full_emergency_flow()
    test_circuit_breaker_simulation()

    # Mostra resumo
    return results.print_summary()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
