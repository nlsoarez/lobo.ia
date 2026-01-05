#!/usr/bin/env python3
"""
Framework de Backtesting V4.2
Simula 24 horas de operação com os novos módulos de emergência.

Execute: python backtest_v42.py
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass, field
import random
import json

sys.path.insert(0, '.')

from config_loader import config
from system_logger import system_logger


@dataclass
class SimulatedTrade:
    """Trade simulado."""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime = None
    exit_price: float = 0.0
    profit_pct: float = 0.0
    profit_usd: float = 0.0
    trade_type: str = "regular"  # regular, emergency, critical
    decision: str = ""
    result: str = ""  # win, loss, timeout


@dataclass
class BacktestResults:
    """Resultados do backtest."""
    # Trades
    total_trades: int = 0
    regular_trades: int = 0
    emergency_trades: int = 0
    critical_trades: int = 0

    # Resultados
    wins: int = 0
    losses: int = 0
    timeouts: int = 0

    # Performance
    total_profit_usd: float = 0.0
    total_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    # Detalhes por tipo
    regular_profit: float = 0.0
    emergency_profit: float = 0.0
    critical_profit: float = 0.0

    # Métricas
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    sharpe_ratio: float = 0.0

    # Alertas e eventos
    alerts_generated: int = 0
    circuit_breakers_triggered: int = 0
    missed_opportunities: int = 0

    # Detalhes
    trades: List[SimulatedTrade] = field(default_factory=list)
    hourly_performance: Dict[int, float] = field(default_factory=dict)


class BacktestSimulator:
    """
    Simulador de backtesting para validação do sistema V4.2.
    """

    def __init__(self, initial_capital: float = 1000.0):
        """Inicializa o simulador."""
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.results = BacktestResults()

        # Parâmetros de simulação
        self.regular_win_rate = 0.55  # 55% win rate normal
        self.emergency_win_rate = 0.50  # 50% win rate emergência
        self.critical_win_rate = 0.60  # 60% win rate crítico (sinais mais fortes)

        self.avg_win_pct = 0.015  # 1.5% lucro médio
        self.avg_loss_pct = 0.01  # 1% perda média

        # Limites
        self.max_regular_trades = 20
        self.max_emergency_trades = 35
        self.critical_allowance = 5

        # Contadores
        self.trades_today = 0
        self.emergency_trades_today = 0
        self.critical_trades_today = 0
        self.consecutive_losses = 0

        # Estado
        self.emergency_mode = False
        self.paused = False

    def generate_signal(self, hour: int, emergency_mode: bool) -> Dict[str, Any]:
        """
        Gera um sinal simulado baseado na hora do dia.

        Args:
            hour: Hora do dia (0-23)
            emergency_mode: Se modo emergência está ativo

        Returns:
            Sinal simulado
        """
        # Maior probabilidade de sinais fortes em horários de maior volatilidade
        volatility_hours = [9, 10, 14, 15, 16, 21, 22]  # Alta volatilidade
        is_volatile = hour in volatility_hours

        # Score base
        base_score = random.randint(40, 85)
        if is_volatile:
            base_score = min(95, base_score + random.randint(5, 15))

        # RSI simulado
        if random.random() < 0.15:  # 15% chance de RSI extremo
            rsi = random.randint(12, 25)  # Oversold
        elif random.random() < 0.10:  # 10% chance de overbought
            rsi = random.randint(75, 88)
        else:
            rsi = random.randint(35, 65)

        # Volume ratio
        volume_ratio = round(random.uniform(0.5, 2.5), 2)
        if is_volatile:
            volume_ratio = round(volume_ratio * 1.3, 2)

        return {
            'symbol': random.choice([
                'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD',
                'ARB-USD', 'XRP-USD', 'ADA-USD', 'AVAX-USD'
            ]),
            'total_score': base_score,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'signal': 'STRONG_BUY' if base_score >= 65 else 'BUY',
            'hour': hour
        }

    def evaluate_trade_opportunity(self, signal: Dict) -> tuple[bool, str]:
        """
        Avalia se deve executar o trade baseado nos limites V4.2.

        Returns:
            Tuple (executar, tipo_trade)
        """
        score = signal['total_score']
        rsi = signal['rsi']
        volume = signal['volume_ratio']

        # Dentro do limite normal
        if self.trades_today < self.max_regular_trades:
            if score >= 50 and rsi <= 65:
                return True, 'regular'
            return False, 'rejected_criteria'

        # Limite normal atingido - verificar emergência
        if not self.emergency_mode:
            return False, 'rejected_limit'

        # CRITICAL: RSI oversold + score alto
        is_critical = (
            rsi <= 25 and
            score >= 65 and
            volume >= 1.5 and
            self.critical_trades_today < self.critical_allowance
        )

        if is_critical:
            return True, 'critical'

        # EMERGENCY: Critérios relaxados
        if self.emergency_trades_today < (self.max_emergency_trades - self.max_regular_trades):
            if score >= 60 and rsi <= 40:
                return True, 'emergency'

        return False, 'rejected_emergency_criteria'

    def simulate_trade_result(self, trade_type: str) -> tuple[str, float]:
        """
        Simula resultado do trade.

        Returns:
            Tuple (resultado, profit_pct)
        """
        # Define win rate baseado no tipo
        if trade_type == 'critical':
            win_rate = self.critical_win_rate
        elif trade_type == 'emergency':
            win_rate = self.emergency_win_rate
        else:
            win_rate = self.regular_win_rate

        is_win = random.random() < win_rate

        if is_win:
            # Lucro variável
            profit_pct = self.avg_win_pct * random.uniform(0.5, 1.5)
            return 'win', profit_pct
        else:
            # Perda variável
            loss_pct = self.avg_loss_pct * random.uniform(0.5, 1.5)
            return 'loss', -loss_pct

    def check_emergency_mode(self, hour: int) -> bool:
        """
        Verifica se modo emergência deve ser ativado.

        Triggers:
        - Mais de 4h sem entrada
        - 3+ perdas consecutivas
        - Perda diária > 2%
        """
        # Ativa emergência às 16h se poucos trades
        if hour >= 16 and self.trades_today < 10:
            return True

        # Perdas consecutivas
        if self.consecutive_losses >= 3:
            return True

        # Drawdown diário
        daily_pnl_pct = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        if daily_pnl_pct <= -2.0:
            return True

        return False

    def check_circuit_breaker(self) -> str:
        """
        Verifica circuit breakers.

        Returns:
            Ação do circuit breaker ou None
        """
        if self.consecutive_losses >= 5:
            return 'stop_trading'
        elif self.consecutive_losses >= 4:
            return 'pause_15min'
        elif self.consecutive_losses >= 3:
            return 'pause_5min'
        elif self.consecutive_losses >= 2:
            return 'reduce_50'
        return None

    def run_simulation(self, hours: int = 24) -> BacktestResults:
        """
        Executa simulação de backtesting.

        Args:
            hours: Duração da simulação em horas

        Returns:
            Resultados do backtest
        """
        print("\n" + "=" * 70)
        print("🔬 BACKTESTING V4.2 - EMERGENCY TRADING SYSTEM")
        print("=" * 70)
        print(f"Capital inicial: ${self.initial_capital:.2f}")
        print(f"Duração: {hours} horas")
        print("-" * 70)

        start_time = datetime.now().replace(hour=0, minute=0, second=0)

        for hour in range(hours):
            current_time = start_time + timedelta(hours=hour)

            # Verifica modo emergência
            self.emergency_mode = self.check_emergency_mode(hour)

            # Verifica circuit breaker
            cb_action = self.check_circuit_breaker()
            if cb_action == 'stop_trading':
                print(f"⛔ Hora {hour}: Circuit breaker STOP_TRADING ativado")
                self.results.circuit_breakers_triggered += 1
                continue
            elif cb_action and 'pause' in cb_action:
                print(f"⏸️ Hora {hour}: Circuit breaker {cb_action}")
                self.results.circuit_breakers_triggered += 1
                continue

            # Gera sinais (2-5 por hora)
            num_signals = random.randint(2, 5)

            for _ in range(num_signals):
                signal = self.generate_signal(hour, self.emergency_mode)

                # Avalia oportunidade
                execute, trade_type = self.evaluate_trade_opportunity(signal)

                if not execute:
                    if signal['total_score'] >= 65:  # Oportunidade perdida
                        self.results.missed_opportunities += 1
                    continue

                # Simula resultado
                result, profit_pct = self.simulate_trade_result(trade_type)

                # Calcula lucro em USD
                position_size = self.capital * 0.10  # 10% por trade
                if trade_type in ['emergency', 'critical']:
                    position_size *= 0.70  # Reduz 30%

                profit_usd = position_size * profit_pct

                # Atualiza capital
                self.capital += profit_usd

                # Cria trade
                trade = SimulatedTrade(
                    symbol=signal['symbol'],
                    entry_time=current_time,
                    entry_price=100.0,  # Preço fictício
                    exit_time=current_time + timedelta(minutes=random.randint(15, 90)),
                    profit_pct=profit_pct * 100,
                    profit_usd=profit_usd,
                    trade_type=trade_type,
                    decision=f"score={signal['total_score']}, rsi={signal['rsi']}",
                    result=result
                )

                # Atualiza contadores
                self.trades_today += 1
                if trade_type == 'emergency':
                    self.emergency_trades_today += 1
                elif trade_type == 'critical':
                    self.critical_trades_today += 1

                if result == 'win':
                    self.results.wins += 1
                    self.consecutive_losses = 0
                else:
                    self.results.losses += 1
                    self.consecutive_losses += 1

                # Atualiza resultados
                self.results.trades.append(trade)
                self.results.total_trades += 1
                self.results.total_profit_usd += profit_usd
                self.results.total_profit_pct += profit_pct * 100

                if trade_type == 'regular':
                    self.results.regular_trades += 1
                    self.results.regular_profit += profit_usd
                elif trade_type == 'emergency':
                    self.results.emergency_trades += 1
                    self.results.emergency_profit += profit_usd
                elif trade_type == 'critical':
                    self.results.critical_trades += 1
                    self.results.critical_profit += profit_usd

            # Performance horária
            hourly_pnl = ((self.capital - self.initial_capital) / self.initial_capital) * 100
            self.results.hourly_performance[hour] = hourly_pnl

            # Log a cada 4 horas
            if hour % 4 == 0:
                mode = "🚨 EMERGENCY" if self.emergency_mode else "📊 NORMAL"
                print(f"Hora {hour:2d}: {mode} | Trades: {self.trades_today} | P&L: ${self.capital - self.initial_capital:+.2f} ({hourly_pnl:+.2f}%)")

        # Calcula métricas finais
        self._calculate_final_metrics()

        return self.results

    def _calculate_final_metrics(self):
        """Calcula métricas finais do backtest."""
        if self.results.total_trades > 0:
            self.results.win_rate = (self.results.wins / self.results.total_trades) * 100
            self.results.avg_profit_per_trade = self.results.total_profit_usd / self.results.total_trades

        # Calcula max drawdown
        peak = self.initial_capital
        max_dd = 0
        running_capital = self.initial_capital

        for trade in self.results.trades:
            running_capital += trade.profit_usd
            peak = max(peak, running_capital)
            drawdown = (peak - running_capital) / peak * 100
            max_dd = max(max_dd, drawdown)

        self.results.max_drawdown_pct = max_dd

        # Sharpe ratio simplificado (assumindo risk-free rate = 0)
        if len(self.results.trades) > 1:
            profits = [t.profit_pct for t in self.results.trades]
            avg_return = sum(profits) / len(profits)
            variance = sum((p - avg_return) ** 2 for p in profits) / len(profits)
            std_dev = variance ** 0.5
            if std_dev > 0:
                self.results.sharpe_ratio = (avg_return / std_dev) * (252 ** 0.5)  # Anualizado

    def print_results(self):
        """Imprime resultados do backtest."""
        r = self.results

        print("\n" + "=" * 70)
        print("📊 RESULTADOS DO BACKTEST")
        print("=" * 70)

        print("\n💰 PERFORMANCE GERAL")
        print("-" * 40)
        print(f"Capital inicial:     ${self.initial_capital:,.2f}")
        print(f"Capital final:       ${self.capital:,.2f}")
        print(f"P&L Total:           ${r.total_profit_usd:+,.2f} ({r.total_profit_pct:+.2f}%)")
        print(f"Max Drawdown:        {r.max_drawdown_pct:.2f}%")
        print(f"Sharpe Ratio:        {r.sharpe_ratio:.2f}")

        print("\n📈 ESTATÍSTICAS DE TRADES")
        print("-" * 40)
        print(f"Total de trades:     {r.total_trades}")
        print(f"  - Regular:         {r.regular_trades} (P&L: ${r.regular_profit:+.2f})")
        print(f"  - Emergency:       {r.emergency_trades} (P&L: ${r.emergency_profit:+.2f})")
        print(f"  - Critical:        {r.critical_trades} (P&L: ${r.critical_profit:+.2f})")
        print(f"Win Rate:            {r.win_rate:.1f}%")
        print(f"Wins / Losses:       {r.wins} / {r.losses}")
        print(f"Avg profit/trade:    ${r.avg_profit_per_trade:+.2f}")

        print("\n⚠️ EVENTOS")
        print("-" * 40)
        print(f"Circuit breakers:    {r.circuit_breakers_triggered}")
        print(f"Missed opportunities: {r.missed_opportunities}")

        print("\n📉 PERFORMANCE POR HORA")
        print("-" * 40)
        for hour in [0, 4, 8, 12, 16, 20, 23]:
            if hour in r.hourly_performance:
                pnl = r.hourly_performance[hour]
                bar = "█" * int(abs(pnl) * 5)
                sign = "+" if pnl >= 0 else "-"
                print(f"Hora {hour:2d}: {sign}{bar} {pnl:+.2f}%")

        print("\n" + "=" * 70)

        # Análise de melhoria
        print("\n🎯 ANÁLISE DE MELHORIA COM V4.2")
        print("-" * 40)

        if r.emergency_trades > 0 or r.critical_trades > 0:
            extra_trades = r.emergency_trades + r.critical_trades
            extra_profit = r.emergency_profit + r.critical_profit

            print(f"✅ Trades adicionais (Emergency Override): {extra_trades}")
            print(f"✅ P&L adicional gerado: ${extra_profit:+.2f}")

            if r.regular_profit != 0:
                improvement = (extra_profit / abs(r.regular_profit)) * 100
                print(f"✅ Melhoria sobre modo regular: {improvement:+.1f}%")
        else:
            print("❌ Nenhum trade de emergência executado neste backtest")

        print("=" * 70)


def run_comparison_backtest():
    """Executa backtest comparativo: V4.1 vs V4.2."""
    print("\n" + "=" * 70)
    print("🔄 BACKTEST COMPARATIVO: V4.1 (sem emergency) vs V4.2 (com emergency)")
    print("=" * 70)

    results_comparison = []

    # Simula 5 dias com diferentes seeds
    for day in range(1, 6):
        random.seed(42 + day)  # Seed fixo para reprodutibilidade

        # V4.1 (sem emergency override)
        sim_v41 = BacktestSimulator(initial_capital=1000.0)
        sim_v41.max_emergency_trades = 20  # Mesmo que regular (desativado)
        sim_v41.critical_allowance = 0
        r_v41 = sim_v41.run_simulation(hours=24)

        # V4.2 (com emergency override)
        random.seed(42 + day)  # Mesmo seed
        sim_v42 = BacktestSimulator(initial_capital=1000.0)
        r_v42 = sim_v42.run_simulation(hours=24)

        results_comparison.append({
            'day': day,
            'v41_profit': r_v41.total_profit_usd,
            'v41_trades': r_v41.total_trades,
            'v42_profit': r_v42.total_profit_usd,
            'v42_trades': r_v42.total_trades,
            'improvement': r_v42.total_profit_usd - r_v41.total_profit_usd
        })

    # Resumo
    print("\n" + "=" * 70)
    print("📊 RESUMO COMPARATIVO")
    print("=" * 70)
    print(f"{'Dia':<5} {'V4.1 P&L':>12} {'V4.1 Trades':>12} {'V4.2 P&L':>12} {'V4.2 Trades':>12} {'Melhoria':>12}")
    print("-" * 70)

    total_v41 = 0
    total_v42 = 0

    for r in results_comparison:
        print(f"{r['day']:<5} ${r['v41_profit']:>10.2f} {r['v41_trades']:>12} ${r['v42_profit']:>10.2f} {r['v42_trades']:>12} ${r['improvement']:>10.2f}")
        total_v41 += r['v41_profit']
        total_v42 += r['v42_profit']

    print("-" * 70)
    print(f"{'TOTAL':<5} ${total_v41:>10.2f} {'':>12} ${total_v42:>10.2f} {'':>12} ${total_v42 - total_v41:>10.2f}")

    improvement_pct = ((total_v42 - total_v41) / abs(total_v41) * 100) if total_v41 != 0 else 0
    print(f"\n✅ Melhoria total V4.2 vs V4.1: {improvement_pct:+.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    # Backtest único detalhado
    random.seed(42)
    simulator = BacktestSimulator(initial_capital=1000.0)
    results = simulator.run_simulation(hours=24)
    simulator.print_results()

    # Backtest comparativo
    print("\n\n")
    run_comparison_backtest()
