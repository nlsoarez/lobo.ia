"""
Framework de backtesting para testar estratégias em dados históricos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
import json

from data_collector import DataCollector
from signal_analyzer import SignalAnalyzer
from portfolio_manager import PortfolioManager
from config_loader import config
from system_logger import system_logger


class BacktestResult:
    """
    Armazena e analisa resultados de backtesting.
    """

    def __init__(self, trades: List[Dict], initial_capital: float, final_capital: float):
        """
        Inicializa resultado de backtest.

        Args:
            trades: Lista de trades executados.
            initial_capital: Capital inicial.
            final_capital: Capital final.
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self.final_capital = final_capital
        self.total_return = ((final_capital - initial_capital) / initial_capital) * 100

    def calculate_metrics(self) -> Dict:
        """
        Calcula métricas de performance.

        Returns:
            Dicionário com métricas completas.
        """
        if not self.trades:
            return self._empty_metrics()

        df = pd.DataFrame(self.trades)

        # Separa wins e losses
        wins = df[df['profit'] > 0]
        losses = df[df['profit'] < 0]

        # Métricas básicas
        total_trades = len(df)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # Profit metrics
        total_profit = df['profit'].sum()
        avg_profit = df['profit'].mean()
        max_profit = df['profit'].max()
        max_loss = df['profit'].min()

        # Win/Loss specifics
        avg_win = wins['profit'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['profit'].mean()) if len(losses) > 0 else 0

        # Profit factor
        gross_profit = wins['profit'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['profit'].sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Drawdown
        df['cumulative_profit'] = df['profit'].cumsum()
        running_max = df['cumulative_profit'].cummax()
        drawdown = (df['cumulative_profit'] - running_max)
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        max_drawdown_pct = (max_drawdown / self.initial_capital) * 100

        # Sharpe ratio (simplified - assumes risk-free rate = 0)
        returns = df['profit'] / self.initial_capital
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Recovery factor
        recovery_factor = total_profit / max_drawdown if max_drawdown > 0 else 0

        # Average holding period
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['holding_period'] = (pd.to_datetime(df['exit_time']) - pd.to_datetime(df['entry_time']))
            avg_holding_hours = df['holding_period'].dt.total_seconds().mean() / 3600
        else:
            avg_holding_hours = 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return_pct': self.total_return,
            'avg_profit_per_trade': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'recovery_factor': recovery_factor,
            'avg_holding_hours': avg_holding_hours,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital
        }

    def _empty_metrics(self) -> Dict:
        """Retorna métricas vazias quando não há trades."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'total_return_pct': 0,
            'avg_profit_per_trade': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_profit': 0,
            'max_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'recovery_factor': 0,
            'avg_holding_hours': 0,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital
        }

    def print_summary(self):
        """Imprime resumo formatado dos resultados."""
        metrics = self.calculate_metrics()

        print("\n" + "=" * 70)
        print("📊 RESULTADOS DO BACKTESTING")
        print("=" * 70)

        print(f"\n💰 PERFORMANCE GERAL:")
        print(f"  Capital Inicial:      R$ {metrics['initial_capital']:,.2f}")
        print(f"  Capital Final:        R$ {metrics['final_capital']:,.2f}")
        print(f"  Lucro Total:          R$ {metrics['total_profit']:,.2f}")
        print(f"  Retorno Total:        {metrics['total_return_pct']:.2f}%")

        print(f"\n📈 ESTATÍSTICAS DE TRADES:")
        print(f"  Total de Trades:      {metrics['total_trades']}")
        print(f"  Trades Vencedores:    {metrics['winning_trades']}")
        print(f"  Trades Perdedores:    {metrics['losing_trades']}")
        print(f"  Win Rate:             {metrics['win_rate']:.2f}%")

        print(f"\n💵 MÉTRICAS DE LUCRO:")
        print(f"  Lucro Médio/Trade:    R$ {metrics['avg_profit_per_trade']:.2f}")
        print(f"  Lucro Médio (Wins):   R$ {metrics['avg_win']:.2f}")
        print(f"  Perda Média (Losses): R$ {metrics['avg_loss']:.2f}")
        print(f"  Maior Lucro:          R$ {metrics['max_profit']:.2f}")
        print(f"  Maior Perda:          R$ {metrics['max_loss']:.2f}")

        print(f"\n📊 MÉTRICAS AVANÇADAS:")
        print(f"  Profit Factor:        {metrics['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:         R$ {metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        print(f"  Recovery Factor:      {metrics['recovery_factor']:.2f}")
        print(f"  Holding Médio:        {metrics['avg_holding_hours']:.1f}h")

        print("=" * 70 + "\n")

    def export_to_csv(self, filename: str = 'backtest_results.csv'):
        """
        Exporta trades para CSV.

        Args:
            filename: Nome do arquivo de saída.
        """
        df = pd.DataFrame(self.trades)
        df.to_csv(filename, index=False)
        print(f"✅ Resultados exportados para {filename}")


class Backtester:
    """
    Motor de backtesting para testar estratégias em dados históricos.
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        interval: str = '5m'
    ):
        """
        Inicializa o backtester.

        Args:
            symbol: Símbolo do ativo.
            start_date: Data inicial (YYYY-MM-DD).
            end_date: Data final (YYYY-MM-DD).
            initial_capital: Capital inicial.
            interval: Intervalo dos candles.
        """
        self.symbol = symbol
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.interval = interval

        self.trades: List[Dict] = []
        self.portfolio: Optional[PortfolioManager] = None

        system_logger.info(
            f"Backtester inicializado: {symbol} | "
            f"{start_date} → {end_date} | "
            f"Capital: R$ {initial_capital:,.2f}"
        )

    def load_data(self) -> pd.DataFrame:
        """
        Carrega dados históricos para o período especificado.

        Returns:
            DataFrame com dados OHLCV.
        """
        system_logger.info(f"Carregando dados históricos para {self.symbol}...")

        # Calcula período em dias
        days = (self.end_date - self.start_date).days

        # Ajusta período para yfinance
        if days <= 7:
            period = '7d'
        elif days <= 30:
            period = '1mo'
        elif days <= 90:
            period = '3mo'
        elif days <= 180:
            period = '6mo'
        else:
            period = '1y'

        collector = DataCollector(
            symbol=self.symbol,
            period=period,
            interval=self.interval
        )

        data = collector.get_data(use_cache=False)

        # Filtra pelo range de datas
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data[
                (data['datetime'] >= self.start_date) &
                (data['datetime'] <= self.end_date)
            ]

        system_logger.info(f"Dados carregados: {len(data)} candles")
        return data

    def run(self, strategy_config: Optional[Dict] = None) -> BacktestResult:
        """
        Executa backtest completo.

        Args:
            strategy_config: Configurações de estratégia (opcional).

        Returns:
            BacktestResult com resultados completos.
        """
        system_logger.info("🚀 Iniciando backtesting...")

        # Carrega dados
        data = self.load_data()

        if data.empty:
            system_logger.error("Nenhum dado disponível para backtest")
            return BacktestResult([], self.initial_capital, self.initial_capital)

        # Inicializa portfólio
        self.portfolio = PortfolioManager(initial_capital=self.initial_capital)

        # Simula trading bar-a-bar
        window_size = 100  # Janela mínima para indicadores

        for i in range(window_size, len(data)):
            # Pega janela de dados
            window_data = data.iloc[i - window_size:i].copy()
            current_bar = data.iloc[i]

            # Verifica posições abertas (stop-loss / take-profit)
            self._check_exit_conditions(current_bar)

            # Se não tem posição, procura sinal de entrada
            if not self.portfolio.has_position(self.symbol):
                signal = self._generate_signal(window_data)

                if signal and signal['action'] == 'BUY':
                    self._execute_entry(signal, current_bar)

        # Fecha posições abertas ao final
        if self.portfolio.has_position(self.symbol):
            final_bar = data.iloc[-1]
            self._force_exit(final_bar)

        # Cria resultado
        result = BacktestResult(
            trades=self.trades,
            initial_capital=self.initial_capital,
            final_capital=self.portfolio.current_capital
        )

        system_logger.info("✅ Backtesting concluído")
        return result

    def _generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Gera sinal de trading usando SignalAnalyzer."""
        try:
            analyzer = SignalAnalyzer(data, self.symbol)
            return analyzer.generate_signal()
        except Exception as e:
            system_logger.debug(f"Erro ao gerar sinal: {e}")
            return None

    def _execute_entry(self, signal: Dict, bar: pd.Series):
        """Executa entrada em posição."""
        price = float(bar['close'])
        quantity = self.portfolio.calculate_position_size(self.symbol, price)

        if quantity > 0:
            success = self.portfolio.open_position(self.symbol, quantity, price)

            if success:
                system_logger.debug(
                    f"📈 ENTRADA: {quantity} @ R$ {price:.2f} | "
                    f"Tempo: {bar.get('datetime', 'N/A')}"
                )

    def _check_exit_conditions(self, bar: pd.Series):
        """Verifica condições de saída (stop-loss / take-profit)."""
        if not self.portfolio.has_position(self.symbol):
            return

        price = float(bar['close'])
        reason = self.portfolio.check_stop_loss_take_profit(self.symbol, price)

        if reason:
            self._execute_exit(price, bar, reason)

    def _execute_exit(self, price: float, bar: pd.Series, reason: str):
        """Executa saída de posição."""
        result = self.portfolio.close_position(self.symbol, price, reason)

        if result:
            result['exit_datetime'] = bar.get('datetime', None)
            self.trades.append(result)

            system_logger.debug(
                f"📉 SAÍDA ({reason}): {result['quantity']} @ R$ {price:.2f} | "
                f"Lucro: R$ {result['profit']:.2f} | "
                f"Tempo: {bar.get('datetime', 'N/A')}"
            )

    def _force_exit(self, bar: pd.Series):
        """Força saída ao final do backtest."""
        price = float(bar['close'])
        self._execute_exit(price, bar, 'end_of_backtest')


def run_backtest_example():
    """Exemplo de uso do backtester."""
    backtester = Backtester(
        symbol='PETR4.SA',
        start_date='2024-01-01',
        end_date='2024-12-31',
        initial_capital=10000.0,
        interval='1d'
    )

    result = backtester.run()
    result.print_summary()
    result.export_to_csv('backtest_petr4.csv')


if __name__ == '__main__':
    run_backtest_example()
