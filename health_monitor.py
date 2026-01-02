"""
Sistema de monitoramento de saúde do trading.
V4.1 - Monitora performance, latência e disponibilidade.
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from threading import Lock, Thread
from collections import deque
from enum import Enum

from config_loader import config
from system_logger import system_logger


class HealthStatus(Enum):
    """Status de saúde do sistema."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricCollector:
    """
    Coleta e agrega métricas ao longo do tempo.
    """

    def __init__(self, window_size: int = 100):
        """
        Inicializa coletor de métricas.

        Args:
            window_size: Número de amostras a manter.
        """
        self._metrics: Dict[str, deque] = {}
        self._timestamps: Dict[str, deque] = {}
        self._lock = Lock()
        self.window_size = window_size

    def record(self, metric_name: str, value: float):
        """
        Registra uma métrica.

        Args:
            metric_name: Nome da métrica.
            value: Valor a registrar.
        """
        with self._lock:
            if metric_name not in self._metrics:
                self._metrics[metric_name] = deque(maxlen=self.window_size)
                self._timestamps[metric_name] = deque(maxlen=self.window_size)

            self._metrics[metric_name].append(value)
            self._timestamps[metric_name].append(datetime.now())

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Obtém estatísticas de uma métrica.

        Args:
            metric_name: Nome da métrica.

        Returns:
            Dicionário com min, max, avg, count, last.
        """
        with self._lock:
            if metric_name not in self._metrics or len(self._metrics[metric_name]) == 0:
                return {'min': 0, 'max': 0, 'avg': 0, 'count': 0, 'last': 0}

            values = list(self._metrics[metric_name])
            return {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'count': len(values),
                'last': values[-1]
            }

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Retorna estatísticas de todas as métricas."""
        with self._lock:
            return {
                name: self.get_stats(name)
                for name in self._metrics.keys()
            }


class SymbolHealthTracker:
    """
    Rastreia saúde individual de cada símbolo.
    """

    def __init__(self):
        """Inicializa tracker de símbolos."""
        # {symbol: {'successes': int, 'failures': int, 'latencies': deque, 'last_success': datetime}}
        self._symbols: Dict[str, Dict] = {}
        self._lock = Lock()

    def record_success(self, symbol: str, latency_ms: float):
        """Registra sucesso na coleta."""
        with self._lock:
            if symbol not in self._symbols:
                self._symbols[symbol] = {
                    'successes': 0,
                    'failures': 0,
                    'latencies': deque(maxlen=50),
                    'last_success': None,
                    'last_error': None
                }

            self._symbols[symbol]['successes'] += 1
            self._symbols[symbol]['latencies'].append(latency_ms)
            self._symbols[symbol]['last_success'] = datetime.now()

    def record_failure(self, symbol: str, error: str):
        """Registra falha na coleta."""
        with self._lock:
            if symbol not in self._symbols:
                self._symbols[symbol] = {
                    'successes': 0,
                    'failures': 0,
                    'latencies': deque(maxlen=50),
                    'last_success': None,
                    'last_error': None
                }

            self._symbols[symbol]['failures'] += 1
            self._symbols[symbol]['last_error'] = error

    def get_reliability(self, symbol: str) -> float:
        """
        Retorna taxa de sucesso do símbolo (0-1).
        """
        with self._lock:
            if symbol not in self._symbols:
                return 0.5  # Neutro para desconhecidos

            s = self._symbols[symbol]
            total = s['successes'] + s['failures']
            return s['successes'] / total if total > 0 else 0.5

    def get_avg_latency(self, symbol: str) -> float:
        """Retorna latência média em ms."""
        with self._lock:
            if symbol not in self._symbols or len(self._symbols[symbol]['latencies']) == 0:
                return 0

            latencies = list(self._symbols[symbol]['latencies'])
            return sum(latencies) / len(latencies)

    def get_problematic_symbols(self, threshold: float = 0.5) -> List[str]:
        """
        Retorna símbolos com taxa de sucesso abaixo do limiar.
        """
        with self._lock:
            problematic = []
            for symbol, stats in self._symbols.items():
                total = stats['successes'] + stats['failures']
                if total >= 3:  # Mínimo de tentativas
                    reliability = stats['successes'] / total
                    if reliability < threshold:
                        problematic.append(symbol)
            return problematic

    def get_all_stats(self) -> Dict[str, Dict]:
        """Retorna estatísticas de todos os símbolos."""
        with self._lock:
            result = {}
            for symbol, stats in self._symbols.items():
                total = stats['successes'] + stats['failures']
                latencies = list(stats['latencies'])

                result[symbol] = {
                    'reliability': stats['successes'] / total if total > 0 else 0,
                    'successes': stats['successes'],
                    'failures': stats['failures'],
                    'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
                    'last_success': stats['last_success'].isoformat() if stats['last_success'] else None,
                    'last_error': stats['last_error']
                }
            return result


class HealthMonitor:
    """
    Monitor central de saúde do sistema de trading.

    Monitora:
    - Performance de coleta de dados
    - Latência de requisições
    - Taxa de sucesso por símbolo
    - Métricas de trading
    - Alertas proativos
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Inicializa o monitor."""
        self.metrics = MetricCollector()
        self.symbol_tracker = SymbolHealthTracker()

        # Configurações de alertas
        health_config = config.get('health', {})
        self.latency_warning_ms = health_config.get('latency_warning_ms', 2000)
        self.latency_critical_ms = health_config.get('latency_critical_ms', 5000)
        self.min_success_rate = health_config.get('min_success_rate', 0.7)

        # Estado do sistema
        self._status = HealthStatus.UNKNOWN
        self._last_check = None
        self._alerts: List[Dict] = []
        self._alerts_lock = Lock()

        # Métricas de trading
        self._trading_metrics = {
            'trades_executed': 0,
            'trades_failed': 0,
            'signals_generated': 0,
            'signals_ignored': 0,
            'last_trade_time': None
        }
        self._trading_lock = Lock()

        system_logger.info("HealthMonitor V4.1 inicializado")

    def record_fetch(self, symbol: str, success: bool, latency_ms: float, error: str = None):
        """
        Registra resultado de uma coleta de dados.

        Args:
            symbol: Símbolo coletado.
            success: Se a coleta foi bem-sucedida.
            latency_ms: Latência em milissegundos.
            error: Mensagem de erro (se falhou).
        """
        # Registra métricas gerais
        self.metrics.record('fetch_latency_ms', latency_ms)
        self.metrics.record('fetch_success', 1 if success else 0)

        # Registra por símbolo
        if success:
            self.symbol_tracker.record_success(symbol, latency_ms)
        else:
            self.symbol_tracker.record_failure(symbol, error or "Unknown error")

        # Verifica alertas
        if latency_ms > self.latency_critical_ms:
            self._add_alert('CRITICAL', f"Latência crítica para {symbol}: {latency_ms:.0f}ms")
        elif latency_ms > self.latency_warning_ms:
            self._add_alert('WARNING', f"Latência alta para {symbol}: {latency_ms:.0f}ms")

    def record_trade(self, success: bool, symbol: str = None, details: Dict = None):
        """
        Registra execução de trade.

        Args:
            success: Se o trade foi executado com sucesso.
            symbol: Símbolo do trade.
            details: Detalhes adicionais.
        """
        with self._trading_lock:
            if success:
                self._trading_metrics['trades_executed'] += 1
                self._trading_metrics['last_trade_time'] = datetime.now()
            else:
                self._trading_metrics['trades_failed'] += 1

        self.metrics.record('trade_success', 1 if success else 0)

    def record_signal(self, generated: bool, ignored: bool = False):
        """
        Registra geração de sinal.

        Args:
            generated: Se um sinal foi gerado.
            ignored: Se o sinal foi ignorado.
        """
        with self._trading_lock:
            if generated:
                self._trading_metrics['signals_generated'] += 1
            if ignored:
                self._trading_metrics['signals_ignored'] += 1

    def _add_alert(self, level: str, message: str):
        """Adiciona alerta à lista."""
        with self._alerts_lock:
            self._alerts.append({
                'level': level,
                'message': message,
                'timestamp': datetime.now().isoformat()
            })

            # Mantém apenas últimos 100 alertas
            if len(self._alerts) > 100:
                self._alerts = self._alerts[-100:]

        system_logger.warning(f"[ALERT {level}] {message}")

    def check_system_health(self) -> Dict[str, Any]:
        """
        Verifica saúde geral do sistema.

        Returns:
            Dicionário com status detalhado.
        """
        self._last_check = datetime.now()

        # Coleta métricas
        fetch_stats = self.metrics.get_stats('fetch_latency_ms')
        success_stats = self.metrics.get_stats('fetch_success')

        # Calcula status
        issues = []

        # Verifica latência
        if fetch_stats['avg'] > self.latency_critical_ms:
            issues.append(f"Latência crítica: {fetch_stats['avg']:.0f}ms")
            self._status = HealthStatus.CRITICAL
        elif fetch_stats['avg'] > self.latency_warning_ms:
            issues.append(f"Latência alta: {fetch_stats['avg']:.0f}ms")
            if self._status != HealthStatus.CRITICAL:
                self._status = HealthStatus.WARNING

        # Verifica taxa de sucesso
        if success_stats['count'] > 0:
            success_rate = success_stats['avg']
            if success_rate < self.min_success_rate:
                issues.append(f"Taxa de sucesso baixa: {success_rate*100:.1f}%")
                if self._status != HealthStatus.CRITICAL:
                    self._status = HealthStatus.WARNING

        # Se não há issues, está saudável
        if not issues:
            self._status = HealthStatus.HEALTHY

        # Obtém símbolos problemáticos
        problematic_symbols = self.symbol_tracker.get_problematic_symbols()

        # Trading metrics
        with self._trading_lock:
            trading = self._trading_metrics.copy()

        # Alertas recentes
        with self._alerts_lock:
            recent_alerts = self._alerts[-10:]

        return {
            'status': self._status.value,
            'checked_at': self._last_check.isoformat(),
            'data_sources': {
                'avg_latency_ms': fetch_stats['avg'],
                'max_latency_ms': fetch_stats['max'],
                'success_rate': success_stats['avg'] * 100 if success_stats['count'] > 0 else 0,
                'total_fetches': int(success_stats['count'])
            },
            'symbol_reliability': {
                'total_tracked': len(self.symbol_tracker.get_all_stats()),
                'problematic_count': len(problematic_symbols),
                'problematic_symbols': problematic_symbols[:10]
            },
            'trading': trading,
            'issues': issues,
            'recent_alerts': recent_alerts,
            'performance_metrics': self.metrics.get_all_metrics()
        }

    def get_symbol_reliability(self) -> Dict[str, Dict]:
        """Retorna confiabilidade de todos os símbolos."""
        return self.symbol_tracker.get_all_stats()

    def get_performance_summary(self) -> str:
        """Retorna resumo de performance em texto."""
        health = self.check_system_health()

        lines = [
            f"=== HealthMonitor - {health['checked_at']} ===",
            f"Status: {health['status'].upper()}",
            "",
            "Data Sources:",
            f"  Latência média: {health['data_sources']['avg_latency_ms']:.0f}ms",
            f"  Taxa de sucesso: {health['data_sources']['success_rate']:.1f}%",
            f"  Total fetches: {health['data_sources']['total_fetches']}",
            "",
            "Símbolos:",
            f"  Rastreados: {health['symbol_reliability']['total_tracked']}",
            f"  Problemáticos: {health['symbol_reliability']['problematic_count']}",
            "",
            "Trading:",
            f"  Trades executados: {health['trading']['trades_executed']}",
            f"  Trades falhados: {health['trading']['trades_failed']}",
            f"  Sinais gerados: {health['trading']['signals_generated']}",
        ]

        if health['issues']:
            lines.append("")
            lines.append("Issues:")
            for issue in health['issues']:
                lines.append(f"  - {issue}")

        return "\n".join(lines)

    def reset(self):
        """Reseta todas as métricas."""
        self.metrics = MetricCollector()
        self.symbol_tracker = SymbolHealthTracker()

        with self._trading_lock:
            self._trading_metrics = {
                'trades_executed': 0,
                'trades_failed': 0,
                'signals_generated': 0,
                'signals_ignored': 0,
                'last_trade_time': None
            }

        with self._alerts_lock:
            self._alerts.clear()

        self._status = HealthStatus.UNKNOWN


# Instância global
health_monitor = HealthMonitor()


# Decorador para monitorar tempo de execução
def monitor_execution(operation_name: str):
    """
    Decorador para monitorar tempo de execução de funções.

    Args:
        operation_name: Nome da operação para métricas.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start) * 1000
                health_monitor.metrics.record(f'{operation_name}_latency_ms', elapsed_ms)
                health_monitor.metrics.record(f'{operation_name}_success', 1)
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                health_monitor.metrics.record(f'{operation_name}_latency_ms', elapsed_ms)
                health_monitor.metrics.record(f'{operation_name}_success', 0)
                raise

        return wrapper
    return decorator


if __name__ == "__main__":
    # Teste do monitor
    monitor = HealthMonitor()

    # Simula algumas coletas
    import random
    symbols = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'EMBR3.SA']

    for _ in range(20):
        symbol = random.choice(symbols)
        latency = random.uniform(100, 3000)
        success = random.random() > 0.2

        monitor.record_fetch(symbol, success, latency, "Test error" if not success else None)

    # Simula trades
    monitor.record_trade(True, 'PETR4.SA')
    monitor.record_trade(False, 'VALE3.SA')

    # Verifica saúde
    print(monitor.get_performance_summary())
