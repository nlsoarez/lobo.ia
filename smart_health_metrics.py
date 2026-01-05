"""
Smart Health Metrics - Sistema de Métricas de Saúde V4.2
Corrige inconsistências em métricas e fornece status real do sistema.

Features:
- Cálculo correto de success rate baseado em trades executados
- Status multidimensional (latência, sucesso, conexões)
- Detecção de inconsistências automática
- Histórico de métricas para análise
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from system_logger import system_logger


class HealthStatus(Enum):
    """Status de saúde do sistema."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricStatus(Enum):
    """Status individual de métrica."""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class MetricPoint:
    """Ponto de métrica com timestamp."""
    timestamp: datetime
    value: float
    status: MetricStatus


@dataclass
class HealthDimension:
    """Dimensão de saúde do sistema."""
    name: str
    value: float
    status: MetricStatus
    threshold_green: float
    threshold_yellow: float
    message: str


@dataclass
class InconsistencyAlert:
    """Alerta de inconsistência detectada."""
    timestamp: datetime
    metric_name: str
    expected: str
    actual: str
    severity: str
    suggested_action: str


class SmartHealthMetrics:
    """
    Sistema de métricas de saúde inteligente.
    Detecta inconsistências e fornece status real do sistema.
    """

    def __init__(self):
        """Inicializa o sistema de métricas."""
        # Histórico de métricas (últimas 100 entradas)
        self.latency_history: deque = deque(maxlen=100)
        self.success_history: deque = deque(maxlen=100)
        self.trade_history: deque = deque(maxlen=500)

        # Contadores de trades
        self.total_trades_attempted = 0
        self.total_trades_executed = 0
        self.total_trades_successful = 0  # Com lucro
        self.total_trades_failed = 0  # Com prejuízo

        # Contadores de API
        self.api_calls_total = 0
        self.api_calls_success = 0
        self.api_calls_failed = 0

        # Alertas de inconsistência
        self.inconsistency_alerts: List[InconsistencyAlert] = []

        # Última verificação
        self.last_check_time = datetime.now()
        self.last_status = HealthStatus.UNKNOWN

        system_logger.info("SmartHealthMetrics V4.2 inicializado")

    def record_api_call(self, success: bool, latency_ms: float, source: str = "unknown"):
        """
        Registra uma chamada de API.

        Args:
            success: Se a chamada foi bem-sucedida
            latency_ms: Latência em milissegundos
            source: Fonte da chamada (ex: "yfinance", "binance")
        """
        self.api_calls_total += 1

        if success:
            self.api_calls_success += 1
        else:
            self.api_calls_failed += 1

        # Registra latência
        status = self._get_latency_status(latency_ms)
        self.latency_history.append(MetricPoint(
            timestamp=datetime.now(),
            value=latency_ms,
            status=status
        ))

    def record_trade_result(self, profit: float, symbol: str, trade_type: str = "crypto"):
        """
        Registra resultado de um trade.

        Args:
            profit: Lucro/prejuízo do trade
            symbol: Símbolo do ativo
            trade_type: Tipo do trade
        """
        self.total_trades_executed += 1

        if profit > 0:
            self.total_trades_successful += 1
        else:
            self.total_trades_failed += 1

        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'profit': profit,
            'success': profit > 0,
            'type': trade_type
        })

    def record_trade_attempt(self, symbol: str, executed: bool, reason: str = ""):
        """
        Registra tentativa de trade.

        Args:
            symbol: Símbolo do ativo
            executed: Se foi executado
            reason: Motivo se não executado
        """
        self.total_trades_attempted += 1

        if not executed:
            system_logger.debug(f"Trade não executado: {symbol} - {reason}")

    def calculate_success_rate(self) -> float:
        """
        Calcula taxa de sucesso REAL baseada em trades executados.

        Returns:
            Taxa de sucesso (0-100%)
        """
        if self.total_trades_executed == 0:
            return 0.0

        return (self.total_trades_successful / self.total_trades_executed) * 100

    def calculate_api_success_rate(self) -> float:
        """
        Calcula taxa de sucesso de chamadas de API.

        Returns:
            Taxa de sucesso (0-100%)
        """
        if self.api_calls_total == 0:
            return 100.0  # Sem chamadas = sem falhas

        return (self.api_calls_success / self.api_calls_total) * 100

    def calculate_avg_latency(self) -> float:
        """
        Calcula latência média recente.

        Returns:
            Latência média em ms
        """
        if not self.latency_history:
            return 0.0

        recent = [p.value for p in list(self.latency_history)[-20:]]
        return sum(recent) / len(recent) if recent else 0.0

    def _get_latency_status(self, latency_ms: float) -> MetricStatus:
        """Determina status baseado em latência."""
        if latency_ms < 50:
            return MetricStatus.GREEN
        elif latency_ms < 200:
            return MetricStatus.YELLOW
        return MetricStatus.RED

    def _get_success_status(self, success_rate: float) -> MetricStatus:
        """Determina status baseado em taxa de sucesso."""
        if success_rate >= 60:
            return MetricStatus.GREEN
        elif success_rate >= 40:
            return MetricStatus.YELLOW
        return MetricStatus.RED

    def _get_api_status(self, api_rate: float) -> MetricStatus:
        """Determina status baseado em taxa de sucesso de API."""
        if api_rate >= 95:
            return MetricStatus.GREEN
        elif api_rate >= 80:
            return MetricStatus.YELLOW
        return MetricStatus.RED

    def get_health_dimensions(self) -> List[HealthDimension]:
        """
        Retorna todas as dimensões de saúde do sistema.

        Returns:
            Lista de dimensões com status
        """
        avg_latency = self.calculate_avg_latency()
        success_rate = self.calculate_success_rate()
        api_rate = self.calculate_api_success_rate()

        dimensions = [
            HealthDimension(
                name="latency",
                value=avg_latency,
                status=self._get_latency_status(avg_latency),
                threshold_green=50,
                threshold_yellow=200,
                message=f"Latência média: {avg_latency:.0f}ms"
            ),
            HealthDimension(
                name="trade_success",
                value=success_rate,
                status=self._get_success_status(success_rate),
                threshold_green=60,
                threshold_yellow=40,
                message=f"Taxa de sucesso: {success_rate:.1f}% ({self.total_trades_successful}/{self.total_trades_executed})"
            ),
            HealthDimension(
                name="api_success",
                value=api_rate,
                status=self._get_api_status(api_rate),
                threshold_green=95,
                threshold_yellow=80,
                message=f"API sucesso: {api_rate:.1f}% ({self.api_calls_success}/{self.api_calls_total})"
            )
        ]

        return dimensions

    def update_health_status(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """
        Atualiza e retorna status de saúde do sistema.
        Usa lógica multidimensional: 2/3 GREEN = HEALTHY

        Returns:
            Tuple (status, detalhes)
        """
        dimensions = self.get_health_dimensions()

        green_count = sum(1 for d in dimensions if d.status == MetricStatus.GREEN)
        yellow_count = sum(1 for d in dimensions if d.status == MetricStatus.YELLOW)
        red_count = sum(1 for d in dimensions if d.status == MetricStatus.RED)

        # Lógica multidimensional
        if green_count >= 2:
            status = HealthStatus.HEALTHY
        elif red_count >= 2:
            status = HealthStatus.CRITICAL
        else:
            status = HealthStatus.DEGRADED

        # Verifica inconsistências
        self._check_inconsistencies(status, dimensions)

        self.last_status = status
        self.last_check_time = datetime.now()

        details = {
            'status': status.value,
            'dimensions': {
                d.name: {
                    'value': d.value,
                    'status': d.status.value,
                    'message': d.message
                }
                for d in dimensions
            },
            'summary': {
                'green': green_count,
                'yellow': yellow_count,
                'red': red_count
            },
            'trades': {
                'attempted': self.total_trades_attempted,
                'executed': self.total_trades_executed,
                'successful': self.total_trades_successful,
                'failed': self.total_trades_failed,
                'success_rate': self.calculate_success_rate()
            },
            'api': {
                'total': self.api_calls_total,
                'success': self.api_calls_success,
                'failed': self.api_calls_failed,
                'success_rate': self.calculate_api_success_rate()
            },
            'latency_avg_ms': self.calculate_avg_latency(),
            'last_check': self.last_check_time.isoformat(),
            'inconsistencies': len(self.inconsistency_alerts)
        }

        return status, details

    def _check_inconsistencies(self, status: HealthStatus, dimensions: List[HealthDimension]):
        """
        Verifica inconsistências nas métricas.

        Args:
            status: Status atual
            dimensions: Dimensões de saúde
        """
        # Inconsistência 1: HEALTHY mas success rate 0%
        success_dim = next((d for d in dimensions if d.name == "trade_success"), None)
        if success_dim:
            if status == HealthStatus.HEALTHY and success_dim.value == 0 and self.total_trades_executed == 0:
                # Não é inconsistência se não houve trades
                pass
            elif status == HealthStatus.HEALTHY and success_dim.value < 40 and self.total_trades_executed > 0:
                self._add_inconsistency(
                    metric_name="trade_success",
                    expected=f"Success rate >= 40% for HEALTHY status",
                    actual=f"Success rate = {success_dim.value:.1f}%",
                    severity="WARNING",
                    suggested_action="Verificar cálculo de success_rate e critérios de trades"
                )

        # Inconsistência 2: Muitas falhas de API mas status não CRITICAL
        api_dim = next((d for d in dimensions if d.name == "api_success"), None)
        if api_dim and api_dim.value < 50 and status != HealthStatus.CRITICAL:
            self._add_inconsistency(
                metric_name="api_success",
                expected="Status CRITICAL com API success < 50%",
                actual=f"Status {status.value} com API success = {api_dim.value:.1f}%",
                severity="WARNING",
                suggested_action="Verificar conexões de API e fallbacks"
            )

    def _add_inconsistency(
        self,
        metric_name: str,
        expected: str,
        actual: str,
        severity: str,
        suggested_action: str
    ):
        """Adiciona alerta de inconsistência."""
        alert = InconsistencyAlert(
            timestamp=datetime.now(),
            metric_name=metric_name,
            expected=expected,
            actual=actual,
            severity=severity,
            suggested_action=suggested_action
        )
        self.inconsistency_alerts.append(alert)

        # Mantém apenas últimos 50 alertas
        if len(self.inconsistency_alerts) > 50:
            self.inconsistency_alerts = self.inconsistency_alerts[-50:]

        system_logger.warning(
            f"Inconsistência detectada em {metric_name}: "
            f"Expected={expected}, Actual={actual}. "
            f"Ação: {suggested_action}"
        )

    def get_recent_inconsistencies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retorna inconsistências recentes.

        Args:
            limit: Número máximo de resultados

        Returns:
            Lista de inconsistências
        """
        recent = self.inconsistency_alerts[-limit:]
        return [
            {
                'timestamp': a.timestamp.isoformat(),
                'metric': a.metric_name,
                'expected': a.expected,
                'actual': a.actual,
                'severity': a.severity,
                'action': a.suggested_action
            }
            for a in recent
        ]

    def reset_counters(self):
        """Reseta todos os contadores (para novo período)."""
        self.total_trades_attempted = 0
        self.total_trades_executed = 0
        self.total_trades_successful = 0
        self.total_trades_failed = 0
        self.api_calls_total = 0
        self.api_calls_success = 0
        self.api_calls_failed = 0
        self.latency_history.clear()
        self.success_history.clear()
        self.trade_history.clear()
        self.inconsistency_alerts.clear()

        system_logger.info("SmartHealthMetrics: contadores resetados")


# Instância global
smart_health_metrics = SmartHealthMetrics()
