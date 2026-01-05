"""
Smart Alert System - Sistema de Alertas Inteligentes V4.2
Detecta anomalias, inconsistências e oportunidades perdidas.

Features:
- Alertas para métricas contraditórias
- Monitoramento de oportunidades não executadas
- RSI extremo por períodos prolongados
- Notificação multi-canal (log, dashboard, webhook)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

from config_loader import config
from system_logger import system_logger


class AlertLevel(Enum):
    """Níveis de severidade de alertas."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(Enum):
    """Categorias de alertas."""
    METRIC_INCONSISTENCY = "metric_inconsistency"
    MISSED_OPPORTUNITY = "missed_opportunity"
    RSI_EXTREME = "rsi_extreme"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    TRADE_LIMIT = "trade_limit"
    API_HEALTH = "api_health"


@dataclass
class Alert:
    """Estrutura de um alerta."""
    id: str
    timestamp: datetime
    level: AlertLevel
    category: AlertCategory
    title: str
    message: str
    details: Dict[str, Any]
    suggested_action: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None


@dataclass
class AlertStats:
    """Estatísticas de alertas."""
    total: int = 0
    by_level: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)
    unresolved: int = 0
    last_24h: int = 0


class SmartAlertSystem:
    """
    Sistema de alertas inteligentes com detecção automática de anomalias.
    """

    def __init__(self):
        """Inicializa o sistema de alertas."""
        # Histórico de alertas
        self.alerts: deque = deque(maxlen=1000)
        self.alert_counter = 0

        # Configurações
        self.notification_channels = config.get(
            'emergency.alerts.notification_channels',
            ['log', 'dashboard']
        )
        self.enable_metric_alerts = config.get('emergency.alerts.enable_metric_alerts', True)
        self.enable_opportunity_alerts = config.get('emergency.alerts.enable_opportunity_alerts', True)

        # Rastreamento de condições
        self.rsi_extreme_tracker: Dict[str, List[datetime]] = {}  # symbol -> timestamps
        self.missed_opportunities: deque = deque(maxlen=100)

        # Callbacks para notificação
        self._notification_callbacks: List[Callable] = []

        system_logger.info("SmartAlertSystem V4.2 inicializado")

    def _generate_alert_id(self) -> str:
        """Gera ID único para alerta."""
        self.alert_counter += 1
        return f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self.alert_counter:04d}"

    def create_alert(
        self,
        level: AlertLevel,
        category: AlertCategory,
        title: str,
        message: str,
        details: Dict[str, Any] = None,
        suggested_action: str = ""
    ) -> Alert:
        """
        Cria e registra um novo alerta.

        Args:
            level: Severidade do alerta
            category: Categoria do alerta
            title: Título curto
            message: Mensagem detalhada
            details: Detalhes adicionais
            suggested_action: Ação sugerida

        Returns:
            Alerta criado
        """
        alert = Alert(
            id=self._generate_alert_id(),
            timestamp=datetime.now(),
            level=level,
            category=category,
            title=title,
            message=message,
            details=details or {},
            suggested_action=suggested_action
        )

        self.alerts.append(alert)
        self._notify(alert)

        return alert

    def _notify(self, alert: Alert):
        """
        Envia notificação do alerta pelos canais configurados.

        Args:
            alert: Alerta a notificar
        """
        # Log
        if 'log' in self.notification_channels:
            log_method = {
                AlertLevel.INFO: system_logger.info,
                AlertLevel.WARNING: system_logger.warning,
                AlertLevel.CRITICAL: system_logger.error,
                AlertLevel.EMERGENCY: system_logger.critical
            }.get(alert.level, system_logger.info)

            log_method(
                f"[{alert.level.value.upper()}] {alert.category.value}: {alert.title} - {alert.message}"
            )

        # Callbacks customizados
        for callback in self._notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                system_logger.error(f"Erro em callback de alerta: {e}")

    def register_notification_callback(self, callback: Callable):
        """
        Registra callback para notificações.

        Args:
            callback: Função que recebe Alert como argumento
        """
        self._notification_callbacks.append(callback)

    # ==================== VERIFICAÇÕES AUTOMÁTICAS ====================

    def check_metric_consistency(
        self,
        health_status: str,
        success_rate: float,
        api_success_rate: float,
        total_trades: int
    ) -> List[Alert]:
        """
        Verifica consistência das métricas de saúde.

        Args:
            health_status: Status reportado (healthy, degraded, critical)
            success_rate: Taxa de sucesso de trades
            api_success_rate: Taxa de sucesso de API
            total_trades: Total de trades executados

        Returns:
            Lista de alertas gerados
        """
        if not self.enable_metric_alerts:
            return []

        alerts = []

        # Alerta 1: HEALTHY mas success rate muito baixo (com trades)
        if health_status.lower() == "healthy" and total_trades > 0 and success_rate < 30:
            alert = self.create_alert(
                level=AlertLevel.WARNING,
                category=AlertCategory.METRIC_INCONSISTENCY,
                title="Métrica de sucesso inconsistente",
                message=f"Status HEALTHY mas success rate de {success_rate:.1f}% ({total_trades} trades)",
                details={
                    'health_status': health_status,
                    'success_rate': success_rate,
                    'total_trades': total_trades
                },
                suggested_action="Verificar cálculo de success_rate e critérios de trades"
            )
            alerts.append(alert)

        # Alerta 2: Success rate 0% mas HEALTHY (sem trades não é problema)
        if health_status.lower() == "healthy" and success_rate == 0 and total_trades > 5:
            alert = self.create_alert(
                level=AlertLevel.CRITICAL,
                category=AlertCategory.METRIC_INCONSISTENCY,
                title="Nenhum trade bem-sucedido",
                message=f"0% de sucesso em {total_trades} trades mas status HEALTHY",
                details={
                    'health_status': health_status,
                    'success_rate': success_rate,
                    'total_trades': total_trades
                },
                suggested_action="Verificar estratégia de trading e parâmetros de entrada"
            )
            alerts.append(alert)

        # Alerta 3: API com muitas falhas
        if api_success_rate < 80:
            level = AlertLevel.CRITICAL if api_success_rate < 50 else AlertLevel.WARNING
            alert = self.create_alert(
                level=level,
                category=AlertCategory.API_HEALTH,
                title="Alta taxa de falhas de API",
                message=f"Taxa de sucesso de API: {api_success_rate:.1f}%",
                details={'api_success_rate': api_success_rate},
                suggested_action="Verificar conectividade e status das APIs"
            )
            alerts.append(alert)

        return alerts

    def check_missed_opportunities(
        self,
        signals: List[Dict[str, Any]],
        executed_symbols: List[str],
        reason: str = "limit_reached"
    ) -> List[Alert]:
        """
        Verifica oportunidades perdidas.

        Args:
            signals: Sinais disponíveis
            executed_symbols: Símbolos que foram executados
            reason: Motivo de não execução

        Returns:
            Lista de alertas
        """
        if not self.enable_opportunity_alerts:
            return []

        alerts = []

        # Filtra sinais fortes não executados
        strong_missed = [
            s for s in signals
            if s.get('symbol') not in executed_symbols
            and s.get('total_score', s.get('score', 0)) >= 70
        ]

        if len(strong_missed) >= 3:
            # Registra oportunidades perdidas
            for signal in strong_missed[:5]:
                self.missed_opportunities.append({
                    'timestamp': datetime.now(),
                    'symbol': signal.get('symbol'),
                    'score': signal.get('total_score', signal.get('score', 0)),
                    'rsi': signal.get('rsi', 50),
                    'reason': reason
                })

            alert = self.create_alert(
                level=AlertLevel.WARNING,
                category=AlertCategory.MISSED_OPPORTUNITY,
                title="Múltiplas oportunidades perdidas",
                message=f"{len(strong_missed)} sinais fortes (score >= 70) não executados",
                details={
                    'count': len(strong_missed),
                    'signals': [
                        {
                            'symbol': s.get('symbol'),
                            'score': s.get('total_score', s.get('score', 0)),
                            'rsi': s.get('rsi', 50)
                        }
                        for s in strong_missed[:5]
                    ],
                    'reason': reason
                },
                suggested_action="Considerar ativar modo emergência ou aumentar limites"
            )
            alerts.append(alert)

        return alerts

    def check_rsi_extreme(
        self,
        symbol: str,
        rsi: float,
        threshold_low: float = 20,
        threshold_high: float = 80,
        sustained_minutes: int = 30
    ) -> Optional[Alert]:
        """
        Verifica se RSI está em nível extremo por período prolongado.

        Args:
            symbol: Símbolo do ativo
            rsi: Valor atual do RSI
            threshold_low: Limite inferior (oversold)
            threshold_high: Limite superior (overbought)
            sustained_minutes: Minutos de sustentação para alerta

        Returns:
            Alerta se condição sustentada
        """
        now = datetime.now()
        is_extreme = rsi <= threshold_low or rsi >= threshold_high

        if symbol not in self.rsi_extreme_tracker:
            self.rsi_extreme_tracker[symbol] = []

        # Limpa entradas antigas (> 1 hora)
        self.rsi_extreme_tracker[symbol] = [
            t for t in self.rsi_extreme_tracker[symbol]
            if now - t < timedelta(hours=1)
        ]

        if is_extreme:
            self.rsi_extreme_tracker[symbol].append(now)

            # Verifica se é sustentado
            if len(self.rsi_extreme_tracker[symbol]) >= 3:
                first_extreme = self.rsi_extreme_tracker[symbol][0]
                duration = (now - first_extreme).total_seconds() / 60

                if duration >= sustained_minutes:
                    condition = "OVERSOLD" if rsi <= threshold_low else "OVERBOUGHT"

                    alert = self.create_alert(
                        level=AlertLevel.WARNING,
                        category=AlertCategory.RSI_EXTREME,
                        title=f"RSI extremo sustentado: {symbol}",
                        message=f"{symbol} em {condition} (RSI={rsi:.1f}) por {duration:.0f} minutos",
                        details={
                            'symbol': symbol,
                            'rsi': rsi,
                            'condition': condition,
                            'duration_minutes': duration
                        },
                        suggested_action=f"Avaliar entrada {'compra' if condition == 'OVERSOLD' else 'venda'} em {symbol}"
                    )

                    # Limpa tracker após alerta
                    self.rsi_extreme_tracker[symbol] = []

                    return alert
        else:
            # RSI voltou ao normal, limpa tracker
            self.rsi_extreme_tracker[symbol] = []

        return None

    def check_trade_limits(
        self,
        trades_today: int,
        max_trades: int,
        emergency_mode: bool,
        pending_signals: int
    ) -> Optional[Alert]:
        """
        Verifica se limites de trades estão impedindo oportunidades.

        Args:
            trades_today: Trades executados hoje
            max_trades: Limite máximo
            emergency_mode: Se modo emergência está ativo
            pending_signals: Sinais pendentes de execução

        Returns:
            Alerta se limite é problema
        """
        if trades_today >= max_trades and pending_signals > 0:
            level = AlertLevel.CRITICAL if pending_signals >= 5 else AlertLevel.WARNING

            alert = self.create_alert(
                level=level,
                category=AlertCategory.TRADE_LIMIT,
                title="Limite de trades atingido",
                message=f"{trades_today}/{max_trades} trades executados, {pending_signals} sinais pendentes",
                details={
                    'trades_today': trades_today,
                    'max_trades': max_trades,
                    'pending_signals': pending_signals,
                    'emergency_mode': emergency_mode
                },
                suggested_action="Ativar modo emergência para permitir trades adicionais" if not emergency_mode
                else "Considerar aumentar critical_trade_allowance"
            )
            return alert

        return None

    # ==================== CONSULTAS ====================

    def get_alerts(
        self,
        level: AlertLevel = None,
        category: AlertCategory = None,
        since: datetime = None,
        unresolved_only: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Consulta alertas com filtros.

        Args:
            level: Filtro por nível
            category: Filtro por categoria
            since: Alertas após esta data
            unresolved_only: Apenas não resolvidos
            limit: Número máximo

        Returns:
            Lista de alertas como dicionários
        """
        filtered = []

        for alert in reversed(list(self.alerts)):
            if len(filtered) >= limit:
                break

            if level and alert.level != level:
                continue
            if category and alert.category != category:
                continue
            if since and alert.timestamp < since:
                continue
            if unresolved_only and alert.resolved:
                continue

            filtered.append({
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'category': alert.category.value,
                'title': alert.title,
                'message': alert.message,
                'details': alert.details,
                'suggested_action': alert.suggested_action,
                'resolved': alert.resolved,
                'acknowledged': alert.acknowledged
            })

        return filtered

    def get_stats(self) -> AlertStats:
        """
        Retorna estatísticas de alertas.

        Returns:
            Estatísticas agregadas
        """
        now = datetime.now()
        stats = AlertStats()

        for alert in self.alerts:
            stats.total += 1

            # Por nível
            level_key = alert.level.value
            stats.by_level[level_key] = stats.by_level.get(level_key, 0) + 1

            # Por categoria
            cat_key = alert.category.value
            stats.by_category[cat_key] = stats.by_category.get(cat_key, 0) + 1

            # Não resolvidos
            if not alert.resolved:
                stats.unresolved += 1

            # Últimas 24h
            if now - alert.timestamp < timedelta(hours=24):
                stats.last_24h += 1

        return stats

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Marca alerta como resolvido.

        Args:
            alert_id: ID do alerta

        Returns:
            True se encontrado e resolvido
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                system_logger.info(f"Alerta {alert_id} marcado como resolvido")
                return True
        return False

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Marca alerta como reconhecido.

        Args:
            alert_id: ID do alerta

        Returns:
            True se encontrado e reconhecido
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                system_logger.info(f"Alerta {alert_id} reconhecido")
                return True
        return False

    def get_recent_missed_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retorna oportunidades perdidas recentes.

        Returns:
            Lista de oportunidades perdidas
        """
        recent = list(self.missed_opportunities)[-limit:]
        return [
            {
                'timestamp': o['timestamp'].isoformat(),
                'symbol': o['symbol'],
                'score': o['score'],
                'rsi': o['rsi'],
                'reason': o['reason']
            }
            for o in recent
        ]


# Instância global
smart_alert_system = SmartAlertSystem()
