"""
Emergency Dashboard V4.2
Dashboard de monitoramento em tempo real para o sistema de emergência.

Execute: streamlit run emergency_dashboard.py --server.port 8502
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import json
import time

sys.path.insert(0, '.')

from config_loader import config

# Importa módulos V4.2
try:
    from emergency_trade_manager import emergency_trade_manager
    from smart_health_metrics import smart_health_metrics
    from emergency_signal_prioritizer import emergency_signal_prioritizer
    from smart_alert_system import smart_alert_system, AlertLevel
    HAS_V42 = True
except ImportError:
    HAS_V42 = False

try:
    from crypto_scanner import CryptoScanner
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Lobo IA V4.2 - Emergency Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS customizado
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(145deg, #161b22 0%, #21262d 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f6fc;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
    }

    /* Status badges */
    .badge-healthy { background: #238636; color: white; }
    .badge-degraded { background: #d29922; color: black; }
    .badge-critical { background: #da3633; color: white; }
    .badge-emergency { background: #f85149; color: white; animation: pulse 1s infinite; }

    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* Alert cards */
    .alert-critical {
        border-left: 4px solid #da3633;
        background: rgba(218, 54, 51, 0.1);
    }

    .alert-warning {
        border-left: 4px solid #d29922;
        background: rgba(210, 153, 34, 0.1);
    }

    .alert-info {
        border-left: 4px solid #58a6ff;
        background: rgba(88, 166, 255, 0.1);
    }

    .alert-item {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }

    /* Progress bars */
    .progress-container {
        background: #21262d;
        border-radius: 10px;
        overflow: hidden;
        height: 24px;
        margin: 0.5rem 0;
    }

    .progress-bar {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .progress-green { background: linear-gradient(90deg, #238636, #2ea043); }
    .progress-yellow { background: linear-gradient(90deg, #d29922, #e3b341); }
    .progress-red { background: linear-gradient(90deg, #da3633, #f85149); }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f0f6fc;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def get_trade_manager_status():
    """Obtém status do trade manager."""
    if not HAS_V42:
        return {
            'trades_today': 0,
            'max_regular_trades': 20,
            'emergency_trades_today': 0,
            'critical_overrides_today': 0,
            'critical_allowance': 5,
            'remaining_regular': 20,
            'remaining_emergency': 15,
            'remaining_critical': 5
        }
    return emergency_trade_manager.get_status()


def get_health_status():
    """Obtém status de saúde."""
    if not HAS_V42:
        return {
            'status': 'unknown',
            'dimensions': {},
            'trades': {'success_rate': 0},
            'api': {'success_rate': 100}
        }

    status, details = smart_health_metrics.update_health_status()
    details['status'] = status.value
    return details


def get_alerts():
    """Obtém alertas recentes."""
    if not HAS_V42:
        return []

    return smart_alert_system.get_alerts(limit=20)


def get_alert_stats():
    """Obtém estatísticas de alertas."""
    if not HAS_V42:
        return {'total': 0, 'unresolved': 0, 'last_24h': 0, 'by_level': {}}

    stats = smart_alert_system.get_stats()
    return {
        'total': stats.total,
        'unresolved': stats.unresolved,
        'last_24h': stats.last_24h,
        'by_level': stats.by_level
    }


def get_critical_signals():
    """Obtém sinais críticos atuais."""
    if not HAS_CRYPTO or not HAS_V42:
        return []

    try:
        scanner = CryptoScanner()
        signals = scanner.scan_crypto_market()

        # Prioriza sinais
        prioritized = emergency_signal_prioritizer.get_critical_signals(
            signals, emergency_mode=True
        )

        return [
            {
                'symbol': p.symbol,
                'score': p.priority_score,
                'priority': p.priority.value,
                'criteria_met': p.criteria_met[:3],
                'bypass': p.bypass_allowed
            }
            for p in prioritized[:5]
        ]
    except Exception as e:
        return []


def get_missed_opportunities():
    """Obtém oportunidades perdidas."""
    if not HAS_V42:
        return []

    return smart_alert_system.get_recent_missed_opportunities(limit=5)


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Renderiza cabeçalho."""
    col1, col2, col3 = st.columns([2, 4, 2])

    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 2.5rem;">🐺</span>
            <div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #f0f6fc;">LOBO IA V4.2</div>
                <div style="font-size: 0.75rem; color: #8b949e;">Emergency Dashboard</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        health = get_health_status()
        status = health.get('status', 'unknown')

        badge_class = {
            'healthy': 'badge-healthy',
            'degraded': 'badge-degraded',
            'critical': 'badge-critical'
        }.get(status, 'badge-critical')

        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem;">
            <span class="badge {badge_class}">{status.upper()}</span>
            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #8b949e;">
                {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        tm = get_trade_manager_status()
        emergency_active = tm.get('trades_today', 0) > tm.get('max_regular_trades', 20)

        if emergency_active:
            st.markdown("""
            <div style="text-align: right;">
                <span class="badge badge-emergency">🚨 EMERGENCY MODE</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: right;">
                <span class="badge badge-healthy">✅ NORMAL MODE</span>
            </div>
            """, unsafe_allow_html=True)


def render_trade_limits():
    """Renderiza limites de trades."""
    st.markdown('<div class="section-header">📊 Limites de Trades</div>', unsafe_allow_html=True)

    tm = get_trade_manager_status()

    col1, col2, col3 = st.columns(3)

    with col1:
        trades = tm.get('trades_today', 0)
        max_regular = tm.get('max_regular_trades', 20)
        pct = min(100, (trades / max_regular) * 100) if max_regular > 0 else 0

        color = 'green' if pct < 75 else 'yellow' if pct < 100 else 'red'

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Trades Regulares</div>
            <div class="metric-value">{trades} / {max_regular}</div>
            <div class="progress-container">
                <div class="progress-bar progress-{color}" style="width: {pct}%;">{pct:.0f}%</div>
            </div>
            <div style="color: #8b949e; font-size: 0.75rem;">Restantes: {tm.get('remaining_regular', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        emergency = tm.get('emergency_trades_today', 0)
        max_emergency = tm.get('max_emergency_trades', 35) - tm.get('max_regular_trades', 20)
        pct = min(100, (emergency / max_emergency) * 100) if max_emergency > 0 else 0

        color = 'green' if pct < 50 else 'yellow' if pct < 80 else 'red'

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Emergency Trades</div>
            <div class="metric-value">{emergency} / {max_emergency}</div>
            <div class="progress-container">
                <div class="progress-bar progress-{color}" style="width: {pct}%;">{pct:.0f}%</div>
            </div>
            <div style="color: #8b949e; font-size: 0.75rem;">Restantes: {tm.get('remaining_emergency', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        critical = tm.get('critical_overrides_today', 0)
        max_critical = tm.get('critical_allowance', 5)
        pct = min(100, (critical / max_critical) * 100) if max_critical > 0 else 0

        color = 'green' if pct < 50 else 'yellow' if pct < 80 else 'red'

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Critical Overrides</div>
            <div class="metric-value">{critical} / {max_critical}</div>
            <div class="progress-container">
                <div class="progress-bar progress-{color}" style="width: {pct}%;">{pct:.0f}%</div>
            </div>
            <div style="color: #8b949e; font-size: 0.75rem;">Bypass disponíveis: {tm.get('remaining_critical', 0)}</div>
        </div>
        """, unsafe_allow_html=True)


def render_health_metrics():
    """Renderiza métricas de saúde."""
    st.markdown('<div class="section-header">❤️ Health Metrics</div>', unsafe_allow_html=True)

    health = get_health_status()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        success_rate = health.get('trades', {}).get('success_rate', 0)
        color = '#238636' if success_rate >= 60 else '#d29922' if success_rate >= 40 else '#da3633'

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value" style="color: {color};">{success_rate:.1f}%</div>
            <div style="color: #8b949e; font-size: 0.75rem;">
                {health.get('trades', {}).get('successful', 0)} wins / {health.get('trades', {}).get('executed', 0)} trades
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        api_rate = health.get('api', {}).get('success_rate', 100)
        color = '#238636' if api_rate >= 95 else '#d29922' if api_rate >= 80 else '#da3633'

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">API Success</div>
            <div class="metric-value" style="color: {color};">{api_rate:.1f}%</div>
            <div style="color: #8b949e; font-size: 0.75rem;">
                {health.get('api', {}).get('success', 0)} / {health.get('api', {}).get('total', 0)} calls
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        latency = health.get('latency_avg_ms', 0)
        color = '#238636' if latency < 50 else '#d29922' if latency < 200 else '#da3633'

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Latência Média</div>
            <div class="metric-value" style="color: {color};">{latency:.0f}ms</div>
            <div style="color: #8b949e; font-size: 0.75rem;">
                {'✅ OK' if latency < 100 else '⚠️ Elevada'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        inconsistencies = health.get('inconsistencies', 0)
        color = '#238636' if inconsistencies == 0 else '#d29922' if inconsistencies < 3 else '#da3633'

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Inconsistências</div>
            <div class="metric-value" style="color: {color};">{inconsistencies}</div>
            <div style="color: #8b949e; font-size: 0.75rem;">
                {'✅ Nenhuma' if inconsistencies == 0 else '⚠️ Verificar'}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_critical_signals():
    """Renderiza sinais críticos."""
    st.markdown('<div class="section-header">🔥 Sinais Críticos</div>', unsafe_allow_html=True)

    signals = get_critical_signals()

    if not signals:
        st.info("Nenhum sinal crítico no momento")
        return

    for signal in signals:
        bypass_badge = "🔓 BYPASS" if signal.get('bypass') else ""
        criteria = ", ".join(signal.get('criteria_met', [])[:2])

        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid #f85149;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.2rem; font-weight: 600;">{signal['symbol']}</span>
                    <span class="badge badge-critical" style="margin-left: 0.5rem;">{signal['priority'].upper()}</span>
                    <span style="margin-left: 0.5rem; color: #f85149;">{bypass_badge}</span>
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #f0f6fc;">
                    {signal['score']:.0f}
                </div>
            </div>
            <div style="color: #8b949e; font-size: 0.75rem; margin-top: 0.5rem;">
                {criteria}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_alerts():
    """Renderiza alertas."""
    st.markdown('<div class="section-header">🔔 Alertas Recentes</div>', unsafe_allow_html=True)

    alerts = get_alerts()
    stats = get_alert_stats()

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total (24h)", stats.get('last_24h', 0))
    with col2:
        st.metric("Não resolvidos", stats.get('unresolved', 0))
    with col3:
        critical_count = stats.get('by_level', {}).get('critical', 0)
        st.metric("Críticos", critical_count)

    # Lista de alertas
    if not alerts:
        st.info("Nenhum alerta registrado")
        return

    for alert in alerts[:10]:
        level = alert.get('level', 'info')
        alert_class = f"alert-{level}"
        icon = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}.get(level, '⚪')

        timestamp = alert.get('timestamp', '')[:19]

        st.markdown(f"""
        <div class="alert-item {alert_class}">
            <div style="display: flex; justify-content: space-between;">
                <span>{icon} <strong>{alert.get('title', 'Alerta')}</strong></span>
                <span style="color: #8b949e; font-size: 0.75rem;">{timestamp}</span>
            </div>
            <div style="color: #8b949e; font-size: 0.85rem; margin-top: 0.25rem;">
                {alert.get('message', '')}
            </div>
            <div style="color: #58a6ff; font-size: 0.75rem; margin-top: 0.25rem;">
                💡 {alert.get('suggested_action', 'Verificar logs')}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_missed_opportunities():
    """Renderiza oportunidades perdidas."""
    st.markdown('<div class="section-header">❌ Oportunidades Perdidas</div>', unsafe_allow_html=True)

    opportunities = get_missed_opportunities()

    if not opportunities:
        st.success("✅ Nenhuma oportunidade perdida recentemente")
        return

    for opp in opportunities:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid #d29922;">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-weight: 600;">{opp.get('symbol', 'N/A')}</span>
                <span style="color: #8b949e; font-size: 0.75rem;">{opp.get('timestamp', '')[:19]}</span>
            </div>
            <div style="color: #8b949e; font-size: 0.85rem;">
                Score: {opp.get('score', 0):.0f} | RSI: {opp.get('rsi', 0):.1f} | Motivo: {opp.get('reason', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_circuit_breaker_status():
    """Renderiza status do circuit breaker."""
    st.markdown('<div class="section-header">⚡ Circuit Breaker</div>', unsafe_allow_html=True)

    # Configuração atual
    cb_config = config.get('circuit_breakers', {})
    levels = cb_config.get('levels', [])

    if not levels:
        st.info("Circuit breakers não configurados")
        return

    # Mostra níveis
    for level in levels:
        losses = level.get('losses', 0)
        action = level.get('action', 'unknown')

        icon = {
            'reduce_50': '📉',
            'pause_5min': '⏸️',
            'pause_15min': '⏸️',
            'stop_trading': '⛔'
        }.get(action, '⚠️')

        st.markdown(f"""
        <div style="padding: 0.5rem; background: #21262d; border-radius: 6px; margin-bottom: 0.5rem;">
            {icon} <strong>{losses} perdas</strong> → {action}
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    render_header()

    st.markdown("<hr style='border-color: #30363d;'>", unsafe_allow_html=True)

    # Limites de trades
    render_trade_limits()

    st.markdown("<hr style='border-color: #30363d;'>", unsafe_allow_html=True)

    # Health metrics
    render_health_metrics()

    st.markdown("<hr style='border-color: #30363d;'>", unsafe_allow_html=True)

    # Layout em duas colunas
    col_left, col_right = st.columns([3, 2])

    with col_left:
        render_critical_signals()
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        render_alerts()

    with col_right:
        render_missed_opportunities()
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        render_circuit_breaker_status()

    # Auto-refresh
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; background: #21262d;
                padding: 8px 12px; border-radius: 6px; font-size: 0.75rem; color: #8b949e;">
        <span style="display: inline-block; width: 6px; height: 6px;
                     background: #3fb950; border-radius: 50%; margin-right: 6px;
                     animation: blink 1s infinite;"></span>
        Auto-refresh: 30s
    </div>
    <style>
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
    </style>
    """, unsafe_allow_html=True)

    # Refresh automático
    time.sleep(30)
    st.rerun()


if __name__ == "__main__":
    main()
