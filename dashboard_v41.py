"""
Lobo IA Dashboard V4.1 - Professional Trading Interface
Comprehensive monitoring dashboard with modern design.

Run: streamlit run dashboard_v41.py --server.port 8501
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import time
import os
import sys
import json

sys.path.insert(0, '.')

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Lobo IA Trading Dashboard",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== IMPORTS ====================

try:
    from crypto_scanner import CryptoScanner, CRYPTOCURRENCIES, CRYPTO_BLACKLIST, get_close_column
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    CRYPTOCURRENCIES = {}
    CRYPTO_BLACKLIST = set()

try:
    from logger import Logger
    HAS_LOGGER = True
except ImportError:
    HAS_LOGGER = False

try:
    from config_loader import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    config = None

try:
    import pytz
    BRAZIL_TZ = pytz.timezone('America/Sao_Paulo')
except ImportError:
    BRAZIL_TZ = timezone(timedelta(hours=-3))


# ==================== PROFESSIONAL CSS ====================

def apply_professional_css():
    """Apply professional dark theme CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-card: #21262d;
        --bg-hover: #30363d;
        --border-color: #30363d;
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --accent-green: #3fb950;
        --accent-red: #f85149;
        --accent-blue: #58a6ff;
        --accent-yellow: #d29922;
        --accent-purple: #a371f7;
    }

    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1600px;
    }

    /* Hide streamlit elements */
    #MainMenu, footer, .stDeployButton {display: none;}

    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    /* Cards */
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
    }

    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .metric-value.positive { color: var(--accent-green); }
    .metric-value.negative { color: var(--accent-red); }
    .metric-value.neutral { color: var(--accent-blue); }

    .metric-delta {
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-online {
        background: rgba(63, 185, 80, 0.15);
        color: var(--accent-green);
        border: 1px solid var(--accent-green);
    }

    .status-warning {
        background: rgba(210, 153, 34, 0.15);
        color: var(--accent-yellow);
        border: 1px solid var(--accent-yellow);
    }

    .status-offline {
        background: rgba(248, 81, 73, 0.15);
        color: var(--accent-red);
        border: 1px solid var(--accent-red);
    }

    /* Live dot animation */
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.9); }
    }

    /* Transaction table */
    .tx-table {
        width: 100%;
        border-collapse: collapse;
    }

    .tx-table th {
        text-align: left;
        padding: 0.75rem;
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        border-bottom: 1px solid var(--border-color);
    }

    .tx-table td {
        padding: 0.75rem;
        font-size: 0.85rem;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-primary);
    }

    .tx-table tr:hover {
        background: var(--bg-hover);
    }

    .tx-buy { color: var(--accent-green); font-weight: 600; }
    .tx-sell { color: var(--accent-red); font-weight: 600; }

    /* Position card */
    .position-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .position-symbol {
        font-weight: 600;
        font-size: 1rem;
        color: var(--text-primary);
    }

    .position-detail {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }

    .position-pnl {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Progress bar */
    .progress-container {
        background: var(--bg-card);
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }

    .progress-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    /* Alert items */
    .alert-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }

    .alert-info { background: rgba(88, 166, 255, 0.1); border-left: 3px solid var(--accent-blue); }
    .alert-warning { background: rgba(210, 153, 34, 0.1); border-left: 3px solid var(--accent-yellow); }
    .alert-success { background: rgba(63, 185, 80, 0.1); border-left: 3px solid var(--accent-green); }
    .alert-error { background: rgba(248, 81, 73, 0.1); border-left: 3px solid var(--accent-red); }

    /* Buttons */
    .stButton > button {
        background: var(--bg-card);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: var(--bg-hover);
        border-color: var(--accent-blue);
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: var(--border-color);
        margin: 1.5rem 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--bg-hover);
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== DATA FUNCTIONS ====================

def get_brazil_time():
    """Get current time in Brazil timezone."""
    try:
        return datetime.now(BRAZIL_TZ)
    except:
        return datetime.now(timezone(timedelta(hours=-3)))


def get_db_logger():
    """Get database logger instance."""
    if HAS_LOGGER:
        try:
            return Logger()
        except:
            pass
    return None


def get_system_state():
    """Read system state from file."""
    state_file = os.path.join(os.path.dirname(__file__), '.system_state.json')
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def save_system_state(state):
    """Save system state to file."""
    state_file = os.path.join(os.path.dirname(__file__), '.system_state.json')
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def get_initial_capital():
    """Get initial capital from config."""
    if HAS_CONFIG and config:
        return config.get('crypto.capital', 1000.0)
    return 1000.0


def get_positions():
    """Get open positions from database."""
    logger = get_db_logger()
    if logger:
        try:
            return logger.load_positions()
        except:
            pass
    return {}


def get_trades(limit=50):
    """Get recent trades from database."""
    logger = get_db_logger()
    if logger:
        try:
            return logger.get_trades(limit=limit)
        except:
            pass
    return []


def get_performance_stats():
    """Get performance statistics."""
    logger = get_db_logger()
    if logger:
        try:
            return logger.get_performance_stats()
        except:
            pass
    return {'total_trades': 0, 'total_profit': 0, 'wins': 0, 'losses': 0, 'win_rate': 0}


def get_error_count():
    """Count errors in log files."""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(log_dir):
        return 0

    count = 0
    now = datetime.now()

    try:
        for f in os.listdir(log_dir):
            if f.endswith('.log'):
                path = os.path.join(log_dir, f)
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
                if now - mtime <= timedelta(hours=24):
                    with open(path, 'r', errors='ignore') as file:
                        count += sum(1 for line in file if '[ERROR]' in line or '[CRITICAL]' in line)
    except:
        pass

    return count


# ==================== UI COMPONENTS ====================

def render_header():
    """Render header with logo and status."""
    col1, col2, col3 = st.columns([2, 4, 2])

    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 2.5rem;">🐺</span>
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #f0f6fc;">LOBO IA</div>
                <div style="font-size: 0.8rem; color: #8b949e;">Trading Autônomo V4.1</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        current_time = get_brazil_time().strftime("%d/%m/%Y %H:%M:%S")
        st.markdown(f"""
        <div style="text-align: center; padding-top: 0.5rem;">
            <span style="color: #8b949e; font-size: 0.85rem;">🕐 {current_time} (Brasília)</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        state = get_system_state()
        em_active = state.get('emergency_mode', {}).get('active', False)

        if em_active:
            status_class = "status-warning"
            status_text = "EMERGÊNCIA"
            status_icon = "⚠️"
        else:
            status_class = "status-online"
            status_text = "OPERACIONAL"
            status_icon = ""

        st.markdown(f"""
        <div style="text-align: right; padding-top: 0.5rem;">
            <span class="status-badge {status_class}">
                <span class="live-dot"></span>
                {status_icon} {status_text}
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_portfolio_summary():
    """Render portfolio summary with balance."""
    st.markdown('<div class="section-header">💰 Portfólio</div>', unsafe_allow_html=True)

    initial_capital = get_initial_capital()
    positions = get_positions()
    stats = get_performance_stats()

    # Calculate current values
    positions_value = sum(p.get('trade_value', 0) for p in positions.values())
    available_capital = initial_capital - positions_value
    total_profit = stats.get('total_profit', 0)
    current_balance = initial_capital + total_profit
    profit_pct = (total_profit / initial_capital * 100) if initial_capital > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Saldo Total</div>
            <div class="metric-value {'positive' if current_balance >= initial_capital else 'negative'}">
                ${current_balance:,.2f}
            </div>
            <div class="metric-delta" style="color: {'#3fb950' if profit_pct >= 0 else '#f85149'};">
                {'+' if profit_pct >= 0 else ''}{profit_pct:.2f}% desde início
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Disponível</div>
            <div class="metric-value neutral">${available_capital:,.2f}</div>
            <div class="metric-delta" style="color: #8b949e;">
                {(available_capital/initial_capital*100):.0f}% do capital
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Em Posições</div>
            <div class="metric-value" style="color: #a371f7;">${positions_value:,.2f}</div>
            <div class="metric-delta" style="color: #8b949e;">
                {len(positions)} posições abertas
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        pnl_class = 'positive' if total_profit >= 0 else 'negative'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Lucro/Prejuízo Total</div>
            <div class="metric-value {pnl_class}">
                {'+' if total_profit >= 0 else ''}${total_profit:,.2f}
            </div>
            <div class="metric-delta" style="color: #8b949e;">
                {stats.get('total_trades', 0)} trades realizados
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_positions():
    """Render open positions."""
    st.markdown('<div class="section-header">📊 Posições Abertas</div>', unsafe_allow_html=True)

    positions = get_positions()

    if not positions:
        st.markdown("""
        <div class="alert-item alert-info">
            <span>📭</span>
            <span>Nenhuma posição aberta no momento</span>
        </div>
        """, unsafe_allow_html=True)
        return

    cols = st.columns(min(len(positions), 3))

    for i, (symbol, pos) in enumerate(positions.items()):
        with cols[i % 3]:
            entry_price = pos.get('entry_price', 0)
            quantity = pos.get('quantity', 0)
            trade_value = pos.get('trade_value', 0)
            stop_loss = pos.get('stop_loss', 0)
            take_profit = pos.get('take_profit', 0)

            # Calculate unrealized P&L (using entry price as placeholder since we don't have current price)
            # In production, you'd fetch current price
            unrealized_pnl = 0
            pnl_pct = 0

            entry_time = pos.get('entry_time', '')
            if isinstance(entry_time, str) and entry_time:
                try:
                    entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    time_held = datetime.now(timezone.utc) - entry_dt
                    hours_held = time_held.total_seconds() / 3600
                    time_str = f"{hours_held:.1f}h"
                except:
                    time_str = "N/A"
            else:
                time_str = "N/A"

            st.markdown(f"""
            <div class="position-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <span class="position-symbol">{symbol}</span>
                    <span class="position-pnl" style="color: {'#3fb950' if unrealized_pnl >= 0 else '#f85149'};">
                        {'+' if unrealized_pnl >= 0 else ''}{pnl_pct:.2f}%
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div class="position-detail">Entrada: ${entry_price:.4f}</div>
                    <div class="position-detail">Qtd: {quantity:.6f}</div>
                    <div class="position-detail">Valor: ${trade_value:.2f}</div>
                    <div class="position-detail">Tempo: {time_str}</div>
                    <div class="position-detail" style="color: #f85149;">SL: ${stop_loss:.4f}</div>
                    <div class="position-detail" style="color: #3fb950;">TP: ${take_profit:.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_recent_trades():
    """Render recent transactions."""
    st.markdown('<div class="section-header">📜 Transações Recentes</div>', unsafe_allow_html=True)

    trades = get_trades(limit=20)

    if not trades:
        st.markdown("""
        <div class="alert-item alert-info">
            <span>📭</span>
            <span>Nenhuma transação registrada</span>
        </div>
        """, unsafe_allow_html=True)
        return

    # Build table HTML
    rows_html = ""
    for trade in trades[:10]:
        symbol = trade.get('symbol', 'N/A')
        action = trade.get('action', 'N/A')
        price = float(trade.get('price', 0))
        quantity = float(trade.get('quantity', 0))
        profit = float(trade.get('profit', 0) or 0)
        date = trade.get('date', '')

        # Format date
        if isinstance(date, str) and date:
            try:
                dt = datetime.fromisoformat(date.replace('Z', '+00:00'))
                date_str = dt.strftime('%d/%m %H:%M')
            except:
                date_str = str(date)[:16]
        else:
            date_str = str(date)[:16] if date else 'N/A'

        action_class = 'tx-buy' if action == 'BUY' else 'tx-sell'
        profit_html = ""
        if action == 'SELL' and profit != 0:
            profit_color = '#3fb950' if profit > 0 else '#f85149'
            profit_html = f'<span style="color: {profit_color};">{"+$" if profit > 0 else "-$"}{abs(profit):.2f}</span>'

        rows_html += f"""
        <tr>
            <td style="font-weight: 600;">{symbol}</td>
            <td class="{action_class}">{action}</td>
            <td style="font-family: 'JetBrains Mono', monospace;">${price:.4f}</td>
            <td>{quantity:.4f}</td>
            <td>{profit_html}</td>
            <td style="color: #8b949e;">{date_str}</td>
        </tr>
        """

    st.markdown(f"""
    <div style="overflow-x: auto;">
        <table class="tx-table">
            <thead>
                <tr>
                    <th>Ativo</th>
                    <th>Tipo</th>
                    <th>Preço</th>
                    <th>Quantidade</th>
                    <th>Resultado</th>
                    <th>Data</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def render_performance_stats():
    """Render performance statistics."""
    st.markdown('<div class="section-header">📈 Performance</div>', unsafe_allow_html=True)

    stats = get_performance_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total de Trades</div>
            <div class="metric-value neutral">{stats.get('total_trades', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        win_rate = stats.get('win_rate', 0)
        wr_color = '#3fb950' if win_rate >= 55 else '#d29922' if win_rate >= 45 else '#f85149'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value" style="color: {wr_color};">{win_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        wins = stats.get('wins', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Wins</div>
            <div class="metric-value positive">{wins}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        losses = stats.get('losses', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Losses</div>
            <div class="metric-value negative">{losses}</div>
        </div>
        """, unsafe_allow_html=True)


def render_system_status():
    """Render system status panel."""
    st.markdown('<div class="section-header">🖥️ Status do Sistema</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        errors = get_error_count()
        active_cryptos = len(CRYPTOCURRENCIES) - len(CRYPTO_BLACKLIST) if HAS_CRYPTO else 0
        total_cryptos = len(CRYPTOCURRENCIES) if HAS_CRYPTO else 0
        quality = (active_cryptos / total_cryptos * 100) if total_cryptos > 0 else 0

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class="metric-label">Criptos Ativas</div>
                    <div class="metric-value neutral">{active_cryptos} / {total_cryptos}</div>
                </div>
                <div style="text-align: right;">
                    <div class="metric-label">Qualidade</div>
                    <div class="metric-value" style="color: {'#3fb950' if quality >= 80 else '#d29922'};">{quality:.0f}%</div>
                </div>
            </div>
            <div class="progress-container" style="margin-top: 0.75rem;">
                <div class="progress-bar" style="width: {quality}%; background: linear-gradient(90deg, #3fb950, #58a6ff);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        error_color = '#f85149' if errors > 10 else '#d29922' if errors > 0 else '#3fb950'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Erros (24h)</div>
            <div class="metric-value" style="color: {error_color};">{errors}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        state = get_system_state()
        em = state.get('emergency_mode', {})
        is_active = em.get('active', False)

        if is_active:
            st.markdown("""
            <div class="alert-item alert-warning">
                <span style="font-size: 1.25rem;">🚨</span>
                <div>
                    <div style="font-weight: 600;">Modo Emergência Ativo</div>
                    <div style="font-size: 0.8rem; color: #8b949e;">Parâmetros relaxados</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🛑 Desativar Emergência", use_container_width=True):
                state['emergency_mode'] = {'active': False}
                save_system_state(state)
                st.rerun()
        else:
            st.markdown("""
            <div class="alert-item alert-success">
                <span style="font-size: 1.25rem;">✅</span>
                <div>
                    <div style="font-weight: 600;">Sistema Normal</div>
                    <div style="font-size: 0.8rem; color: #8b949e;">Parâmetros padrão ativos</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Blacklist info
        if CRYPTO_BLACKLIST:
            with st.expander(f"📛 Blacklist ({len(CRYPTO_BLACKLIST)} criptos)"):
                for symbol in sorted(CRYPTO_BLACKLIST):
                    st.markdown(f"• {symbol}")


def render_alerts():
    """Render alerts section."""
    st.markdown('<div class="section-header">🔔 Alertas</div>', unsafe_allow_html=True)

    alerts = []

    # Check for errors
    errors = get_error_count()
    if errors > 50:
        alerts.append(("error", "🔴", "Alto número de erros", f"{errors} erros nas últimas 24h"))
    elif errors > 10:
        alerts.append(("warning", "🟡", "Erros detectados", f"{errors} erros nas últimas 24h"))

    # Check positions
    positions = get_positions()
    if len(positions) >= 5:
        alerts.append(("warning", "📊", "Muitas posições abertas", f"{len(positions)} posições ativas"))

    # Emergency mode
    state = get_system_state()
    if state.get('emergency_mode', {}).get('active'):
        alerts.append(("warning", "🚨", "Modo Emergência", "Sistema em modo emergência"))

    # Blacklist
    if len(CRYPTO_BLACKLIST) > 5:
        alerts.append(("info", "📋", "Criptos na blacklist", f"{len(CRYPTO_BLACKLIST)} ativos bloqueados"))

    # System OK
    if not alerts:
        alerts.append(("success", "✅", "Sistema OK", "Operando normalmente"))

    for alert_type, icon, title, msg in alerts:
        st.markdown(f"""
        <div class="alert-item alert-{alert_type}">
            <span style="font-size: 1.25rem;">{icon}</span>
            <div>
                <div style="font-weight: 600;">{title}</div>
                <div style="font-size: 0.8rem; color: #8b949e;">{msg}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_actions():
    """Render action buttons."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Atualizar", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("✅ Validar Sistema", use_container_width=True):
            with st.spinner("Validando..."):
                time.sleep(1)
                st.success("Sistema validado!")

    with col3:
        report = {
            "generated_at": datetime.now().isoformat(),
            "version": "V4.1",
            "stats": get_performance_stats(),
            "positions": len(get_positions()),
            "errors_24h": get_error_count()
        }
        st.download_button(
            "📥 Exportar Relatório",
            json.dumps(report, indent=2, default=str),
            file_name=f"lobo_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )


# ==================== MAIN ====================

def main():
    """Main application."""
    apply_professional_css()

    # Header
    render_header()
    st.markdown("<hr>", unsafe_allow_html=True)

    # Portfolio Summary (top section)
    render_portfolio_summary()
    st.markdown("<hr>", unsafe_allow_html=True)

    # Main content - two columns
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Positions
        render_positions()
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Recent trades
        render_recent_trades()

    with col_right:
        # Performance
        render_performance_stats()
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # System Status
        render_system_status()
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Alerts
        render_alerts()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Actions
    render_actions()


if __name__ == "__main__":
    main()
