"""
Lobo IA - Dashboard de Trading
Interface moderna, minimalista e profissional.

Execute com: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, '.')

from logger import Logger
from config_loader import config

# Imports opcionais
try:
    from b3_calendar import is_holiday, is_weekend, is_trading_day, get_next_trading_day
    HAS_CALENDAR = True
except ImportError:
    HAS_CALENDAR = False

try:
    from crypto_scanner import CryptoScanner
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

try:
    from market_scanner import MarketScanner
    HAS_SCANNER = True
except ImportError:
    HAS_SCANNER = False

try:
    from binance_client import BinanceClient
    HAS_BINANCE = True
except ImportError:
    HAS_BINANCE = False

try:
    from coinmarketcap_client import CoinMarketCapClient
    HAS_CMC = True
except ImportError:
    HAS_CMC = False


# =============================================================================
# CONFIGURACAO DA PAGINA
# =============================================================================

st.set_page_config(
    page_title="Lobo IA",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CSS - DESIGN SYSTEM MODERNO
# =============================================================================

st.markdown("""
<style>
    /* Reset e variaveis */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #16161f;
        --bg-hover: #1e1e2a;
        --border: #2a2a3a;
        --text-primary: #ffffff;
        --text-secondary: #8b8b9e;
        --text-muted: #5a5a6e;
        --accent: #6366f1;
        --accent-hover: #818cf8;
        --success: #22c55e;
        --danger: #ef4444;
        --warning: #f59e0b;
        --gradient-1: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        --gradient-2: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        --gradient-3: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }

    /* Base */
    .stApp {
        background: var(--bg-primary);
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary);
    }

    /* Cards */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
    }

    .card-title {
        font-size: 14px;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .card-value {
        font-size: 32px;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }

    .card-delta {
        font-size: 14px;
        font-weight: 500;
        margin-top: 8px;
    }

    .delta-positive { color: var(--success); }
    .delta-negative { color: var(--danger); }
    .delta-neutral { color: var(--text-muted); }

    /* Metric cards compact */
    .metric-mini {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .metric-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }

    .metric-content {
        flex: 1;
    }

    .metric-label {
        font-size: 12px;
        color: var(--text-muted);
        margin-bottom: 2px;
    }

    .metric-value {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* Status badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }

    .badge-success {
        background: rgba(34, 197, 94, 0.15);
        color: var(--success);
    }

    .badge-danger {
        background: rgba(239, 68, 68, 0.15);
        color: var(--danger);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: var(--warning);
    }

    .badge-neutral {
        background: rgba(99, 102, 241, 0.15);
        color: var(--accent);
    }

    /* Header */
    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 0;
        margin-bottom: 24px;
        border-bottom: 1px solid var(--border);
    }

    .logo {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .logo-icon {
        font-size: 32px;
    }

    .logo-text {
        font-size: 24px;
        font-weight: 700;
        color: var(--text-primary);
    }

    .logo-sub {
        font-size: 12px;
        color: var(--text-muted);
    }

    /* Nav tabs */
    .nav-tabs {
        display: flex;
        gap: 8px;
        background: var(--bg-secondary);
        padding: 6px;
        border-radius: 12px;
        margin-bottom: 24px;
    }

    .nav-tab {
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s;
    }

    .nav-tab:hover {
        color: var(--text-primary);
        background: var(--bg-hover);
    }

    .nav-tab.active {
        color: var(--text-primary);
        background: var(--accent);
    }

    /* Tables */
    .stDataFrame {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }

    [data-testid="stDataFrame"] > div {
        background: transparent !important;
    }

    /* Trade row styles */
    .trade-row {
        display: flex;
        align-items: center;
        padding: 16px;
        border-bottom: 1px solid var(--border);
    }

    .trade-row:last-child {
        border-bottom: none;
    }

    /* Section titles */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: var(--border);
        margin: 24px 0;
    }

    /* Streamlit overrides */
    .stMetric {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }

    .stButton > button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: var(--accent-hover);
        border: none;
    }

    .stSelectbox > div > div {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-secondary);
        padding: 4px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
    }

    /* Chart styling */
    .js-plotly-plot .plotly .modebar {
        background: var(--bg-card) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }

    /* Live indicator */
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--success);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 48px;
        color: var(--text-muted);
    }

    .empty-state-icon {
        font-size: 48px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNCOES DE DADOS
# =============================================================================

@st.cache_data(ttl=30)
def load_trades(limit=500):
    """Carrega trades do banco de dados."""
    try:
        with Logger() as logger:
            trades = logger.get_trades(limit=limit)
        df = pd.DataFrame(trades)
        if not df.empty:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            numeric_cols = ['price', 'quantity', 'profit']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].astype(float)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_performance_stats():
    """Carrega estatisticas de performance."""
    try:
        with Logger() as logger:
            stats = logger.get_performance_stats()
        return stats
    except Exception:
        return {}


@st.cache_data(ttl=60)
def get_binance_balance():
    """Obtem saldo da Binance."""
    if not HAS_BINANCE:
        return None
    try:
        client = BinanceClient()
        return client.get_total_balance_usdt()
    except Exception as e:
        return {'error': str(e)}


@st.cache_data(ttl=120)
def get_cmc_market_overview():
    """Obtem overview do mercado via CMC."""
    if not HAS_CMC:
        return None
    try:
        client = CoinMarketCapClient()
        if not client.api_key:
            return None
        return client.get_market_overview()
    except Exception:
        return None


def get_market_status():
    """Retorna status dos mercados."""
    now = datetime.now()

    b3_open = False
    b3_reason = "Fechado"

    if HAS_CALENDAR:
        if is_weekend(now):
            b3_reason = "Fim de semana"
        elif is_holiday(now):
            b3_reason = "Feriado"
        elif 10 <= now.hour < 18:
            b3_open = True
            b3_reason = "Aberto"
        else:
            b3_reason = "Fora do horario"

    return {
        'b3': {'open': b3_open, 'status': b3_reason},
        'crypto': {'open': True, 'status': '24/7'}
    }


# =============================================================================
# COMPONENTES UI
# =============================================================================

def render_header():
    """Renderiza header da aplicacao."""
    col1, col2, col3 = st.columns([2, 4, 2])

    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 36px;">🐺</span>
            <div>
                <div style="font-size: 24px; font-weight: 700; color: white;">Lobo IA</div>
                <div style="font-size: 12px; color: #5a5a6e;">Trading Autonomo</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        pass  # Espaco central

    with col3:
        market = get_market_status()
        now = datetime.now()

        st.markdown(f"""
        <div style="text-align: right;">
            <div style="font-size: 12px; color: #5a5a6e;">Ultima atualizacao</div>
            <div style="font-size: 16px; font-weight: 600; color: white;">{now.strftime("%H:%M:%S")}</div>
        </div>
        """, unsafe_allow_html=True)


def render_metric_card(label, value, delta=None, delta_type="neutral", icon="📊"):
    """Renderiza card de metrica."""
    delta_class = f"delta-{delta_type}"
    delta_html = f'<div class="card-delta {delta_class}">{delta}</div>' if delta else ''

    st.markdown(f"""
    <div class="card">
        <div class="card-title">{icon} {label}</div>
        <div class="card-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_status_badges():
    """Renderiza badges de status."""
    market = get_market_status()
    mode = config.get('execution.mode', 'simulation')

    b3_class = "badge-success" if market['b3']['open'] else "badge-danger"
    mode_class = "badge-warning" if mode == "simulation" else "badge-success"

    st.markdown(f"""
    <div style="display: flex; gap: 12px; margin-bottom: 24px;">
        <span class="badge {b3_class}">
            <span style="font-size: 8px;">●</span> B3 {market['b3']['status']}
        </span>
        <span class="badge badge-success">
            <span style="font-size: 8px;">●</span> Crypto {market['crypto']['status']}
        </span>
        <span class="badge {mode_class}">
            {'🎮 Simulacao' if mode == 'simulation' else '💰 Real'}
        </span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGINAS
# =============================================================================

def page_overview():
    """Pagina principal - Overview."""

    render_status_badges()

    # Metricas principais
    stats = load_performance_stats()
    trades_df = load_trades()

    capital_inicial = config.get('trading.capital', 10000)
    total_profit = stats.get('total_profit', 0)
    capital_atual = capital_inicial + total_profit
    win_rate = stats.get('win_rate', 0)
    total_trades = stats.get('total_trades', 0)

    # Saldo Binance
    binance = get_binance_balance()
    if binance and binance.get('success'):
        capital_atual = binance.get('total_usdt', 0)
        capital_label = "Saldo USDT"
        currency = "$"
    else:
        capital_label = "Capital"
        currency = "R$"

    # Cards de metricas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pct_change = ((capital_atual / capital_inicial) - 1) * 100 if capital_inicial > 0 else 0
        delta_type = "positive" if pct_change >= 0 else "negative"
        st.metric(
            label=f"💰 {capital_label}",
            value=f"{currency} {capital_atual:,.2f}",
            delta=f"{pct_change:+.2f}%"
        )

    with col2:
        delta_type = "positive" if total_profit >= 0 else "negative"
        st.metric(
            label="📈 Lucro Total",
            value=f"R$ {total_profit:,.2f}",
            delta="Ganho" if total_profit >= 0 else "Perda"
        )

    with col3:
        delta_type = "positive" if win_rate >= 50 else "negative"
        st.metric(
            label="🎯 Win Rate",
            value=f"{win_rate:.1f}%",
            delta=f"{win_rate - 50:+.1f}% vs 50%"
        )

    with col4:
        st.metric(
            label="📊 Total Trades",
            value=f"{total_trades}",
            delta=f"{stats.get('wins', 0)}W / {stats.get('losses', 0)}L"
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Graficos
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">📈 Evolucao do Capital</div>', unsafe_allow_html=True)

        if not trades_df.empty and 'profit' in trades_df.columns:
            df_sorted = trades_df.sort_values('date')
            df_sorted['cumulative'] = df_sorted['profit'].cumsum()
            df_sorted['capital'] = capital_inicial + df_sorted['cumulative']

            fig = go.Figure()

            # Area gradient
            fig.add_trace(go.Scatter(
                x=df_sorted['date'],
                y=df_sorted['capital'],
                mode='lines',
                name='Capital',
                line=dict(color='#6366f1', width=2),
                fill='tozeroy',
                fillcolor='rgba(99, 102, 241, 0.1)'
            ))

            fig.add_hline(
                y=capital_inicial,
                line_dash="dash",
                line_color="#5a5a6e",
                annotation_text="Inicial"
            )

            fig.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=False,
                    color='#5a5a6e'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#2a2a3a',
                    color='#5a5a6e'
                ),
                showlegend=False,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <div>Aguardando dados...</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">📊 Lucro por Trade</div>', unsafe_allow_html=True)

        if not trades_df.empty and 'profit' in trades_df.columns:
            recent = trades_df.sort_values('date', ascending=False).head(20)

            colors = ['#22c55e' if p >= 0 else '#ef4444' for p in recent['profit']]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(len(recent))),
                y=recent['profit'],
                marker_color=colors,
                marker_line_width=0
            ))

            fig.add_hline(y=0, line_color="#5a5a6e", line_width=1)

            fig.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=False,
                    color='#5a5a6e',
                    title="Ultimos trades"
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#2a2a3a',
                    color='#5a5a6e'
                ),
                showlegend=False,
                bargap=0.3
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <div>Aguardando dados...</div>
            </div>
            """, unsafe_allow_html=True)

    # Transacoes recentes
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔄 Transacoes Recentes</div>', unsafe_allow_html=True)

    if not trades_df.empty:
        recent = trades_df.sort_values('date', ascending=False).head(8)

        display_df = recent[['symbol', 'date', 'action', 'price', 'quantity', 'profit']].copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%d/%m %H:%M')

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'symbol': st.column_config.TextColumn('Ativo', width='medium'),
                'date': st.column_config.TextColumn('Data', width='small'),
                'action': st.column_config.TextColumn('Tipo', width='small'),
                'price': st.column_config.NumberColumn('Preco', format="$ %.2f", width='small'),
                'quantity': st.column_config.NumberColumn('Qtd', format="%.4f", width='small'),
                'profit': st.column_config.NumberColumn('P&L', format="$ %.2f", width='small')
            }
        )
    else:
        st.info("Nenhuma transacao registrada.")


def page_trades():
    """Pagina de historico de trades."""

    st.markdown('<div class="section-title">🔍 Historico de Trades</div>', unsafe_allow_html=True)

    trades_df = load_trades(limit=1000)

    if trades_df.empty:
        st.info("Nenhum trade registrado.")
        return

    # Filtros em linha
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        symbols = ['Todos'] + sorted(trades_df['symbol'].unique().tolist())
        selected_symbol = st.selectbox("Ativo", symbols, label_visibility="collapsed")

    with col2:
        selected_action = st.selectbox("Tipo", ['Todos', 'BUY', 'SELL'], label_visibility="collapsed")

    with col3:
        result_filter = st.selectbox("Resultado", ['Todos', 'Lucro', 'Prejuizo'], label_visibility="collapsed")

    with col4:
        date_range = st.selectbox("Periodo", ['Todos', 'Hoje', '7 dias', '30 dias'], label_visibility="collapsed")

    # Aplica filtros
    filtered = trades_df.copy()

    if selected_symbol != 'Todos':
        filtered = filtered[filtered['symbol'] == selected_symbol]

    if selected_action != 'Todos':
        filtered = filtered[filtered['action'] == selected_action]

    if result_filter == 'Lucro':
        filtered = filtered[filtered['profit'] > 0]
    elif result_filter == 'Prejuizo':
        filtered = filtered[filtered['profit'] < 0]

    if date_range != 'Todos' and 'date' in filtered.columns:
        now = datetime.now()
        if date_range == 'Hoje':
            filtered = filtered[filtered['date'].dt.date == now.date()]
        elif date_range == '7 dias':
            filtered = filtered[filtered['date'] >= now - timedelta(days=7)]
        elif date_range == '30 dias':
            filtered = filtered[filtered['date'] >= now - timedelta(days=30)]

    # Metricas do filtro
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total", len(filtered))
    with col2:
        st.metric("P&L", f"R$ {filtered['profit'].sum():,.2f}" if not filtered.empty else "R$ 0")
    with col3:
        wins = len(filtered[filtered['profit'] > 0]) if not filtered.empty else 0
        st.metric("Ganhos", wins)
    with col4:
        losses = len(filtered[filtered['profit'] < 0]) if not filtered.empty else 0
        st.metric("Perdas", losses)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Tabela
    if not filtered.empty:
        display = filtered[['symbol', 'date', 'action', 'price', 'quantity', 'profit']].copy()
        display = display.sort_values('date', ascending=False)
        display['date'] = pd.to_datetime(display['date']).dt.strftime('%d/%m/%Y %H:%M')

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                'symbol': 'Ativo',
                'date': 'Data',
                'action': 'Tipo',
                'price': st.column_config.NumberColumn('Preco', format="$ %.2f"),
                'quantity': st.column_config.NumberColumn('Qtd', format="%.6f"),
                'profit': st.column_config.NumberColumn('P&L', format="$ %.2f")
            }
        )

        # Export
        csv = filtered.to_csv(index=False)
        st.download_button("📥 Exportar CSV", csv, "trades.csv", "text/csv")
    else:
        st.warning("Nenhum trade encontrado com os filtros.")


def page_analytics():
    """Pagina de analytics."""

    st.markdown('<div class="section-title">📈 Analytics</div>', unsafe_allow_html=True)

    trades_df = load_trades()

    if trades_df.empty:
        st.info("Aguardando dados para analise.")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Performance", "Drawdown", "Por Dia"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Lucros**")
            st.metric("Medio", f"R$ {trades_df['profit'].mean():.2f}")
            st.metric("Maximo", f"R$ {trades_df['profit'].max():.2f}")
            st.metric("Minimo", f"R$ {trades_df['profit'].min():.2f}")

        with col2:
            wins = trades_df[trades_df['profit'] > 0]
            losses = trades_df[trades_df['profit'] < 0]

            avg_win = wins['profit'].mean() if len(wins) > 0 else 0
            avg_loss = losses['profit'].mean() if len(losses) > 0 else 0

            st.markdown("**Medias**")
            st.metric("Ganho Medio", f"R$ {avg_win:.2f}")
            st.metric("Perda Media", f"R$ {avg_loss:.2f}")

            if avg_loss != 0:
                st.metric("Ratio", f"{abs(avg_win / avg_loss):.2f}")

        with col3:
            gross_profit = wins['profit'].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses['profit'].sum()) if len(losses) > 0 else 1

            st.markdown("**Totais**")
            st.metric("Profit Factor", f"{gross_profit / gross_loss:.2f}")
            st.metric("Lucro Bruto", f"R$ {gross_profit:.2f}")
            st.metric("Perda Bruta", f"R$ {gross_loss:.2f}")

    with tab2:
        df_sorted = trades_df.sort_values('date')
        df_sorted['cumulative'] = df_sorted['profit'].cumsum()
        df_sorted['running_max'] = df_sorted['cumulative'].cummax()
        df_sorted['drawdown'] = df_sorted['cumulative'] - df_sorted['running_max']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sorted['date'],
            y=df_sorted['drawdown'],
            mode='lines',
            line=dict(color='#ef4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.1)'
        ))

        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, color='#5a5a6e'),
            yaxis=dict(showgrid=True, gridcolor='#2a2a3a', color='#5a5a6e'),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
        st.metric("Drawdown Maximo", f"R$ {abs(df_sorted['drawdown'].min()):.2f}")

    with tab3:
        if 'date' in trades_df.columns:
            trades_df['day'] = trades_df['date'].dt.date
            daily = trades_df.groupby('day').agg({'profit': 'sum', 'id': 'count'}).reset_index()
            daily.columns = ['Data', 'Lucro', 'Trades']

            colors = ['#22c55e' if p >= 0 else '#ef4444' for p in daily['Lucro']]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily['Data'],
                y=daily['Lucro'],
                marker_color=colors
            ))

            fig.add_hline(y=0, line_color="#5a5a6e", line_width=1)

            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, color='#5a5a6e'),
                yaxis=dict(showgrid=True, gridcolor='#2a2a3a', color='#5a5a6e'),
                showlegend=False,
                bargap=0.3
            )

            st.plotly_chart(fig, use_container_width=True)


def page_scanner():
    """Pagina do scanner de mercado."""

    st.markdown('<div class="section-title">🔍 Scanner de Mercado</div>', unsafe_allow_html=True)

    # Market overview CMC
    cmc_data = get_cmc_market_overview()

    if cmc_data:
        fear_greed = cmc_data.get('fear_greed', {})
        if fear_greed:
            fg_score = fear_greed.get('score', 50)
            fg_class = fear_greed.get('classification', 'Neutral')
            fg_emoji = fear_greed.get('emoji', '😐')

            st.markdown(f"""
            <div style="background: #16161f; border: 1px solid #2a2a3a; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
                <div style="display: flex; align-items: center; gap: 24px;">
                    <div style="font-size: 48px;">{fg_emoji}</div>
                    <div>
                        <div style="color: #5a5a6e; font-size: 12px;">Fear & Greed Index</div>
                        <div style="color: white; font-size: 32px; font-weight: 700;">{fg_score}</div>
                        <div style="color: #8b8b9e;">{fg_class}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["₿ Crypto", "🇧🇷 B3"])

    with tab1:
        if HAS_CRYPTO:
            if st.button("🔄 Escanear Mercado Crypto", use_container_width=True):
                with st.spinner("Analisando criptomoedas..."):
                    scanner = CryptoScanner()
                    results = scanner.scan_crypto_market()

                    if results:
                        # BTC e ETH
                        col1, col2 = st.columns(2)
                        btc = next((r for r in results if r['symbol'] == 'BTC-USD'), None)
                        eth = next((r for r in results if r['symbol'] == 'ETH-USD'), None)

                        with col1:
                            if btc:
                                delta_color = "normal" if btc['change_24h'] >= 0 else "inverse"
                                st.metric(
                                    "₿ Bitcoin",
                                    f"${btc['price']:,.2f}",
                                    f"{btc['change_24h']:+.2f}%",
                                    delta_color=delta_color
                                )

                        with col2:
                            if eth:
                                delta_color = "normal" if eth['change_24h'] >= 0 else "inverse"
                                st.metric(
                                    "Ξ Ethereum",
                                    f"${eth['price']:,.2f}",
                                    f"{eth['change_24h']:+.2f}%",
                                    delta_color=delta_color
                                )

                        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

                        # Tabela
                        df = pd.DataFrame(results[:15])

                        st.dataframe(
                            df[['symbol', 'name', 'price', 'total_score', 'signal', 'change_24h']],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'symbol': 'Ativo',
                                'name': 'Nome',
                                'price': st.column_config.NumberColumn('Preco', format="$ %.2f"),
                                'total_score': st.column_config.NumberColumn('Score', format="%.1f"),
                                'signal': 'Sinal',
                                'change_24h': st.column_config.NumberColumn('24h', format="%.2f%%")
                            }
                        )
        else:
            st.warning("Scanner crypto nao disponivel.")

    with tab2:
        if HAS_SCANNER:
            if st.button("🔄 Escanear B3", use_container_width=True):
                with st.spinner("Analisando acoes..."):
                    scanner = MarketScanner()
                    results = scanner.scan_market()

                    if results:
                        df = pd.DataFrame(results[:20])

                        st.dataframe(
                            df[['symbol', 'price', 'total_score', 'signal', 'rsi', 'change_5d']],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'symbol': 'Ativo',
                                'price': st.column_config.NumberColumn('Preco', format="R$ %.2f"),
                                'total_score': st.column_config.NumberColumn('Score', format="%.1f"),
                                'signal': 'Sinal',
                                'rsi': st.column_config.NumberColumn('RSI', format="%.1f"),
                                'change_5d': st.column_config.NumberColumn('5d', format="%.2f%%")
                            }
                        )
        else:
            st.warning("Scanner B3 nao disponivel.")


def page_settings():
    """Pagina de configuracoes."""

    st.markdown('<div class="section-title">⚙️ Configuracoes</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Trading**")
        st.write(f"Capital: R$ {config.get('trading.capital', 10000):,.2f}")
        st.write(f"Exposicao: {config.get('trading.exposure', 0.03)*100:.1f}%")
        st.write(f"Max Exposicao: {config.get('trading.max_total_exposure', 0.2)*100:.1f}%")

        st.markdown("**Risco**")
        st.write(f"Stop Loss: {config.get('risk.stop_loss', 0.02)*100:.1f}%")
        st.write(f"Take Profit: {config.get('risk.take_profit', 0.05)*100:.1f}%")

    with col2:
        st.markdown("**Sistema**")
        st.write(f"Modo: {config.get('execution.mode', 'simulation')}")
        st.write(f"Horario B3: {config.get('market.open_hour', 10)}h - {config.get('market.close_hour', 18)}h")

        st.markdown("**APIs**")
        env_vars = ['BINANCE_API_KEY', 'CMC_API_KEY', 'DATABASE_URL']
        for var in env_vars:
            status = "✅" if os.environ.get(var) else "❌"
            st.write(f"{status} {var}")

    # Binance status
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("**Conexao Binance**")

    binance = get_binance_balance()
    if binance and binance.get('success'):
        testnet = binance.get('testnet', True)
        st.success(f"Conectado - {'Testnet' if testnet else 'Mainnet'}")
        st.write(f"Saldo: ${binance.get('total_usdt', 0):,.2f} USDT")
    elif binance and 'error' in binance:
        st.error(f"Erro: {binance['error']}")
    else:
        st.warning("Nao conectado")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Funcao principal."""

    render_header()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Navegacao com tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔄 Trades",
        "📈 Analytics",
        "🔍 Scanner",
        "⚙️ Config"
    ])

    with tab1:
        page_overview()

    with tab2:
        page_trades()

    with tab3:
        page_analytics()

    with tab4:
        page_scanner()

    with tab5:
        page_settings()

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 24px; color: #5a5a6e; font-size: 12px; margin-top: 48px;">
        Lobo IA • Trading Autonomo • v2.0
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
