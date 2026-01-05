"""
Lobo IA - Crypto Trading Portal
Interface profissional com atualizacao em tempo real.
Versao 4.2 - Crypto Only (24/7)

Execute com: streamlit run dashboard.py --server.runOnSave true
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import os
import sys
import time

# Configura timezone do Brasil (UTC-3)
try:
    import pytz
    BRAZIL_TZ = pytz.timezone('America/Sao_Paulo')
except ImportError:
    BRAZIL_TZ = timezone(timedelta(hours=-3))


def get_brazil_time() -> datetime:
    """Retorna datetime no horario de Brasilia."""
    try:
        return datetime.now(BRAZIL_TZ)
    except:
        return datetime.now(timezone(timedelta(hours=-3)))


sys.path.insert(0, '.')

from logger import Logger
from config_loader import config

# Imports opcionais
try:
    from crypto_scanner import CryptoScanner
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

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
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Lobo IA Trading",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# PROFESSIONAL CSS
# =============================================================================

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Root variables */
    :root {
        --bg-dark: #0b0e11;
        --bg-card: #1a1d23;
        --bg-card-hover: #22262d;
        --bg-input: #12151a;
        --border: #2b3139;
        --border-light: #3a4149;
        --text-white: #eaecef;
        --text-gray: #848e9c;
        --text-dark: #5e6673;
        --green: #0ecb81;
        --green-bg: rgba(14, 203, 129, 0.1);
        --red: #f6465d;
        --red-bg: rgba(246, 70, 93, 0.1);
        --yellow: #f0b90b;
        --yellow-bg: rgba(240, 185, 11, 0.1);
        --blue: #1e80ff;
        --blue-bg: rgba(30, 128, 255, 0.1);
        --purple: #8b5cf6;
    }

    /* Global */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: var(--bg-dark);
    }

    /* Hide streamlit elements */
    #MainMenu, footer, header, .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}

    /* Top bar */
    .top-bar {
        background: var(--bg-card);
        border-bottom: 1px solid var(--border);
        padding: 12px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: -1rem -1rem 1.5rem -1rem;
    }

    .logo-section {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .logo-icon {
        font-size: 28px;
    }

    .logo-text {
        font-size: 20px;
        font-weight: 700;
        color: var(--text-white);
        letter-spacing: -0.5px;
    }

    .logo-badge {
        background: var(--green-bg);
        color: var(--green);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .status-section {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .status-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 13px;
        color: var(--text-gray);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    .status-dot.online { background: var(--green); }
    .status-dot.offline { background: var(--red); }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(0.95); }
    }

    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }

    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.2s;
    }

    .stat-card:hover {
        border-color: var(--border-light);
        background: var(--bg-card-hover);
    }

    .stat-label {
        font-size: 12px;
        color: var(--text-dark);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .stat-value {
        font-size: 24px;
        font-weight: 700;
        color: var(--text-white);
        font-family: 'JetBrains Mono', monospace;
    }

    .stat-delta {
        font-size: 13px;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .stat-delta.positive { color: var(--green); }
    .stat-delta.negative { color: var(--red); }
    .stat-delta.neutral { color: var(--text-gray); }

    /* Chart container */
    .chart-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .chart-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border);
    }

    .chart-title {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-white);
    }

    .chart-subtitle {
        font-size: 12px;
        color: var(--text-dark);
    }

    /* Trade table */
    .trades-table {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    .table-header {
        display: grid;
        grid-template-columns: 1.5fr 1fr 0.8fr 1fr 1fr 1fr;
        padding: 12px 20px;
        background: var(--bg-input);
        border-bottom: 1px solid var(--border);
        font-size: 11px;
        font-weight: 600;
        color: var(--text-dark);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .table-row {
        display: grid;
        grid-template-columns: 1.5fr 1fr 0.8fr 1fr 1fr 1fr;
        padding: 14px 20px;
        border-bottom: 1px solid var(--border);
        font-size: 13px;
        color: var(--text-white);
        transition: background 0.15s;
    }

    .table-row:hover {
        background: var(--bg-card-hover);
    }

    .table-row:last-child {
        border-bottom: none;
    }

    .trade-symbol {
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .trade-symbol .icon {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: var(--bg-input);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
    }

    .trade-type {
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .trade-type.buy {
        background: var(--green-bg);
        color: var(--green);
    }

    .trade-type.sell {
        background: var(--red-bg);
        color: var(--red);
    }

    .trade-pnl {
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .trade-pnl.positive { color: var(--green); }
    .trade-pnl.negative { color: var(--red); }

    /* Market overview */
    .market-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .market-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    .market-icon.fear { background: var(--red-bg); }
    .market-icon.greed { background: var(--green-bg); }
    .market-icon.neutral { background: var(--yellow-bg); }

    .market-info {
        flex: 1;
    }

    .market-label {
        font-size: 12px;
        color: var(--text-dark);
        margin-bottom: 4px;
    }

    .market-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-white);
        font-family: 'JetBrains Mono', monospace;
    }

    .market-status {
        font-size: 13px;
        color: var(--text-gray);
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
    }

    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: var(--text-white);
    }

    .section-action {
        font-size: 12px;
        color: var(--blue);
        cursor: pointer;
    }

    /* Live badge */
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--red-bg);
        color: var(--red);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }

    .live-badge::before {
        content: '';
        width: 6px;
        height: 6px;
        background: var(--red);
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }

    /* Streamlit overrides */
    .stMetric {
        background: transparent !important;
        padding: 0 !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        color: var(--text-white) !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        color: var(--text-dark) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 13px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-card);
        padding: 4px;
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        color: var(--text-gray);
        padding: 10px 20px;
        font-size: 13px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: var(--blue) !important;
        color: white !important;
    }

    .stButton > button {
        background: var(--blue) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }

    .stButton > button:hover {
        background: #1a6fd4 !important;
    }

    .stSelectbox > div > div {
        background: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    .stDataFrame {
        border: none !important;
    }

    [data-testid="stDataFrame"] {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-dark); }

    /* Auto refresh indicator */
    .refresh-indicator {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 12px;
        color: var(--text-gray);
        display: flex;
        align-items: center;
        gap: 8px;
        z-index: 1000;
    }

    .refresh-indicator .dot {
        width: 8px;
        height: 8px;
        background: var(--green);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=15)
def load_trades(limit=500):
    """Carrega trades do banco."""
    try:
        with Logger() as logger:
            trades = logger.get_trades(limit=limit)
        df = pd.DataFrame(trades)
        if not df.empty:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            for col in ['price', 'quantity', 'profit']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=15)
def load_stats():
    """Carrega estatisticas."""
    try:
        with Logger() as logger:
            return logger.get_performance_stats()
    except Exception:
        return {}


@st.cache_data(ttl=30)
def get_binance_balance():
    """Saldo Binance."""
    if not HAS_BINANCE:
        return None
    try:
        client = BinanceClient()
        return client.get_total_balance_usdt()
    except Exception as e:
        return {'error': str(e)}


@st.cache_data(ttl=60)
def get_cmc_overview():
    """Overview CoinMarketCap."""
    if not HAS_CMC:
        return None
    try:
        client = CoinMarketCapClient()
        if not client.api_key:
            return None
        return client.get_market_overview()
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_crypto_scan():
    """Scan de criptomoedas."""
    if not HAS_CRYPTO:
        return []
    try:
        scanner = CryptoScanner()
        return scanner.scan_crypto_market()
    except Exception:
        return []


def get_market_status():
    """Status do mercado crypto (sempre aberto 24/7)."""
    now = get_brazil_time()

    return {
        'crypto': {'open': True, 'status': '24/7'},
        'time': now.strftime("%H:%M:%S"),
        'date': now.strftime("%d/%m/%Y")
    }


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_topbar():
    """Barra superior."""
    market = get_market_status()
    mode = config.get('execution.mode', 'simulation')

    mode_text = "SIMULACAO" if mode == "simulation" else "REAL"
    mode_color = "var(--yellow)" if mode == "simulation" else "var(--green)"

    st.markdown(f"""
    <div class="top-bar">
        <div class="logo-section">
            <span class="logo-icon">🐺</span>
            <span class="logo-text">Lobo IA Crypto</span>
            <span class="logo-badge" style="background: {mode_color}20; color: {mode_color};">{mode_text}</span>
        </div>
        <div class="status-section">
            <div class="status-item">
                <span class="status-dot online"></span>
                Crypto 24/7
            </div>
            <div class="status-item" style="color: var(--text-white); font-family: 'JetBrains Mono', monospace;">
                {market['time']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stats_cards():
    """Cards de estatisticas."""
    stats = load_stats()
    trades_df = load_trades()

    capital_inicial = config.get('crypto.capital', 1000)
    total_profit = float(stats.get('total_profit', 0))
    capital_atual = capital_inicial + total_profit
    win_rate = float(stats.get('win_rate', 0))
    total_trades = int(stats.get('total_trades', 0))
    wins = int(stats.get('wins', 0))
    losses = int(stats.get('losses', 0))

    # Binance
    binance = get_binance_balance()
    if binance and binance.get('success'):
        capital_atual = float(binance.get('total_usdt', 0))

    pct_change = ((capital_atual / capital_inicial) - 1) * 100 if capital_inicial > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        delta_color = "normal" if pct_change >= 0 else "inverse"
        st.metric("Capital", f"$ {capital_atual:,.2f}", f"{pct_change:+.2f}%", delta_color=delta_color)

    with col2:
        delta_color = "normal" if total_profit >= 0 else "inverse"
        st.metric("P&L Total", f"$ {total_profit:,.2f}", "Lucro" if total_profit >= 0 else "Perda", delta_color=delta_color)

    with col3:
        delta_color = "normal" if win_rate >= 50 else "inverse"
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{win_rate - 50:+.1f}%", delta_color=delta_color)

    with col4:
        st.metric("Trades", f"{total_trades}", f"{wins}W / {losses}L")

    with col5:
        # Profit factor
        if losses > 0 and not trades_df.empty:
            wins_df = trades_df[trades_df['profit'] > 0]
            losses_df = trades_df[trades_df['profit'] < 0]
            gross_win = wins_df['profit'].sum() if len(wins_df) > 0 else 0
            gross_loss = abs(losses_df['profit'].sum()) if len(losses_df) > 0 else 1
            pf = gross_win / gross_loss if gross_loss > 0 else 0
        else:
            pf = 0
        st.metric("Profit Factor", f"{pf:.2f}", "Ratio W/L")


def render_market_overview():
    """Overview do mercado."""
    cmc = get_cmc_overview()

    if cmc:
        fear_greed = cmc.get('fear_greed', {})
        fg_score = fear_greed.get('score', 50)
        fg_class = fear_greed.get('classification', 'Neutral')
        fg_emoji = fear_greed.get('emoji', '😐')

        if fg_score < 35:
            icon_class = "fear"
        elif fg_score > 65:
            icon_class = "greed"
        else:
            icon_class = "neutral"

        st.markdown(f"""
        <div class="market-card">
            <div class="market-icon {icon_class}">{fg_emoji}</div>
            <div class="market-info">
                <div class="market-label">Fear & Greed Index</div>
                <div class="market-value">{fg_score}</div>
            </div>
            <div class="market-status">{fg_class}</div>
        </div>
        """, unsafe_allow_html=True)


def render_portfolio_chart():
    """Grafico de evolucao do portfolio."""
    trades_df = load_trades()
    capital_inicial = config.get('trading.capital', 10000)

    if trades_df.empty or 'profit' not in trades_df.columns:
        st.info("Aguardando trades...")
        return

    df = trades_df.sort_values('date').copy()
    df['cumulative'] = df['profit'].cumsum()
    df['capital'] = capital_inicial + df['cumulative']

    fig = go.Figure()

    # Area do capital
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['capital'],
        mode='lines',
        name='Capital',
        line=dict(color='#1e80ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(30, 128, 255, 0.1)'
    ))

    # Linha base
    fig.add_hline(
        y=capital_inicial,
        line_dash="dash",
        line_color="#3a4149",
        annotation_text="Base"
    )

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#5e6673', showline=False),
        yaxis=dict(showgrid=True, gridcolor='#2b3139', color='#5e6673', showline=False),
        showlegend=False,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def render_pnl_chart():
    """Grafico de P&L por trade."""
    trades_df = load_trades()

    if trades_df.empty or 'profit' not in trades_df.columns:
        st.info("Aguardando trades...")
        return

    recent = trades_df.sort_values('date', ascending=False).head(30)
    colors = ['#0ecb81' if p >= 0 else '#f6465d' for p in recent['profit']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(recent))),
        y=recent['profit'],
        marker_color=colors,
        marker_line_width=0
    ))

    fig.add_hline(y=0, line_color="#3a4149", line_width=1)

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#5e6673', showline=False, title=""),
        yaxis=dict(showgrid=True, gridcolor='#2b3139', color='#5e6673', showline=False),
        showlegend=False,
        bargap=0.3
    )

    st.plotly_chart(fig, use_container_width=True)


def render_trades_table():
    """Tabela de trades recentes."""
    trades_df = load_trades()

    if trades_df.empty:
        st.info("Nenhum trade registrado.")
        return

    recent = trades_df.sort_values('date', ascending=False).head(10)

    # Header
    st.markdown("""
    <div class="trades-table">
        <div class="table-header">
            <div>Ativo</div>
            <div>Data</div>
            <div>Tipo</div>
            <div>Preco</div>
            <div>Quantidade</div>
            <div>P&L</div>
        </div>
    """, unsafe_allow_html=True)

    # Rows
    for _, row in recent.iterrows():
        symbol = row['symbol']
        date = pd.to_datetime(row['date']).strftime('%d/%m %H:%M')
        action = row['action']
        price = row['price']
        qty = row['quantity']
        profit = row['profit']

        action_class = "buy" if action == "BUY" else "sell"
        pnl_class = "positive" if profit >= 0 else "negative"
        pnl_sign = "+" if profit >= 0 else ""

        # Icon baseado no tipo
        icon = "₿" if "BTC" in symbol else "Ξ" if "ETH" in symbol else "◆"

        st.markdown(f"""
        <div class="table-row">
            <div class="trade-symbol">
                <span class="icon">{icon}</span>
                {symbol.replace('-USD', '')}
            </div>
            <div style="color: var(--text-gray);">{date}</div>
            <div><span class="trade-type {action_class}">{action}</span></div>
            <div style="font-family: 'JetBrains Mono', monospace;">${price:,.2f}</div>
            <div style="color: var(--text-gray);">{qty:.6f}</div>
            <div class="trade-pnl {pnl_class}">{pnl_sign}${profit:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_crypto_scanner():
    """Scanner de criptomoedas."""
    results = get_crypto_scan()

    if not results:
        if st.button("🔄 Escanear Mercado", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        return

    # BTC e ETH
    btc = next((r for r in results if r['symbol'] == 'BTC-USD'), None)
    eth = next((r for r in results if r['symbol'] == 'ETH-USD'), None)

    col1, col2 = st.columns(2)

    with col1:
        if btc:
            delta_color = "normal" if btc['change_24h'] >= 0 else "inverse"
            st.metric("₿ Bitcoin", f"${btc['price']:,.2f}", f"{btc['change_24h']:+.2f}%", delta_color=delta_color)

    with col2:
        if eth:
            delta_color = "normal" if eth['change_24h'] >= 0 else "inverse"
            st.metric("Ξ Ethereum", f"${eth['price']:,.2f}", f"{eth['change_24h']:+.2f}%", delta_color=delta_color)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Tabela
    df = pd.DataFrame(results[:15])

    st.dataframe(
        df[['symbol', 'name', 'price', 'total_score', 'signal', 'change_24h']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'symbol': st.column_config.TextColumn('Ativo'),
            'name': st.column_config.TextColumn('Nome'),
            'price': st.column_config.NumberColumn('Preco', format="$ %.2f"),
            'total_score': st.column_config.NumberColumn('Score', format="%.1f"),
            'signal': st.column_config.TextColumn('Sinal'),
            'change_24h': st.column_config.NumberColumn('24h', format="%.2f%%")
        }
    )


def render_settings():
    """Configuracoes."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Crypto Trading**")
        st.write(f"Capital: $ {config.get('crypto.capital', 1000):,.2f} USD")
        st.write(f"Exposicao: {config.get('crypto.exposure', 0.10)*100:.1f}%")
        st.write(f"Stop Loss: {config.get('risk.stop_loss', 0.02)*100:.1f}%")
        st.write(f"Take Profit: {config.get('risk.take_profit', 0.05)*100:.1f}%")

    with col2:
        st.markdown("**APIs**")

        # Verifica Binance (testa conexao real)
        binance_status = "❌"
        if os.environ.get('BINANCE_API_KEY') and os.environ.get('BINANCE_SECRET_KEY'):
            if HAS_BINANCE:
                try:
                    client = BinanceClient()
                    test = client.test_connection()
                    if test.get('authenticated') or test.get('public_api'):
                        binance_status = "✅"
                except:
                    pass
        st.write(f"{binance_status} BINANCE_API_KEY")

        # CMC API
        cmc_status = "✅" if os.environ.get('CMC_API_KEY') else "❌"
        st.write(f"{cmc_status} CMC_API_KEY")

        # Database
        db_status = "✅" if os.environ.get('DATABASE_URL') else "❌"
        st.write(f"{db_status} DATABASE_URL")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main."""

    # Auto refresh
    refresh_interval = st.sidebar.selectbox(
        "Auto Refresh",
        [0, 15, 30, 60],
        index=2,
        format_func=lambda x: "Desativado" if x == 0 else f"{x}s"
    )

    if st.sidebar.button("🔄 Atualizar"):
        st.cache_data.clear()
        st.rerun()

    # Top bar
    render_topbar()

    # Stats
    render_stats_cards()

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Layout principal
    col1, col2 = st.columns([2, 1])

    with col1:
        # Charts
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">
                <div>
                    <div class="chart-title">Evolucao do Portfolio</div>
                    <div class="chart-subtitle">Performance acumulada</div>
                </div>
                <span class="live-badge">LIVE</span>
            </div>
        """, unsafe_allow_html=True)
        render_portfolio_chart()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">
                <div>
                    <div class="chart-title">P&L por Trade</div>
                    <div class="chart-subtitle">Ultimos 30 trades</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        render_pnl_chart()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Market overview
        render_market_overview()

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Mini stats
        stats = load_stats()
        trades_df = load_trades()

        if not trades_df.empty:
            avg_profit = trades_df['profit'].mean()
            max_profit = trades_df['profit'].max()
            max_loss = trades_df['profit'].min()

            st.markdown(f"""
            <div class="chart-container">
                <div class="chart-title" style="margin-bottom: 16px;">Metricas Rapidas</div>
                <div style="display: flex; flex-direction: column; gap: 12px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-gray);">Lucro Medio</span>
                        <span style="color: var(--text-white); font-family: 'JetBrains Mono';">R$ {avg_profit:.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-gray);">Maior Lucro</span>
                        <span style="color: var(--green); font-family: 'JetBrains Mono';">R$ {max_profit:.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-gray);">Maior Perda</span>
                        <span style="color: var(--red); font-family: 'JetBrains Mono';">R$ {max_loss:.2f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Trades Recentes", "🔍 Scanner", "⚙️ Config"])

    with tab1:
        render_trades_table()

    with tab2:
        render_crypto_scanner()

    with tab3:
        render_settings()

    # Auto refresh indicator
    if refresh_interval > 0:
        st.markdown(f"""
        <div class="refresh-indicator">
            <span class="dot"></span>
            Atualizando a cada {refresh_interval}s
        </div>
        """, unsafe_allow_html=True)

        time.sleep(refresh_interval)
        st.cache_data.clear()
        st.rerun()


if __name__ == '__main__':
    main()
