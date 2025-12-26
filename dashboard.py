"""
Dashboard interativo do Lobo IA - Monitoramento em Tempo Real.
Visualize transacoes, ganhos/perdas, saldo e status do sistema.

Execute com: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Adiciona diretorio raiz ao path
sys.path.insert(0, '.')

from logger import Logger
from config_loader import config

# Tenta importar modulos opcionais
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


# Configuracao da pagina
st.set_page_config(
    page_title="Lobo IA - Dashboard",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
    .positive { color: #00ff00 !important; }
    .negative { color: #ff4444 !important; }
    .neutral { color: #ffaa00 !important; }
    .status-online { color: #00ff00; }
    .status-offline { color: #ff4444; }
    .trade-buy { background-color: rgba(0, 255, 0, 0.1); }
    .trade-sell { background-color: rgba(255, 0, 0, 0.1); }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 5px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNCOES DE CARREGAMENTO DE DADOS
# ============================================================================

@st.cache_data(ttl=30)
def load_trades(limit=500):
    """Carrega trades do banco de dados."""
    try:
        with Logger() as logger:
            trades = logger.get_trades(limit=limit)
        df = pd.DataFrame(trades)
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_performance_stats():
    """Carrega estatisticas de performance."""
    try:
        with Logger() as logger:
            stats = logger.get_performance_stats()
        return stats
    except Exception as e:
        return {}


@st.cache_data(ttl=60)
def get_binance_balance():
    """Obtem saldo real da Binance."""
    if not HAS_BINANCE:
        return None

    try:
        client = BinanceClient()
        result = client.get_total_balance_usdt()
        return result
    except Exception as e:
        return {'error': str(e)}


def get_market_status():
    """Retorna status dos mercados."""
    now = datetime.now()

    b3_status = {
        'name': 'B3',
        'is_open': False,
        'reason': 'Indisponivel',
        'next_open': 'N/A'
    }

    crypto_status = {
        'name': 'Crypto',
        'is_open': True,  # Crypto 24/7
        'reason': '24/7',
        'next_open': 'Sempre aberto'
    }

    if HAS_CALENDAR:
        if is_weekend(now):
            b3_status['reason'] = 'Fim de semana'
            b3_status['is_open'] = False
        elif is_holiday(now):
            b3_status['reason'] = 'Feriado'
            b3_status['is_open'] = False
        elif 10 <= now.hour < 18:
            b3_status['is_open'] = True
            b3_status['reason'] = 'Aberto'
        else:
            b3_status['reason'] = 'Fora do horario'
            b3_status['is_open'] = False

        b3_status['next_open'] = get_next_trading_day(now).strftime('%d/%m/%Y')

    return {'b3': b3_status, 'crypto': crypto_status}


# ============================================================================
# PAGINA PRINCIPAL - DASHBOARD
# ============================================================================

def show_main_dashboard():
    """Dashboard principal com metricas em tempo real."""

    # Header
    st.markdown('<div class="main-header">🐺 LOBO IA - Dashboard de Trading</div>', unsafe_allow_html=True)

    # Status dos mercados
    market_status = get_market_status()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        b3 = market_status['b3']
        status_icon = "🟢" if b3['is_open'] else "🔴"
        st.metric(
            f"{status_icon} B3",
            b3['reason'],
            f"Proximo: {b3['next_open']}" if not b3['is_open'] else "Operando"
        )

    with col2:
        crypto = market_status['crypto']
        st.metric(
            "🟢 Crypto",
            crypto['reason'],
            "BTC, ETH, SOL..."
        )

    with col3:
        mode = config.get('execution.mode', 'simulation')
        mode_display = "SIMULACAO" if mode == "simulation" else "REAL"
        mode_icon = "🎮" if mode == "simulation" else "💰"
        st.metric(f"{mode_icon} Modo", mode_display)

    with col4:
        st.metric("🕐 Atualizado", datetime.now().strftime("%H:%M:%S"))

    st.divider()

    # Metricas financeiras principais
    stats = load_performance_stats()
    trades_df = load_trades()

    capital_inicial = config.get('trading.capital', 10000)
    capital_atual = capital_inicial + stats.get('total_profit', 0)

    # Saldo real da Binance
    binance_balance = get_binance_balance()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        # Mostrar saldo Binance se disponivel
        if binance_balance and binance_balance.get('success'):
            total_usdt = binance_balance.get('total_usdt', 0)
            testnet_label = " (Testnet)" if binance_balance.get('testnet') else ""
            st.metric(
                f"💰 Binance{testnet_label}",
                f"$ {total_usdt:,.2f}",
                f"USDT"
            )
        else:
            st.metric(
                "💰 Saldo Simulado",
                f"R$ {capital_atual:,.2f}",
                f"{((capital_atual/capital_inicial)-1)*100:+.2f}%"
            )

    with col2:
        total_profit = stats.get('total_profit', 0)
        st.metric(
            "📈 Lucro/Prejuizo",
            f"R$ {total_profit:,.2f}",
            f"{'Ganho' if total_profit >= 0 else 'Perda'}",
            delta_color="normal" if total_profit >= 0 else "inverse"
        )

    with col3:
        win_rate = stats.get('win_rate', 0)
        st.metric(
            "🎯 Win Rate",
            f"{win_rate:.1f}%",
            f"{win_rate - 50:+.1f}% vs 50%",
            delta_color="normal" if win_rate >= 50 else "inverse"
        )

    with col4:
        total_trades = stats.get('total_trades', 0)
        st.metric("📊 Total Trades", total_trades)

    with col5:
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        st.metric("✅/❌ W/L", f"{wins}/{losses}")

    st.divider()

    # Graficos principais
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Evolucao do Capital")

        if not trades_df.empty and 'profit' in trades_df.columns:
            trades_df_sorted = trades_df.sort_values('date')
            trades_df_sorted['cumulative_profit'] = trades_df_sorted['profit'].cumsum()
            trades_df_sorted['capital'] = capital_inicial + trades_df_sorted['cumulative_profit']

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df_sorted['date'],
                y=trades_df_sorted['capital'],
                mode='lines',
                name='Capital',
                line=dict(color='#00ff00', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ))

            # Linha de capital inicial
            fig.add_hline(y=capital_inicial, line_dash="dash",
                         line_color="yellow", annotation_text="Capital Inicial")

            fig.update_layout(
                xaxis_title="Data",
                yaxis_title="Capital (R$)",
                hovermode='x unified',
                height=350,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aguardando trades para exibir grafico...")

    with col2:
        st.subheader("📊 Lucro por Trade")

        if not trades_df.empty and 'profit' in trades_df.columns:
            # Ultimos 20 trades
            recent_trades = trades_df.sort_values('date', ascending=False).head(20)

            colors = ['#00ff00' if p >= 0 else '#ff4444' for p in recent_trades['profit']]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(len(recent_trades))),
                y=recent_trades['profit'],
                marker_color=colors,
                name='Lucro'
            ))

            fig.add_hline(y=0, line_color="white", line_width=1)

            fig.update_layout(
                xaxis_title="Trade #",
                yaxis_title="Lucro (R$)",
                height=350,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aguardando trades para exibir grafico...")

    st.divider()

    # Transacoes recentes
    st.subheader("🔄 Transacoes Recentes")

    if not trades_df.empty:
        recent = trades_df.sort_values('date', ascending=False).head(10)

        # Formata tabela
        display_df = recent[['symbol', 'date', 'action', 'price', 'quantity', 'profit']].copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%d/%m %H:%M')
        display_df['price'] = display_df['price'].apply(lambda x: f"R$ {x:.2f}")
        display_df['profit'] = display_df['profit'].apply(
            lambda x: f"R$ {x:+.2f}" if x != 0 else "-"
        )
        display_df.columns = ['Simbolo', 'Data', 'Acao', 'Preco', 'Qtd', 'Resultado']

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("📭 Nenhuma transacao registrada ainda.")


# ============================================================================
# PAGINA DE TRANSACOES
# ============================================================================

def show_transactions():
    """Pagina de historico de transacoes."""
    st.header("🔍 Historico de Transacoes")

    trades_df = load_trades(limit=1000)

    if trades_df.empty:
        st.info("📭 Nenhum trade registrado ainda.")
        return

    # Filtros
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        symbols = ['Todos'] + sorted(trades_df['symbol'].unique().tolist())
        selected_symbol = st.selectbox("Simbolo", symbols)

    with col2:
        actions = ['Todos', 'BUY', 'SELL']
        selected_action = st.selectbox("Acao", actions)

    with col3:
        result_filter = st.selectbox("Resultado", ['Todos', 'Lucro', 'Prejuizo', 'Neutro'])

    with col4:
        date_range = st.selectbox("Periodo", ['Todos', 'Hoje', '7 dias', '30 dias'])

    # Aplica filtros
    filtered_df = trades_df.copy()

    if selected_symbol != 'Todos':
        filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]

    if selected_action != 'Todos':
        filtered_df = filtered_df[filtered_df['action'] == selected_action]

    if result_filter == 'Lucro':
        filtered_df = filtered_df[filtered_df['profit'] > 0]
    elif result_filter == 'Prejuizo':
        filtered_df = filtered_df[filtered_df['profit'] < 0]
    elif result_filter == 'Neutro':
        filtered_df = filtered_df[filtered_df['profit'] == 0]

    if date_range != 'Todos' and 'date' in filtered_df.columns:
        now = datetime.now()
        if date_range == 'Hoje':
            filtered_df = filtered_df[filtered_df['date'].dt.date == now.date()]
        elif date_range == '7 dias':
            filtered_df = filtered_df[filtered_df['date'] >= now - timedelta(days=7)]
        elif date_range == '30 dias':
            filtered_df = filtered_df[filtered_df['date'] >= now - timedelta(days=30)]

    # Metricas do filtro
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Filtrado", len(filtered_df))
    with col2:
        profit_sum = filtered_df['profit'].sum() if not filtered_df.empty else 0
        st.metric("Lucro Total", f"R$ {profit_sum:,.2f}")
    with col3:
        wins = len(filtered_df[filtered_df['profit'] > 0]) if not filtered_df.empty else 0
        st.metric("Ganhos", wins)
    with col4:
        losses = len(filtered_df[filtered_df['profit'] < 0]) if not filtered_df.empty else 0
        st.metric("Perdas", losses)

    st.divider()

    # Tabela de transacoes
    if not filtered_df.empty:
        display_df = filtered_df[['symbol', 'date', 'action', 'price', 'quantity', 'profit']].copy()
        display_df = display_df.sort_values('date', ascending=False)
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%d/%m/%Y %H:%M')

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'symbol': 'Simbolo',
                'date': 'Data',
                'action': 'Acao',
                'price': st.column_config.NumberColumn('Preco', format="R$ %.2f"),
                'quantity': 'Qtd',
                'profit': st.column_config.NumberColumn('Lucro', format="R$ %.2f")
            }
        )

        # Exportar
        col1, col2 = st.columns([1, 4])
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Exportar CSV",
                csv,
                "trades_export.csv",
                "text/csv",
                use_container_width=True
            )
    else:
        st.warning("Nenhum trade encontrado com os filtros selecionados.")


# ============================================================================
# PAGINA DE ANALISE DE PERFORMANCE
# ============================================================================

def show_performance():
    """Pagina de analise de performance."""
    st.header("📈 Analise de Performance")

    trades_df = load_trades()
    stats = load_performance_stats()

    if trades_df.empty:
        st.info("📭 Nenhum trade registrado ainda.")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Metricas", "📉 Drawdown", "📅 Por Periodo"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Lucros")
            st.metric("Lucro Medio", f"R$ {trades_df['profit'].mean():.2f}")
            st.metric("Maior Lucro", f"R$ {trades_df['profit'].max():.2f}")
            st.metric("Maior Perda", f"R$ {trades_df['profit'].min():.2f}")

        with col2:
            wins = trades_df[trades_df['profit'] > 0]
            losses = trades_df[trades_df['profit'] < 0]

            avg_win = wins['profit'].mean() if len(wins) > 0 else 0
            avg_loss = losses['profit'].mean() if len(losses) > 0 else 0

            st.subheader("Medias")
            st.metric("Media Ganhos", f"R$ {avg_win:.2f}")
            st.metric("Media Perdas", f"R$ {avg_loss:.2f}")

            if avg_loss != 0:
                ratio = abs(avg_win / avg_loss)
                st.metric("Ratio W/L", f"{ratio:.2f}")

        with col3:
            gross_profit = wins['profit'].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses['profit'].sum()) if len(losses) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            st.subheader("Totais")
            st.metric("Profit Factor", f"{profit_factor:.2f}")
            st.metric("Lucro Bruto", f"R$ {gross_profit:.2f}")
            st.metric("Perda Bruta", f"R$ {gross_loss:.2f}")

    with tab2:
        st.subheader("Analise de Drawdown")

        trades_sorted = trades_df.sort_values('date')
        trades_sorted['cumulative_profit'] = trades_sorted['profit'].cumsum()
        trades_sorted['running_max'] = trades_sorted['cumulative_profit'].cummax()
        trades_sorted['drawdown'] = trades_sorted['cumulative_profit'] - trades_sorted['running_max']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_sorted['date'],
            y=trades_sorted['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.2)'
        ))

        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Drawdown (R$)",
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        max_dd = trades_sorted['drawdown'].min()
        st.metric("Drawdown Maximo", f"R$ {abs(max_dd):.2f}")

    with tab3:
        st.subheader("Performance por Periodo")

        if 'date' in trades_df.columns:
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df['day'] = trades_df['date'].dt.date

            daily_profit = trades_df.groupby('day').agg({
                'profit': 'sum',
                'id': 'count'
            }).reset_index()
            daily_profit.columns = ['Data', 'Lucro', 'Trades']

            fig = go.Figure()
            colors = ['#00ff00' if p >= 0 else '#ff4444' for p in daily_profit['Lucro']]

            fig.add_trace(go.Bar(
                x=daily_profit['Data'],
                y=daily_profit['Lucro'],
                marker_color=colors,
                name='Lucro Diario'
            ))

            fig.add_hline(y=0, line_color="white", line_width=1)

            fig.update_layout(
                xaxis_title="Data",
                yaxis_title="Lucro (R$)",
                height=400,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGINA DE SCANNER DE MERCADO
# ============================================================================

def show_market_scanner():
    """Pagina do scanner de mercado."""
    st.header("🔍 Scanner de Mercado")

    tab1, tab2 = st.tabs(["🇧🇷 B3 - Acoes", "₿ Criptomoedas"])

    with tab1:
        if HAS_SCANNER:
            if st.button("🔄 Escanear B3", use_container_width=True):
                with st.spinner("Escaneando mercado B3..."):
                    scanner = MarketScanner()
                    results = scanner.scan_market()

                    if results:
                        df = pd.DataFrame(results[:20])

                        st.success(f"Encontradas {len(results)} acoes!")

                        st.dataframe(
                            df[['symbol', 'price', 'total_score', 'signal', 'rsi', 'change_5d']],
                            use_container_width=True,
                            column_config={
                                'symbol': 'Simbolo',
                                'price': st.column_config.NumberColumn('Preco', format="R$ %.2f"),
                                'total_score': st.column_config.NumberColumn('Score', format="%.1f"),
                                'signal': 'Sinal',
                                'rsi': st.column_config.NumberColumn('RSI', format="%.1f"),
                                'change_5d': st.column_config.NumberColumn('Var 5d', format="%.2f%%")
                            }
                        )
                    else:
                        st.warning("Nenhum resultado encontrado.")
        else:
            st.warning("Scanner B3 nao disponivel.")

    with tab2:
        if HAS_CRYPTO:
            if st.button("🔄 Escanear Crypto", use_container_width=True):
                with st.spinner("Escaneando mercado crypto..."):
                    scanner = CryptoScanner()
                    results = scanner.scan_crypto_market()

                    if results:
                        df = pd.DataFrame(results[:15])

                        st.success(f"Analisadas {len(results)} criptomoedas!")

                        # BTC e ETH destacados
                        col1, col2 = st.columns(2)
                        btc = next((r for r in results if r['symbol'] == 'BTC-USD'), None)
                        eth = next((r for r in results if r['symbol'] == 'ETH-USD'), None)

                        with col1:
                            if btc:
                                st.metric(
                                    "₿ Bitcoin",
                                    f"${btc['price']:,.2f}",
                                    f"{btc['change_24h']:+.2f}%"
                                )

                        with col2:
                            if eth:
                                st.metric(
                                    "Ξ Ethereum",
                                    f"${eth['price']:,.2f}",
                                    f"{eth['change_24h']:+.2f}%"
                                )

                        st.divider()

                        st.dataframe(
                            df[['symbol', 'name', 'price', 'total_score', 'signal', 'change_24h']],
                            use_container_width=True,
                            column_config={
                                'symbol': 'Simbolo',
                                'name': 'Nome',
                                'price': st.column_config.NumberColumn('Preco', format="$ %.2f"),
                                'total_score': st.column_config.NumberColumn('Score', format="%.1f"),
                                'signal': 'Sinal',
                                'change_24h': st.column_config.NumberColumn('24h', format="%.2f%%")
                            }
                        )
                    else:
                        st.warning("Nenhum resultado encontrado.")
        else:
            st.warning("Scanner Crypto nao disponivel.")


# ============================================================================
# PAGINA DE CONFIGURACOES
# ============================================================================

def show_settings():
    """Pagina de configuracoes."""
    st.header("⚙️ Configuracoes")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Trading")
        st.write(f"**Capital Inicial:** R$ {config.get('trading.capital', 10000):,.2f}")
        st.write(f"**Exposicao por Trade:** {config.get('trading.exposure', 0.03)*100:.1f}%")
        st.write(f"**Exposicao Maxima:** {config.get('trading.max_total_exposure', 0.2)*100:.1f}%")

        st.subheader("Risco")
        st.write(f"**Stop Loss:** {config.get('risk.stop_loss', 0.02)*100:.1f}%")
        st.write(f"**Take Profit:** {config.get('risk.take_profit', 0.05)*100:.1f}%")
        st.write(f"**Max Drawdown:** {config.get('risk.max_drawdown', 0.1)*100:.1f}%")

    with col2:
        st.subheader("Execucao")
        st.write(f"**Modo:** {config.get('execution.mode', 'simulation')}")

        st.subheader("Mercado")
        st.write(f"**Horario B3:** {config.get('market.open_hour', 10)}h - {config.get('market.close_hour', 18)}h")
        st.write(f"**Verificar Feriados:** {config.get('market.check_holidays', True)}")

        st.subheader("Crypto")
        st.write(f"**Habilitado:** {config.get('crypto.enabled', True)}")
        st.write(f"**Broker:** {config.get('crypto.broker', 'simulation')}")

    st.divider()

    st.subheader("Variaveis de Ambiente")
    env_vars = ['BINANCE_API_KEY', 'BINANCE_SECRET_KEY', 'DATABASE_URL']
    for var in env_vars:
        value = os.environ.get(var)
        status = "✅ Configurado" if value else "❌ Nao configurado"
        st.write(f"**{var}:** {status}")

    st.divider()

    # Status da conexao Binance
    st.subheader("💰 Conexao Binance")

    binance_balance = get_binance_balance()

    if binance_balance:
        if binance_balance.get('success'):
            testnet = binance_balance.get('testnet', True)
            mode_text = "🟡 TESTNET" if testnet else "🟢 MAINNET (REAL)"
            st.info(f"**Modo:** {mode_text}")

            st.write(f"**Saldo Total:** $ {binance_balance.get('total_usdt', 0):,.2f} USDT")

            details = binance_balance.get('details', [])
            if details:
                st.write("**Ativos:**")
                for asset in details:
                    st.write(f"  - **{asset['asset']}:** {asset['amount']:.8f} (~${asset['value_usdt']:.2f})")
            else:
                st.write("Nenhum ativo com saldo")
        elif 'error' in binance_balance:
            st.error(f"Erro na conexao: {binance_balance['error']}")
    else:
        st.warning("Cliente Binance nao disponivel. Verifique as API Keys.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funcao principal."""

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/wolf.png", width=80)
        st.title("Lobo IA")
        st.caption("Trading Inteligente")

        st.divider()

        page = st.radio(
            "Navegacao",
            [
                "🏠 Dashboard",
                "🔄 Transacoes",
                "📈 Performance",
                "🔍 Scanner",
                "⚙️ Configuracoes"
            ],
            label_visibility="collapsed"
        )

        st.divider()

        # Auto refresh
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)

        if st.button("🔄 Atualizar Agora", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # Status rapido
        stats = load_performance_stats()
        capital_inicial = config.get('trading.capital', 10000)
        capital_atual = capital_inicial + stats.get('total_profit', 0)

        st.metric("💰 Saldo", f"R$ {capital_atual:,.2f}")
        st.metric("📊 Trades", stats.get('total_trades', 0))

    # Paginas
    if page == "🏠 Dashboard":
        show_main_dashboard()
    elif page == "🔄 Transacoes":
        show_transactions()
    elif page == "📈 Performance":
        show_performance()
    elif page == "🔍 Scanner":
        show_market_scanner()
    elif page == "⚙️ Configuracoes":
        show_settings()

    # Auto refresh
    if auto_refresh:
        import time
        time.sleep(30)
        st.cache_data.clear()
        st.rerun()


if __name__ == '__main__':
    main()
