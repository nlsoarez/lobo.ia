"""
Dashboard interativo do Lobo IA usando Streamlit.
Visualize performance, posições, histórico e execute backtests.

Execute com: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import sys

# Adiciona diretório raiz ao path
sys.path.insert(0, '.')

from logger import Logger
from portfolio_manager import PortfolioManager
from backtesting import Backtester
from learning_module import LearningModule
from config_loader import config


# Configuração da página
st.set_page_config(
    page_title="Lobo IA Dashboard",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .big-metric {
        font-size: 2rem;
        font-weight: bold;
    }
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_trades():
    """Carrega trades do banco de dados."""
    try:
        with Logger() as logger:
            trades = logger.get_trades(limit=1000)
        return pd.DataFrame(trades)
    except Exception as e:
        st.error(f"Erro ao carregar trades: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_performance_stats():
    """Carrega estatísticas de performance."""
    try:
        with Logger() as logger:
            stats = logger.get_performance_stats()
        return stats
    except Exception as e:
        st.error(f"Erro ao carregar estatísticas: {e}")
        return {}


def main():
    """Função principal do dashboard."""

    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🐺 LOBO IA - Dashboard de Trading")
        st.caption("Sistema Autônomo de Trading Inteligente")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")

        page = st.radio(
            "Navegação",
            ["📊 Overview", "📈 Performance", "💼 Posições", "🔍 Histórico", "🔬 Backtesting", "🤖 Machine Learning"],
            label_visibility="collapsed"
        )

        st.divider()

        st.subheader("Status do Sistema")
        status_col1, status_col2 = st.columns(2)

        # Simula status (em produção, ler de arquivo/API)
        with status_col1:
            st.metric("Status", "🟢 ATIVO")
        with status_col2:
            st.metric("Modo", "SIMULAÇÃO")

        st.divider()

        # Botão de refresh
        if st.button("🔄 Atualizar Dados", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Páginas
    if page == "📊 Overview":
        show_overview()
    elif page == "📈 Performance":
        show_performance()
    elif page == "💼 Posições":
        show_positions()
    elif page == "🔍 Histórico":
        show_history()
    elif page == "🔬 Backtesting":
        show_backtesting()
    elif page == "🤖 Machine Learning":
        show_machine_learning()


def show_overview():
    """Mostra overview geral do sistema."""
    st.header("📊 Overview Geral")

    # Carrega dados
    stats = load_performance_stats()
    trades_df = load_trades()

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_profit = stats.get('total_profit', 0)
        color = "normal" if total_profit >= 0 else "inverse"
        st.metric(
            "💰 Lucro Total",
            f"R$ {total_profit:,.2f}",
            delta=f"{(total_profit / 10000) * 100:.2f}%" if total_profit != 0 else None,
            delta_color=color
        )

    with col2:
        win_rate = stats.get('win_rate', 0)
        st.metric(
            "🎯 Win Rate",
            f"{win_rate:.1f}%",
            delta=f"{win_rate - 50:.1f}% vs 50%",
            delta_color="normal" if win_rate >= 50 else "inverse"
        )

    with col3:
        total_trades = stats.get('total_trades', 0)
        st.metric("📊 Total de Trades", total_trades)

    with col4:
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        st.metric("✅ Wins / ❌ Losses", f"{wins} / {losses}")

    st.divider()

    # Gráficos
    if not trades_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Evolução do Capital")

            # Calcula lucro acumulado
            trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
            trades_df['capital'] = 10000 + trades_df['cumulative_profit']

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df.index,
                y=trades_df['capital'],
                mode='lines',
                name='Capital',
                line=dict(color='#00ff00', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ))

            fig.update_layout(
                xaxis_title="Trade #",
                yaxis_title="Capital (R$)",
                hovermode='x unified',
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 Distribuição de Lucros")

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=trades_df['profit'],
                nbinsx=20,
                name='Lucros',
                marker_color='#00ff00',
                opacity=0.75
            ))

            fig.update_layout(
                xaxis_title="Lucro (R$)",
                yaxis_title="Frequência",
                hovermode='x unified',
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📭 Nenhum trade registrado ainda. Execute o sistema para ver dados.")


def show_performance():
    """Mostra análise detalhada de performance."""
    st.header("📈 Análise de Performance")

    trades_df = load_trades()

    if trades_df.empty:
        st.info("📭 Nenhum trade registrado.")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Métricas", "📉 Drawdown", "🕒 Análise Temporal"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("💵 Lucro Médio", f"R$ {trades_df['profit'].mean():.2f}")
            st.metric("📈 Maior Lucro", f"R$ {trades_df['profit'].max():.2f}")
            st.metric("📉 Maior Perda", f"R$ {trades_df['profit'].min():.2f}")

        with col2:
            wins = trades_df[trades_df['profit'] > 0]
            losses = trades_df[trades_df['profit'] < 0]

            avg_win = wins['profit'].mean() if len(wins) > 0 else 0
            avg_loss = losses['profit'].mean() if len(losses) > 0 else 0

            st.metric("💚 Lucro Médio (Wins)", f"R$ {avg_win:.2f}")
            st.metric("💔 Perda Média (Losses)", f"R$ {avg_loss:.2f}")

            if avg_loss != 0:
                ratio = abs(avg_win / avg_loss)
                st.metric("⚖️ Ratio Win/Loss", f"{ratio:.2f}")

        with col3:
            gross_profit = wins['profit'].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses['profit'].sum()) if len(losses) > 0 else 1

            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            st.metric("🏆 Profit Factor", f"{profit_factor:.2f}")
            st.metric("💰 Lucro Bruto", f"R$ {gross_profit:.2f}")
            st.metric("💸 Perda Bruta", f"R$ {gross_loss:.2f}")

    with tab2:
        st.subheader("📉 Análise de Drawdown")

        # Calcula drawdown
        trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
        trades_df['running_max'] = trades_df['cumulative_profit'].cummax()
        trades_df['drawdown'] = trades_df['cumulative_profit'] - trades_df['running_max']

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=trades_df.index,
            y=trades_df['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))

        fig.update_layout(
            xaxis_title="Trade #",
            yaxis_title="Drawdown (R$)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        max_dd = trades_df['drawdown'].min()
        st.metric("📉 Drawdown Máximo", f"R$ {abs(max_dd):.2f}")

    with tab3:
        st.subheader("🕒 Performance ao Longo do Tempo")

        if 'date' in trades_df.columns:
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df['day'] = trades_df['date'].dt.date

            daily_profit = trades_df.groupby('day')['profit'].sum().reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily_profit['day'],
                y=daily_profit['profit'],
                name='Lucro Diário',
                marker_color=['green' if p >= 0 else 'red' for p in daily_profit['profit']]
            ))

            fig.update_layout(
                xaxis_title="Data",
                yaxis_title="Lucro (R$)",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)


def show_positions():
    """Mostra posições abertas."""
    st.header("💼 Posições Abertas")

    st.info("ℹ️ Em modo simulação. Posições não são persistidas entre execuções.")

    # Exemplo de estrutura (em produção, ler de arquivo ou API)
    positions_data = {
        'Símbolo': ['PETR4.SA', 'VALE3.SA'],
        'Quantidade': [10, 5],
        'Preço Médio': [35.50, 75.20],
        'Preço Atual': [36.20, 74.80],
        'P&L': [7.00, -2.00],
        'P&L %': [1.97, -0.53]
    }

    df_pos = pd.DataFrame(positions_data)

    st.dataframe(
        df_pos.style.applymap(
            lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else ('color: red' if isinstance(x, (int, float)) and x < 0 else ''),
            subset=['P&L', 'P&L %']
        ),
        use_container_width=True,
        hide_index=True
    )


def show_history():
    """Mostra histórico completo de trades."""
    st.header("🔍 Histórico de Trades")

    trades_df = load_trades()

    if trades_df.empty:
        st.info("📭 Nenhum trade registrado.")
        return

    # Filtros
    col1, col2, col3 = st.columns(3)

    with col1:
        symbols = ['Todos'] + list(trades_df['symbol'].unique())
        selected_symbol = st.selectbox("Símbolo", symbols)

    with col2:
        actions = ['Todos', 'BUY', 'SELL']
        selected_action = st.selectbox("Ação", actions)

    with col3:
        profit_filter = st.selectbox("Resultado", ['Todos', 'Lucro', 'Prejuízo'])

    # Aplica filtros
    filtered_df = trades_df.copy()

    if selected_symbol != 'Todos':
        filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]

    if selected_action != 'Todos':
        filtered_df = filtered_df[filtered_df['action'] == selected_action]

    if profit_filter == 'Lucro':
        filtered_df = filtered_df[filtered_df['profit'] > 0]
    elif profit_filter == 'Prejuízo':
        filtered_df = filtered_df[filtered_df['profit'] < 0]

    # Mostra tabela
    st.dataframe(
        filtered_df[['symbol', 'date', 'action', 'price', 'quantity', 'profit']],
        use_container_width=True,
        hide_index=True
    )

    # Botão de exportação
    if st.button("💾 Exportar para CSV"):
        filtered_df.to_csv('trades_export.csv', index=False)
        st.success("✅ Trades exportados para trades_export.csv")


def show_backtesting():
    """Interface de backtesting."""
    st.header("🔬 Backtesting")

    st.markdown("""
    Execute backtests para testar estratégias em dados históricos.
    """)

    col1, col2 = st.columns(2)

    with col1:
        symbol = st.text_input("Símbolo", "PETR4.SA")
        start_date = st.date_input("Data Inicial", datetime.now() - timedelta(days=90))

    with col2:
        interval = st.selectbox("Intervalo", ["1d", "1h", "5m", "15m", "30m"])
        end_date = st.date_input("Data Final", datetime.now())

    initial_capital = st.number_input("Capital Inicial (R$)", min_value=1000, value=10000, step=1000)

    if st.button("🚀 Executar Backtest", use_container_width=True):
        with st.spinner("Executando backtest..."):
            try:
                backtester = Backtester(
                    symbol=symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    initial_capital=initial_capital,
                    interval=interval
                )

                result = backtester.run()
                metrics = result.calculate_metrics()

                st.success("✅ Backtest concluído!")

                # Mostra resultados
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("💰 Lucro Total", f"R$ {metrics['total_profit']:.2f}")
                with col2:
                    st.metric("📈 Retorno", f"{metrics['total_return_pct']:.2f}%")
                with col3:
                    st.metric("🎯 Win Rate", f"{metrics['win_rate']:.1f}%")
                with col4:
                    st.metric("📊 Trades", metrics['total_trades'])

                # Métricas detalhadas
                st.subheader("📊 Métricas Detalhadas")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Lucro:**")
                    st.write(f"- Lucro Médio/Trade: R$ {metrics['avg_profit_per_trade']:.2f}")
                    st.write(f"- Maior Lucro: R$ {metrics['max_profit']:.2f}")
                    st.write(f"- Maior Perda: R$ {metrics['max_loss']:.2f}")

                with col2:
                    st.write("**Risco:**")
                    st.write(f"- Profit Factor: {metrics['profit_factor']:.2f}")
                    st.write(f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    st.write(f"- Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

            except Exception as e:
                st.error(f"❌ Erro durante backtest: {e}")


def show_machine_learning():
    """Interface de machine learning."""
    st.header("🤖 Machine Learning")

    st.markdown("""
    O módulo de ML usa Random Forest para prever sucesso de trades baseado em indicadores técnicos.
    """)

    st.info("ℹ️ Para treinar o modelo, são necessários pelo menos 50 trades no histórico.")

    # Status do modelo
    learning = LearningModule()

    col1, col2, col3 = st.columns(3)

    with col1:
        status = "✅ Treinado" if learning.is_trained else "❌ Não Treinado"
        st.metric("Status do Modelo", status)

    with col2:
        st.metric("Trades no Histórico", len(learning.history))

    with col3:
        if learning.is_trained:
            st.metric("Acurácia", "Modelo OK")
        else:
            st.metric("Acurácia", "N/A")

    # Botão de treinamento
    if st.button("🏋️ Treinar Modelo", use_container_width=True):
        with st.spinner("Treinando modelo..."):
            # Carrega histórico do banco
            trades_df = load_trades()

            if len(trades_df) < 50:
                st.error("❌ Dados insuficientes. Necessário pelo menos 50 trades.")
            else:
                # Adiciona trades ao learning module
                for _, trade in trades_df.iterrows():
                    learning.record_trade(trade.to_dict())

                # Treina
                success = learning.train_model()

                if success:
                    st.success("✅ Modelo treinado com sucesso!")

                    # Mostra feature importance
                    importance = learning.get_feature_importance()

                    if importance:
                        st.subheader("📊 Importância das Features")

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(importance.keys()),
                            y=list(importance.values()),
                            marker_color='lightblue'
                        ))

                        fig.update_layout(
                            xaxis_title="Feature",
                            yaxis_title="Importância",
                            height=300
                        )

                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Falha ao treinar modelo.")


if __name__ == '__main__':
    main()
