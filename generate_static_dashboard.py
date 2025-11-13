"""
Gerador de Dashboard HTML estático para GitHub Pages.
Cria visualizações interativas usando Plotly.
"""

import os
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

def generate_static_dashboard():
    """Gera dashboard HTML estático com gráficos interativos."""
    
    # Cria diretório de saída
    os.makedirs('dashboard_output', exist_ok=True)
    
    try:
        # Conecta ao banco
        conn = sqlite3.connect('trades.db')
        
        # Carrega todos os trades
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY date DESC", conn)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            
            # Calcula lucro acumulado
            df_sorted = df.sort_values('date')
            df_sorted['cumulative_profit'] = df_sorted['profit'].cumsum()
            df_sorted['capital'] = 10000 + df_sorted['cumulative_profit']
            
            # Estatísticas
            total_profit = df['profit'].sum()
            win_rate = (len(df[df['profit'] > 0]) / len(df) * 100) if len(df) > 0 else 0
            
            # Gráfico de evolução do capital
            fig_capital = go.Figure()
            fig_capital.add_trace(go.Scatter(
                x=df_sorted['date'],
                y=df_sorted['capital'],
                mode='lines',
                name='Capital',
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            fig_capital.update_layout(
                title='Evolução do Capital',
                xaxis_title='Data',
                yaxis_title='Capital (R$)',
                height=400,
                template='plotly_white'
            )
            
            # Gráfico de distribuição de lucros
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=df['profit'],
                nbinsx=20,
                name='Distribuição',
                marker_color='#764ba2'
            ))
            fig_dist.update_layout(
                title='Distribuição de Lucros/Perdas',
                xaxis_title='Lucro (R$)',
                yaxis_title='Frequência',
                height=400,
                template='plotly_white'
            )
            
            # Performance por símbolo
            by_symbol = df.groupby('symbol')['profit'].sum().reset_index()
            fig_symbol = px.bar(
                by_symbol,
                x='symbol',
                y='profit',
                title='Performance por Ativo',
                color='profit',
                color_continuous_scale=['red', 'yellow', 'green'],
                height=400
            )
            
            # Heatmap de atividade (dias vs horas)
            df['day'] = df['date'].dt.day_name()
            df['hour'] = df['date'].dt.hour
            activity_matrix = df.pivot_table(
                index='hour',
                columns='day',
                values='profit',
                aggfunc='count',
                fill_value=0
            )
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=activity_matrix.values,
                x=activity_matrix.columns,
                y=activity_matrix.index,
                colorscale='Viridis'
            ))
            fig_heatmap.update_layout(
                title='Mapa de Atividade de Trading',
                xaxis_title='Dia da Semana',
                yaxis_title='Hora',
                height=400
            )
            
            # Converte figuras para JSON
            capital_json = fig_capital.to_json()
            dist_json = fig_dist.to_json()
            symbol_json = fig_symbol.to_json()
            heatmap_json = fig_heatmap.to_json()
            
        else:
            # Gráficos vazios se não há dados
            empty_fig = go.Figure().add_annotation(
                text="Sem dados disponíveis",
                showarrow=False,
                font=dict(size=20)
            )
            capital_json = empty_fig.to_json()
            dist_json = capital_json
            symbol_json = capital_json
            heatmap_json = capital_json
            total_profit = 0
            win_rate = 0
            df = pd.DataFrame()
        
        conn.close()
        
    except Exception as e:
        print(f"Erro ao processar dados: {e}")
        # Gráficos vazios em caso de erro
        empty_fig = go.Figure().add_annotation(
            text="Erro ao carregar dados",
            showarrow=False,
            font=dict(size=20)
        )
        capital_json = empty_fig.to_json()
        dist_json = capital_json
        symbol_json = capital_json
        heatmap_json = capital_json
        total_profit = 0
        win_rate = 0
        df = pd.DataFrame()
    
    # HTML do dashboard
    html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐺 Lobo IA - Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            opacity: 0.9;
            font-size: 1.2em;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .table-container {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f5f5f5;
            font-weight: 600;
            color: #333;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .profit-positive {{
            color: #10b981;
            font-weight: bold;
        }}
        .profit-negative {{
            color: #ef4444;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            color: white;
            margin-top: 50px;
            padding: 20px;
            opacity: 0.9;
        }}
        .update-time {{
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 10px;
            display: inline-block;
            margin-top: 10px;
        }}
        @media (max-width: 768px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🐺 Lobo IA Trading Dashboard</h1>
        <p>Sistema Autônomo de Trading para B3</p>
        <div class="update-time">
            🕒 Atualizado: {datetime.now().strftime('%d/%m/%Y às %H:%M')}
        </div>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total de Trades</div>
                <div class="stat-value">{len(df) if not df.empty else 0}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Lucro Total</div>
                <div class="stat-value">R$ {total_profit:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Taxa de Acerto</div>
                <div class="stat-value">{win_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Capital Atual</div>
                <div class="stat-value">R$ {(10000 + total_profit):.2f}</div>
            </div>
        </div>
        
        <div class="chart-grid">
            <div class="chart-container">
                <div id="capital-chart"></div>
            </div>
            <div class="chart-container">
                <div id="dist-chart"></div>
            </div>
            <div class="chart-container">
                <div id="symbol-chart"></div>
            </div>
            <div class="chart-container">
                <div id="heatmap-chart"></div>
            </div>
        </div>
        
        <div class="table-container">
            <h2>📋 Últimos 20 Trades</h2>
            {df.head(20)[['date', 'symbol', 'action', 'price', 'quantity', 'profit']].to_html(
                index=False,
                classes='trades-table',
                escape=False
            ) if not df.empty else '<p>Nenhum trade registrado</p>'}
        </div>
    </div>
    
    <div class="footer">
        <p>Desenvolvido com 🐺 por Lobo IA Team</p>
        <p>Powered by GitHub Actions + GitHub Pages</p>
    </div>
    
    <script>
        // Renderiza gráficos
        Plotly.newPlot('capital-chart', {capital_json});
        Plotly.newPlot('dist-chart', {dist_json});
        Plotly.newPlot('symbol-chart', {symbol_json});
        Plotly.newPlot('heatmap-chart', {heatmap_json});
        
        // Aplica classes de cor aos lucros
        document.querySelectorAll('td').forEach(cell => {{
            if (cell.textContent.includes('R$') || !isNaN(parseFloat(cell.textContent))) {{
                const value = parseFloat(cell.textContent.replace('R$', '').replace(',', '.'));
                if (!isNaN(value) && cell.cellIndex === 5) {{ // Coluna de profit
                    if (value > 0) {{
                        cell.classList.add('profit-positive');
                        cell.textContent = '+' + cell.textContent;
                    }} else if (value < 0) {{
                        cell.classList.add('profit-negative');
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
    """
    
    # Salva o dashboard
    with open('dashboard_output/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Dashboard HTML gerado com sucesso!")
    print(f"📊 Total de trades processados: {len(df) if not df.empty else 0}")

if __name__ == "__main__":
    generate_static_dashboard()
