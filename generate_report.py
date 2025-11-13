"""
Gerador de relatórios para GitHub Actions.
Cria relatórios em texto e HTML para visualização.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
import json

def generate_report():
    """Gera relatório de performance do trading."""
    
    # Cria diretório de relatórios
    os.makedirs('reports', exist_ok=True)
    
    try:
        # Conecta ao banco
        conn = sqlite3.connect('trades.db')
        
        # Carrega trades
        query = """
        SELECT * FROM trades 
        WHERE date >= date('now', '-7 days')
        ORDER BY date DESC
        """
        df = pd.read_sql_query(query, conn)
        
        # Calcula estatísticas
        total_trades = len(df)
        
        if total_trades > 0:
            total_profit = df['profit'].sum()
            wins = len(df[df['profit'] > 0])
            losses = len(df[df['profit'] < 0])
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            # Maior lucro e perda
            max_profit = df['profit'].max()
            max_loss = df['profit'].min()
            
            # Por símbolo
            by_symbol = df.groupby('symbol').agg({
                'profit': ['sum', 'count', 'mean']
            }).round(2)
            
            # Trades recentes
            recent_trades = df.head(10)
        else:
            total_profit = 0
            wins = 0
            losses = 0
            win_rate = 0
            max_profit = 0
            max_loss = 0
            by_symbol = pd.DataFrame()
            recent_trades = pd.DataFrame()
        
        conn.close()
        
    except Exception as e:
        print(f"Erro ao acessar banco: {e}")
        # Valores padrão se não há banco
        total_trades = 0
        total_profit = 0
        wins = 0
        losses = 0
        win_rate = 0
        max_profit = 0
        max_loss = 0
        by_symbol = pd.DataFrame()
        recent_trades = pd.DataFrame()
    
    # Gera relatório em texto
    report_text = f"""# 🐺 LOBO IA - Relatório de Trading

**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M')}

## 📊 Resumo da Semana

- **Total de Trades:** {total_trades}
- **Lucro Total:** R$ {total_profit:.2f}
- **Taxa de Acerto:** {win_rate:.1f}%
- **Vitórias/Derrotas:** {wins}/{losses}

## 💰 Extremos

- **Maior Lucro:** R$ {max_profit:.2f}
- **Maior Perda:** R$ {max_loss:.2f}

## 📈 Performance por Ativo

"""
    
    if not by_symbol.empty:
        for symbol in by_symbol.index:
            profit_sum = by_symbol.loc[symbol, ('profit', 'sum')]
            trade_count = by_symbol.loc[symbol, ('profit', 'count')]
            profit_mean = by_symbol.loc[symbol, ('profit', 'mean')]
            report_text += f"- **{symbol}:** R$ {profit_sum:.2f} ({int(trade_count)} trades, média R$ {profit_mean:.2f})\n"
    else:
        report_text += "Nenhum trade executado ainda.\n"
    
    report_text += "\n## 🔄 Últimos 10 Trades\n\n"
    
    if not recent_trades.empty:
        report_text += "| Data | Símbolo | Ação | Preço | Qtd | Lucro |\n"
        report_text += "|------|---------|------|-------|-----|-------|\n"
        
        for _, trade in recent_trades.iterrows():
            date_str = pd.to_datetime(trade['date']).strftime('%d/%m %H:%M')
            profit_emoji = "🟢" if trade['profit'] >= 0 else "🔴"
            report_text += f"| {date_str} | {trade['symbol']} | {trade['action']} | R$ {trade['price']:.2f} | {trade['quantity']} | {profit_emoji} R$ {trade['profit']:.2f} |\n"
    else:
        report_text += "Nenhum trade recente.\n"
    
    report_text += "\n---\n*Relatório gerado automaticamente por GitHub Actions*"
    
    # Salva relatório em texto
    with open('reports/summary.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Gera HTML para visualização melhor
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Lobo IA - Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        .positive {{ color: #10b981; }}
        .negative {{ color: #ef4444; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f9fafb;
            font-weight: 600;
        }}
        .emoji {{ font-size: 1.2em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🐺 Lobo IA - Dashboard de Trading</h1>
        <p>Atualizado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total de Trades</div>
                <div class="stat-value">{total_trades}</div>
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
                <div class="stat-label">Win/Loss</div>
                <div class="stat-value">{wins}/{losses}</div>
            </div>
        </div>
        
        <h2>📈 Performance por Ativo</h2>
        {by_symbol.to_html(classes='performance-table') if not by_symbol.empty else '<p>Sem dados</p>'}
        
        <h2>🔄 Trades Recentes</h2>
        {recent_trades[['date', 'symbol', 'action', 'price', 'quantity', 'profit']].to_html(index=False) if not recent_trades.empty else '<p>Sem trades recentes</p>'}
    </div>
</body>
</html>
    """
    
    # Salva HTML
    with open('reports/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Salva CSV para download
    if not recent_trades.empty:
        recent_trades.to_csv('reports/trades.csv', index=False)
    
    print("✅ Relatórios gerados com sucesso!")
    print(f"📊 Total de trades: {total_trades}")
    print(f"💰 Lucro total: R$ {total_profit:.2f}")
    print(f"🎯 Taxa de acerto: {win_rate:.1f}%")

if __name__ == "__main__":
    generate_report()
