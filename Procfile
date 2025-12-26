# Procfile para Railway / Heroku
# Define múltiplos processos que podem rodar separadamente

# Worker: Bot de trading autônomo (processo principal)
worker: python start.py

# Web: Dashboard Streamlit (interface de monitoramento)
web: streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false

# Alternativa: Rodar ambos com supervisord ou usar serviços separados no Railway
