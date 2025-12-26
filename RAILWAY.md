# Deploy no Railway

Este guia explica como fazer deploy do Lobo IA no Railway.

## Pre-requisitos

1. Conta no [Railway](https://railway.app)
2. Railway CLI instalado (opcional): `npm install -g @railway/cli`

## Deploy via Dashboard

### 1. Criar novo projeto

1. Acesse [railway.app](https://railway.app)
2. Clique em "New Project"
3. Selecione "Deploy from GitHub repo"
4. Autorize acesso ao repositório

### 2. Adicionar PostgreSQL

1. No projeto, clique em "+ New"
2. Selecione "Database" > "PostgreSQL"
3. A variavel `DATABASE_URL` sera configurada automaticamente

### 3. Configurar Variaveis de Ambiente

No painel do servico, va em "Variables" e adicione:

```
# Trading
TRADING_SYMBOLS=PETR4.SA,VALE3.SA,ITUB4.SA
TRADING_CAPITAL=10000.0
EXECUTION_MODE=simulation

# Risk
STOP_LOSS=0.02
TAKE_PROFIT=0.05
MAX_DRAWDOWN=0.10

# Market (B3)
MARKET_OPEN_HOUR=10
MARKET_CLOSE_HOUR=18
CHECK_INTERVAL=60

# Logging
LOG_LEVEL=INFO
```

### 4. Configurar Servicos

O projeto pode rodar de duas formas:

#### Opcao A: Bot de Trading (Worker)

No "Settings" do servico:
- Start Command: `python start.py`

#### Opcao B: Dashboard (Web)

No "Settings" do servico:
- Start Command: `streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

#### Opcao C: Ambos (Recomendado)

Crie dois servicos separados no Railway:
1. **lobo-worker**: Bot de trading (`python start.py`)
2. **lobo-dashboard**: Dashboard web (comando Streamlit acima)

Ambos podem usar o mesmo banco PostgreSQL.

## Deploy via CLI

```bash
# Login
railway login

# Inicializar projeto
railway init

# Link ao projeto existente
railway link

# Deploy
railway up

# Ver logs
railway logs
```

## Health Checks

O sistema inclui endpoints de health check:

- `/health` - Liveness probe (processo rodando)
- `/ready` - Readiness probe (banco conectado)
- `/status` - Status detalhado
- `/metrics` - Metricas Prometheus

## Arquitetura no Railway

```
┌─────────────────────────────────────────────────┐
│                   Railway                        │
│  ┌──────────────┐     ┌──────────────────────┐  │
│  │ lobo-worker  │     │   lobo-dashboard     │  │
│  │ (start.py)   │     │   (streamlit)        │  │
│  │              │     │                      │  │
│  │ Health: 8080 │     │   Port: $PORT        │  │
│  └──────┬───────┘     └──────────┬───────────┘  │
│         │                        │              │
│         └────────┬───────────────┘              │
│                  │                              │
│         ┌────────▼────────┐                     │
│         │   PostgreSQL    │                     │
│         │   (DATABASE_URL)│                     │
│         └─────────────────┘                     │
└─────────────────────────────────────────────────┘
```

## Monitoramento

### Logs
```bash
railway logs --service lobo-worker
```

### Metricas
Acesse `https://seu-app.railway.app/metrics` para metricas Prometheus.

### Status
Acesse `https://seu-app.railway.app/status` para status detalhado.

## Custos

Railway oferece $5/mes de credito gratis. Estimativa de uso:
- Worker: ~$2-3/mes (rodando 24/7)
- Dashboard: ~$1-2/mes
- PostgreSQL: ~$0.50/mes (uso minimo)

## Troubleshooting

### Bot nao executa trades

1. Verifique se `EXECUTION_MODE=simulation`
2. Verifique logs: `railway logs`
3. Verifique horario de mercado (10h-18h BRT)

### Dashboard nao carrega

1. Verifique se a porta esta configurada: `--server.port=$PORT`
2. Verifique variavel PORT no Railway

### Erro de conexao com banco

1. Verifique se PostgreSQL esta rodando
2. Verifique se `DATABASE_URL` esta configurada
3. Teste conexao: `railway run python -c "from logger import Logger; Logger()"`

## Suporte

- [Railway Docs](https://docs.railway.app)
- [Issues do Projeto](https://github.com/seu-usuario/lobo.ia/issues)
