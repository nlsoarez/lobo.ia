# Lobo IA – Sistema Inteligente de Trading de Ações

Lobo IA é um sistema autônomo de trading para ações da B3, com aprendizado contínuo, análise técnica e gestão de risco.

## Módulos

- `data_collector.py`: coleta dados de mercado via API da Brapi.dev
- `signal_analyzer.py`: analisa os dados e gera sinais de compra/venda
- `trade_executor.py`: executa ordens simuladas
- `learning_module.py`: avalia desempenho e ajusta estratégias
- `portfolio_manager.py`: gerencia capital e risco
- `logger.py`: registra operações em banco de dados SQLite
- `main.py`: orquestra o funcionamento do sistema

## Requisitos

- Python 3
- Bibliotecas: `requests`, `pandas`, `ta`, `sqlite3`

## Como usar

1. Clone o repositório
2. Instale as dependências com `pip install -r requirements.txt`
3. Execute `main.py`

## Objetivo

Criar um sistema de trading com IA adaptativa, capaz de aprender, agir com autonomia e se comunicar de forma transparente, buscando lucros consistentes com controle de risco e disciplina.
