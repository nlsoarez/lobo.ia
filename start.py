import time
import traceback
from datetime import datetime
from logger import Logger

# Inicializa o logger SQLite
logger = Logger()

# Função para verificar se o mercado está aberto
def mercado_aberto():
    agora = datetime.now()
    return agora.weekday() < 5 and 10 <= agora.hour < 18  # Segunda a sexta, das 10h às 18h

# Simulação de execução do Lobo IA
def executar_lobo():
    try:
        print("Executando Lobo IA...")
        # Aqui você chamaria os módulos reais, como:
        # analise_tecnica(), decisao_trade(), etc.
        logger.log_trade({
            'symbol': 'LOBO',
            'date': str(datetime.now()),
            'action': 'EXECUTE',
            'price': 0,
            'quantity': 0,
            'profit': 0,
            'indicators': 'sistema iniciado',
            'notes': 'Lobo IA executado com sucesso.'
        })
    except Exception as e:
        erro = traceback.format_exc()
        logger.log_trade({
            'symbol': 'LOBO',
            'date': str(datetime.now()),
            'action': 'ERROR',
            'price': 0,
            'quantity': 0,
            'profit': 0,
            'indicators': 'erro',
            'notes': erro
        })

# Loop principal com melhorias
if __name__ == "__main__":
    logger.log_trade({
        'symbol': 'LOBO',
        'date': str(datetime.now()),
        'action': 'START',
        'price': 0,
        'quantity': 0,
        'profit': 0,
        'indicators': 'sistema iniciado',
        'notes': 'Lobo IA iniciado.'
    })
    try:
        while True:
            if mercado_aberto():
                executar_lobo()
            else:
                print("Mercado fechado. Aguardando...")
            time.sleep(60)
    except KeyboardInterrupt:
        logger.log_trade({
            'symbol': 'LOBO',
            'date': str(datetime.now()),
            'action': 'STOP',
            'price': 0,
            'quantity': 0,
            'profit': 0,
            'indicators': 'sistema encerrado',
            'notes': 'Lobo IA encerrado manualmente.'
        })
