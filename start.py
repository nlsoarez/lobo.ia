import time
import traceback
import subprocess
from datetime import datetime
from logger import Logger

logger = Logger()

def mercado_aberto():
    agora = datetime.now()
    return agora.weekday() < 5 and 10 <= agora.hour < 18  # Segunda a sexta, das 10h às 18h

def executar_lobo():
    try:
        print("Executando Lobo IA...")
        subprocess.run(["python3", "main.py"])  # ✅ Corrigido para python3
        logger.log_trade({
            'symbol': 'LOBO',
            'date': str(datetime.now()),
            'action': 'EXECUTE',
            'price': 0,
            'quantity': 0,
            'profit': 0,
            'indicators': 'execução',
            'notes': 'main.py executado com sucesso.'
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

if __name__ == "__main__":
    logger.log_trade({
        'symbol': 'LOBO',
        'date': str(datetime.now()),
        'action': 'START',
        'price': 0,
        'quantity': 0,
        'profit': 0,
        'indicators': 'início',
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
            'indicators': 'encerrado',
            'notes': 'Lobo IA encerrado manualmente.'
        })
