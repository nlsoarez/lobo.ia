import time
import traceback
from logger import log_event

# Simulação de módulos principais do Lobo IA
def executar_lobo():
    try:
        print("Executando Lobo IA...")
        # Aqui você chamaria os módulos reais, como:
        # analise_tecnica(), decisao_trade(), etc.
        log_event("Lobo IA executado com sucesso.")
    except Exception as e:
        erro = traceback.format_exc()
        log_event(f"Erro na execução: {erro}")

if __name__ == "__main__":
    while True:
        executar_lobo()
        time.sleep(60)  # Espera 60 segundos entre execuções
