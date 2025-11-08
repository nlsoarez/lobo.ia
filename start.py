import time
import subprocess

while True:
    print("Executando Lobo IA...")
    subprocess.run(["python", "main.py"])
    print("Aguardando 5 minutos antes da próxima execução...")
    time.sleep(300)  # 5 minutos
