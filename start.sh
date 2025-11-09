#!/bin/bash
nohup python3 start.py > lobo.log 2>&1 &
echo "Lobo IA iniciado com nohup. Logs em lobo.log"
