# Integracao com Corretoras

Este guia explica como conectar o Lobo IA com corretoras reais para executar trades automaticamente.

> **AVISO**: Trading automatizado envolve riscos financeiros significativos. Use com cautela e sempre comece em modo simulacao.

## Corretoras para Acoes B3

### 1. XP Investimentos

```python
# Instalar SDK
pip install xpinvest

# Configurar
XP_API_KEY=sua_chave
XP_SECRET=seu_secret
XP_ACCOUNT=sua_conta
```

**Documentacao**: https://developers.xpi.com.br

### 2. Clear Corretora

```python
# Via MetaTrader 5
pip install MetaTrader5

import MetaTrader5 as mt5

mt5.initialize()
mt5.login(account, password=senha, server="Clear-Demo")
```

### 3. Rico Investimentos

- Usa mesma API da XP (grupo XP Inc.)
- Contato: api@rico.com.vc

### 4. BTG Pactual

```python
# API REST
BTG_API_URL=https://api.btgpactual.com
BTG_CLIENT_ID=seu_client_id
BTG_CLIENT_SECRET=seu_secret
```

### 5. MetaTrader 5 (Universal)

Funciona com varias corretoras:
- Clear
- Modal
- Orama
- Genial

```python
import MetaTrader5 as mt5

# Inicializar
if not mt5.initialize():
    print("Erro ao inicializar MT5")
    quit()

# Login
authorized = mt5.login(
    login=12345678,
    password="sua_senha",
    server="Nome-Servidor"
)

# Enviar ordem
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": "PETR4",
    "volume": 100,
    "type": mt5.ORDER_TYPE_BUY,
    "price": mt5.symbol_info_tick("PETR4").ask,
    "deviation": 20,
    "magic": 234000,
    "comment": "Lobo IA Trade",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

result = mt5.order_send(request)
```

## Corretoras para Criptomoedas

### 1. Binance (Recomendado)

```python
pip install python-binance

from binance.client import Client

client = Client(api_key, api_secret)

# Comprar BTC
order = client.order_market_buy(
    symbol='BTCUSDT',
    quantity=0.001
)

# Vender ETH
order = client.order_market_sell(
    symbol='ETHUSDT',
    quantity=0.1
)
```

**Variaveis de ambiente:**
```
BINANCE_API_KEY=sua_api_key
BINANCE_SECRET_KEY=sua_secret_key
BINANCE_TESTNET=true  # Para testes
```

### 2. Mercado Bitcoin (Brasil)

```python
pip install mercadobitcoin

from mercadobitcoin import TradeApi

api = TradeApi(
    tapi_id='seu_id',
    tapi_secret='seu_secret'
)

# Comprar Bitcoin
api.place_buy_order(
    coin_pair='BRLBTC',
    quantity='0.001',
    limit_price='150000'
)
```

### 3. Foxbit (Brasil)

```python
# API REST
FOXBIT_API_KEY=sua_key
FOXBIT_API_SECRET=seu_secret

# Endpoint
https://api.foxbit.com.br/rest/v3
```

### 4. Bitfinex

```python
pip install bitfinex-api-py

from bfxapi import Client

bfx = Client(
    api_key='sua_key',
    api_secret='seu_secret'
)
```

## Configuracao no Lobo IA

### 1. Variaveis de Ambiente (.env)

```bash
# Modo de execucao
EXECUTION_MODE=live  # simulation, paper, live

# B3 - MetaTrader 5
MT5_LOGIN=12345678
MT5_PASSWORD=sua_senha
MT5_SERVER=Clear-Real

# Crypto - Binance
BINANCE_API_KEY=sua_api_key
BINANCE_SECRET_KEY=sua_secret_key
BINANCE_TESTNET=false
```

### 2. Arquivo config.yaml

```yaml
execution:
  mode: "live"  # Mudar de simulation para live

  # B3
  broker_b3: "metatrader5"  # ou "xp", "clear"

  # Crypto
  broker_crypto: "binance"  # ou "mercadobitcoin"
```

### 3. Implementar Trade Executor Real

Crie um arquivo `broker_executor.py`:

```python
from abc import ABC, abstractmethod

class BrokerExecutor(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def buy(self, symbol, quantity, price=None):
        pass

    @abstractmethod
    def sell(self, symbol, quantity, price=None):
        pass

    @abstractmethod
    def get_balance(self):
        pass


class BinanceExecutor(BrokerExecutor):
    def __init__(self):
        from binance.client import Client
        self.client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_SECRET_KEY')
        )

    def buy(self, symbol, quantity, price=None):
        if price:
            return self.client.order_limit_buy(
                symbol=symbol,
                quantity=quantity,
                price=str(price)
            )
        return self.client.order_market_buy(
            symbol=symbol,
            quantity=quantity
        )

    def sell(self, symbol, quantity, price=None):
        if price:
            return self.client.order_limit_sell(
                symbol=symbol,
                quantity=quantity,
                price=str(price)
            )
        return self.client.order_market_sell(
            symbol=symbol,
            quantity=quantity
        )


class MT5Executor(BrokerExecutor):
    def __init__(self):
        import MetaTrader5 as mt5
        self.mt5 = mt5

    def connect(self):
        self.mt5.initialize()
        return self.mt5.login(
            login=int(os.getenv('MT5_LOGIN')),
            password=os.getenv('MT5_PASSWORD'),
            server=os.getenv('MT5_SERVER')
        )
```

## Fluxo para Ativar Trading Real

1. **Teste em Simulacao** (1-2 semanas)
   ```yaml
   execution:
     mode: "simulation"
   ```

2. **Teste em Paper Trading** (1-2 semanas)
   ```yaml
   execution:
     mode: "paper"
   ```
   - Binance Testnet
   - MT5 conta demo

3. **Trading Real com Valor Minimo**
   ```yaml
   execution:
     mode: "live"
   trading:
     capital: 1000  # Comece pequeno!
   ```

4. **Aumente Gradualmente**
   - Monitore performance
   - Ajuste parametros
   - Aumente capital conforme confianca

## Seguranca

### Boas Praticas

1. **Nunca commite suas chaves**
   ```gitignore
   .env
   *.key
   credentials.json
   ```

2. **Use variaveis de ambiente**
   ```bash
   export BINANCE_API_KEY="sua_key"
   ```

3. **Restrinja permissoes da API**
   - Binance: Apenas trading, sem saque
   - MT5: Apenas trading automatizado

4. **Defina limites**
   ```yaml
   risk:
     max_daily_loss: 500  # R$ maximo de perda diaria
     max_position_size: 0.05  # 5% do capital por trade
   ```

5. **Monitore 24/7**
   - Alertas por Telegram
   - Dashboard em tempo real
   - Logs detalhados

## Custos e Taxas

| Corretora | Taxa Trading | Taxa Custodia |
|-----------|--------------|---------------|
| XP | 0.03% | Gratis |
| Clear | 0.00% | Gratis |
| Binance | 0.10% | N/A |
| Mercado Bitcoin | 0.30-0.70% | N/A |

## Suporte

- **Binance API**: https://binance-docs.github.io/apidocs/
- **MT5 Python**: https://www.mql5.com/en/docs/python_metatrader5
- **XP API**: developers@xpi.com.br
