# Lobo IA V4.2 - Sistema de Trading de Emergência

## Visão Geral

O Sistema V4.2 introduz um conjunto de módulos avançados para gerenciamento de emergências, métricas inteligentes e alertas proativos. Este sistema foi projetado para maximizar oportunidades durante períodos de baixa atividade sem comprometer a segurança.

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOBO IA V4.2 - Crypto Only                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  EmergencyTrade  │    │  SmartHealth     │                  │
│  │     Manager      │◄───│    Metrics       │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  EmergencySignal │    │   SmartAlert     │                  │
│  │   Prioritizer    │───►│     System       │                  │
│  └──────────────────┘    └──────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Módulos

### 1. EmergencyTradeManager (`emergency_trade_manager.py`)

Gerencia override de limites durante modo emergência.

**Funcionalidades:**
- Override de limites de trades (20 → 35)
- Critical trade allowance (5 trades adicionais)
- Redução automática de tamanho de posição
- Tracking de trades críticos

**Decisões de Trade:**
| Tipo | Descrição | Multiplier |
|------|-----------|------------|
| `APPROVED` | Trade normal aprovado | 1.0 |
| `EMERGENCY` | Trade em modo emergência | 0.7 |
| `CRITICAL` | Trade crítico (bypass) | 0.56 |
| `DENIED_LIMIT` | Negado por limite | - |
| `DENIED_CRITICAL_EXHAUSTED` | Sem bypass disponível | - |

**Exemplo de Uso:**
```python
from emergency_trade_manager import emergency_trade_manager

# Verificar se pode operar
can_trade, decision, reasons = emergency_trade_manager.can_trade(
    signal={'symbol': 'BTC-USD', 'total_score': 72, 'rsi': 18},
    emergency_mode=True
)

if can_trade:
    # Obter multiplicador de posição
    multiplier = emergency_trade_manager.get_position_size_multiplier(decision)
    position_size = base_size * multiplier
```

### 2. SmartHealthMetrics (`smart_health_metrics.py`)

Métricas de saúde corrigidas com cálculo multi-dimensional.

**Correção Principal:**
```python
# ANTES (incorreto)
success_rate = total_successes / total_attempts  # Incluía tentativas não executadas

# DEPOIS (correto)
success_rate = wins / executed_trades  # Apenas trades executados
```

**Dimensões de Saúde:**
1. **Latency**: Tempo de resposta das APIs
2. **Success Rate**: Taxa de trades bem-sucedidos
3. **API Success**: Taxa de sucesso de chamadas API

**Status Geral:**
- `HEALTHY`: 2+ dimensões GREEN
- `DEGRADED`: 1 dimensão GREEN ou 2+ YELLOW
- `CRITICAL`: 2+ dimensões RED

**Exemplo:**
```python
from smart_health_metrics import smart_health_metrics

# Registrar resultado de trade
smart_health_metrics.record_trade_result(success=True)
smart_health_metrics.record_latency(45.0)

# Obter status
health = smart_health_metrics.get_health_status()
print(f"Status: {health['overall_status']}")
print(f"Success Rate: {health['success_rate']:.1f}%")
```

### 3. EmergencySignalPrioritizer (`emergency_signal_prioritizer.py`)

Prioriza sinais com classificação inteligente.

**Classificações:**
| Prioridade | Critérios | Ação |
|------------|-----------|------|
| `CRITICAL` | RSI ≤ 20, Score ≥ 65, Volume ≥ 1.5x | Bypass permitido |
| `HIGH` | Score ≥ 70 ou RSI ≤ 25 | Prioridade máxima |
| `MEDIUM` | Score ≥ 55 | Normal |
| `LOW` | Score ≥ 40 | Baixa prioridade |
| `SKIP` | Score < 40 | Ignorar |

**Scoring por Peso:**
```yaml
weights:
  rsi_extreme: 30     # RSI em nível extremo
  score_high: 25      # Score alto (>70)
  volume_high: 20     # Volume acima da média
  trend_aligned: 15   # Alinhado com tendência
  momentum_strong: 10 # Momentum forte
```

**Exemplo:**
```python
from emergency_signal_prioritizer import emergency_signal_prioritizer

signals = [
    {'symbol': 'BTC-USD', 'total_score': 75, 'rsi': 18, 'volume_ratio': 1.8},
    {'symbol': 'ETH-USD', 'total_score': 65, 'rsi': 35, 'volume_ratio': 1.2},
]

prioritized = emergency_signal_prioritizer.prioritize_signals(
    signals,
    emergency_mode=True
)

for sig in prioritized:
    print(f"{sig.symbol}: {sig.priority.name} - Bypass: {sig.bypass_recommended}")
```

### 4. SmartAlertSystem (`smart_alert_system.py`)

Sistema de alertas inteligentes com detecção de anomalias.

**Tipos de Alertas:**
1. **Metric Inconsistency**: Status HEALTHY mas success rate baixo
2. **Missed Opportunity**: Sinais fortes não executados
3. **RSI Extreme**: RSI em nível extremo por período prolongado
4. **Trade Limit**: Limite atingido com sinais pendentes
5. **API Health**: Alta taxa de falhas de API

**Níveis:**
- `INFO`: Informativo
- `WARNING`: Atenção necessária
- `CRITICAL`: Ação imediata
- `EMERGENCY`: Crítico urgente

**Exemplo:**
```python
from smart_alert_system import smart_alert_system

# Verificar consistência de métricas
alerts = smart_alert_system.check_metric_consistency(
    health_status="healthy",
    success_rate=25.0,  # Baixo!
    api_success_rate=95.0,
    total_trades=10
)

# Verificar oportunidades perdidas
alerts = smart_alert_system.check_missed_opportunities(
    signals=all_signals,
    executed_symbols=['BTC-USD'],
    reason="limit_reached"
)

# Obter alertas ativos
active = smart_alert_system.get_alerts(unresolved_only=True)
```

## Configuração

### config.yaml

```yaml
# Modo Emergência
emergency:
  enabled: true
  max_duration_hours: 6
  trade_limit_multiplier: 1.75
  position_size_multiplier: 0.70
  filter_relaxation_percent: 40
  max_trades: 35
  max_positions_override: 7

  critical_signals:
    min_score: 65
    max_rsi: 20
    min_volume_ratio: 1.5
    allowed_override_trades: 5

  triggers:
    hours_without_entry: 4
    consecutive_losses: 3
    daily_loss_threshold: -0.02

# Limites Diários
daily_limits:
  regular:
    max_trades: 20
    max_positions: 5
    max_daily_loss_percent: 2.0

  emergency:
    max_trades: 35
    max_positions: 7
    max_daily_loss_percent: 3.0
    critical_trade_allowance: 5

# Circuit Breakers
circuit_breakers:
  enabled: true
  levels:
    - losses: 2
      action: "reduce_50"
      multiplier: 0.5
    - losses: 3
      action: "pause_5min"
      pause_minutes: 5
    - losses: 5
      action: "stop_trading"
      stop: true
```

### Configuração Otimizada (config_v42_optimized.yaml)

Para melhor performance, use as configurações otimizadas:

```yaml
# Sinais críticos mais seletivos
critical_signals:
  min_score: 68        # ↑ de 65
  max_rsi: 22          # ↑ de 20
  min_volume_ratio: 1.3 # ↓ de 1.5

# Circuit breakers mais agressivos
circuit_breakers:
  levels:
    - losses: 2
      action: "reduce_30"  # Era 50%
      multiplier: 0.7
    - losses: 3
      action: "pause_2min" # Era 5min
      pause_minutes: 2
```

## Ativação do Modo Emergência

### Triggers Automáticos

O sistema ativa automaticamente quando:

1. **Sem entradas por 4 horas**
   - Detecta inatividade prolongada
   - Relaxa filtros em 40%

2. **3 perdas consecutivas**
   - Ativa para recuperação
   - Reduz tamanho de posição

3. **Perda diária de -2%**
   - Proteção contra drawdown
   - Permite trades de recuperação

### Ativação Manual

```python
from emergency_trade_manager import emergency_trade_manager

# Ativar emergência
emergency_trade_manager.activate_emergency(reason="Manual override")

# Verificar status
if emergency_trade_manager.is_emergency_active():
    print("Modo emergência ativo")

# Desativar
emergency_trade_manager.deactivate_emergency()
```

## Monitoramento

### Dashboard de Emergência

Execute o dashboard para monitoramento em tempo real:

```bash
python emergency_dashboard.py
```

**Painéis:**
- Trade Limits: Uso atual vs limite
- Health Metrics: Status de cada dimensão
- Critical Signals: Sinais com bypass
- Alerts: Últimos alertas ativos

### Logs

Os logs incluem informações detalhadas:

```
[INFO] EmergencyTradeManager: Trade CRITICAL aprovado para BTC-USD (bypass 1/5)
[WARNING] SmartAlertSystem: Métrica inconsistente - HEALTHY mas 25% success rate
[INFO] EmergencySignalPrioritizer: 3 sinais CRITICAL, 5 HIGH, 12 MEDIUM
```

## Testes

### Executar Suite de Testes

```bash
# Todos os testes
python -m pytest tests/test_v42_emergency_system.py -v

# Testes específicos
python -m pytest tests/test_v42_emergency_system.py::TestEmergencyTradeManager -v
python -m pytest tests/test_v42_emergency_system.py::TestSmartHealthMetrics -v
```

### Backtesting

```bash
# Simulação de 24 horas
python backtest_v42.py

# Comparação V4.1 vs V4.2
python backtest_v42.py --compare
```

## Troubleshooting

### Problema: Sistema não ativa emergência

**Verificar:**
1. `emergency.enabled: true` no config
2. Triggers configurados corretamente
3. Logs para mensagens de erro

### Problema: Muitos alertas

**Solução:**
```yaml
emergency:
  alerts:
    enable_metric_alerts: false  # Desativar temporariamente
```

### Problema: Success rate sempre 0%

**Causa:** Nenhum trade executado ainda
**Verificar:** `smart_health_metrics.total_trades_executed`

### Problema: Circuit breaker ativando muito

**Solução:** Ajustar thresholds
```yaml
circuit_breakers:
  levels:
    - losses: 3  # Aumentar de 2 para 3
      action: "reduce_50"
```

## Rollout Plan

### Fase 1: Validação (1-2 dias)
- [ ] Executar suite de testes
- [ ] Rodar backtesting comparativo
- [ ] Verificar logs de decisão

### Fase 2: Shadow Mode (3-5 dias)
- [ ] Ativar módulos sem execução real
- [ ] Coletar métricas
- [ ] Ajustar thresholds

### Fase 3: Produção Gradual (1 semana)
- [ ] Começar com 50% dos limites
- [ ] Monitorar alertas
- [ ] Aumentar gradualmente

### Fase 4: Full Production
- [ ] Ativar todos os recursos
- [ ] Monitoramento contínuo
- [ ] Ajustes baseados em performance

## Métricas de Sucesso

| Métrica | Meta | Crítico |
|---------|------|---------|
| Win Rate | > 50% | < 35% |
| Max Drawdown | < 3% | > 5% |
| Emergency Trades/Day | < 10 | > 20 |
| Critical Bypasses/Day | < 3 | > 5 |
| Alertas Críticos/Day | < 5 | > 15 |

## Changelog

### V4.2.0 (2026-01-05)
- Adicionado EmergencyTradeManager
- Adicionado SmartHealthMetrics
- Adicionado EmergencySignalPrioritizer
- Adicionado SmartAlertSystem
- Removido suporte a B3 (Crypto Only)
- Novo dashboard de emergência
- Framework de backtesting
- Configurações otimizadas
