"""
V4.0 Phase 4: Reinforcement Learning Agent
Agente RL para otimização de decisões de trading.
Implementação com Q-Learning tabular e aproximação de função simples.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import IntEnum
import random
import json

from system_logger import system_logger


class TradingAction(IntEnum):
    """Ações possíveis do agente."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


@dataclass
class Experience:
    """Experiência para replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class AgentDecision:
    """Decisão do agente RL."""
    action: TradingAction
    confidence: float
    q_values: Dict[str, float]
    state_features: Dict[str, float]
    exploration: bool


class SimpleNeuralNetwork:
    """
    Rede neural simples (2 camadas) implementada do zero.
    Sem dependências externas de ML.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialização Xavier
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, x: np.ndarray, y_true: np.ndarray,
                learning_rate: float = 0.001) -> float:
        """Backward pass com gradiente descendente."""
        # Forward
        y_pred = self.forward(x)

        # Loss (MSE)
        loss = np.mean((y_pred - y_true) ** 2)

        # Gradientes
        dz2 = y_pred - y_true
        dW2 = np.dot(self.a1.T, dz2) / len(x)
        db2 = np.mean(dz2, axis=0)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self._relu_derivative(self.z1)
        dW1 = np.dot(x.T, dz1) / len(x)
        db1 = np.mean(dz1, axis=0)

        # Atualiza pesos
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return loss

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def copy_from(self, other: 'SimpleNeuralNetwork'):
        """Copia pesos de outra rede."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

    def save(self, filepath: str):
        """Salva pesos em arquivo."""
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, filepath: str):
        """Carrega pesos de arquivo."""
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']


class TradingRLAgent:
    """
    V4.0 Phase 4: Agente de Reinforcement Learning.
    Usa Deep Q-Learning com Experience Replay.
    """

    def __init__(self, state_size: int = 20, action_size: int = 4):
        """Inicializa o agente RL."""
        self.state_size = state_size
        self.action_size = action_size

        # Replay buffer
        self.memory = deque(maxlen=5000)

        # Hiperparâmetros
        self.gamma = 0.95  # Fator de desconto
        self.epsilon = 1.0  # Taxa de exploração inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.target_update_freq = 100

        # Redes neurais (Q-network e target network)
        hidden_size = 64
        self.q_network = SimpleNeuralNetwork(state_size, hidden_size, action_size)
        self.target_network = SimpleNeuralNetwork(state_size, hidden_size, action_size)
        self.target_network.copy_from(self.q_network)

        # Contadores
        self.training_step = 0
        self.total_reward = 0
        self.episode_rewards = []

        # Histórico de decisões
        self.decision_history = deque(maxlen=1000)

        # Estatísticas
        self.stats = {
            'total_decisions': 0,
            'exploration_decisions': 0,
            'exploitation_decisions': 0,
            'action_distribution': {a.name: 0 for a in TradingAction}
        }

        system_logger.info(f"TradingRLAgent V4.0 inicializado (state={state_size}, actions={action_size})")

    def prepare_state(self, market_data: Dict, position_data: Optional[Dict] = None) -> np.ndarray:
        """
        Prepara vetor de estado para o agente.

        Args:
            market_data: Dados de mercado (preços, indicadores)
            position_data: Dados da posição atual (se houver)
        """
        features = []

        # Features de mercado
        features.append(market_data.get('rsi', 50) / 100)  # Normalizado 0-1
        features.append((market_data.get('macd', 0) + 0.05) / 0.1)  # Normalizado ~0-1
        features.append(market_data.get('volume_ratio', 1.0) / 3.0)  # Normalizado
        features.append(market_data.get('price_change_1h', 0) / 0.05 + 0.5)  # Centralizado
        features.append(market_data.get('price_change_24h', 0) / 0.1 + 0.5)

        # EMAs relativas
        features.append(market_data.get('ema_ratio_12_26', 1.0))
        features.append(market_data.get('ema_ratio_50_200', 1.0))

        # Volatilidade
        features.append(market_data.get('volatility', 0.02) / 0.05)
        features.append(market_data.get('atr_pct', 0.02) / 0.05)

        # Bollinger Bands position
        features.append(market_data.get('bb_position', 0.5))

        # Momentum
        features.append(market_data.get('momentum_score', 50) / 100)

        # Features de posição
        if position_data:
            features.append(1.0)  # Tem posição
            features.append(position_data.get('pnl_pct', 0) / 0.05 + 0.5)  # PnL normalizado
            features.append(min(position_data.get('hold_time_hours', 0) / 4, 1.0))  # Tempo hold
            features.append(position_data.get('distance_to_tp', 0.5))
            features.append(position_data.get('distance_to_sl', 0.5))
        else:
            features.extend([0.0, 0.5, 0.0, 0.5, 0.5])  # Sem posição

        # Regime de mercado (one-hot simplificado)
        regime = market_data.get('regime', 'sideways')
        features.append(1.0 if regime == 'bull' else 0.0)
        features.append(1.0 if regime == 'bear' else 0.0)
        features.append(1.0 if regime == 'high_vol' else 0.0)

        # Padding se necessário
        while len(features) < self.state_size:
            features.append(0.0)

        # Trunca se muito grande
        features = features[:self.state_size]

        return np.array(features, dtype=np.float32)

    def act(self, state: np.ndarray, training: bool = True) -> AgentDecision:
        """
        Escolhe ação baseado no estado.

        Args:
            state: Vetor de estado
            training: Se True, usa epsilon-greedy
        """
        exploration = False

        if training and np.random.random() < self.epsilon:
            # Exploração: ação aleatória
            action = random.randrange(self.action_size)
            exploration = True
            self.stats['exploration_decisions'] += 1
        else:
            # Exploração: usa Q-network
            state_batch = state.reshape(1, -1)
            q_values = self.q_network.forward(state_batch)[0]
            action = int(np.argmax(q_values))
            self.stats['exploitation_decisions'] += 1

        # Calcula Q-values para logging
        state_batch = state.reshape(1, -1)
        q_values = self.q_network.forward(state_batch)[0]
        q_dict = {TradingAction(i).name: float(q_values[i]) for i in range(self.action_size)}

        # Confiança baseada na diferença entre Q-values
        q_sorted = sorted(q_values, reverse=True)
        confidence = (q_sorted[0] - q_sorted[1]) / (abs(q_sorted[0]) + 0.001) if len(q_sorted) > 1 else 0.5
        confidence = max(0, min(1, confidence))

        # Atualiza estatísticas
        self.stats['total_decisions'] += 1
        self.stats['action_distribution'][TradingAction(action).name] += 1

        decision = AgentDecision(
            action=TradingAction(action),
            confidence=confidence,
            q_values=q_dict,
            state_features={'state_norm': float(np.linalg.norm(state))},
            exploration=exploration
        )

        self.decision_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': decision.action.name,
            'confidence': decision.confidence,
            'exploration': exploration,
            'epsilon': self.epsilon
        })

        return decision

    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool):
        """Armazena experiência no replay buffer."""
        self.memory.append(Experience(state, action, reward, next_state, done))
        self.total_reward += reward

    def replay(self) -> Optional[float]:
        """
        Treina o modelo com experiências do buffer.

        Returns:
            Loss médio do batch, ou None se buffer insuficiente
        """
        if len(self.memory) < self.batch_size:
            return None

        # Amostra batch aleatório
        batch = random.sample(list(self.memory), self.batch_size)

        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])

        # Calcula Q-targets
        current_q = self.q_network.forward(states)
        next_q = self.target_network.forward(next_states)

        targets = current_q.copy()

        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        # Treina Q-network
        loss = self.q_network.backward(states, targets, self.learning_rate)

        # Atualiza target network periodicamente
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def calculate_reward(self, action: TradingAction, trade_result: Dict) -> float:
        """
        Calcula recompensa baseada na ação e resultado.

        Args:
            action: Ação tomada
            trade_result: Resultado do trade (profit_pct, hit_tp, hit_sl, etc.)
        """
        profit_pct = trade_result.get('profit_pct', 0)
        hit_tp = trade_result.get('hit_tp', False)
        hit_sl = trade_result.get('hit_sl', False)
        hold_time = trade_result.get('hold_time_minutes', 30)

        reward = 0.0

        if action == TradingAction.BUY:
            if profit_pct > 0:
                reward = profit_pct * 100  # Lucro é bom
                if hit_tp:
                    reward += 5  # Bônus por atingir TP
            else:
                reward = profit_pct * 150  # Perda penalizada mais

            # Penalidade por segurar muito tempo
            if hold_time > 60:
                reward -= 1

        elif action == TradingAction.SELL:
            # Recompensa por evitar perda
            if trade_result.get('avoided_loss', False):
                reward = 3
            else:
                reward = -profit_pct * 50  # Penalidade se vendeu errado

        elif action == TradingAction.CLOSE:
            # Fechar posição
            reward = profit_pct * 80
            if hit_sl:
                reward -= 2  # Penalidade por stop

        elif action == TradingAction.HOLD:
            # Hold - pequena recompensa por estabilidade
            reward = 0.1 if profit_pct >= 0 else -0.2

        return float(reward)

    def train_on_historical_data(self, historical_trades: List[Dict],
                                 epochs: int = 10) -> Dict[str, Any]:
        """
        Treina agente com dados históricos de trades.

        Args:
            historical_trades: Lista de trades históricos
            epochs: Número de épocas de treinamento
        """
        system_logger.info(f"\n🤖 TREINANDO AGENTE RL com {len(historical_trades)} trades")

        training_stats = {
            'epochs': epochs,
            'total_samples': 0,
            'avg_loss': 0,
            'final_epsilon': self.epsilon,
            'losses': []
        }

        for epoch in range(epochs):
            epoch_loss = 0
            samples = 0

            for i, trade in enumerate(historical_trades[:-1]):
                # Prepara estado atual
                state = self._trade_to_state(trade)

                # Determina ação que foi tomada
                action = self._infer_action(trade)

                # Próximo estado
                next_trade = historical_trades[i + 1]
                next_state = self._trade_to_state(next_trade)

                # Calcula reward
                reward = self.calculate_reward(action, trade)

                # Armazena experiência
                done = (i == len(historical_trades) - 2)
                self.remember(state, action.value, reward, next_state, done)

                samples += 1

                # Replay periódico
                if samples % 10 == 0:
                    loss = self.replay()
                    if loss is not None:
                        epoch_loss += loss

            avg_epoch_loss = epoch_loss / max(1, samples // 10)
            training_stats['losses'].append(avg_epoch_loss)

            system_logger.info(f"   Época {epoch+1}/{epochs}: Loss={avg_epoch_loss:.4f}, "
                             f"Epsilon={self.epsilon:.3f}")

        training_stats['total_samples'] = len(historical_trades) * epochs
        training_stats['avg_loss'] = np.mean(training_stats['losses'])
        training_stats['final_epsilon'] = self.epsilon

        return training_stats

    def _trade_to_state(self, trade: Dict) -> np.ndarray:
        """Converte dados de trade para estado."""
        market_data = {
            'rsi': trade.get('entry_rsi', 50),
            'macd': trade.get('macd', 0),
            'volume_ratio': trade.get('volume_ratio', 1.0),
            'price_change_1h': trade.get('price_change_1h', 0),
            'price_change_24h': trade.get('price_change_24h', 0),
            'volatility': trade.get('volatility', 0.02),
            'momentum_score': trade.get('momentum_score', 50),
            'regime': trade.get('regime', 'sideways')
        }

        position_data = None
        if trade.get('has_position', False):
            position_data = {
                'pnl_pct': trade.get('pnl_pct', 0),
                'hold_time_hours': trade.get('hold_time_minutes', 30) / 60
            }

        return self.prepare_state(market_data, position_data)

    def _infer_action(self, trade: Dict) -> TradingAction:
        """Infere ação tomada baseado nos dados do trade."""
        action_str = trade.get('action', 'HOLD').upper()

        if 'BUY' in action_str:
            return TradingAction.BUY
        elif 'SELL' in action_str:
            return TradingAction.SELL
        elif 'CLOSE' in action_str:
            return TradingAction.CLOSE
        else:
            return TradingAction.HOLD

    def get_action_recommendation(self, market_data: Dict,
                                  position_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Retorna recomendação de ação para uso no sistema de trading.
        """
        state = self.prepare_state(market_data, position_data)
        decision = self.act(state, training=False)

        # Traduz para recomendação
        recommendation = {
            'action': decision.action.name,
            'confidence': decision.confidence,
            'should_execute': decision.confidence > 0.3,  # Threshold de confiança
            'q_values': decision.q_values,
            'exploration': decision.exploration
        }

        # Adiciona contexto
        if decision.action == TradingAction.BUY:
            recommendation['reason'] = "RL Agent recomenda entrada baseado em padrão aprendido"
        elif decision.action == TradingAction.SELL:
            recommendation['reason'] = "RL Agent recomenda saída para evitar perda"
        elif decision.action == TradingAction.CLOSE:
            recommendation['reason'] = "RL Agent recomenda fechar posição"
        else:
            recommendation['reason'] = "RL Agent recomenda manter posição atual"

        return recommendation

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do agente."""
        exploration_rate = (self.stats['exploration_decisions'] /
                          max(1, self.stats['total_decisions']))

        return {
            'total_decisions': self.stats['total_decisions'],
            'exploration_rate': exploration_rate,
            'current_epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_steps': self.training_step,
            'total_reward': self.total_reward,
            'action_distribution': self.stats['action_distribution']
        }

    def save_model(self, filepath: str):
        """Salva modelo."""
        self.q_network.save(filepath)
        system_logger.info(f"RL Agent model salvo em {filepath}")

    def load_model(self, filepath: str):
        """Carrega modelo."""
        try:
            self.q_network.load(filepath)
            self.target_network.copy_from(self.q_network)
            system_logger.info(f"RL Agent model carregado de {filepath}")
        except Exception as e:
            system_logger.warning(f"Erro carregando modelo RL: {e}")

    def log_agent_status(self):
        """Loga status do agente."""
        stats = self.get_stats()

        system_logger.info(f"\n🤖 RL AGENT STATUS:")
        system_logger.info(f"   Decisões totais: {stats['total_decisions']}")
        system_logger.info(f"   Taxa exploração: {stats['exploration_rate']*100:.1f}%")
        system_logger.info(f"   Epsilon atual: {stats['current_epsilon']:.3f}")
        system_logger.info(f"   Memória: {stats['memory_size']} experiências")
        system_logger.info(f"   Reward total: {stats['total_reward']:.2f}")

