"""
Módulo de aprendizado de máquina para otimização de estratégias.
Usa Random Forest e feature engineering para melhorar decisões de trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle
import os

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn não disponível. Machine Learning desabilitado.")

from system_logger import system_logger


class LearningModule:
    """
    Módulo de machine learning para aprendizado e otimização de estratégias.
    Usa Random Forest para prever sucesso de trades baseado em indicadores.
    """

    def __init__(self, model_path: str = 'models/trading_model.pkl'):
        """
        Inicializa o módulo de aprendizado.

        Args:
            model_path: Caminho para salvar/carregar modelo treinado.
        """
        self.history: List[Dict] = []
        self.model_path = model_path
        self.model: Optional['RandomForestClassifier'] = None
        self.scaler: Optional['StandardScaler'] = None
        self.is_trained = False

        if SKLEARN_AVAILABLE:
            self._ensure_model_dir()
            self._load_model()
        else:
            system_logger.warning("Machine Learning desabilitado - sklearn não disponível")

    def _ensure_model_dir(self):
        """Cria diretório de modelos se não existir."""
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def _load_model(self):
        """Carrega modelo treinado se existir."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
                    self.is_trained = True
                system_logger.info(f"✅ Modelo carregado de {self.model_path}")
            except Exception as e:
                system_logger.error(f"Erro ao carregar modelo: {e}")

    def _save_model(self):
        """Salva modelo treinado."""
        if not SKLEARN_AVAILABLE or self.model is None:
            return

        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler
                }, f)
            system_logger.info(f"✅ Modelo salvo em {self.model_path}")
        except Exception as e:
            system_logger.error(f"Erro ao salvar modelo: {e}")

    def record_trade(self, trade_result: Dict):
        """
        Registra resultado de um trade para aprendizado.

        Args:
            trade_result: Dicionário com dados do trade executado.
        """
        self.history.append(trade_result)
        system_logger.debug(f"Trade registrado: {trade_result.get('symbol', 'N/A')}")

    def evaluate_performance(self) -> Dict:
        """
        Avalia performance geral baseado no histórico.

        Returns:
            Dicionário com métricas de performance.
        """
        if not self.history:
            return {
                'total_trades': 0,
                'total_profit': 0,
                'win_rate': 0,
                'avg_profit': 0
            }

        df = pd.DataFrame(self.history)

        total_trades = len(df)
        total_profit = df['profit'].sum()
        wins = len(df[df['profit'] > 0])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        avg_profit = df['profit'].mean()

        metrics = {
            'total_trades': total_trades,
            'total_profit': total_profit,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'wins': wins,
            'losses': total_trades - wins
        }

        system_logger.info(
            f"📊 Performance: {total_trades} trades | "
            f"Win Rate: {win_rate:.1f}% | "
            f"Lucro Total: R$ {total_profit:.2f}"
        )

        return metrics

    def prepare_features(self, trade_data: Dict) -> Optional[np.ndarray]:
        """
        Prepara features para ML baseado em indicadores.

        Args:
            trade_data: Dados do trade com indicadores.

        Returns:
            Array numpy com features ou None se dados insuficientes.
        """
        if not SKLEARN_AVAILABLE:
            return None

        try:
            # Extrai indicadores
            indicators = trade_data.get('indicators', {})

            if isinstance(indicators, str):
                # Parse se for string JSON
                import json
                indicators = json.loads(indicators)

            features = [
                indicators.get('rsi', 50),
                indicators.get('ema_fast', 0),
                indicators.get('ema_slow', 0),
                indicators.get('macd_diff', 0),
                indicators.get('volume_ratio', 1),
                trade_data.get('price', 0),
                trade_data.get('quantity', 0)
            ]

            return np.array(features).reshape(1, -1)

        except Exception as e:
            system_logger.debug(f"Erro ao preparar features: {e}")
            return None

    def train_model(self, min_samples: int = 50) -> bool:
        """
        Treina modelo de ML com histórico de trades.

        Args:
            min_samples: Número mínimo de amostras para treinar.

        Returns:
            True se treinamento foi bem-sucedido.
        """
        if not SKLEARN_AVAILABLE:
            system_logger.warning("Treinamento cancelado - sklearn não disponível")
            return False

        if len(self.history) < min_samples:
            system_logger.info(
                f"Dados insuficientes para treinar: {len(self.history)}/{min_samples}"
            )
            return False

        system_logger.info(f"🤖 Treinando modelo com {len(self.history)} trades...")

        try:
            # Prepara dataset
            X = []
            y = []

            for trade in self.history:
                features = self.prepare_features(trade)
                if features is not None:
                    X.append(features.flatten())
                    # Label: 1 se lucro > 0, 0 caso contrário
                    y.append(1 if trade.get('profit', 0) > 0 else 0)

            if len(X) < min_samples:
                system_logger.warning("Features insuficientes após processamento")
                return False

            X = np.array(X)
            y = np.array(y)

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Normaliza features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Treina Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            self.model.fit(X_train_scaled, y_train)

            # Avalia modelo
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            system_logger.info(
                f"✅ Modelo treinado | "
                f"Accuracy: {accuracy:.2%} | "
                f"Precision: {precision:.2%} | "
                f"Recall: {recall:.2%} | "
                f"F1: {f1:.2%}"
            )

            self.is_trained = True
            self._save_model()

            return True

        except Exception as e:
            system_logger.error(f"Erro durante treinamento: {e}", exc_info=True)
            return False

    def predict_trade_success(self, trade_data: Dict) -> Optional[Tuple[bool, float]]:
        """
        Prediz probabilidade de sucesso de um trade.

        Args:
            trade_data: Dados do trade proposto.

        Returns:
            Tupla (deve_executar, probabilidade) ou None se modelo não disponível.
        """
        if not SKLEARN_AVAILABLE or not self.is_trained or self.model is None:
            return None

        try:
            features = self.prepare_features(trade_data)

            if features is None:
                return None

            # Normaliza features
            features_scaled = self.scaler.transform(features)

            # Prediz probabilidade
            proba = self.model.predict_proba(features_scaled)[0]
            success_prob = proba[1]  # Probabilidade de classe 1 (sucesso)

            # Decide baseado em threshold
            threshold = 0.55  # 55% de confiança mínima
            should_trade = success_prob >= threshold

            system_logger.debug(
                f"🎯 Predição ML: {success_prob:.2%} | "
                f"Decisão: {'✅ EXECUTAR' if should_trade else '❌ PULAR'}"
            )

            return should_trade, success_prob

        except Exception as e:
            system_logger.debug(f"Erro na predição: {e}")
            return None

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Retorna importância de cada feature no modelo.

        Returns:
            Dicionário feature -> importância ou None.
        """
        if not SKLEARN_AVAILABLE or not self.is_trained or self.model is None:
            return None

        feature_names = ['rsi', 'ema_fast', 'ema_slow', 'macd_diff', 'volume_ratio', 'price', 'quantity']
        importances = self.model.feature_importances_

        return dict(zip(feature_names, importances))

    def adjust_strategy(self) -> Dict:
        """
        Ajusta parâmetros de estratégia baseado no aprendizado.

        Returns:
            Dicionário com recomendações de ajuste.
        """
        performance = self.evaluate_performance()

        recommendations = {
            'retrain_needed': len(self.history) >= 100 and not self.is_trained,
            'performance_acceptable': performance['win_rate'] >= 50,
            'adjustments': []
        }

        # Analisa win rate
        if performance['win_rate'] < 40:
            recommendations['adjustments'].append({
                'parameter': 'risk_threshold',
                'suggestion': 'Aumentar critério de entrada (maior confiança)',
                'reason': f"Win rate muito baixo: {performance['win_rate']:.1f}%"
            })

        # Analisa número de trades
        if performance['total_trades'] < 10:
            recommendations['adjustments'].append({
                'parameter': 'signal_sensitivity',
                'suggestion': 'Reduzir restrições para gerar mais sinais',
                'reason': f"Poucos trades executados: {performance['total_trades']}"
            })

        # Analisa lucro médio
        if performance['avg_profit'] < 0:
            recommendations['adjustments'].append({
                'parameter': 'stop_loss',
                'suggestion': 'Revisar stop-loss e take-profit',
                'reason': f"Lucro médio negativo: R$ {performance['avg_profit']:.2f}"
            })

        system_logger.info(f"🔧 Ajustes recomendados: {len(recommendations['adjustments'])}")

        return recommendations

    def export_history(self, filename: str = 'trade_history.csv'):
        """
        Exporta histórico de trades para CSV.

        Args:
            filename: Nome do arquivo de saída.
        """
        if not self.history:
            system_logger.warning("Nenhum histórico para exportar")
            return

        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)
        system_logger.info(f"✅ Histórico exportado para {filename}")
