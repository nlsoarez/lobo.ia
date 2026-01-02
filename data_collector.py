"""
Coletor de dados de mercado com retry logic, cache e blacklist automático.
V4.1 - Adicionado sistema de blacklist para símbolos problemáticos.
"""

import time
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta
from threading import Lock
from config_loader import config
from system_logger import system_logger


class SymbolBlacklist:
    """
    Gerencia blacklist de símbolos com falhas consecutivas.
    Thread-safe para uso em coleta paralela.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern para compartilhar blacklist globalmente."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Inicializa a blacklist."""
        # {symbol: {'fail_count': int, 'last_fail': datetime, 'last_error': str}}
        self._blacklist: Dict[str, Dict] = {}
        self._lock = Lock()

        # Configurações
        self.max_consecutive_fails = config.get('blacklist.max_fails', 3)
        self.blacklist_duration_hours = config.get('blacklist.duration_hours', 24)
        self.reset_fail_count_hours = config.get('blacklist.reset_hours', 1)

        system_logger.info(
            f"Blacklist inicializada: max_fails={self.max_consecutive_fails}, "
            f"duração={self.blacklist_duration_hours}h"
        )

    def register_failure(self, symbol: str, error: str) -> bool:
        """
        Registra uma falha para o símbolo.

        Args:
            symbol: Símbolo que falhou.
            error: Mensagem de erro.

        Returns:
            True se símbolo foi blacklisted após esta falha.
        """
        with self._lock:
            now = datetime.now()

            if symbol in self._blacklist:
                entry = self._blacklist[symbol]

                # Verifica se deve resetar contador (falha antiga)
                hours_since_last_fail = (now - entry['last_fail']).total_seconds() / 3600
                if hours_since_last_fail > self.reset_fail_count_hours:
                    entry['fail_count'] = 1
                else:
                    entry['fail_count'] += 1

                entry['last_fail'] = now
                entry['last_error'] = error
            else:
                self._blacklist[symbol] = {
                    'fail_count': 1,
                    'last_fail': now,
                    'last_error': error,
                    'blacklisted_at': None
                }

            entry = self._blacklist[symbol]

            # Verifica se atingiu limite
            if entry['fail_count'] >= self.max_consecutive_fails:
                if entry.get('blacklisted_at') is None:
                    entry['blacklisted_at'] = now
                    system_logger.warning(
                        f"🚫 Símbolo {symbol} BLACKLISTED após {entry['fail_count']} falhas consecutivas. "
                        f"Erro: {error[:100]}"
                    )
                return True

            return False

    def register_success(self, symbol: str):
        """
        Registra sucesso e reseta contador de falhas.

        Args:
            symbol: Símbolo que teve sucesso.
        """
        with self._lock:
            if symbol in self._blacklist:
                entry = self._blacklist[symbol]
                # Mantém no histórico mas reseta contadores
                entry['fail_count'] = 0
                entry['blacklisted_at'] = None
                system_logger.debug(f"✅ Símbolo {symbol} removido da blacklist (sucesso)")

    def is_blacklisted(self, symbol: str) -> bool:
        """
        Verifica se símbolo está na blacklist.

        Args:
            symbol: Símbolo a verificar.

        Returns:
            True se está blacklisted e dentro do período de duração.
        """
        with self._lock:
            if symbol not in self._blacklist:
                return False

            entry = self._blacklist[symbol]

            if entry.get('blacklisted_at') is None:
                return False

            # Verifica se período de blacklist expirou
            hours_blacklisted = (datetime.now() - entry['blacklisted_at']).total_seconds() / 3600

            if hours_blacklisted >= self.blacklist_duration_hours:
                # Período expirou, remove da blacklist
                entry['fail_count'] = 0
                entry['blacklisted_at'] = None
                system_logger.info(f"⏰ Blacklist expirada para {symbol} após {self.blacklist_duration_hours}h")
                return False

            return True

    def get_blacklisted_symbols(self) -> List[str]:
        """Retorna lista de símbolos atualmente blacklisted."""
        with self._lock:
            return [
                symbol for symbol, entry in self._blacklist.items()
                if entry.get('blacklisted_at') is not None
            ]

    def get_status(self, symbol: str) -> Optional[Dict]:
        """Retorna status detalhado de um símbolo."""
        with self._lock:
            if symbol not in self._blacklist:
                return None
            return self._blacklist[symbol].copy()

    def get_all_status(self) -> Dict[str, Dict]:
        """Retorna status de todos os símbolos rastreados."""
        with self._lock:
            return {s: e.copy() for s, e in self._blacklist.items()}

    def clear(self, symbol: Optional[str] = None):
        """
        Limpa blacklist.

        Args:
            symbol: Se fornecido, limpa apenas este símbolo.
        """
        with self._lock:
            if symbol:
                self._blacklist.pop(symbol, None)
                system_logger.info(f"Blacklist limpa para {symbol}")
            else:
                self._blacklist.clear()
                system_logger.info("Toda blacklist limpa")


# Instância global da blacklist
symbol_blacklist = SymbolBlacklist()


class DataCollector:
    """
    Coleta dados de mercado do Yahoo Finance com tratamento de erros robusto.
    Implementa retry logic, cache e validações.
    """

    # Cache simples em memória (símbolo -> (timestamp, dataframe))
    _cache = {}

    def __init__(
        self,
        symbol: str = "PETR4.SA",
        period: str = "1d",
        interval: str = "5m"
    ):
        """
        Inicializa o coletor de dados.

        Args:
            symbol: Símbolo do ativo (ex: "PETR4.SA" para B3).
            period: Período de dados históricos (1d, 5d, 1mo, etc.).
            interval: Intervalo dos candles (1m, 5m, 15m, 1h, 1d, etc.).
        """
        self.symbol = symbol
        self.period = period
        self.interval = interval

        # Carrega configurações
        self.max_retries = config.get('data.max_retries', 3)
        self.retry_delay = config.get('data.retry_delay', 2)
        self.cache_ttl = config.get('data.cache_ttl', 300)  # 5 minutos

    def get_data(self, use_cache: bool = True, skip_blacklist_check: bool = False) -> pd.DataFrame:
        """
        Coleta dados do Yahoo Finance com retry logic, cache e verificação de blacklist.

        Args:
            use_cache: Se True, usa cache quando disponível.
            skip_blacklist_check: Se True, ignora verificação de blacklist.

        Returns:
            DataFrame com dados OHLCV normalizados.

        Raises:
            ValueError: Se símbolo está na blacklist ou dados insuficientes.
            ConnectionError: Se houver problema de conexão persistente.
        """
        # Verifica blacklist primeiro
        if not skip_blacklist_check and symbol_blacklist.is_blacklisted(self.symbol):
            raise ValueError(
                f"Símbolo {self.symbol} está na blacklist. "
                f"Próxima tentativa em {symbol_blacklist.blacklist_duration_hours}h."
            )

        # Verifica cache
        if use_cache and self._is_cache_valid(self.symbol):
            system_logger.debug(f"Usando dados em cache para {self.symbol}")
            return self._cache[self.symbol][1].copy()

        # Tenta baixar dados com retry logic
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                system_logger.info(
                    f"Coletando dados para {self.symbol} "
                    f"(tentativa {attempt}/{self.max_retries})"
                )

                df = self._download_data()

                # Valida dados
                self._validate_data(df)

                # Normaliza colunas
                df = self._normalize_columns(df)

                # Limpa dados
                df = self._clean_data(df)

                # Armazena no cache
                self._cache[self.symbol] = (datetime.now(), df.copy())

                # Registra sucesso na blacklist (reseta contador de falhas)
                symbol_blacklist.register_success(self.symbol)

                system_logger.info(
                    f"Dados coletados com sucesso: {len(df)} candles para {self.symbol}"
                )

                return df

            except Exception as e:
                last_error = e
                error_str = str(e)
                system_logger.warning(
                    f"Tentativa {attempt} falhou para {self.symbol}: {error_str}"
                )

                if attempt < self.max_retries:
                    sleep_time = self.retry_delay * (2 ** (attempt - 1))  # Backoff exponencial
                    system_logger.info(f"Aguardando {sleep_time}s antes de tentar novamente...")
                    time.sleep(sleep_time)

        # Se chegou aqui, todas as tentativas falharam
        error_msg = (
            f"Falha ao coletar dados para {self.symbol} após {self.max_retries} tentativas. "
            f"Último erro: {last_error}"
        )

        # Registra falha na blacklist
        was_blacklisted = symbol_blacklist.register_failure(self.symbol, str(last_error))

        if was_blacklisted:
            system_logger.error(
                f"❌ {self.symbol} adicionado à blacklist após {self.max_retries} falhas"
            )

        system_logger.error(error_msg)
        raise ConnectionError(error_msg)

    @staticmethod
    def is_symbol_blacklisted(symbol: str) -> bool:
        """
        Verifica se um símbolo está na blacklist.

        Args:
            symbol: Símbolo a verificar.

        Returns:
            True se símbolo está blacklisted.
        """
        return symbol_blacklist.is_blacklisted(symbol)

    @staticmethod
    def get_blacklisted_symbols() -> List[str]:
        """Retorna lista de símbolos atualmente na blacklist."""
        return symbol_blacklist.get_blacklisted_symbols()

    @staticmethod
    def clear_blacklist(symbol: Optional[str] = None):
        """
        Limpa blacklist.

        Args:
            symbol: Se fornecido, limpa apenas este símbolo.
        """
        symbol_blacklist.clear(symbol)

    def _download_data(self) -> pd.DataFrame:
        """
        Faz o download dos dados do Yahoo Finance.

        Returns:
            DataFrame bruto do yfinance.

        Raises:
            ValueError: Se não retornar dados.
        """
        # Nota: yfinance >= 0.2.28 removeu o parametro show_errors
        df = yf.download(
            self.symbol,
            period=self.period,
            interval=self.interval,
            progress=False
        )

        if df.empty:
            raise ValueError(
                f"Nenhum dado retornado para {self.symbol}. "
                "Verifique se o símbolo está correto e se o mercado está aberto."
            )

        return df

    def _validate_data(self, df: pd.DataFrame):
        """
        Valida os dados recebidos.

        Args:
            df: DataFrame a validar.

        Raises:
            ValueError: Se os dados forem inválidos.
        """
        if df.empty:
            raise ValueError("DataFrame vazio")

        # Verifica número mínimo de candles (necessário para indicadores)
        min_candles = 50
        if len(df) < min_candles:
            raise ValueError(
                f"Dados insuficientes: {len(df)} candles (mínimo: {min_candles})"
            )

        # Verifica colunas essenciais
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # yfinance pode retornar MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            actual_columns = df.columns.get_level_values(0)
        else:
            actual_columns = df.columns

        missing = [col for col in required_columns if col not in actual_columns]
        if missing:
            raise ValueError(f"Colunas faltando: {missing}")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza nomes de colunas para lowercase.

        Args:
            df: DataFrame original.

        Returns:
            DataFrame com colunas normalizadas.
        """
        # Se for MultiIndex (múltiplos símbolos), pega apenas o primeiro nível
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Renomeia para lowercase
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close"
        }

        df = df.rename(columns=column_mapping)

        # Remove colunas não essenciais
        essential_columns = ['open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in essential_columns if col in df.columns]]

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa dados removendo nulos e valores inválidos.

        Args:
            df: DataFrame a limpar.

        Returns:
            DataFrame limpo.
        """
        # Remove linhas com valores nulos
        before = len(df)
        df = df.dropna()
        after = len(df)

        if before - after > 0:
            system_logger.debug(f"Removidas {before - after} linhas com valores nulos")

        # Remove linhas com valores negativos ou zero em preço/volume
        df = df[
            (df['open'] > 0) &
            (df['high'] > 0) &
            (df['low'] > 0) &
            (df['close'] > 0) &
            (df['volume'] >= 0)
        ]

        # Remove linhas onde high < low (dados inválidos)
        df = df[df['high'] >= df['low']]

        # Reseta o índice
        df = df.reset_index(drop=False)
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'datetime'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'datetime'})

        return df

    def _is_cache_valid(self, symbol: str) -> bool:
        """
        Verifica se o cache é válido para o símbolo.

        Args:
            symbol: Símbolo a verificar.

        Returns:
            True se cache é válido, False caso contrário.
        """
        if symbol not in self._cache:
            return False

        timestamp, _ = self._cache[symbol]
        age = (datetime.now() - timestamp).total_seconds()

        return age < self.cache_ttl

    @classmethod
    def clear_cache(cls, symbol: Optional[str] = None):
        """
        Limpa cache de dados.

        Args:
            symbol: Se fornecido, limpa apenas este símbolo. Caso contrário, limpa tudo.
        """
        if symbol:
            cls._cache.pop(symbol, None)
            system_logger.debug(f"Cache limpo para {symbol}")
        else:
            cls._cache.clear()
            system_logger.debug("Todo cache limpo")
