"""
Sistema de cache inteligente multi-nível para dados de mercado.
V4.1 - Cache hierárquico com memória, disco e prefetch.
"""

import os
import json
import pickle
import hashlib
import time
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from threading import Lock, Thread
from pathlib import Path
from collections import OrderedDict

import pandas as pd

from config_loader import config
from system_logger import system_logger


class LRUCache:
    """
    Cache LRU (Least Recently Used) em memória.
    Thread-safe com limite de tamanho.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """
        Inicializa cache LRU.

        Args:
            max_size: Número máximo de itens.
            ttl_seconds: Tempo de vida dos itens em segundos.
        """
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, datetime] = {}
        self._lock = Lock()
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)

        # Estatísticas
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Obtém item do cache.

        Args:
            key: Chave do item.

        Returns:
            Valor ou None se não encontrado/expirado.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Verifica TTL
            if datetime.now() - self._timestamps[key] > self.ttl:
                self._remove_item(key)
                self._misses += 1
                return None

            # Move para o final (mais recente)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def set(self, key: str, value: Any):
        """
        Armazena item no cache.

        Args:
            key: Chave do item.
            value: Valor a armazenar.
        """
        with self._lock:
            # Remove se já existe
            if key in self._cache:
                self._remove_item(key)

            # Verifica limite de tamanho
            while len(self._cache) >= self.max_size:
                # Remove o mais antigo (primeiro)
                oldest_key = next(iter(self._cache))
                self._remove_item(oldest_key)

            # Adiciona novo item
            self._cache[key] = value
            self._timestamps[key] = datetime.now()

    def _remove_item(self, key: str):
        """Remove item do cache (sem lock)."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)

    def clear(self):
        """Limpa todo o cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def get_stats(self) -> Dict:
        """Retorna estatísticas do cache."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate
            }


class DiskCache:
    """
    Cache em disco para dados históricos.
    Usa pickle para serialização eficiente de DataFrames.
    """

    def __init__(
        self,
        cache_dir: str = None,
        ttl_hours: int = 24,
        max_size_mb: int = 100
    ):
        """
        Inicializa cache em disco.

        Args:
            cache_dir: Diretório para armazenar cache.
            ttl_hours: Tempo de vida em horas.
            max_size_mb: Tamanho máximo do cache em MB.
        """
        self.cache_dir = Path(cache_dir or config.get('cache.disk_dir', '/tmp/lobo_cache'))
        self.ttl = timedelta(hours=ttl_hours)
        self.max_size_bytes = max_size_mb * 1024 * 1024

        # Cria diretório se não existir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()

        # Limpa cache expirado na inicialização
        self._cleanup_expired()

    def _get_cache_path(self, key: str) -> Path:
        """Gera path do arquivo de cache."""
        # Usa hash para evitar problemas com caracteres especiais
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Obtém DataFrame do cache em disco.

        Args:
            key: Chave do item (geralmente símbolo + período).

        Returns:
            DataFrame ou None se não encontrado/expirado.
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        with self._lock:
            try:
                # Verifica TTL
                mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                if datetime.now() - mtime > self.ttl:
                    cache_path.unlink()
                    return None

                # Carrega DataFrame
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

            except Exception as e:
                system_logger.debug(f"Erro ao ler cache de disco: {e}")
                return None

    def set(self, key: str, df: pd.DataFrame):
        """
        Armazena DataFrame no cache em disco.

        Args:
            key: Chave do item.
            df: DataFrame a armazenar.
        """
        cache_path = self._get_cache_path(key)

        with self._lock:
            try:
                # Verifica tamanho total do cache
                self._ensure_space()

                # Salva DataFrame
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f)

            except Exception as e:
                system_logger.debug(f"Erro ao salvar cache em disco: {e}")

    def _ensure_space(self):
        """Garante que há espaço no cache."""
        try:
            total_size = sum(
                f.stat().st_size
                for f in self.cache_dir.glob('*.pkl')
            )

            if total_size > self.max_size_bytes:
                # Remove arquivos mais antigos até liberar 20% do espaço
                files = sorted(
                    self.cache_dir.glob('*.pkl'),
                    key=lambda f: f.stat().st_mtime
                )

                target_size = self.max_size_bytes * 0.8
                current_size = total_size

                for f in files:
                    if current_size <= target_size:
                        break
                    current_size -= f.stat().st_size
                    f.unlink()

        except Exception as e:
            system_logger.debug(f"Erro ao limpar cache: {e}")

    def _cleanup_expired(self):
        """Remove arquivos expirados."""
        try:
            now = datetime.now()
            for cache_file in self.cache_dir.glob('*.pkl'):
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if now - mtime > self.ttl:
                    cache_file.unlink()
        except Exception as e:
            system_logger.debug(f"Erro ao limpar cache expirado: {e}")

    def clear(self):
        """Limpa todo o cache em disco."""
        with self._lock:
            for cache_file in self.cache_dir.glob('*.pkl'):
                try:
                    cache_file.unlink()
                except Exception:
                    pass

    def get_stats(self) -> Dict:
        """Retorna estatísticas do cache em disco."""
        try:
            files = list(self.cache_dir.glob('*.pkl'))
            total_size = sum(f.stat().st_size for f in files)

            return {
                'files': len(files),
                'size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'usage_pct': (total_size / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0
            }
        except Exception:
            return {'files': 0, 'size_mb': 0, 'max_size_mb': 0, 'usage_pct': 0}


class SmartCache:
    """
    Cache hierárquico inteligente com múltiplos níveis.

    Níveis:
    1. Memória (rápido, curta duração)
    2. Disco (médio, longa duração)
    3. Fonte de dados (fallback)
    """

    def __init__(
        self,
        memory_size: int = None,
        memory_ttl: int = None,
        disk_ttl_hours: int = None,
        enable_disk: bool = True,
        enable_prefetch: bool = True
    ):
        """
        Inicializa cache inteligente.

        Args:
            memory_size: Tamanho máximo do cache em memória.
            memory_ttl: TTL do cache em memória (segundos).
            disk_ttl_hours: TTL do cache em disco (horas).
            enable_disk: Habilita cache em disco.
            enable_prefetch: Habilita prefetch de símbolos frequentes.
        """
        cache_config = config.get('cache', {})

        # Cache em memória
        self.memory_cache = LRUCache(
            max_size=memory_size or cache_config.get('memory_size', 100),
            ttl_seconds=memory_ttl or cache_config.get('memory_ttl', 300)
        )

        # Cache em disco (opcional)
        self.enable_disk = enable_disk
        if enable_disk:
            self.disk_cache = DiskCache(
                ttl_hours=disk_ttl_hours or cache_config.get('disk_ttl_hours', 24)
            )
        else:
            self.disk_cache = None

        # Prefetch
        self.enable_prefetch = enable_prefetch
        self._prefetch_symbols: List[str] = []
        self._prefetch_thread: Optional[Thread] = None

        # Estatísticas
        self._stats_lock = Lock()
        self._stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'source_fetches': 0
        }

        system_logger.info(
            f"SmartCache inicializado: memoria={self.memory_cache.max_size}, "
            f"disco={'habilitado' if enable_disk else 'desabilitado'}"
        )

    def get_with_fallback(
        self,
        key: str,
        fetch_func: Callable[[], pd.DataFrame],
        use_disk: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Obtém dados com fallback hierárquico.

        Ordem de busca:
        1. Cache em memória
        2. Cache em disco
        3. Fonte de dados (fetch_func)

        Args:
            key: Chave do cache.
            fetch_func: Função para buscar dados se cache miss.
            use_disk: Se True, usa cache em disco como fallback.

        Returns:
            DataFrame ou None se todas as fontes falharem.
        """
        # 1. Tenta memória
        df = self.memory_cache.get(key)
        if df is not None:
            with self._stats_lock:
                self._stats['memory_hits'] += 1
            return df

        # 2. Tenta disco
        if use_disk and self.disk_cache is not None:
            df = self.disk_cache.get(key)
            if df is not None:
                # Promove para memória
                self.memory_cache.set(key, df)
                with self._stats_lock:
                    self._stats['disk_hits'] += 1
                return df

        # 3. Busca da fonte
        try:
            df = fetch_func()
            if df is not None and not df.empty:
                # Armazena em todos os níveis
                self.memory_cache.set(key, df)
                if self.disk_cache is not None:
                    self.disk_cache.set(key, df)

                with self._stats_lock:
                    self._stats['source_fetches'] += 1

                return df
        except Exception as e:
            system_logger.debug(f"Erro ao buscar dados para {key}: {e}")

        return None

    def set(self, key: str, df: pd.DataFrame, persist_to_disk: bool = True):
        """
        Armazena dados em todos os níveis de cache.

        Args:
            key: Chave do cache.
            df: DataFrame a armazenar.
            persist_to_disk: Se True, também salva em disco.
        """
        self.memory_cache.set(key, df)

        if persist_to_disk and self.disk_cache is not None:
            self.disk_cache.set(key, df)

    def invalidate(self, key: str):
        """Invalida cache em todos os níveis."""
        self.memory_cache._remove_item(key)
        if self.disk_cache is not None:
            cache_path = self.disk_cache._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()

    def clear(self):
        """Limpa todos os níveis de cache."""
        self.memory_cache.clear()
        if self.disk_cache is not None:
            self.disk_cache.clear()

    def prefetch(self, symbols: List[str], fetch_func: Callable[[str], pd.DataFrame]):
        """
        Pré-carrega símbolos frequentes em background.

        Args:
            symbols: Lista de símbolos para prefetch.
            fetch_func: Função para buscar dados (recebe símbolo).
        """
        if not self.enable_prefetch:
            return

        self._prefetch_symbols = symbols

        def _prefetch_worker():
            for symbol in symbols:
                key = f"data_{symbol}"
                if self.memory_cache.get(key) is None:
                    try:
                        df = fetch_func(symbol)
                        if df is not None:
                            self.set(key, df)
                    except Exception:
                        pass

        self._prefetch_thread = Thread(target=_prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    def get_stats(self) -> Dict:
        """Retorna estatísticas combinadas de todos os níveis."""
        with self._stats_lock:
            combined_stats = self._stats.copy()

        combined_stats['memory'] = self.memory_cache.get_stats()

        if self.disk_cache is not None:
            combined_stats['disk'] = self.disk_cache.get_stats()

        # Calcula hit rate total
        total_requests = (
            combined_stats['memory_hits'] +
            combined_stats['disk_hits'] +
            combined_stats['source_fetches']
        )

        if total_requests > 0:
            combined_stats['total_hit_rate'] = (
                (combined_stats['memory_hits'] + combined_stats['disk_hits'])
                / total_requests * 100
            )
        else:
            combined_stats['total_hit_rate'] = 0

        return combined_stats


# Instância global
smart_cache = SmartCache()


def get_cached_data(
    symbol: str,
    period: str = "5d",
    interval: str = "5m",
    fetch_func: Callable = None
) -> Optional[pd.DataFrame]:
    """
    Função de conveniência para obter dados com cache.

    Args:
        symbol: Símbolo do ativo.
        period: Período de dados.
        interval: Intervalo dos candles.
        fetch_func: Função alternativa para fetch.

    Returns:
        DataFrame com dados ou None.
    """
    from data_collector import DataCollector

    key = f"data_{symbol}_{period}_{interval}"

    if fetch_func is None:
        def fetch_func():
            collector = DataCollector(symbol=symbol, period=period, interval=interval)
            return collector.get_data(use_cache=False)

    return smart_cache.get_with_fallback(key, fetch_func)


if __name__ == "__main__":
    # Teste do cache
    print("Testando SmartCache...")

    cache = SmartCache()

    # Simula dados
    import numpy as np
    test_df = pd.DataFrame({
        'close': np.random.randn(100),
        'volume': np.random.randint(1000, 10000, 100)
    })

    # Teste set/get
    cache.set("test_key", test_df)
    retrieved = cache.memory_cache.get("test_key")

    print(f"Cache set/get: {'OK' if retrieved is not None else 'FALHOU'}")

    # Estatísticas
    stats = cache.get_stats()
    print(f"Estatísticas: {stats}")
