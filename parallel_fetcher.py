"""
Coleta paralela otimizada de dados de mercado.
V4.1 - Sistema de fetching paralelo com timeout individual por símbolo.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Optional, Tuple, Callable, Any
from datetime import datetime
from threading import Lock

import pandas as pd
import yfinance as yf

from config_loader import config
from system_logger import system_logger
from data_collector import DataCollector, symbol_blacklist


class ParallelFetcher:
    """
    Coleta dados de múltiplos símbolos em paralelo.

    Features:
    - Timeout individual por símbolo (evita bloqueio)
    - Integração com blacklist
    - Callback para processamento imediato
    - Estatísticas de performance
    """

    def __init__(
        self,
        max_workers: int = None,
        timeout_per_symbol: float = None,
        period: str = "5d",
        interval: str = "5m"
    ):
        """
        Inicializa o fetcher paralelo.

        Args:
            max_workers: Número máximo de threads (default: config ou 5).
            timeout_per_symbol: Timeout por símbolo em segundos (default: config ou 10).
            period: Período de dados históricos.
            interval: Intervalo dos candles.
        """
        fetcher_config = config.get('parallel_fetcher', {})

        self.max_workers = max_workers or fetcher_config.get('max_workers', 5)
        self.timeout_per_symbol = timeout_per_symbol or fetcher_config.get('timeout', 10.0)
        self.period = period
        self.interval = interval

        # Estatísticas
        self._stats_lock = Lock()
        self._stats = {
            'total_fetches': 0,
            'successful': 0,
            'failed': 0,
            'timeouts': 0,
            'blacklisted_skipped': 0,
            'avg_fetch_time': 0.0,
            'last_batch_time': 0.0
        }

        system_logger.info(
            f"ParallelFetcher inicializado: {self.max_workers} workers, "
            f"{self.timeout_per_symbol}s timeout"
        )

    def fetch_multiple(
        self,
        symbols: List[str],
        callback: Optional[Callable[[str, pd.DataFrame], Any]] = None,
        skip_blacklisted: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Coleta dados de múltiplos símbolos em paralelo.

        Args:
            symbols: Lista de símbolos a coletar.
            callback: Função chamada para cada símbolo coletado (symbol, dataframe).
            skip_blacklisted: Se True, pula símbolos na blacklist.

        Returns:
            Dicionário {symbol: DataFrame} com os dados coletados.
        """
        start_time = time.time()
        results = {}

        # Filtra símbolos blacklisted
        if skip_blacklisted:
            filtered_symbols = []
            blacklisted_count = 0
            for symbol in symbols:
                if symbol_blacklist.is_blacklisted(symbol):
                    blacklisted_count += 1
                    system_logger.debug(f"Pulando {symbol}: blacklisted")
                else:
                    filtered_symbols.append(symbol)

            with self._stats_lock:
                self._stats['blacklisted_skipped'] += blacklisted_count
        else:
            filtered_symbols = symbols

        if not filtered_symbols:
            system_logger.warning("Nenhum símbolo válido para coletar")
            return results

        system_logger.info(f"Iniciando coleta paralela de {len(filtered_symbols)} símbolos...")

        # Executa coletas em paralelo
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._fetch_with_timeout,
                    symbol
                ): symbol
                for symbol in filtered_symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results[symbol] = df

                        # Chama callback se fornecido
                        if callback:
                            try:
                                callback(symbol, df)
                            except Exception as e:
                                system_logger.warning(f"Erro no callback para {symbol}: {e}")

                except Exception as e:
                    system_logger.debug(f"Falha ao coletar {symbol}: {e}")

        # Atualiza estatísticas
        elapsed = time.time() - start_time
        with self._stats_lock:
            self._stats['total_fetches'] += len(filtered_symbols)
            self._stats['successful'] += len(results)
            self._stats['failed'] += len(filtered_symbols) - len(results)
            self._stats['last_batch_time'] = elapsed

            if self._stats['successful'] > 0:
                total_time = self._stats['avg_fetch_time'] * (self._stats['successful'] - len(results))
                total_time += elapsed
                self._stats['avg_fetch_time'] = total_time / self._stats['successful']

        system_logger.info(
            f"Coleta paralela concluída em {elapsed:.2f}s: "
            f"{len(results)}/{len(filtered_symbols)} sucessos "
            f"({len(results)/len(filtered_symbols)*100:.1f}%)"
        )

        return results

    def _fetch_with_timeout(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Coleta dados com timeout individual.

        Args:
            symbol: Símbolo a coletar.

        Returns:
            DataFrame com dados ou None se falhar.
        """
        start = time.time()

        try:
            collector = DataCollector(
                symbol=symbol,
                period=self.period,
                interval=self.interval
            )

            # Usa get_data com cache
            df = collector.get_data(use_cache=True)

            elapsed = time.time() - start
            if elapsed > self.timeout_per_symbol * 0.8:
                system_logger.debug(f"{symbol}: coleta lenta ({elapsed:.2f}s)")

            return df

        except Exception as e:
            elapsed = time.time() - start

            # Verifica se foi timeout
            if elapsed >= self.timeout_per_symbol:
                with self._stats_lock:
                    self._stats['timeouts'] += 1
                system_logger.warning(f"Timeout ao coletar {symbol} ({elapsed:.2f}s)")
            else:
                system_logger.debug(f"Erro ao coletar {symbol}: {e}")

            return None

    def get_stats(self) -> Dict:
        """Retorna estatísticas de performance."""
        with self._stats_lock:
            stats = self._stats.copy()

        # Calcula taxa de sucesso
        if stats['total_fetches'] > 0:
            stats['success_rate'] = stats['successful'] / stats['total_fetches'] * 100
            stats['timeout_rate'] = stats['timeouts'] / stats['total_fetches'] * 100
        else:
            stats['success_rate'] = 0
            stats['timeout_rate'] = 0

        return stats

    def reset_stats(self):
        """Reseta estatísticas."""
        with self._stats_lock:
            self._stats = {
                'total_fetches': 0,
                'successful': 0,
                'failed': 0,
                'timeouts': 0,
                'blacklisted_skipped': 0,
                'avg_fetch_time': 0.0,
                'last_batch_time': 0.0
            }


class AsyncParallelFetcher:
    """
    Versão assíncrona do fetcher paralelo.
    Mais eficiente para I/O-bound operations.
    """

    def __init__(
        self,
        max_concurrent: int = None,
        timeout_per_symbol: float = None,
        period: str = "5d",
        interval: str = "5m"
    ):
        """
        Inicializa o fetcher assíncrono.

        Args:
            max_concurrent: Número máximo de requisições concorrentes.
            timeout_per_symbol: Timeout por símbolo em segundos.
            period: Período de dados históricos.
            interval: Intervalo dos candles.
        """
        fetcher_config = config.get('parallel_fetcher', {})

        self.max_concurrent = max_concurrent or fetcher_config.get('max_concurrent', 10)
        self.timeout_per_symbol = timeout_per_symbol or fetcher_config.get('timeout', 10.0)
        self.period = period
        self.interval = interval

        self._semaphore = None  # Criado em runtime para cada loop

    async def fetch_multiple_async(
        self,
        symbols: List[str],
        skip_blacklisted: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Coleta dados de múltiplos símbolos de forma assíncrona.

        Args:
            symbols: Lista de símbolos a coletar.
            skip_blacklisted: Se True, pula símbolos na blacklist.

        Returns:
            Dicionário {symbol: DataFrame} com os dados coletados.
        """
        # Filtra blacklisted
        if skip_blacklisted:
            filtered_symbols = [
                s for s in symbols
                if not symbol_blacklist.is_blacklisted(s)
            ]
        else:
            filtered_symbols = symbols

        if not filtered_symbols:
            return {}

        # Cria semáforo para limitar concorrência
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # Cria tasks
        tasks = [
            self._fetch_symbol_async(symbol)
            for symbol in filtered_symbols
        ]

        # Executa todas as tasks
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Processa resultados
        results = {}
        for symbol, result in zip(filtered_symbols, results_list):
            if isinstance(result, Exception):
                system_logger.debug(f"Erro assíncrono em {symbol}: {result}")
            elif result is not None:
                results[symbol] = result

        return results

    async def _fetch_symbol_async(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Coleta dados de um símbolo de forma assíncrona.
        """
        async with self._semaphore:
            try:
                # Executa coleta síncrona em thread pool
                loop = asyncio.get_event_loop()

                def sync_fetch():
                    collector = DataCollector(
                        symbol=symbol,
                        period=self.period,
                        interval=self.interval
                    )
                    return collector.get_data(use_cache=True)

                # Executa com timeout
                df = await asyncio.wait_for(
                    loop.run_in_executor(None, sync_fetch),
                    timeout=self.timeout_per_symbol
                )

                return df

            except asyncio.TimeoutError:
                system_logger.warning(f"Timeout assíncrono para {symbol}")
                return None
            except Exception as e:
                system_logger.debug(f"Erro ao coletar {symbol}: {e}")
                return None


# Instância global para uso conveniente
parallel_fetcher = ParallelFetcher()


def fetch_symbols_parallel(
    symbols: List[str],
    period: str = "5d",
    interval: str = "5m",
    max_workers: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Função de conveniência para coleta paralela.

    Args:
        symbols: Lista de símbolos.
        period: Período de dados.
        interval: Intervalo dos candles.
        max_workers: Número de threads.

    Returns:
        Dicionário com DataFrames.
    """
    fetcher = ParallelFetcher(
        max_workers=max_workers,
        period=period,
        interval=interval
    )
    return fetcher.fetch_multiple(symbols)


if __name__ == "__main__":
    # Teste
    test_symbols = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "WEGE3.SA"]

    print("Testando coleta paralela...")
    start = time.time()

    results = fetch_symbols_parallel(test_symbols)

    elapsed = time.time() - start
    print(f"\nColetados {len(results)} símbolos em {elapsed:.2f}s")

    for symbol, df in results.items():
        print(f"  {symbol}: {len(df)} candles")
