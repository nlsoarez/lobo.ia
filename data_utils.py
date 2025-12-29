"""
V4.1: Data Utilities
Funções utilitárias para normalização e validação de dados.
Resolve o erro 'close' column em todos os módulos.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from system_logger import system_logger


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nomes de colunas para lowercase.
    Resolve problemas de case-sensitivity com yfinance.

    Args:
        df: DataFrame original

    Returns:
        DataFrame com colunas em lowercase
    """
    if df is None or df.empty:
        return df

    column_mapping = {}
    for col in df.columns:
        if isinstance(col, str):
            column_mapping[col] = col.lower()

    return df.rename(columns=column_mapping)


def get_close_column(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Retorna a coluna de preço de fechamento (case-insensitive).

    Args:
        df: DataFrame com dados OHLCV

    Returns:
        Series com preços de fechamento ou None
    """
    if df is None or df.empty:
        return None

    # Tenta diferentes variações
    for col in ['Close', 'close', 'CLOSE', 'last', 'Last', 'LAST', 'Adj Close', 'adj close']:
        if col in df.columns:
            return df[col]

    # Fallback: busca case-insensitive
    for col in df.columns:
        if isinstance(col, str) and col.lower() in ['close', 'last']:
            return df[col]

    return None


def get_ohlcv_columns(df: pd.DataFrame) -> Dict[str, Optional[pd.Series]]:
    """
    Extrai todas as colunas OHLCV de forma segura.

    Args:
        df: DataFrame com dados OHLCV

    Returns:
        Dict com Series para cada coluna OHLCV
    """
    if df is None or df.empty:
        return {'open': None, 'high': None, 'low': None, 'close': None, 'volume': None}

    def find_column(names):
        for name in names:
            if name in df.columns:
                return df[name]
        # Fallback case-insensitive
        for col in df.columns:
            if isinstance(col, str) and col.lower() == names[0].lower():
                return df[col]
        return None

    return {
        'open': find_column(['open', 'Open', 'OPEN']),
        'high': find_column(['high', 'High', 'HIGH']),
        'low': find_column(['low', 'Low', 'LOW']),
        'close': find_column(['close', 'Close', 'CLOSE', 'Adj Close']),
        'volume': find_column(['volume', 'Volume', 'VOLUME'])
    }


def validate_ohlcv_data(df: pd.DataFrame, min_rows: int = 10) -> Dict[str, Any]:
    """
    Valida dados OHLCV e retorna status detalhado.

    Args:
        df: DataFrame para validar
        min_rows: Mínimo de linhas requeridas

    Returns:
        Dict com status da validação
    """
    result = {
        'valid': False,
        'rows': 0,
        'missing_columns': [],
        'nan_percentage': 0,
        'errors': []
    }

    if df is None:
        result['errors'].append('DataFrame é None')
        return result

    if df.empty:
        result['errors'].append('DataFrame está vazio')
        return result

    result['rows'] = len(df)

    if len(df) < min_rows:
        result['errors'].append(f'Menos de {min_rows} linhas ({len(df)})')
        return result

    # Verifica colunas OHLCV
    ohlcv = get_ohlcv_columns(df)

    for col_name, series in ohlcv.items():
        if series is None:
            result['missing_columns'].append(col_name)

    if result['missing_columns']:
        result['errors'].append(f"Colunas faltando: {result['missing_columns']}")
        return result

    # Verifica NaN
    close_col = ohlcv['close']
    if close_col is not None:
        nan_count = close_col.isna().sum()
        result['nan_percentage'] = (nan_count / len(close_col)) * 100

        if result['nan_percentage'] > 10:
            result['errors'].append(f"Muitos NaN: {result['nan_percentage']:.1f}%")
            return result

    result['valid'] = True
    return result


def safe_get_ohlcv_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extrai arrays OHLCV de forma segura para cálculos.

    Args:
        df: DataFrame com dados OHLCV

    Returns:
        Dict com arrays numpy para cada coluna
    """
    # Primeiro normaliza as colunas
    df_norm = normalize_dataframe_columns(df)

    result = {
        'open': np.array([]),
        'high': np.array([]),
        'low': np.array([]),
        'close': np.array([]),
        'volume': np.array([])
    }

    if df_norm is None or df_norm.empty:
        return result

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_norm.columns:
            result[col] = df_norm[col].values

    return result


def prepare_dataframe_for_analysis(df: pd.DataFrame, symbol: str = '') -> Optional[pd.DataFrame]:
    """
    Prepara DataFrame para análise, normalizando colunas e validando dados.
    Função principal a ser usada pelos módulos Phase 2.

    Args:
        df: DataFrame original
        symbol: Símbolo do ativo (para logging)

    Returns:
        DataFrame normalizado ou None se inválido
    """
    if df is None or df.empty:
        system_logger.warning(f"[{symbol}] DataFrame vazio ou None")
        return None

    # Normaliza colunas
    df_norm = normalize_dataframe_columns(df)

    # Valida dados
    validation = validate_ohlcv_data(df_norm)

    if not validation['valid']:
        system_logger.warning(f"[{symbol}] Dados inválidos: {validation['errors']}")
        return None

    return df_norm


# Alias para compatibilidade
normalize_df = normalize_dataframe_columns
