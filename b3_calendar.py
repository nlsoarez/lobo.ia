"""
Calendario de feriados da B3 (Bolsa de Valores do Brasil).
Atualizado anualmente.
"""

from datetime import date, datetime
from typing import List, Set


# Feriados B3 - 2024
FERIADOS_2024 = [
    date(2024, 1, 1),   # Confraternizacao Universal
    date(2024, 2, 12),  # Carnaval
    date(2024, 2, 13),  # Carnaval
    date(2024, 3, 29),  # Sexta-feira Santa
    date(2024, 4, 21),  # Tiradentes
    date(2024, 5, 1),   # Dia do Trabalho
    date(2024, 5, 30),  # Corpus Christi
    date(2024, 7, 9),   # Revolucao Constitucionalista (SP)
    date(2024, 9, 7),   # Independencia do Brasil
    date(2024, 10, 12), # Nossa Senhora Aparecida
    date(2024, 11, 2),  # Finados
    date(2024, 11, 15), # Proclamacao da Republica
    date(2024, 11, 20), # Consciencia Negra
    date(2024, 12, 24), # Vespera de Natal
    date(2024, 12, 25), # Natal
    date(2024, 12, 31), # Vespera de Ano Novo
]

# Feriados B3 - 2025
FERIADOS_2025 = [
    date(2025, 1, 1),   # Confraternizacao Universal
    date(2025, 3, 3),   # Carnaval
    date(2025, 3, 4),   # Carnaval
    date(2025, 4, 18),  # Sexta-feira Santa
    date(2025, 4, 21),  # Tiradentes
    date(2025, 5, 1),   # Dia do Trabalho
    date(2025, 6, 19),  # Corpus Christi
    date(2025, 7, 9),   # Revolucao Constitucionalista (SP)
    date(2025, 9, 7),   # Independencia do Brasil
    date(2025, 10, 12), # Nossa Senhora Aparecida
    date(2025, 11, 2),  # Finados
    date(2025, 11, 15), # Proclamacao da Republica
    date(2025, 11, 20), # Consciencia Negra
    date(2025, 12, 24), # Vespera de Natal
    date(2025, 12, 25), # Natal
    date(2025, 12, 31), # Vespera de Ano Novo
]

# Feriados B3 - 2026
FERIADOS_2026 = [
    date(2026, 1, 1),   # Confraternizacao Universal
    date(2026, 2, 16),  # Carnaval
    date(2026, 2, 17),  # Carnaval
    date(2026, 4, 3),   # Sexta-feira Santa
    date(2026, 4, 21),  # Tiradentes
    date(2026, 5, 1),   # Dia do Trabalho
    date(2026, 6, 4),   # Corpus Christi
    date(2026, 7, 9),   # Revolucao Constitucionalista (SP)
    date(2026, 9, 7),   # Independencia do Brasil
    date(2026, 10, 12), # Nossa Senhora Aparecida
    date(2026, 11, 2),  # Finados
    date(2026, 11, 15), # Proclamacao da Republica
    date(2026, 11, 20), # Consciencia Negra
    date(2026, 12, 24), # Vespera de Natal
    date(2026, 12, 25), # Natal
    date(2026, 12, 31), # Vespera de Ano Novo
]

# Todos os feriados
ALL_HOLIDAYS: Set[date] = set(FERIADOS_2024 + FERIADOS_2025 + FERIADOS_2026)


def is_holiday(check_date: date = None) -> bool:
    """
    Verifica se uma data e feriado da B3.

    Args:
        check_date: Data a verificar. Se None, usa data atual.

    Returns:
        True se for feriado.
    """
    if check_date is None:
        check_date = date.today()

    if isinstance(check_date, datetime):
        check_date = check_date.date()

    return check_date in ALL_HOLIDAYS


def is_weekend(check_date: date = None) -> bool:
    """
    Verifica se uma data e fim de semana.

    Args:
        check_date: Data a verificar. Se None, usa data atual.

    Returns:
        True se for sabado ou domingo.
    """
    if check_date is None:
        check_date = date.today()

    if isinstance(check_date, datetime):
        check_date = check_date.date()

    # 5 = Sabado, 6 = Domingo
    return check_date.weekday() >= 5


def is_trading_day(check_date: date = None) -> bool:
    """
    Verifica se e um dia de pregao da B3.

    Args:
        check_date: Data a verificar. Se None, usa data atual.

    Returns:
        True se for dia util de pregao.
    """
    if check_date is None:
        check_date = date.today()

    return not is_weekend(check_date) and not is_holiday(check_date)


def get_next_trading_day(from_date: date = None) -> date:
    """
    Retorna o proximo dia de pregao.

    Args:
        from_date: Data inicial. Se None, usa data atual.

    Returns:
        Proximo dia util de pregao.
    """
    from datetime import timedelta

    if from_date is None:
        from_date = date.today()

    if isinstance(from_date, datetime):
        from_date = from_date.date()

    next_day = from_date + timedelta(days=1)

    while not is_trading_day(next_day):
        next_day += timedelta(days=1)

    return next_day


def get_holidays_for_year(year: int) -> List[date]:
    """
    Retorna lista de feriados para um ano especifico.

    Args:
        year: Ano desejado.

    Returns:
        Lista de datas de feriados.
    """
    return [h for h in ALL_HOLIDAYS if h.year == year]


# Informacoes sobre o mercado
B3_INFO = {
    'name': 'B3 - Brasil Bolsa Balcao',
    'timezone': 'America/Sao_Paulo',
    'open_time': '10:00',
    'close_time': '18:00',
    'pre_market': '09:45',
    'after_market': '18:25',
    'currency': 'BRL',
}
