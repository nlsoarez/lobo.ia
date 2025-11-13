"""
Logger de trades para banco de dados SQLite.
Thread-safe e com context manager.
"""

import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from config_loader import config


class Logger:
    """
    Logger thread-safe para persistir trades em banco de dados SQLite.
    Usa connection pooling e locks para garantir segurança em ambiente concorrente.
    """

    def __init__(self, db_name: Optional[str] = None):
        """
        Inicializa o logger de banco de dados.

        Args:
            db_name: Nome do arquivo do banco de dados. Se None, usa config.yaml.
        """
        if db_name is None:
            db_name = config.get('database.db_name', 'trades.db')

        self.db_name = db_name
        self.lock = threading.Lock()

        # Usa check_same_thread=False para permitir uso multi-thread
        # Mas protege com Lock para garantir thread-safety
        self.conn = sqlite3.connect(
            db_name,
            check_same_thread=False,
            isolation_level='DEFERRED'
        )
        self.conn.row_factory = sqlite3.Row  # Retorna resultados como dicionários
        self.create_table()

    def __enter__(self):
        """Suporte a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Fecha conexão ao sair do context manager."""
        self.close()

    def create_table(self):
        """
        Cria tabela de trades se não existir.
        Adiciona índices para melhorar performance de queries.
        """
        with self.lock:
            cursor = self.conn.cursor()

            # Cria tabela principal
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TIMESTAMP NOT NULL,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    profit REAL DEFAULT 0,
                    indicators TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Cria índices para melhorar performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol
                ON trades(symbol)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_date
                ON trades(date)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_action
                ON trades(action)
            """)

            self.conn.commit()

    def log_trade(self, trade: Dict[str, Any]) -> int:
        """
        Registra um trade no banco de dados.

        Args:
            trade: Dicionário com dados do trade (symbol, date, action, price, quantity, profit, indicators, notes).

        Returns:
            ID do trade inserido.

        Raises:
            sqlite3.Error: Se houver erro ao inserir no banco.
        """
        with self.lock:
            cursor = self.conn.cursor()

            # Converte date para timestamp se for string
            trade_date = trade.get('date')
            if isinstance(trade_date, str):
                try:
                    trade_date = datetime.fromisoformat(trade_date)
                except (ValueError, TypeError):
                    trade_date = datetime.now()
            elif trade_date is None:
                trade_date = datetime.now()

            cursor.execute("""
                INSERT INTO trades (symbol, date, action, price, quantity, profit, indicators, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get('symbol', ''),
                trade_date,
                trade.get('action', ''),
                trade.get('price', 0.0),
                trade.get('quantity', 0),
                trade.get('profit', 0.0),
                trade.get('indicators', ''),
                trade.get('notes', '')
            ))

            self.conn.commit()
            return cursor.lastrowid

    def get_trades(
        self,
        symbol: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Recupera trades do banco de dados.

        Args:
            symbol: Filtrar por símbolo (opcional).
            action: Filtrar por ação (BUY, SELL, etc.) (opcional).
            limit: Número máximo de resultados.

        Returns:
            Lista de dicionários com dados dos trades.
        """
        with self.lock:
            cursor = self.conn.cursor()

            query = "SELECT * FROM trades WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if action:
                query += " AND action = ?"
                params.append(action)

            query += " ORDER BY date DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_performance_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Calcula estatísticas de performance.

        Args:
            symbol: Calcular stats para símbolo específico (opcional).

        Returns:
            Dicionário com métricas de performance.
        """
        with self.lock:
            cursor = self.conn.cursor()

            where_clause = "WHERE action IN ('BUY', 'SELL')"
            params = []

            if symbol:
                where_clause += " AND symbol = ?"
                params.append(symbol)

            # Total de trades
            cursor.execute(f"SELECT COUNT(*) as total FROM trades {where_clause}", params)
            total_trades = cursor.fetchone()['total']

            # Lucro total
            cursor.execute(f"SELECT SUM(profit) as total_profit FROM trades {where_clause}", params)
            total_profit = cursor.fetchone()['total_profit'] or 0

            # Win rate
            cursor.execute(f"SELECT COUNT(*) as wins FROM trades {where_clause} AND profit > 0", params)
            wins = cursor.fetchone()['wins']

            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            return {
                'total_trades': total_trades,
                'total_profit': total_profit,
                'wins': wins,
                'losses': total_trades - wins,
                'win_rate': win_rate
            }

    def close(self):
        """Fecha a conexão com o banco de dados."""
        if self.conn:
            with self.lock:
                self.conn.close()
