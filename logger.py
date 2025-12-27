"""
Logger de trades para banco de dados.
Suporta SQLite (local) e PostgreSQL (Railway/produção).
Thread-safe e com context manager.
"""

import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from config_loader import config

# Tenta importar psycopg2 para PostgreSQL
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False


class Logger:
    """
    Logger thread-safe para persistir trades em banco de dados.
    Suporta SQLite (local) e PostgreSQL (Railway/cloud).
    """

    def __init__(self, db_name: Optional[str] = None):
        """
        Inicializa o logger de banco de dados.

        Args:
            db_name: Nome do arquivo do banco de dados (SQLite) ou URL de conexão.
                     Se None, usa DATABASE_URL ou config.yaml.
        """
        self.lock = threading.Lock()
        self.conn = None
        self.db_type = 'sqlite'

        # Determina tipo de banco e string de conexão
        database_url = os.environ.get('DATABASE_URL')

        if database_url:
            # Railway fornece DATABASE_URL com PostgreSQL
            self._init_postgres(database_url)
        elif db_name and db_name.startswith('postgres'):
            self._init_postgres(db_name)
        else:
            # Fallback para SQLite
            if db_name is None:
                db_name = config.get('database.db_name', 'trades.db')
            self._init_sqlite(db_name)

    def _init_sqlite(self, db_name: str):
        """Inicializa conexão SQLite."""
        self.db_type = 'sqlite'
        self.db_name = db_name

        self.conn = sqlite3.connect(
            db_name,
            check_same_thread=False,
            isolation_level='DEFERRED'
        )
        self.conn.row_factory = sqlite3.Row
        self.create_table()

    def _init_postgres(self, database_url: str):
        """Inicializa conexão PostgreSQL."""
        if not HAS_POSTGRES:
            raise ImportError(
                "psycopg2 não está instalado. "
                "Instale com: pip install psycopg2-binary"
            )

        self.db_type = 'postgresql'
        self.database_url = database_url

        # Parse URL para conexão
        parsed = urlparse(database_url)

        self.conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path[1:],  # Remove leading '/'
            user=parsed.username,
            password=parsed.password,
            cursor_factory=RealDictCursor
        )
        self.conn.autocommit = False
        self.create_table()

    def __enter__(self):
        """Suporte a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Fecha conexão ao sair do context manager."""
        self.close()

    def create_table(self):
        """
        Cria tabelas se não existirem.
        Compatível com SQLite e PostgreSQL.
        """
        with self.lock:
            cursor = self.conn.cursor()

            if self.db_type == 'sqlite':
                # SQLite usa AUTOINCREMENT
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
                # Tabela para posicoes abertas (persistencia)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS crypto_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL UNIQUE,
                        quantity REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        trade_value REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        take_profit REAL NOT NULL,
                        max_price REAL NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            else:
                # PostgreSQL usa SERIAL
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        date TIMESTAMP NOT NULL,
                        action VARCHAR(10) NOT NULL,
                        price DECIMAL(15, 4) NOT NULL,
                        quantity INTEGER NOT NULL,
                        profit DECIMAL(15, 4) DEFAULT 0,
                        indicators TEXT,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Tabela para posicoes abertas (persistencia)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS crypto_positions (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL UNIQUE,
                        quantity DECIMAL(20, 8) NOT NULL,
                        entry_price DECIMAL(15, 4) NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        trade_value DECIMAL(15, 4) NOT NULL,
                        stop_loss DECIMAL(15, 4) NOT NULL,
                        take_profit DECIMAL(15, 4) NOT NULL,
                        max_price DECIMAL(15, 4) NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Cria índices para melhorar performance
            if self.db_type == 'sqlite':
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_date ON trades(date)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_action ON trades(action)
                """)
            else:
                # PostgreSQL - usa DO block para evitar erro se índice existe
                cursor.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_symbol') THEN
                            CREATE INDEX idx_symbol ON trades(symbol);
                        END IF;
                        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_date') THEN
                            CREATE INDEX idx_date ON trades(date);
                        END IF;
                        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_action') THEN
                            CREATE INDEX idx_action ON trades(action);
                        END IF;
                    END $$;
                """)

            self.conn.commit()

    def _get_placeholder(self) -> str:
        """Retorna placeholder para queries (%s para PostgreSQL, ? para SQLite)."""
        return '%s' if self.db_type == 'postgresql' else '?'

    def log_trade(self, trade: Dict[str, Any]) -> int:
        """
        Registra um trade no banco de dados.

        Args:
            trade: Dicionário com dados do trade.

        Returns:
            ID do trade inserido.
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

            ph = self._get_placeholder()

            if self.db_type == 'postgresql':
                cursor.execute(f"""
                    INSERT INTO trades (symbol, date, action, price, quantity, profit, indicators, notes)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                    RETURNING id
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
                trade_id = cursor.fetchone()['id']
            else:
                cursor.execute(f"""
                    INSERT INTO trades (symbol, date, action, price, quantity, profit, indicators, notes)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
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
                trade_id = cursor.lastrowid

            self.conn.commit()
            return trade_id

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
            ph = self._get_placeholder()

            query = "SELECT * FROM trades WHERE 1=1"
            params = []

            if symbol:
                query += f" AND symbol = {ph}"
                params.append(symbol)

            if action:
                query += f" AND action = {ph}"
                params.append(action)

            query += f" ORDER BY date DESC LIMIT {ph}"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Converte para lista de dicionários
            if self.db_type == 'postgresql':
                return [dict(row) for row in rows]
            else:
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
            ph = self._get_placeholder()

            where_clause = "WHERE action IN ('BUY', 'SELL')"
            params = []

            if symbol:
                where_clause += f" AND symbol = {ph}"
                params.append(symbol)

            # Total de trades
            cursor.execute(f"SELECT COUNT(*) as total FROM trades {where_clause}", params)
            result = cursor.fetchone()
            total_trades = result['total'] if isinstance(result, dict) else result[0]

            # Lucro total
            cursor.execute(f"SELECT COALESCE(SUM(profit), 0) as total_profit FROM trades {where_clause}", params)
            result = cursor.fetchone()
            total_profit = result['total_profit'] if isinstance(result, dict) else result[0]
            total_profit = float(total_profit) if total_profit else 0

            # Win rate
            cursor.execute(f"SELECT COUNT(*) as wins FROM trades {where_clause} AND profit > 0", params)
            result = cursor.fetchone()
            wins = result['wins'] if isinstance(result, dict) else result[0]

            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            return {
                'total_trades': total_trades,
                'total_profit': total_profit,
                'wins': wins,
                'losses': total_trades - wins,
                'win_rate': win_rate
            }

    def get_last_trades(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Retorna os últimos N trades.

        Args:
            count: Número de trades a retornar.

        Returns:
            Lista de trades.
        """
        return self.get_trades(limit=count)

    def health_check(self) -> Dict[str, Any]:
        """
        Verifica saúde da conexão com o banco.

        Returns:
            Dicionário com status da conexão.
        """
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()

            return {
                'status': 'healthy',
                'database_type': self.db_type,
                'connected': True
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'database_type': self.db_type,
                'connected': False,
                'error': str(e)
            }

    def save_position(self, symbol: str, position: Dict[str, Any]) -> bool:
        """
        Salva ou atualiza uma posicao aberta no banco.

        Args:
            symbol: Simbolo do ativo
            position: Dados da posicao

        Returns:
            True se salvou com sucesso
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()
                ph = self._get_placeholder()

                if self.db_type == 'postgresql':
                    # UPSERT para PostgreSQL
                    cursor.execute(f"""
                        INSERT INTO crypto_positions
                        (symbol, quantity, entry_price, entry_time, trade_value, stop_loss, take_profit, max_price, updated_at)
                        VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, NOW())
                        ON CONFLICT (symbol) DO UPDATE SET
                            quantity = EXCLUDED.quantity,
                            entry_price = EXCLUDED.entry_price,
                            entry_time = EXCLUDED.entry_time,
                            trade_value = EXCLUDED.trade_value,
                            stop_loss = EXCLUDED.stop_loss,
                            take_profit = EXCLUDED.take_profit,
                            max_price = EXCLUDED.max_price,
                            updated_at = NOW()
                    """, (
                        symbol,
                        position['quantity'],
                        position['entry_price'],
                        position['entry_time'],
                        position['trade_value'],
                        position['stop_loss'],
                        position['take_profit'],
                        position.get('max_price', position['entry_price'])
                    ))
                else:
                    # UPSERT para SQLite
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO crypto_positions
                        (symbol, quantity, entry_price, entry_time, trade_value, stop_loss, take_profit, max_price, updated_at)
                        VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, CURRENT_TIMESTAMP)
                    """, (
                        symbol,
                        position['quantity'],
                        position['entry_price'],
                        position['entry_time'],
                        position['trade_value'],
                        position['stop_loss'],
                        position['take_profit'],
                        position.get('max_price', position['entry_price'])
                    ))

                self.conn.commit()
                return True
            except Exception as e:
                print(f"Erro ao salvar posicao {symbol}: {e}")
                return False

    def load_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Carrega todas as posicoes abertas do banco.

        Returns:
            Dicionario {symbol: position_data}
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT * FROM crypto_positions")
                rows = cursor.fetchall()

                positions = {}
                for row in rows:
                    row_dict = dict(row)
                    symbol = row_dict['symbol']
                    positions[symbol] = {
                        'quantity': float(row_dict['quantity']),
                        'entry_price': float(row_dict['entry_price']),
                        'entry_time': row_dict['entry_time'],
                        'trade_value': float(row_dict['trade_value']),
                        'stop_loss': float(row_dict['stop_loss']),
                        'take_profit': float(row_dict['take_profit']),
                        'max_price': float(row_dict['max_price'])
                    }

                return positions
            except Exception as e:
                print(f"Erro ao carregar posicoes: {e}")
                return {}

    def delete_position(self, symbol: str) -> bool:
        """
        Remove uma posicao fechada do banco.

        Args:
            symbol: Simbolo do ativo

        Returns:
            True se removeu com sucesso
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()
                ph = self._get_placeholder()
                cursor.execute(f"DELETE FROM crypto_positions WHERE symbol = {ph}", (symbol,))
                self.conn.commit()
                return True
            except Exception as e:
                print(f"Erro ao deletar posicao {symbol}: {e}")
                return False

    def recover_open_positions_from_trades(self) -> Dict[str, Dict[str, Any]]:
        """
        Recupera posicoes abertas analisando a tabela trades.
        Identifica BUYs que nao tem SELL correspondente.

        Returns:
            Dicionario {symbol: position_data}
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()

                # Busca todos os trades de crypto (simbolos com -USD)
                cursor.execute("""
                    SELECT symbol, action, price, quantity, date
                    FROM trades
                    WHERE symbol LIKE '%-USD'
                    ORDER BY date ASC
                """)
                rows = cursor.fetchall()

                # Rastreia posicoes por simbolo
                positions = {}

                for row in rows:
                    row_dict = dict(row)
                    symbol = row_dict['symbol']
                    action = row_dict['action']
                    price = float(row_dict['price'])
                    quantity = float(row_dict['quantity'])
                    date = row_dict['date']

                    if action == 'BUY':
                        # Abre ou adiciona a posicao
                        if symbol not in positions:
                            positions[symbol] = {
                                'quantity': quantity,
                                'entry_price': price,
                                'entry_time': date,
                                'trade_value': quantity * price,
                                'stop_loss': price * 0.98,  # 2% stop loss
                                'take_profit': price * 1.05,  # 5% take profit
                                'max_price': price
                            }
                        else:
                            # Adiciona a posicao existente (averaging)
                            old = positions[symbol]
                            new_qty = old['quantity'] + quantity
                            new_value = old['trade_value'] + (quantity * price)
                            new_avg_price = new_value / new_qty if new_qty > 0 else price
                            positions[symbol] = {
                                'quantity': new_qty,
                                'entry_price': new_avg_price,
                                'entry_time': old['entry_time'],  # Mantem data original
                                'trade_value': new_value,
                                'stop_loss': new_avg_price * 0.98,
                                'take_profit': new_avg_price * 1.05,
                                'max_price': max(old['max_price'], price)
                            }
                    elif action == 'SELL':
                        # Fecha posicao
                        if symbol in positions:
                            old = positions[symbol]
                            remaining = old['quantity'] - quantity
                            if remaining <= 0.00001:  # Praticamente zero
                                del positions[symbol]
                            else:
                                positions[symbol]['quantity'] = remaining
                                positions[symbol]['trade_value'] = remaining * old['entry_price']

                return positions

            except Exception as e:
                print(f"Erro ao recuperar posicoes de trades: {e}")
                return {}

    def close(self):
        """Fecha a conexão com o banco de dados."""
        if self.conn:
            with self.lock:
                self.conn.close()
