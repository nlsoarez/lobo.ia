import sqlite3

class Logger:
    def __init__(self, db_name='trades.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date TEXT,
                action TEXT,
                price REAL,
                quantity INTEGER,
                profit REAL,
                indicators TEXT,
                notes TEXT
            )
        """)
        self.conn.commit()

    def log_trade(self, trade):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (symbol, date, action, price, quantity, profit, indicators, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade['symbol'], trade['date'], trade['action'], trade['price'],
            trade['quantity'], trade['profit'], trade['indicators'], trade['notes']
        ))
        self.conn.commit()
