#!/usr/bin/env python3
"""
Script para resetar TODOS os dados do Lobo IA V4.2 (Crypto Only).
Remove trades, posições e reinicia capital em USD.
Execute: python reset_all_data.py
"""

import os
import sys

sys.path.insert(0, '.')

from logger import Logger


def reset_all_data():
    """Limpa todos os dados do banco para nova medição."""
    print("=" * 60)
    print("🔄 RESET COMPLETO - Lobo IA V4.2 (Crypto Only)")
    print("=" * 60)

    try:
        logger = Logger()
        cursor = logger.conn.cursor()

        # Conta registros atuais
        cursor.execute("SELECT COUNT(*) FROM trades")
        result = cursor.fetchone()
        trades_count = result[0]

        cursor.execute("SELECT COUNT(*) FROM crypto_positions")
        result = cursor.fetchone()
        positions_count = result[0]

        print(f"\n📊 Dados atuais:")
        print(f"   - Trades: {trades_count}")
        print(f"   - Posições abertas: {positions_count}")

        # Limpa todas as tabelas
        print("\n🗑️ Limpando banco de dados...")

        cursor.execute("DELETE FROM crypto_positions")
        print("   ✅ Tabela crypto_positions limpa")

        cursor.execute("DELETE FROM trades")
        print("   ✅ Tabela trades limpa")

        # Reseta o autoincrement
        cursor.execute("DELETE FROM sqlite_sequence")
        print("   ✅ Sequências resetadas")

        logger.conn.commit()

        print("\n" + "=" * 60)
        print("✅ RESET COMPLETO!")
        print("   Sistema reiniciado com:")
        print("   - 0 posições abertas")
        print("   - 0 trades históricos")
        print("   - $1000.00 USD de capital inicial")
        print("   - Modo: Crypto Only 24/7")
        print("=" * 60)

        logger.close()

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    reset_all_data()
