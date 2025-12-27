"""
Script para resetar posições abertas e começar do zero.
Execute: python reset_positions.py
"""

import os
import sys

sys.path.insert(0, '.')

from logger import Logger


def reset_all_positions():
    """Limpa todas as posições abertas do banco de dados."""
    print("=" * 60)
    print("🔄 RESET DE POSIÇÕES - Lobo IA")
    print("=" * 60)

    try:
        logger = Logger()

        # Mostra posições atuais
        positions = logger.load_positions()
        if positions:
            print(f"\n📂 Posições abertas encontradas: {len(positions)}")
            for symbol, pos in positions.items():
                print(f"   - {symbol}: {pos['quantity']:.6f} @ ${pos['entry_price']:.2f}")
        else:
            print("\n📂 Nenhuma posição aberta na tabela crypto_positions")

        # Verifica também trades não fechados
        recovered = logger.recover_open_positions_from_trades()
        if recovered:
            print(f"\n📂 Posições recuperáveis dos trades: {len(recovered)}")
            for symbol, pos in recovered.items():
                print(f"   - {symbol}: {pos['quantity']:.6f} @ ${pos['entry_price']:.2f}")

        # Limpa tabela crypto_positions
        print("\n🗑️ Limpando tabela crypto_positions...")
        cursor = logger.conn.cursor()
        cursor.execute("DELETE FROM crypto_positions")
        logger.conn.commit()
        print("✅ Tabela crypto_positions limpa!")

        # Pergunta se quer limpar também os trades históricos
        print("\n" + "=" * 60)
        print("⚠️  ATENÇÃO: Os trades históricos na tabela 'trades' ainda existem.")
        print("    Se não limpar, o sistema vai tentar recuperar as posições.")
        print("=" * 60)

        response = input("\nLimpar também os trades de crypto (BUY sem SELL)? [s/N]: ").strip().lower()

        if response == 's':
            # Marca todos os BUYs de crypto como fechados (adiciona SELLs fictícios)
            print("\n🗑️ Limpando trades de crypto...")
            cursor.execute("DELETE FROM trades WHERE symbol LIKE '%-USD'")
            logger.conn.commit()
            print("✅ Trades de crypto removidos!")

        logger.close()

        print("\n" + "=" * 60)
        print("✅ RESET COMPLETO!")
        print("   O sistema vai iniciar com:")
        print("   - 0 posições abertas")
        print("   - $1000.00 de capital")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    reset_all_positions()
