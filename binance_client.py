"""
Cliente Binance para operacoes de trading e consulta de saldo.
"""

import os
import hmac
import hashlib
import time
import requests
from typing import Dict, Optional, List
from decimal import Decimal


class BinanceClient:
    """Cliente para API da Binance."""

    # URLs
    MAINNET_URL = "https://api.binance.com"
    TESTNET_URL = "https://testnet.binance.vision"

    def __init__(self, api_key: str = None, secret_key: str = None, testnet: bool = True):
        """
        Inicializa cliente Binance.

        Args:
            api_key: API Key da Binance
            secret_key: Secret Key da Binance
            testnet: Se True, usa testnet. Se False, usa mainnet (conta real)
        """
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY', '')
        self.secret_key = secret_key or os.environ.get('BINANCE_SECRET_KEY', '')

        # Determina se usa testnet
        testnet_env = os.environ.get('BINANCE_TESTNET', 'true').lower()
        self.testnet = testnet if testnet is not None else (testnet_env == 'true')

        self.base_url = self.TESTNET_URL if self.testnet else self.MAINNET_URL

    def _sign_request(self, params: Dict) -> Dict:
        """Assina requisicao com HMAC SHA256."""
        params['timestamp'] = int(time.time() * 1000)
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        return params

    def _get_headers(self) -> Dict:
        """Retorna headers para requisicao."""
        return {
            'X-MBX-APIKEY': self.api_key
        }

    def get_account_balance(self) -> Dict:
        """
        Obtem saldo da conta.

        Returns:
            Dict com saldos por moeda
        """
        if not self.api_key or not self.secret_key:
            return {'error': 'API keys not configured', 'balances': []}

        try:
            params = self._sign_request({})
            response = requests.get(
                f"{self.base_url}/api/v3/account",
                params=params,
                headers=self._get_headers(),
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                # Filtra apenas saldos > 0
                balances = [
                    {
                        'asset': b['asset'],
                        'free': float(b['free']),
                        'locked': float(b['locked']),
                        'total': float(b['free']) + float(b['locked'])
                    }
                    for b in data.get('balances', [])
                    if float(b['free']) > 0 or float(b['locked']) > 0
                ]
                return {
                    'success': True,
                    'balances': balances,
                    'testnet': self.testnet
                }
            else:
                return {
                    'error': f"API Error: {response.status_code} - {response.text}",
                    'balances': [],
                    'testnet': self.testnet
                }

        except Exception as e:
            return {
                'error': str(e),
                'balances': [],
                'testnet': self.testnet
            }

    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """
        Obtem preco atual de um par.

        Args:
            symbol: Par de trading (ex: BTCUSDT)

        Returns:
            Preco atual ou None
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=10
            )
            if response.status_code == 200:
                return float(response.json()['price'])
        except:
            pass
        return None

    def get_total_balance_usdt(self) -> Dict:
        """
        Calcula saldo total em USDT.

        Returns:
            Dict com saldo total e detalhes
        """
        account = self.get_account_balance()

        if 'error' in account and account.get('balances', []) == []:
            return account

        total_usdt = 0.0
        details = []

        for balance in account.get('balances', []):
            asset = balance['asset']
            total = balance['total']

            if asset == 'USDT':
                value_usdt = total
            elif asset == 'BRL':
                # BRL para USDT
                price = self.get_ticker_price('USDTBRL')
                value_usdt = total / price if price else 0
            else:
                # Converte para USDT
                price = self.get_ticker_price(f"{asset}USDT")
                value_usdt = total * price if price else 0

            if value_usdt > 0.01:  # Ignora valores muito pequenos
                details.append({
                    'asset': asset,
                    'amount': total,
                    'value_usdt': value_usdt
                })
                total_usdt += value_usdt

        return {
            'success': True,
            'total_usdt': total_usdt,
            'details': details,
            'testnet': self.testnet
        }

    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        Obtem ordens abertas.

        Args:
            symbol: Par especifico ou None para todos

        Returns:
            Lista de ordens abertas
        """
        if not self.api_key or not self.secret_key:
            return []

        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            params = self._sign_request(params)

            response = requests.get(
                f"{self.base_url}/api/v3/openOrders",
                params=params,
                headers=self._get_headers(),
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []

    def test_connection(self) -> Dict:
        """
        Testa conexao com a API.

        Returns:
            Dict com status da conexao
        """
        try:
            # Teste publico
            response = requests.get(
                f"{self.base_url}/api/v3/ping",
                timeout=10
            )
            public_ok = response.status_code == 200

            # Teste autenticado
            account = self.get_account_balance()
            auth_ok = 'error' not in account or account.get('balances', []) != []

            return {
                'public_api': public_ok,
                'authenticated': auth_ok,
                'testnet': self.testnet,
                'url': self.base_url
            }
        except Exception as e:
            return {
                'public_api': False,
                'authenticated': False,
                'error': str(e),
                'testnet': self.testnet
            }


def get_binance_client() -> BinanceClient:
    """Factory para criar cliente Binance com configuracoes do ambiente."""
    return BinanceClient()
