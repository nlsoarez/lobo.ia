"""
Servidor de Health Check para Railway e monitoramento.
Fornece endpoints para verificar saúde da aplicação.
"""

import os
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from typing import Optional

from config_loader import config
from logger import Logger


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Handler HTTP para endpoints de health check."""

    # Referência ao sistema de trading (definido externamente)
    trading_system = None
    start_time = datetime.now()

    def log_message(self, format, *args):
        """Suprime logs de requisição HTTP."""
        pass

    def _send_json(self, data: dict, status: int = 200):
        """Envia resposta JSON."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def do_GET(self):
        """Processa requisições GET."""
        if self.path == '/health':
            self._health_check()
        elif self.path == '/ready':
            self._readiness_check()
        elif self.path == '/status':
            self._status_check()
        elif self.path == '/metrics':
            self._metrics()
        else:
            self._send_json({'error': 'Not found'}, 404)

    def _health_check(self):
        """
        Endpoint de liveness probe.
        Retorna 200 se o processo está rodando.
        """
        uptime = (datetime.now() - self.start_time).total_seconds()

        response = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime,
            'version': '2.0.0',
            'environment': os.environ.get('RAILWAY_ENVIRONMENT', 'development')
        }

        self._send_json(response)

    def _readiness_check(self):
        """
        Endpoint de readiness probe.
        Retorna 200 se o sistema está pronto para processar.
        """
        try:
            # Verifica conexão com banco de dados
            with Logger() as logger:
                db_status = logger.health_check()

            if db_status.get('connected'):
                response = {
                    'status': 'ready',
                    'database': db_status,
                    'timestamp': datetime.now().isoformat()
                }
                self._send_json(response)
            else:
                response = {
                    'status': 'not_ready',
                    'database': db_status,
                    'timestamp': datetime.now().isoformat()
                }
                self._send_json(response, 503)

        except Exception as e:
            self._send_json({
                'status': 'not_ready',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 503)

    def _status_check(self):
        """
        Endpoint de status detalhado.
        Retorna informações sobre o sistema de trading.
        """
        try:
            # Obtém estatísticas do banco
            with Logger() as logger:
                db_status = logger.health_check()
                stats = logger.get_performance_stats()
                last_trades = logger.get_last_trades(5)

            response = {
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'database': db_status,
                'trading': {
                    'mode': config.get('execution.mode', 'simulation'),
                    'symbols': config.get('trading.symbols', []),
                    'capital': config.get('trading.capital', 10000)
                },
                'performance': stats,
                'last_trades': last_trades,
                'environment': {
                    'railway': config.is_railway,
                    'production': config.is_production
                }
            }

            self._send_json(response)

        except Exception as e:
            self._send_json({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500)

    def _metrics(self):
        """
        Endpoint de métricas para monitoramento.
        Formato compatível com Prometheus.
        """
        try:
            with Logger() as logger:
                stats = logger.get_performance_stats()

            uptime = (datetime.now() - self.start_time).total_seconds()

            # Formato Prometheus
            metrics = [
                f'# HELP lobo_uptime_seconds Total uptime in seconds',
                f'# TYPE lobo_uptime_seconds gauge',
                f'lobo_uptime_seconds {uptime}',
                f'',
                f'# HELP lobo_trades_total Total number of trades',
                f'# TYPE lobo_trades_total counter',
                f'lobo_trades_total {stats.get("total_trades", 0)}',
                f'',
                f'# HELP lobo_profit_total Total profit in BRL',
                f'# TYPE lobo_profit_total gauge',
                f'lobo_profit_total {stats.get("total_profit", 0)}',
                f'',
                f'# HELP lobo_win_rate Win rate percentage',
                f'# TYPE lobo_win_rate gauge',
                f'lobo_win_rate {stats.get("win_rate", 0)}',
                f'',
                f'# HELP lobo_wins_total Total winning trades',
                f'# TYPE lobo_wins_total counter',
                f'lobo_wins_total {stats.get("wins", 0)}',
                f'',
                f'# HELP lobo_losses_total Total losing trades',
                f'# TYPE lobo_losses_total counter',
                f'lobo_losses_total {stats.get("losses", 0)}',
            ]

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write('\n'.join(metrics).encode())

        except Exception as e:
            self._send_json({
                'error': str(e)
            }, 500)


class HealthServer:
    """
    Servidor de health check que roda em thread separada.
    Não interfere com o sistema de trading principal.
    """

    def __init__(self, port: Optional[int] = None):
        """
        Inicializa o servidor de health check.

        Args:
            port: Porta para o servidor. Se None, usa PORT do ambiente ou 8080.
        """
        if port is None:
            port = int(os.environ.get('HEALTH_PORT', os.environ.get('PORT', 8080)))

        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        """Inicia o servidor em thread separada."""
        self.server = HTTPServer(('0.0.0.0', self.port), HealthCheckHandler)

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        print(f"Health check server running on port {self.port}")

    def stop(self):
        """Para o servidor."""
        if self.server:
            self.server.shutdown()


# Instância global para uso em outros módulos
health_server: Optional[HealthServer] = None


def start_health_server(port: Optional[int] = None) -> HealthServer:
    """
    Função de conveniência para iniciar o servidor.

    Args:
        port: Porta para o servidor.

    Returns:
        Instância do HealthServer.
    """
    global health_server
    health_server = HealthServer(port)
    health_server.start()
    return health_server


if __name__ == '__main__':
    # Teste standalone
    server = start_health_server()
    print(f"Health server running. Test with: curl http://localhost:{server.port}/health")

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
