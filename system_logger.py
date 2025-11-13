"""
Sistema de logging profissional para Lobo IA.
Separado do Logger de banco de dados (logger.py).
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional
import yaml


class SystemLogger:
    """
    Logger profissional para eventos do sistema.
    Usa o módulo logging do Python com rotação de arquivos.
    """

    _instance: Optional['SystemLogger'] = None
    _initialized: bool = False

    def __new__(cls):
        """Implementa Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Inicializa o sistema de logging."""
        if not SystemLogger._initialized:
            self._setup_logger()
            SystemLogger._initialized = True

    def _setup_logger(self):
        """Configura handlers e formatters do logger."""
        # Carrega configurações
        config = self._load_config()
        log_config = config.get('logging', {})

        # Cria diretório de logs se não existir
        log_dir = log_config.get('log_dir', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Define nome do arquivo de log
        date_str = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_dir, f'lobo_{date_str}.log')

        # Cria logger
        self.logger = logging.getLogger('LoboIA')
        level_str = log_config.get('level', 'INFO')
        self.logger.setLevel(getattr(logging, level_str))

        # Remove handlers existentes (evita duplicação)
        self.logger.handlers = []

        # Formato do log
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Handler de arquivo com rotação
        max_bytes = log_config.get('max_file_size_mb', 10) * 1024 * 1024
        backup_count = log_config.get('backup_count', 5)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Handler de console (se habilitado)
        if log_config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _load_config(self) -> dict:
        """Carrega configurações do arquivo YAML."""
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Configuração padrão se arquivo não existir
            return {
                'logging': {
                    'level': 'INFO',
                    'log_dir': 'logs',
                    'max_file_size_mb': 10,
                    'backup_count': 5,
                    'console_output': True
                }
            }

    def debug(self, message: str):
        """Log de nível DEBUG."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log de nível INFO."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log de nível WARNING."""
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False):
        """Log de nível ERROR."""
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False):
        """Log de nível CRITICAL."""
        self.logger.critical(message, exc_info=exc_info)

    def exception(self, message: str):
        """Log de exceção com stack trace."""
        self.logger.exception(message)


# Instância global (Singleton)
system_logger = SystemLogger()
