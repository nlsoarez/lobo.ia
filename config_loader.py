"""
Carregador centralizado de configurações para Lobo IA.
Suporta YAML, variáveis de ambiente e Railway.
"""

import os
import yaml
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    Carrega e gerencia configurações do sistema.
    Implementa Singleton pattern para acesso global.

    Prioridade de configuração:
    1. Variáveis de ambiente (maior prioridade)
    2. Arquivo config.yaml
    3. Valores padrão
    """

    _instance: Optional['ConfigLoader'] = None
    _config: Optional[Dict[str, Any]] = None

    # Mapeamento de variáveis de ambiente para configurações
    ENV_MAPPING = {
        # Database
        'DATABASE_URL': 'database.url',
        'DATABASE_TYPE': 'database.type',  # 'sqlite' ou 'postgresql'
        'DB_NAME': 'database.db_name',

        # Trading
        'TRADING_SYMBOLS': 'trading.symbols',  # Separado por vírgula: PETR4.SA,VALE3.SA
        'TRADING_CAPITAL': 'trading.capital',
        'TRADING_EXPOSURE': 'trading.exposure',
        'MAX_TOTAL_EXPOSURE': 'trading.max_total_exposure',

        # Execution
        'EXECUTION_MODE': 'execution.mode',  # simulation, paper, live

        # Risk
        'STOP_LOSS': 'risk.stop_loss',
        'TAKE_PROFIT': 'risk.take_profit',
        'MAX_DRAWDOWN': 'risk.max_drawdown',

        # Market
        'MARKET_OPEN_HOUR': 'market.open_hour',
        'MARKET_CLOSE_HOUR': 'market.close_hour',
        'CHECK_INTERVAL': 'market.check_interval',

        # Logging
        'LOG_LEVEL': 'logging.level',

        # Notifications
        'TELEGRAM_TOKEN': 'notifications.telegram_token',
        'TELEGRAM_CHAT_ID': 'notifications.telegram_chat_id',

        # Railway/Cloud specific
        'PORT': 'server.port',
        'RAILWAY_ENVIRONMENT': 'environment.name',
    }

    def __new__(cls):
        """Implementa Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_file: str = 'config.yaml'):
        """
        Inicializa o carregador de configurações.

        Args:
            config_file: Caminho para o arquivo de configuração YAML.
        """
        if ConfigLoader._config is None:
            self.config_file = config_file
            self.load()

    def load(self) -> Dict[str, Any]:
        """
        Carrega configurações do arquivo YAML e variáveis de ambiente.

        Returns:
            Dicionário com todas as configurações.
        """
        # 1. Carrega configurações base do YAML
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                ConfigLoader._config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            # Se não existe arquivo, usa configuração vazia
            ConfigLoader._config = {}
            print(f"Aviso: Arquivo '{self.config_file}' não encontrado. Usando variáveis de ambiente.")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Erro ao parsear arquivo de configuração: {e}")

        # 2. Sobrescreve com variáveis de ambiente
        self._apply_environment_variables()

        # 3. Aplica valores padrão se necessário
        self._apply_defaults()

        return ConfigLoader._config

    def _apply_environment_variables(self):
        """Aplica variáveis de ambiente sobre as configurações."""
        for env_var, config_path in self.ENV_MAPPING.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(config_path, self._parse_value(value, config_path))

    def _parse_value(self, value: str, config_path: str) -> Any:
        """
        Converte string de variável de ambiente para tipo apropriado.

        Args:
            value: Valor da variável de ambiente.
            config_path: Caminho da configuração para determinar tipo.

        Returns:
            Valor convertido para tipo apropriado.
        """
        # Listas (separadas por vírgula)
        if config_path == 'trading.symbols':
            return [s.strip() for s in value.split(',')]

        # Floats
        float_paths = [
            'trading.capital', 'trading.exposure', 'trading.max_total_exposure',
            'risk.stop_loss', 'risk.take_profit', 'risk.max_drawdown'
        ]
        if config_path in float_paths:
            return float(value)

        # Integers
        int_paths = ['market.open_hour', 'market.close_hour', 'market.check_interval', 'server.port']
        if config_path in int_paths:
            return int(value)

        # Booleans
        if value.lower() in ('true', 'false', '1', '0', 'yes', 'no'):
            return value.lower() in ('true', '1', 'yes')

        return value

    def _set_nested(self, key_path: str, value: Any):
        """
        Define valor em configuração aninhada.

        Args:
            key_path: Caminho da chave (ex: 'database.url').
            value: Valor a definir.
        """
        keys = key_path.split('.')
        current = ConfigLoader._config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _apply_defaults(self):
        """Aplica valores padrão para configurações essenciais."""
        defaults = {
            'database': {
                'type': 'sqlite',
                'db_name': 'trades.db',
            },
            'trading': {
                'symbols': ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA'],
                'capital': 10000.0,
                'exposure': 0.03,
                'max_total_exposure': 0.20,
            },
            'execution': {
                'mode': 'simulation',
            },
            'risk': {
                'stop_loss': 0.02,
                'take_profit': 0.05,
                'max_drawdown': 0.10,
            },
            'market': {
                'open_hour': 10,
                'close_hour': 18,
                'check_interval': 60,
                'trading_days': [0, 1, 2, 3, 4],
            },
            'logging': {
                'level': 'INFO',
            },
        }

        self._merge_defaults(ConfigLoader._config, defaults)

    def _merge_defaults(self, config: dict, defaults: dict):
        """Merge recursivo de defaults sem sobrescrever valores existentes."""
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config.get(key), dict):
                self._merge_defaults(config[key], value)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Obtém valor de configuração usando notação de ponto.

        Args:
            key_path: Caminho da chave (ex: 'trading.capital', 'strategy.indicators.rsi_period').
            default: Valor padrão se a chave não existir.

        Returns:
            Valor da configuração ou valor padrão.

        Examples:
            >>> config = ConfigLoader()
            >>> capital = config.get('trading.capital')
            >>> rsi_period = config.get('strategy.indicators.rsi_period', 14)
        """
        if ConfigLoader._config is None:
            self.load()

        keys = key_path.split('.')
        value = ConfigLoader._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Obtém uma seção completa da configuração.

        Args:
            section: Nome da seção (ex: 'trading', 'strategy', 'risk').

        Returns:
            Dicionário com todas as configurações da seção.

        Raises:
            KeyError: Se a seção não existir.
        """
        if ConfigLoader._config is None:
            self.load()

        if section not in ConfigLoader._config:
            raise KeyError(f"Seção '{section}' não encontrada na configuração.")

        return ConfigLoader._config[section]

    def reload(self) -> Dict[str, Any]:
        """
        Recarrega configurações do arquivo.

        Returns:
            Dicionário com todas as configurações recarregadas.
        """
        ConfigLoader._config = None
        return self.load()

    @property
    def all(self) -> Dict[str, Any]:
        """
        Retorna todas as configurações.

        Returns:
            Dicionário completo de configurações.
        """
        if ConfigLoader._config is None:
            self.load()
        return ConfigLoader._config

    @property
    def is_railway(self) -> bool:
        """Verifica se está rodando no Railway."""
        return os.environ.get('RAILWAY_ENVIRONMENT') is not None

    @property
    def is_production(self) -> bool:
        """Verifica se está em ambiente de produção."""
        env = os.environ.get('RAILWAY_ENVIRONMENT', os.environ.get('ENVIRONMENT', 'development'))
        return env.lower() in ('production', 'prod')

    def get_database_url(self) -> str:
        """
        Retorna URL de conexão com banco de dados.
        Prioriza DATABASE_URL (padrão Railway) sobre configurações locais.

        Returns:
            String de conexão com o banco de dados.
        """
        # Railway fornece DATABASE_URL diretamente
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
            return database_url

        # Fallback para SQLite local
        db_name = self.get('database.db_name', 'trades.db')
        return f"sqlite:///{db_name}"


# Instância global (Singleton)
config = ConfigLoader()
