"""
Carregador centralizado de configurações para Lobo IA.
"""

import yaml
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    Carrega e gerencia configurações do sistema.
    Implementa Singleton pattern para acesso global.
    """

    _instance: Optional['ConfigLoader'] = None
    _config: Optional[Dict[str, Any]] = None

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
        Carrega configurações do arquivo YAML.

        Returns:
            Dicionário com todas as configurações.

        Raises:
            FileNotFoundError: Se o arquivo de configuração não existir.
            yaml.YAMLError: Se houver erro ao parsear o YAML.
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                ConfigLoader._config = yaml.safe_load(f)
                return ConfigLoader._config
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Arquivo de configuração '{self.config_file}' não encontrado. "
                "Certifique-se de que config.yaml existe no diretório raiz."
            )
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Erro ao parsear arquivo de configuração: {e}"
            )

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


# Instância global (Singleton)
config = ConfigLoader()
