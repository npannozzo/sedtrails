"""
Configuration Controller
========================

Reads simulation configuration files, applies default configuration values,
and provides configurations to other components.
"""

import os
import importlib.resources as pkg_resources
from abc import ABC, abstractmethod
from sedtrails.configuration_interface.validator import YAMLConfigValidator
from typing import Dict, Any


class Controller(ABC):
    """
    Abstract base class for configuration controllers.
    """

    # TODO: Something to keep in mind in this class:
    # if we can and it is convenient to have separted methods to retrieve parts of the configuration
    # that go to the particle tracer and the transport converter. For example, if it is convenient to
    # have a method that returns the configuration values that are only relevant for the particle tracer.

    @abstractmethod
    def read_config(self, config_file: str) -> None:
        """
        Reads the configuration file and applies default values.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Returns the current configuration.
        """
        pass

    @abstractmethod
    def get(self, keys: str, default=None) -> Any:
        """
        Retrieves a value from the configuration data using a dot-separated key.

        """

        pass


class ConfigurationController(Controller):
    """
    A controller for managing the configuration of the simulation.
    """

    def __init__(self):
        self.config_data = {}
        self._config_file: str | None = None

    def read_config(self, config_file: str) -> None:
        """
        Reads the configuration file, validated its contents and set it.

        Parameters
        ----------
        config_file : str
            The path to the configuration file to read.
        """

        if self._config_file is None:
            self._config_file = config_file

        with pkg_resources.as_file(
            pkg_resources.files('sedtrails.config').joinpath('main.schema.json')
        ) as config_schema:
            if not os.path.exists(config_schema):
                raise FileNotFoundError(
                    f'SedTRAILS validation schema file not found: {config_schema}. \
                    Input cannot be parsed.'
                )

            validator = YAMLConfigValidator(str(config_schema))
            self.config_data = validator.validate_yaml(config_file)

        return None

    def get_config(self) -> dict:
        """
        Returns the current configuration.

        Returns
        -------
        dict
            The current configuration.
        """

        if not self.config_data:
            self.read_config(self._config_file)

        return self.config_data

    def get(self, keys: str, default=None) -> Any:
        """
        Retrieves a value from the configuration data using a dot-separated key.

        Parameters
        ----------
        key : str
            The dot-separated key to retrieve from the configuration.
        default : any, optional
            The default value to return if the key is not found.

        Returns
        -------
        any
            The value associated with the key or the default value. None if the key is not found.

        Example
        --------
        >>> controller = ConfigurationController()
        >>> controller.read_config('path/to/config.yaml')
        >>> value = controller.get('some.nested.key')
        >>> print(value)  # Outputs the value for 'some.nested.key'

        """

        from copy import deepcopy  # Lazy import

        config_data = deepcopy(self.get_config())

        for key in keys.split('.'):
            config_data = config_data.get(key, default)
            if not isinstance(config_data, dict):
                return config_data
        return config_data


if __name__ == '__main__':
    # Example usage
    controller = ConfigurationController()
    try:
        controller.read_config('/Users/mgarciaalvarez/devel/sedtrails/examples/config.example.yaml')
        config = controller.get_config()
        # print(config)
    except Exception as e:
        print(f'Error: {e}')

    timestep = controller.get('time.timestep')
    print(f'Timestep: {timestep}')
