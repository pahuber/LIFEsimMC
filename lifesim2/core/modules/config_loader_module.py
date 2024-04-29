from pathlib import Path
from typing import overload

from phringe.core.entities.observation import Observation
from phringe.core.entities.observatory.observatory import Observatory
from phringe.core.entities.settings import Settings
from phringe.io.utils import get_dict_from_path

from lifesim2.core.base_module import BaseModule
from lifesim2.core.context import Context


class ConfigLoaderModule(BaseModule):
    """Class representation of the configuration loader module."""

    @overload
    def __init__(self, config_file_path: Path):
        ...

    @overload
    def __init__(self, settings: Settings, observatory: Observatory, observation: Observation):
        ...

    def __init__(
            self,
            config_file_path: Path = None,
            settings: Settings = None,
            observatory: Observatory = None,
            observation: Observation = None
    ):
        """Constructor method.

        :param config_file_path: The path to the configuration file
        :param settings: The settings object
        :param observatory: The observatory object
        :param observation: The observation object
        """
        self.config_file_path = config_file_path
        self.settings = settings
        self.observatory = observatory
        self.observation = observation

    def apply(self, context: Context) -> Context:
        """Load the configuration file.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        config_dict = get_dict_from_path(self.config_file_path) if self.config_file_path else None

        settings = Settings(**config_dict['settings']) if not self.settings else self.settings
        observatory = Observatory(**config_dict['observatory']) if not self.observatory else self.observatory
        observation = Observation(**config_dict['observation']) if not self.observation else self.observation

        context.config_file_path = self.config_file_path
        context.settings = settings
        context.observatory = observatory
        context.observation = observation
        return context
