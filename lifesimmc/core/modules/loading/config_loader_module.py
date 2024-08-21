from pathlib import Path
from typing import overload

from phringe.api import PHRINGE
from phringe.core.entities.instrument import Instrument
from phringe.core.entities.observation_mode import ObservationMode
from phringe.core.entities.scene import Scene
from phringe.core.entities.simulation import Simulation
from phringe.io.utils import load_config

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.config_resource import ConfigResource


class ConfigLoaderModule(BaseModule):
    """Class representation of the configuration loader module."""

    @overload
    def __init__(self, config_out: str, config_file_path: Path):
        ...

    @overload
    def __init__(
            self,
            r_config_out: str,
            simulation: Simulation,
            observation_mode: ObservationMode,
            instrument: Instrument,
            scene: Scene
    ):
        ...

    def __init__(
            self,
            r_config_out: str,
            config_file_path: Path = None,
            simulation: Simulation = None,
            observation_mode: ObservationMode = None,
            instrument: Instrument = None,
            scene: Scene = None
    ):
        """Constructor method.

        :param config_file_path: The path to the configuration file
        :param simulation: The simulation object
        :param observation_mode: The observation mode object
        :param instrument: The instrument object
        """
        super().__init__()
        self.r_config_out = ConfigResource(name=r_config_out)
        self.config_file_path = config_file_path
        self.simulation = simulation
        self.observation_mode = observation_mode
        self.instrument = instrument
        self.scene = scene

    def apply(self, resources: list[ConfigResource]) -> ConfigResource:
        """Load the configuration file.
        """
        print('Loading configuration...')
        config_dict = load_config(self.config_file_path) if self.config_file_path else None

        simulation = Simulation(**config_dict['simulation']) if not self.simulation else self.simulation
        instrument = Instrument(**config_dict['instrument']) if not self.instrument else self.instrument
        observation_mode = ObservationMode(
            **config_dict['observation_mode']
        ) if not self.observation_mode else self.observation_mode
        scene = Scene(**config_dict['scene']) if not self.scene else self.scene

        self.r_config_out.config_file_path = self.config_file_path
        self.r_config_out.simulation = simulation
        self.r_config_out.observation_mode = observation_mode
        self.r_config_out.instrument = instrument
        self.r_config_out.scene = scene
        self.r_config_out.phringe = PHRINGE()

        print('Done')
        return self.r_config_out
