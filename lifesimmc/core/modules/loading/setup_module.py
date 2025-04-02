from pathlib import Path
from typing import overload

from phringe.entities.configuration import Configuration
from phringe.entities.instrument import Instrument
from phringe.entities.observation import Observation
from phringe.entities.scene import Scene
from phringe.main import PHRINGE

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.config_resource import ConfigResource


# from phringe.core.entities.instrument import Instrument
# from phringe.core.entities.observation_mode import ObservationMode
# from phringe.core.entities.scene import Scene
# from phringe.core.entities.simulation import Simulation
# from phringe.io.utils import load_config


class SetupModule(BaseModule):
    """Class representation of the configuration loader module.

        :param n_config_out: The name of the output configuration resource
        :param config_file_path: The path to the configuration file
        :param simulation: The simulation object
        :param observation: The observation mode object
        :param instrument: The instrument object
        :param scene: The scene object
    """

    @overload
    def __init__(self, n_config_out: str, config_file_path: Path):
        ...

    @overload
    def __init__(
            self,
            n_config_out: str,
            observation: Observation,
            instrument: Instrument,
            scene: Scene
    ):
        ...

    def __init__(
            self,
            n_config_out: str,
            config_file_path: Path = None,
            observation: Observation = None,
            instrument: Instrument = None,
            scene: Scene = None
    ):
        """Constructor method.

        :param n_config_out: The name of the output configuration resource
        :param config_file_path: The path to the configuration file
        :param simulation: The simulation object
        :param observation: The observation mode object
        :param instrument: The instrument object
        :param scene: The scene object
        """
        super().__init__()
        self.n_config_out = n_config_out
        self.config_file_path = config_file_path
        self.observation = observation
        self.instrument = instrument
        self.scene = scene

    def apply(self, resources: list[ConfigResource]) -> ConfigResource:
        """Load the configuration file.

        :param resources: The resources to apply the module to
        :return: The configuration resource
        """
        print('Loading configuration...')
        phringe = PHRINGE(
            seed=self.seed,
            gpu_index=self.gpu_index,
            grid_size=self.grid_size,
            time_step_size=self.time_step_size,
            device=self.device,
            extra_memory=20
        )

        if self.config_file_path:
            config = Configuration(path=self.config_file_path)
            phringe.set(config)

        if self.instrument:
            phringe.set(self.instrument)

        if self.observation:
            phringe.set(self.observation)

        if self.scene:
            phringe.set(self.scene)

        r_config_out = ConfigResource(
            name=self.n_config_out,
            config_file_path=self.config_file_path,
            phringe=phringe,
            instrument=phringe._instrument,
            observation=phringe._observation,
            scene=phringe._scene,
        )

        print('Done')
        return r_config_out
