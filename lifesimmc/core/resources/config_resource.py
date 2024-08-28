from dataclasses import dataclass
from pathlib import Path

from phringe.api import PHRINGE
from phringe.core.entities.instrument import Instrument
from phringe.core.entities.observation_mode import ObservationMode
from phringe.core.entities.scene import Scene
from phringe.core.entities.simulation import Simulation

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class ConfigResource(BaseResource):
    """Class representation of the configuration resource.

    :param config_file_path: The path to the configuration file
    :param instrument: The instrument
    :param observation_mode: The observation mode
    :param phringe: The PHRINGE object
    :param scene: The scene
    :param simulation: The simulation
    """
    config_file_path: Path = None
    instrument: Instrument = None
    observation_mode: ObservationMode = None
    phringe: PHRINGE = None
    scene: Scene = None
    simulation: Simulation = None
