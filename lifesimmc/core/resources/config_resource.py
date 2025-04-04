from dataclasses import dataclass
from pathlib import Path

from phringe.entities.instrument import Instrument
from phringe.entities.observation import Observation
from phringe.entities.scene import Scene
from phringe.main import PHRINGE

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class ConfigResource(BaseResource):
    """Class representation of the configuration resource.

    :param configuration: The path to the configuration file
    :param instrument: The instrument
    :param observation: The observation mode
    :param phringe: The PHRINGE object
    :param scene: The scene
    """
    configuration: Path = None
    instrument: Instrument = None
    observation: Observation = None
    phringe: PHRINGE = None
    scene: Scene = None
