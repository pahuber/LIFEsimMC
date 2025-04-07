from dataclasses import dataclass

from phringe.core.entities.configuration import Configuration
from phringe.core.entities.instrument import Instrument
from phringe.core.entities.observation import Observation
from phringe.core.entities.scene import Scene
from phringe.main import PHRINGE

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class ConfigResource(BaseResource):
    """Class representation of the configuration resource.

    :param phringe: The PHRINGE object
    :param configuration: The configuration
    :param instrument: The instrument
    :param observation: The observation mode
    :param scene: The scene
    """
    configuration: Configuration = None
    instrument: Instrument = None
    observation: Observation = None
    phringe: PHRINGE = None
    scene: Scene = None
