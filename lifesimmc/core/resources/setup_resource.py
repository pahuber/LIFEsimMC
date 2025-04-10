from dataclasses import dataclass

from phringe.core.entities.configuration import Configuration
from phringe.core.entities.instrument import Instrument
from phringe.core.entities.observation import Observation
from phringe.core.entities.scene import Scene
from phringe.main import PHRINGE

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class SetupResource(BaseResource):
    """Class representation of the setup resource.

    Parameters
    ----------
    configuration : Configuration
        The configuration of the simulation.
    instrument : Instrument
        The instrument used for the simulation.
    observation : Observation
        The observation used for the simulation.
    phringe : PHRINGE
        The PHRINGE instance used for the simulation.
    scene : Scene
        The scene used for the simulation.
    """
    configuration: Configuration = None
    instrument: Instrument = None
    observation: Observation = None
    phringe: PHRINGE = None
    scene: Scene = None
