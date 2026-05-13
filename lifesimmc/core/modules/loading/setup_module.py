from typing import overload

from lifesimmc.core.resources.planet_params_resource import PlanetParamsResource, PlanetParams
from phringe.core.configuration import Configuration
from phringe.core.instrument import Instrument
from phringe.core.observation import Observation
from phringe.core.scene import Scene
from phringe.main import PHRINGE

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.planet_resource import PlanetResource
from lifesimmc.core.resources.resource_collection import ResourceCollection
from lifesimmc.core.resources.setup_resource import SetupResource


class SetupModule(BaseModule):
    """Class representation of the configuration loader module.

    Attributes
    ----------
    n_setup_out : str
        Name of the output configuration resource.
    configuration : Configuration
        Configuration object.
    observation : Observation
        Observation object.
    instrument : Instrument
        Instrument object.
    scene : Scene
        Scene object.
    """

    @overload
    def __init__(
            self,
            n_setup_out: str,
            n_planets_out: str,
            configuration: Configuration
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_out : str
            The name of the output configuration resource
        n_planets_out : str
            The name of the output planet parameters resource
        configuration : Configuration
            The configuration object
        """
        ...

    @overload
    def __init__(
            self,
            n_setup_out: str,
            n_planets_out: str,
            observation: Observation,
            instrument: Instrument,
            scene: Scene
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_out : str
            The name of the output configuration resource
        n_planets_out : str
            The name of the output planet parameters resource
        observation : Observation
            The observation mode object
        instrument : Instrument
            The instrument object
        scene : Scene
            The scene object
        """
        ...

    def __init__(
            self,
            n_setup_out: str,
            n_planets_out: str,
            configuration: Configuration = None,
            observation: Observation = None,
            instrument: Instrument = None,
            scene: Scene = None
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_out : str
            The name of the output configuration resource
        n_planets_out : str
            The name of the output planet resource collection
        configuration : Configuration
            The configuration object
        observation : Observation
            The observation mode object
        instrument : Instrument
            The instrument object
        scene : Scene
            The scene object
        """
        super().__init__()
        self.n_setup_out = n_setup_out
        self.n_planets_out = n_planets_out
        self.configuration = configuration
        self.observation = observation
        self.instrument = instrument
        self.scene = scene

    def run(self, pipeline_resources: list[SetupResource]) -> tuple[SetupResource, ResourceCollection[PlanetResource]]:
        print('Loading setup...')
        phringe = PHRINGE(
            seed=self.seed,
            gpu_index=self.gpu_index,
            device=self.device,
            grid_size=self.grid_size,
            time_step_size=self.time_step_size,
            extra_memory=10
        )

        if self.configuration:
            phringe.set(self.configuration)

        if self.instrument:
            phringe.set(self.instrument)

        if self.observation:
            phringe.set(self.observation)

        if self.scene:
            phringe.set(self.scene)

        r_setup_out = SetupResource(
            name=self.n_setup_out,
            phringe=phringe,
        )

        r_planet_out = ResourceCollection[PlanetParamsResource](name=self.n_planets_out)

        for planet in phringe._scene.planets:
            planet_resource = PlanetResource(name=planet.name, planet=planet)
            r_planet_out.collection.append(planet_resource)

        print('Done')
        return r_setup_out, r_planet_out
