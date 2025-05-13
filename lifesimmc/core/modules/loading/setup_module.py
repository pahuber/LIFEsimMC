from typing import overload

import torch
from phringe.core.entities.configuration import Configuration
from phringe.core.entities.instrument import Instrument
from phringe.core.entities.observation import Observation
from phringe.core.entities.scene import Scene
from phringe.main import PHRINGE

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.planet_params_resource import PlanetParamsResource, PlanetParams
from lifesimmc.core.resources.setup_resource import SetupResource


class SetupModule(BaseModule):
    """Class representation of the configuration loader module.

    Parameters
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
            n_planet_params_out: str,
            configuration: Configuration
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_out : str
            The name of the output configuration resource
        n_planet_params_out : str
            The name of the output planet parameters resource
        configuration : Configuration
            The configuration object
        """
        ...

    @overload
    def __init__(
            self,
            n_setup_out: str,
            n_planet_params_out: str,
            observation: Observation,
            instrument: Instrument,
            scene: Scene
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_out : str
            The name of the output configuration resource
        n_planet_params_out : str
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
            n_planet_params_out: str,
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
        n_planet_params_out : str
            The name of the output planet parameters resource
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
        self.n_config_out = n_setup_out
        self.n_planet_params_out = n_planet_params_out
        self.configuration = configuration
        self.observation = observation
        self.instrument = instrument
        self.scene = scene

    def apply(self, resources: list[SetupResource]) -> tuple[SetupResource, PlanetParamsResource]:
        """Load the configuration file.

        Parameters
        ----------
        resources : list[SetupResource]
            List of resources.

        Returns
        -------
        SetupResource
            The configuration resource.
        """
        print('Loading configuration...')
        phringe = PHRINGE(
            seed=self.seed,
            gpu_index=self.gpu_index,
            grid_size=self.grid_size,
            time_step_size=self.time_step_size,
            device=self.device,
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

        r_config_out = SetupResource(
            name=self.n_config_out,
            phringe=phringe,
            configuration=self.configuration,
            instrument=phringe._instrument,
            observation=phringe._observation,
            scene=phringe._scene,
        )

        r_planet_params_out = PlanetParamsResource(
            name=self.n_planet_params_out,
        )

        for planet in phringe._scene.planets:
            # Get planet position from the only pixel in the sky brightness distirbution that is not zero and then from the sky coordinates map at that position the coordinate values
            sky_brightness_distribution = planet._sky_brightness_distribution

            # If planet has orbital motion, use only initial time step
            if planet.has_orbital_motion:
                sky_brightness_distribution = sky_brightness_distribution[0]

            non_zero_indices = torch.nonzero(sky_brightness_distribution[0])
            sky_coordinates = planet._sky_coordinates

            # If planet has orbital motion, i.e. sky_coordinates change with time, then use the first time step
            if planet.has_orbital_motion:
                sky_coordinates = sky_coordinates[:, 0]

            pos_x = sky_coordinates[0][non_zero_indices[0][0], non_zero_indices[0][1]].item()
            pos_y = sky_coordinates[1][non_zero_indices[0][0], non_zero_indices[0][1]].item()

            planet_params = PlanetParams(
                name=planet.name,
                sed_wavelength_bin_centers=phringe.get_wavelength_bin_centers(),
                sed_wavelength_bin_widths=phringe.get_wavelength_bin_widths(),
                sed=phringe.get_source_spectrum(planet.name),
                pos_x=pos_x,
                pos_y=pos_y,
            )
            r_planet_params_out.params.append(planet_params)

        print('Done')
        return r_config_out, r_planet_params_out
