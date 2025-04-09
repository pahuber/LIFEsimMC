from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.planet_params_resource import PlanetParamsResource, PlanetParams


class DataGenerationModule(BaseModule):
    """Class representation of the data generation module.

        Parameters
        ----------
        n_config_in : str
            The name of the input configuration resource.
        n_data_out : str
            The name of the output data resource.
        n_planet_params_out : str
            The name of the output planet parameters resource.
    """

    def __init__(
            self,
            n_config_in: str,
            n_data_out: str,
            n_planet_params_out: str
    ):
        """Constructor method.

        Parameters
        ----------
        n_config_in : str
            The name of the input configuration resource.
        n_data_out : str
            The name of the output data resource.
        n_planet_params_out : str
            The name of the output planet parameters resource.
        """
        super().__init__()
        self.config_in = n_config_in
        self.n_data_out = n_data_out
        self.n_planet_params_out = n_planet_params_out

    def apply(self, resources: list[BaseResource]) -> tuple[DataResource, PlanetParamsResource]:
        """Use PHRINGE to generate synthetic data.

        Parameters
        ----------
        resources : list[BaseResource]
            List of resources to be used in the module.

        Returns
        -------
        tuple[DataResource, PlanetParamsResource]
            Tuple containing the output data resource and planet parameters resource.
        """
        print('Generating synthetic data...')

        r_config_in = self.get_resource_from_name(self.config_in)
        r_data_out = DataResource(self.n_data_out)

        diff_counts = r_config_in.phringe.get_diff_counts()
        r_data_out.set_data(diff_counts)
        r_planet_params_out = PlanetParamsResource(
            name=self.n_planet_params_out,
        )

        for planet in r_config_in.phringe._scene.planets:
            planet_params = PlanetParams(
                name=planet.name,
                sed_wavelength_bin_centers=r_config_in.phringe.get_wavelength_bin_centers(),
                sed_wavelength_bin_widths=r_config_in.phringe.get_wavelength_bin_widths(),
                sed=r_config_in.phringe.get_source_spectrum(planet.name)
            )
            r_planet_params_out.params.append(planet_params)

        print('Done')
        return r_data_out, r_planet_params_out
