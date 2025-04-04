from typing import Union

import torch

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.flux_resource import FluxResource


class DataGenerationModule(BaseModule):
    """Class representation of the data generation module.

        :param n_config_in: The name of the input configuration resource
        :param n_data_out: The name of the output data resource
        :param n_flux_out: The name of the output spectrum resource collection
    """

    def __init__(
            self,
            n_config_in: str,
            n_data_out: str,
            n_flux_out: Union[str, tuple[str]]
    ):
        """Constructor method.

        :param n_config_in: The name of the input configuration resource
        :param n_data_out: The name of the output data resource
        :param n_flux_out: The name of the output spectrum resource collection
        """
        super().__init__()
        self.config_in = n_config_in
        self.n_data_out = n_data_out
        self.n_flux_out = n_flux_out

    def apply(self, resources: list[BaseResource]) -> tuple[DataResource, FluxResource]:
        """Use PHRINGE to generate synthetic data.

        :param resources: The resources to apply the module to
        :return: The data and spectrum resources
        """
        print('Generating synthetic data...')

        r_config_in = self.get_resource_from_name(self.config_in)
        r_data_out = DataResource(self.n_data_out)

        diff_counts = r_config_in.phringe.get_diff_counts()
        r_data_out.set_data(diff_counts)

        spectral_irradiance = []
        planet_name = []

        for planet in r_config_in.phringe._scene.planets:
            planet_name.append(planet.name)
            spectral_irradiance.append(torch.tensor(
                planet._spectral_energy_distribution,
                dtype=torch.float32,
                device=self.device)
            )

        r_flux_out = FluxResource(
            name=self.n_flux_out,
            spectral_irradiance=spectral_irradiance,
            wavelength_bin_centers=r_config_in.phringe.get_wavelength_bin_centers(),
            wavelength_bin_widths=r_config_in.phringe.get_wavelength_bin_widths(),
            planet_name=planet_name
        )

        print('Done')
        return r_data_out, r_flux_out
