from typing import Union

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.flux_resource import FluxResource, FluxResourceCollection


class DataGenerationModule(BaseModule):
    """Class representation of the data generation module.

        :param n_config_in: The name of the input configuration resource
        :param n_data_out: The name of the output data resource
        :param n_flux_out: The name of the output spectrum resource collection
        :param write_to_fits: Whether to write the data to a FITS file
        :param create_copy: Whether to create a copy of the configuration and spectrum files
    """

    def __init__(
            self,
            n_config_in: str,
            n_data_out: str,
            n_flux_out: Union[str, tuple[str]],
            write_to_fits: bool = True,
            create_copy: bool = True
    ):
        """Constructor method.

        :param n_config_in: The name of the input configuration resource
        :param n_data_out: The name of the output data resource
        :param n_flux_out: The name of the output spectrum resource collection
        :param write_to_fits: Whether to write the data to a FITS file
        :param create_copy: Whether to create a copy of the configuration and spectrum files
        """
        super().__init__()
        self.config_in = n_config_in
        self.n_data_out = n_data_out
        self.n_flux_out = n_flux_out
        self.write_to_fits = write_to_fits
        self.create_copy = create_copy

    def apply(self, resources: list[BaseResource]) -> tuple[DataResource, FluxResourceCollection]:
        """Use PHRINGE to generate synthetic data.

        :param resources: The resources to apply the module to
        :return: The data and spectrum resources
        """
        print('Generating synthetic data...')

        r_config_in = self.get_resource_from_name(self.config_in)

        r_config_in.phringe.run(
            config_file_path=r_config_in.config_file_path,
            simulation=r_config_in.simulation,
            observation_mode=r_config_in.observation_mode,
            instrument=r_config_in.instrument,
            scene=r_config_in.scene,
            seed=self.seed,
            gpu=self.gpu,
            write_fits=self.write_to_fits,
            create_copy=self.create_copy,
            extra_memory=20
        )

        r_data_out = DataResource(self.n_data_out)
        r_data_out.set_data(r_config_in.phringe.get_data(as_numpy=False))

        rc_flux_out = FluxResourceCollection(
            name=self.n_flux_out,
            description='Collection of SpectrumResources; one for each planet in the scene'
        )

        for planet in r_config_in.phringe._director._planets:
            rc_flux_out.collection.append(
                FluxResource(
                    name='',
                    spectral_irradiance=planet.spectral_flux_density,
                    wavelength_bin_centers=r_config_in.phringe.get_wavelength_bin_centers(as_numpy=False),
                    wavelength_bin_widths=r_config_in.phringe.get_wavelength_bin_widths(as_numpy=False),
                    planet_name=planet.name
                )
            )

        print('Done')
        return r_data_out, rc_flux_out
