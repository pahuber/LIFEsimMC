from typing import Union

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.spectrum_resource import SpectrumResource
from lifesimmc.util.helpers import Spectrum


class DataGenerationModule(BaseModule):
    """Class representation of the data generation module."""

    def __init__(self, r_config_in: str, r_data_out: str, r_spectrum_out: str, write_to_fits: bool = True,
                 create_copy: bool = True):
        """Constructor method."""
        super().__init__()
        self.config_in = r_config_in
        self.write_to_fits = write_to_fits
        self.create_copy = create_copy
        self.data_out = DataResource(r_data_out)
        self.spectrum_out = SpectrumResource(r_spectrum_out)

    def apply(self, resources: list[BaseResource]) -> Union[DataResource, SpectrumResource]:
        """Use PHRINGE to generate synthetic data.
        """
        print('Generating synthetic data...')

        config = self.get_resource_from_name(self.config_in)

        config.phringe.run(
            config_file_path=config.config_file_path,
            simulation=config.simulation,
            observation_mode=config.observation_mode,
            instrument=config.instrument,
            scene=config.scene,
            gpu=self.gpu,
            write_fits=self.write_to_fits,
            create_copy=self.create_copy
        )

        self.data_out._data = config.phringe.get_data(as_numpy=False)
        spectra = []
        for planet in config.phringe._director._planets:
            spectrum = Spectrum(
                planet.spectral_flux_density,
                None,
                None,
                config.phringe.get_wavelength_bin_centers(as_numpy=False),
                config.phringe.get_wavelength_bin_widths(as_numpy=False)
            )
            spectra.append(spectrum)

        self.spectrum_out.spectra.extend(spectra)
        print('Done')
        return self.data_out, self.spectrum_out
