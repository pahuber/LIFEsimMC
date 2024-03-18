from itertools import product
from pathlib import Path

import numpy as np
from phringe.io.utils import get_dict_from_path
from phringe.phringe import PHRINGE

from lifesim2.core.base_module import BaseModule
from lifesim2.core.context import Context


class TemplateGenerationModule(BaseModule):
    """Class representation of the template generation module."""

    def __init__(
            self,
            config_file_path: Path,
            exoplanetary_system_file_path: Path,
            spectrum_tuple: Path = None,
            write_to_fits: bool = True,
            create_copy: bool = True,
            output_path: Path = Path("."),
    ):
        """Constructor method."""
        self.config_file_path = config_file_path
        self.exoplanetary_system_file_path = exoplanetary_system_file_path
        self.spectrum_tuple = spectrum_tuple
        self.write_to_fits = write_to_fits
        self.create_copy = create_copy
        self.output_path = output_path

    def apply(self, context) -> Context:
        """Apply the module.

        :param context: The context object of the pipelines
        :return: The (updated) context object
        """
        config_dict = get_dict_from_path(self.config_file_path)
        system_dict = get_dict_from_path(self.exoplanetary_system_file_path)

        # Turn off noise sources so the scene.get_all_sources() only returns the planets in the data generator module
        # and the intensity response is ideal
        config_dict['settings']['has_stellar_leakage'] = False
        config_dict['settings']['has_local_zodi_leakage'] = False
        config_dict['settings']['has_exozodi_leakage'] = False
        config_dict['settings']['has_amplitude_perturbations'] = False
        config_dict['settings']['has_phase_perturbations'] = False
        config_dict['settings']['has_polarization_perturbations'] = False

        # For each planet in the scene generate the template data individually, i.e. with only one planet per data set,
        # since multiple planets per template don't make a lot of sense for template matching

        grid_size = config_dict['settings']['grid_size']

        system_dict_copy = system_dict.copy()

        for planet_dict in system_dict['planets']:
            system_dict_copy['planets'] = [planet_dict]

            # Swipe the planet position through every point in the grid and generate the data for each position
            for index_x, index_y in product(range(grid_size), range(grid_size)):
                phringe = PHRINGE()
                phringe.run_with_dict(
                    config_dict=config_dict,
                    exoplanetary_system_dict=system_dict_copy,
                    spectrum_tuple=self.spectrum_tuple,
                    output_dir=self.output_path,
                    write_fits=self.write_to_fits,
                    create_copy=self.create_copy,
                    generate_separate=False
                )

                data = phringe.get_data()
                data /= np.sqrt(np.mean(data ** 2, axis=2))
                a = 0
