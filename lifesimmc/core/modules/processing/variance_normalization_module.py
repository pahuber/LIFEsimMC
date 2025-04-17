from itertools import product

import numpy as np
import torch

from lifesimmc.core.modules.processing.base_transformation_module import BaseTransformationModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.template_resource import TemplateResource
from lifesimmc.core.resources.transformation_resource import TransformationResource


class NoiseVarianceNormalizationModule(BaseTransformationModule):
    """Class representation of the ZCA whitening transformation module. This module applied ZCA whitening to the data
    and templates using a covariance matrix based on a calibration star. The properties of this calibration star are
    assumed to be identical to the properties of the target star.

    Parameters
    ----------

    n_setup_in : str
        Name of the input configuration resource.
    n_data_in : str
        Name of the input data resource.
    n_template_in : str
        Name of the input template resource.
    n_data_out : str
        Name of the output data resource.
    n_template_out : str
        Name of the output template resource.
    n_transformation_out : str
        Name of the output transformation resource.
    diagonal_only : bool
        If True, only the diagonal of the covariance matrix is used for whitening. Default is False.
    """

    def __init__(
            self,
            n_setup_in: str,
            n_data_in: str,
            n_template_in: str,
            n_data_out: str,
            n_template_out: str,
            n_transformation_out: str,
            n_planet_params_in: str
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_in : str
            Name of the input configuration resource.
        n_data_in : str
            Name of the input data resource.
        n_template_in : str
            Name of the input template resource.
        n_data_out : str
            Name of the output data resource.
        n_template_out : str
            Name of the output template resource.
        n_transformation_out : str
            Name of the output transformation resource.
        diagonal_only : bool
            If True, only the diagonal of the covariance matrix is used for whitening. Default is False.
        """
        super().__init__()
        self.n_config_in = n_setup_in
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_data_out = n_data_out
        self.n_template_out = n_template_out
        self.n_transformation_out = n_transformation_out
        self.n_planet_params_in = n_planet_params_in

    def apply(self, resources: list[BaseResource]) -> tuple[DataResource, TemplateResource, TransformationResource]:
        """Apply the module.

        Parameters
        ----------
        resources : list[BaseResource]
            List of resources to be processed.

        Returns
        -------
        tuple[DataResource, TemplateResource, TransformationResource]
            Tuple containing the output data resource, template resource, and transformation resource.
        """
        print('Normalizing with noise variance...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        r_template_in = self.get_resource_from_name(self.n_template_in)
        template_data_in = r_template_in.get_data()
        r_data_out = DataResource(self.n_data_out)
        template_counts_white = torch.zeros(template_data_in.shape, device=self.device)
        planet_params_in = self.get_resource_from_name(self.n_planet_params_in) if self.n_planet_params_in else None

        times = r_config_in.phringe.get_time_steps().cpu().numpy()
        wavelengths = r_config_in.phringe.get_wavelength_bin_centers().cpu().numpy()
        wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths().cpu().numpy()

        flux_init = planet_params_in.params[0].sed.cpu().numpy()
        posx_init = planet_params_in.params[0].pos_x
        posy_init = planet_params_in.params[0].pos_y

        model = r_config_in.phringe._get_template_diff_counts(
            times,
            wavelengths,
            wavelength_bin_widths,
            flux_init,
            posx_init,
            posy_init
        )

        data_empty = data_in - torch.tensor(model, device=self.device, dtype=data_in.dtype)

        # Calculate the variance of the data
        data_variance = torch.zeros(data_in.shape[0], data_in.shape[1], 1, device=self.device)

        # Normalize data and templates by variance
        for k in range(data_in.shape[0]):
            data_variance[k] = torch.var(data_empty[k], dim=1, keepdim=True) ** 0.5
            data_in[k] = data_in[k] / data_variance[k]

            for i, j in product(range(template_data_in.shape[-2]), range(template_data_in.shape[-1])):
                template_counts_white[k, :, :, i, j] = template_data_in[k, :, :, i, j] / data_variance[k]

        # Create the output resources
        r_template_out = TemplateResource(
            name=self.n_template_out,
            grid_coordinates=r_template_in.grid_coordinates
        )
        r_data_out.set_data(data_in)
        r_template_out.set_data(template_counts_white)

        # Save the normalization transformation
        def normalization_transformation(data):
            if isinstance(data, np.ndarray):
                i2 = data_variance.cpu().numpy()
            else:
                i2 = data_variance
            for l in range(data.shape[0]):
                data[l] = data[l] / i2[l]
            return data

        r_transformation_out = TransformationResource(
            name=self.n_transformation_out,
            transformation=normalization_transformation
        )

        print('Done')
        return r_data_out, r_template_out, r_transformation_out
