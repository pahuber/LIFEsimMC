from copy import copy
from itertools import product

import numpy as np
import torch
from phringe.main import PHRINGE
from scipy.linalg import sqrtm, pinv
from tqdm import tqdm

from lifesimmc.core.modules.processing.base_transformation_module import BaseTransformationModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.template_resource import TemplateResource
from lifesimmc.core.resources.transformation_resource import TransformationResource


class ZCAWhiteningCalibrationStarModule(BaseTransformationModule):
    """Class representation of the ZCA whitening transformation module. This module applied ZCA whitening to the data
    and templates using a covariance matrix based on a calibration star. The properties of this calibration star are
    assumed to be identical to the properties of the target star.

    Parameters
    ----------

    n_config_in : str
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
            n_config_in: str,
            n_data_in: str,
            n_template_in: str,
            n_data_out: str,
            n_template_out: str,
            n_transformation_out: str,
            diagonal_only: bool = False
    ):
        """Constructor method.

        Parameters
        ----------
        n_config_in : str
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
        self.n_config_in = n_config_in
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_data_out = n_data_out
        self.n_template_out = n_template_out
        self.n_transformation_out = n_transformation_out
        self.diagonal_only = diagonal_only

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
        print('Applying ZCA whitening...')

        config_in = self.get_resource_from_name(self.n_config_in)

        # Generate a calibration star data set
        if self.seed is None:
            seed = torch.randint(0, 2 ** 31, (1,)).item()
        else:
            seed = (self.seed + 1) * 2
            if seed > 2 ** 31:
                seed = seed // 10

        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        phringe = PHRINGE(
            seed=self.seed,
            gpu_index=self.gpu_index,
            grid_size=self.grid_size,
            time_step_size=self.time_step_size,
            device=self.device,
            extra_memory=20
        )

        phringe.set(config_in.instrument)
        phringe.set(config_in.observation)

        # Remove all planets from the scene to calculate covariance only on noise
        scene = copy(config_in.scene)
        scene.planets = []
        phringe.set(config_in.scene)
        diff_counts = phringe.get_diff_counts()

        # Calculate the whitening matrix
        cov = torch.zeros(
            (diff_counts.shape[0], diff_counts.shape[1], diff_counts.shape[1]),
            device=self.device
        )
        i_cov_sqrt = torch.zeros(
            (diff_counts.shape[0], diff_counts.shape[1], diff_counts.shape[1]),
            device=self.device
        )

        for i in range(len(diff_counts)):
            cov[i] = torch.cov(diff_counts[i])

            if self.diagonal_only:
                cov[i] = torch.diag(torch.diag(cov[i]))

            i_cov_sqrt[i] = torch.tensor(sqrtm(pinv(cov[i].cpu().numpy())), device=self.device)

        # Apply the whitening matrix to the data and templates
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        r_template_in = self.get_resource_from_name(self.n_template_in)
        template_data_in = r_template_in.get_data()
        r_data_out = DataResource(self.n_data_out)

        if data_in is not None:
            for i in range(data_in.shape[0]):
                data_in[i] = i_cov_sqrt[i] @ data_in[i]
            r_data_out.set_data(data_in)

        template_counts_white = torch.zeros(template_data_in.shape, device=self.device)

        for i, j in tqdm(
                product(range(template_data_in.shape[-2]), range(template_data_in.shape[-1])),
                total=template_data_in.shape[-2] * template_data_in.shape[-1]
        ):
            for k in range(template_data_in.shape[0]):
                template_counts_white[k, :, :, i, j] = i_cov_sqrt[k] @ template_data_in[k, :, :, i, j]

        r_template_out = TemplateResource(
            name=self.n_template_out,
            grid_coordinates=r_template_in.grid_coordinates
        )
        r_template_out.set_data(template_counts_white)

        # Save the whitening transformation
        def zca_whitening_transformation(data):
            """Apply the ZCA whitening transformation."""
            if isinstance(data, np.ndarray):
                i2 = i_cov_sqrt.cpu().numpy()
            else:
                i2 = i_cov_sqrt
            for l in range(data.shape[0]):
                data[l] = i2[l] @ data[l]
            return data

        zca = zca_whitening_transformation

        r_transformation_out = TransformationResource(
            name=self.n_transformation_out,
            transformation=zca
        )

        print('Done')
        return r_data_out, r_template_out, r_transformation_out
