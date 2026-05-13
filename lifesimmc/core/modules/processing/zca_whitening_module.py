from copy import deepcopy

import numpy as np
import torch
from phringe.main import PHRINGE

from lifesimmc.core.modules.processing.base_transformation_module import BaseTransformationModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.template_resource import TemplateResource
from lifesimmc.core.resources.transformation_resource import TransformationResource


class ZCAWhiteningModule(BaseTransformationModule):
    """Class representation of the ZCA whitening transformation module. This module applied ZCA whitening to the data
    and templates using a covariance matrix based on a calibration star.

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
            n_data_out: str,
            n_transformation_out: str,
            n_template_in: str = None,
            n_template_out: str = None,
            diagonal_only: bool = False
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
        self.n_setup_in = n_setup_in
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_data_out = n_data_out
        self.n_template_out = n_template_out
        self.n_transformation_out = n_transformation_out
        self.diagonal_only = diagonal_only

    def run(self, pipeline_resources: list[BaseResource]) -> tuple[
                                                                 DataResource, TemplateResource, TransformationResource] | \
                                                             tuple[DataResource, TransformationResource]:
        """Apply the module.

        Parameters
        ----------
        pipeline_resources : list[BaseResource]
            List of resources to be processed.

        Returns
        -------
        tuple[DataResource, TemplateResource, TransformationResource]
            Tuple containing the output data resource, template resource, and transformation resource.
        """
        print('Applying ZCA whitening...')

        r_setup_in = self.get_resource_from_name(self.n_setup_in)

        # Generate a reference data set
        phringe = PHRINGE(
            seed=self.seed,
            gpu_index=self.gpu_index,
            grid_size=self.grid_size,
            time_step_size=self.time_step_size,
            device=self.device,
            extra_memory=20
        )

        # Create a copy of the instrument
        instrument_new = deepcopy(r_setup_in.phringe._instrument)
        phringe.set(instrument_new)

        # Set the observation
        observation_new = deepcopy(r_setup_in.phringe._observation)
        phringe.set(observation_new)

        # Remove all planets from the scene to calculate covariance only on noise
        scene_new = deepcopy(r_setup_in.phringe._scene)

        for planet in r_setup_in.phringe._scene.planets:
            scene_new.remove_source(planet.name)

        phringe.set(scene_new)

        # Get the noise reference counts
        noise_ref = phringe.get_counts(kernels=True)

        # Calculate the whitening matrix
        nk = noise_ref.shape[0]
        nl = noise_ref.shape[1]
        nt = noise_ref.shape[2]

        noise_ref = noise_ref.reshape(nk * nl, nt)
        cov = torch.cov(noise_ref)

        U, Svals, _ = torch.linalg.svd(cov)
        w = U @ torch.diag(1 / torch.sqrt(Svals)) @ U.T

        if self.diagonal_only:
            w = torch.diag(torch.diag(w))

        # Apply the whitening matrix to the data
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        r_data_out = DataResource(self.n_data_out)

        data_in = data_in.reshape(nk * nl, nt)
        data_in_white = w @ data_in
        data_in_white = data_in_white.reshape(nk, nl, nt)

        r_data_out.set_data(data_in_white)

        # Apply whitening to templates
        if self.n_template_in and self.n_template_out:
            r_template_in = self.get_resource_from_name(self.n_template_in)
            template_data_in = r_template_in.get_data()

            nk, nl, nt, ng, _ = template_data_in.shape

            template_data_in_flat = template_data_in.reshape(nk * nl, nt, ng, ng)

            template_data_in_white = torch.einsum(
                "ab,btij->atij",
                w,
                template_data_in_flat,
            )

            template_data_in_white = template_data_in_white.reshape(nk, nl, nt, ng, ng)

            r_template_out = TemplateResource(
                name=self.n_template_out,
                grid_coordinates=r_template_in.grid_coordinates
            )
            r_template_out.set_data(template_data_in_white)

        else:
            r_template_out = None

        # Save the whitening transformation
        def zca_whitening_transformation(data):
            """Apply the ZCA whitening transformation."""
            if isinstance(data, np.ndarray):
                w1 = w.cpu().numpy()
            else:
                w1 = w
            nk, nl, nt = data.shape
            data = w1 @ data.reshape(nk * nl, nt)
            data = data.reshape(nk, nl, nt)

            return data

        zca = zca_whitening_transformation

        r_transformation_out = TransformationResource(
            name=self.n_transformation_out,
            transformation=zca
        )

        print('Done')
        if r_template_out is not None:
            return r_data_out, r_template_out, r_transformation_out

        return r_data_out, r_transformation_out
