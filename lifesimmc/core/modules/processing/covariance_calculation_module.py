from copy import copy

import numpy as np
import torch
from numpy.linalg import pinv
from phringe.main import PHRINGE
from scipy.linalg import sqrtm

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.covariance_resource import CovarianceResource


class CovarianceCalculationModule(BaseModule):
    """Class representation of the base module.

    :param n_config_in: The name of the input configuration resource
    :param n_cov_out: The name of the output covariance resource
    """

    def __init__(
            self,
            n_config_in: str,
            n_cov_out: str,
            diagonal_only: bool = False
    ):
        """Constructor method.

        :param n_config_in: The name of the input configuration resource
        :param n_cov_out: The name of the output covariance resource
        :param diagonal_only: Whether to return the diagonal part of the covariance matrix only
        """
        self.n_config_in = n_config_in
        self.n_cov_out = n_cov_out
        self.diagonal_only = diagonal_only

    def apply(self, resources: list[BaseResource]) -> tuple:
        """Calculate the covariance of the diff_counts without the planet signal. This is done by generating a new diff_counts set
        without a planet. In reality, this could be achieved e.g. by observing a reference star.

        :param resources: The resources to apply the module to
        :return: The resource
        """
        print('Calculating covariance matrix...')

        config_in = self.get_resource_from_name(self.n_config_in)
        cov_out = CovarianceResource(self.n_cov_out)

        # Generate random number and update torcha nd numpy seeds
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

        cov_out.cov = torch.zeros((diff_counts.shape[0], diff_counts.shape[1], diff_counts.shape[1]),
                                  device=self.device)
        cov_out.i_cov_sqrt = torch.zeros((diff_counts.shape[0], diff_counts.shape[1], diff_counts.shape[1]),
                                         device=self.device)

        for i in range(len(diff_counts)):
            cov_out.cov[i] = torch.cov(diff_counts[i])

            if self.diagonal_only:
                cov_out.cov[i] = torch.diag(torch.diag(cov_out.cov[i]))

            cov_out.i_cov_sqrt[i] = torch.tensor(
                sqrtm(pinv(cov_out.cov[i].cpu().numpy())),
                device=self.device
            )

        print('Done')
        return cov_out
