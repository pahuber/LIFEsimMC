from copy import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.linalg import pinv
from phringe.api import PHRINGE
from scipy.linalg import sqrtm

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.covariance_resource import CovarianceResource


# from numpy.linalg import pinv


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
        """Calculate the covariance of the data without the planet signal. This is done by generating a new data set
        without a planet. In reality, this could be achieved e.g. by observing a reference star.

        :param resources: The resources to apply the module to
        :return: The resource
        """
        print('Calculating covariance matrix...')

        config_in = self.get_resource_from_name(self.n_config_in)

        simulation = copy(config_in.simulation)
        simulation.has_planet_signal = False

        # Generate random number and update torcha nd numpy seeds
        if self.seed is None:
            seed = torch.randint(0, 2 ** 32, (1,)).item()
        else:
            seed = (self.seed + 1) * 2
            if seed > 2 ** 32:
                seed = seed // 10

        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        phringe = PHRINGE()
        phringe.run(
            # config_file_path=config_in.config_file_path,
            simulation=simulation,
            instrument=config_in.instrument,
            observation_mode=config_in.observation_mode,
            scene=config_in.scene,
            seed=seed,
            # not the same as in data generation, but predictable
            gpu=self.gpu,
            write_fits=False,
            create_copy=False,
            extra_memory=20
        )

        data = phringe.get_data(as_numpy=False)
        cov_out = CovarianceResource(self.n_cov_out)
        cov_out.cov = torch.zeros((data.shape[0], data.shape[1], data.shape[1]))
        cov_out.i_cov_sqrt = torch.zeros((data.shape[0], data.shape[1], data.shape[1]))

        for i in range(len(data)):
            cov_out.cov[i] = torch.cov(data[i])

            # plt.imshow(cov_out.cov[i].cpu().numpy())
            # plt.colorbar()
            # plt.show()

            if self.diagonal_only:
                cov_out.cov[i] = torch.diag(torch.diag(cov_out.cov[i]))

            cov_out.i_cov_sqrt[i] = torch.tensor(
                sqrtm(pinv(cov_out.cov[i].cpu().numpy())),
                device=config_in.phringe._director._device
            )

        plt.imshow(cov_out.i_cov_sqrt[0].cpu().numpy())
        plt.colorbar()
        plt.show()

        print('Done')
        return cov_out
