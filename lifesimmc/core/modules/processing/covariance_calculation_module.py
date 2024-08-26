from copy import copy

import torch
from phringe.api import PHRINGE

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.covariance_resource import CovarianceResource
from lifesimmc.util.matrix import cov_inv_sqrt


class CovarianceCalculationModule(BaseModule):
    """Class representation of the base module.

    :param n_config_in: The name of the input configuration resource
    :param n_cov_out: The name of the output covariance resource
    """

    def __init__(self, n_config_in: str, n_cov_out: str):
        """Constructor method.

        :param n_config_in: The name of the input configuration resource
        :param n_cov_out: The name of the output covariance resource
        """
        self.n_config_in = n_config_in
        self.n_cov_out = n_cov_out

    def apply(self, resources: list[BaseResource]) -> CovarianceResource:
        """Calculate the covariance of the data without the planet signal. This is done by generating a new data set
        without a planet. In reality, this could be achieved e.g. by observing a reference star.

        :param resources: The resources to apply the module to
        :return: The resource
        """
        print('Calculating covariance matrix...')

        config_in = self.get_resource_from_name(self.n_config_in)

        simulation = copy(config_in.simulation)
        simulation.has_planet_signal = False

        phringe = PHRINGE()
        phringe.run(
            config_file_path=config_in.config_file_path,
            simulation=simulation,
            instrument=config_in.instrument,
            observation_mode=config_in.observation_mode,
            scene=config_in.scene,
            gpu=self.gpu,
            write_fits=False,
            create_copy=False
        )

        data = phringe.get_data(as_numpy=False)

        cov_out = CovarianceResource(self.n_cov_out)

        cov_out.cov = torch.zeros((data.shape[0], data.shape[1], data.shape[1]))
        cov_out.i_cov_sqrt = torch.zeros((data.shape[0], data.shape[1], data.shape[1]))

        for i in range(len(data)):
            cov_out.cov[i] = data[i].cov()

            cov_out.i_cov_sqrt[i] = torch.tensor(
                cov_inv_sqrt(cov_out.cov[i].cpu().numpy()),
                device=config_in.phringe._director._device
            )

        print('Done')
        return cov_out
