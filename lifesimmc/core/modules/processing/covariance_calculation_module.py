from copy import copy

import numpy as np
import torch
from phringe.api import PHRINGE

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.covariance_resource import CovarianceResource


class CovarianceCalculationModule(BaseModule):
    """Class representation of the base module."""

    def __init__(self, config_in: str, cov_out: str):
        """Constructor method."""
        self.config_in = config_in
        self.cov_out = CovarianceResource(cov_out)

    def apply(self, resources: list[BaseResource]) -> CovarianceResource:
        """Calculate the covariance of the data without the planet signal. This is done by generating a new data set
        without a planet. In reality, this could be achieved e.g. by observing a reference star.
        """
        print('Calculating covariance matrix...')

        config = self.get_resource_from_name(self.config_in)

        simulation = copy(config.simulation)
        simulation.has_planet_signal = False

        is_invertible = False
        counter = 0

        while not is_invertible and counter <= 10:

            phringe = PHRINGE()
            phringe.run(
                config_file_path=config.config_file_path,
                simulation=simulation,
                instrument=config.instrument,
                observation_mode=config.observation_mode,
                scene=config.scene,
                gpu=self.gpu,
                write_fits=False,
                create_copy=False
            )

            data = phringe.get_data()

            # Calculate covariance matrix for each differential output
            self.cov_out.cov = torch.zeros((data.shape[0], data.shape[1], data.shape[1]))
            self.cov_out.icov2 = torch.zeros((data.shape[0], data.shape[1], data.shape[1]))

            for i in range(len(data)):
                self.cov_out.cov[i] = data[i].cov()

                if not (self.cov_out.cov[i] < 0).any():
                    try:
                        self.cov_out.icov2[i] = torch.tensor(np.linalg.inv(np.sqrt(self.cov_out.cov[i].cpu().numpy())),
                                                             device=config.phringe._director._device)
                        is_invertible = True
                    except np.linalg.LinAlgError:
                        is_invertible = False
                else:
                    is_invertible = False
                    break

            counter += 1

        if counter == 10:
            raise ValueError('Covariance matrix could not be calculated. Please check the data.')

        print('Done')
        return self.cov_out
