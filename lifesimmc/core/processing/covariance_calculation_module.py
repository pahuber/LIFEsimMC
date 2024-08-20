import numpy as np
import torch

from lifesimmc.core.base_module import BaseModule


class CovarianceCalculationModule(BaseModule):
    """Class representation of the base module."""

    def __init__(self, name: str, config_module: str):
        """Constructor method."""
        super().__init__(name)
        self.name = name
        self.config_module = config_module
        self.covariance_matrix = None

    def apply(self):
        """Calculate the covariance of the data without the planet signal. This is done by generating a new data set
        without a planet. In reality, this could be achieved e.g. by observing a reference star.
        """
        print('Calculating covariance matrix...')

        config_module = self.get_module_from_name(self.config_module)

        simulation = config_module.simulation
        simulation.has_planet_signal = False

        is_invertible = False
        counter = 0

        while not is_invertible and counter <= 10:

            config_module.phringe.run(
                config_file_path=config_module.config_file_path,
                simulation=simulation,
                instrument=config_module.instrument,
                observation_mode=config_module.observation_mode,
                scene=config_module.scene,
                gpu=self.gpu,
                write_fits=False,
                create_copy=False
            )

            data = config_module.phringe.get_data()

            # Calculate covariance matrix for each differential output
            self.covariance_matrix = torch.zeros((data.shape[0], data.shape[1], data.shape[1]))
            for i in range(len(data)):
                self.covariance_matrix[i] = data[i].cov()

                try:
                    self.icov2 = np.linalg.inv(np.sqrt(self.covariance_matrix[i].cpu().numpy()))
                    is_invertible = True
                except np.linalg.LinAlgError:
                    is_invertible = False

            counter += 1

        if counter == 10:
            raise ValueError('Covariance matrix could not be calculated. Please check the data.')

        print('Done')
