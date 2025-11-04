from copy import copy

import torch
from matplotlib import pyplot as plt

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.test_resource import TestResource


class ModelSubtractionModule(BaseModule):
    """Class representation of the model subtraction module.

    Parameters
    ----------
    n_setup_in : str
        Name of the input configuration resource.
    n_data_in : str
        Name of the input data resource.
    n_planet_params_in : str
        Name of the input planet parameters resource.
    n_transformation_in : str or tuple[str]
        Name of the input transformation resource.
    n_test_out : str
        Name of the output test resource.
    pfa : float
        Probability of false alarm.
    """

    def __init__(
            self,
            n_setup_in: str,
            n_data_in: str,
            n_planet_params_in: str,
            n_data_out: str,
    ):
        """Constructor method.

        Parameters
        ----------
        n_setup_in : str
            Name of the input configuration resource.
        n_data_in : str
            Name of the input data resource.
        n_planet_params_in : str
            Name of the input planet parameters resource.
        n_transformation_in : str or tuple[str]
            Name of the input transformation resource.
        """
        self.n_setup_in = n_setup_in
        self.n_data_in = n_data_in
        self.n_planet_params_in = n_planet_params_in
        self.n_data_out = n_data_out

    def apply(self, resources: list[BaseResource]) -> DataResource:
        """Apply the energy detector test.

        Parameters
        ----------
        resources : list[BaseResource]
            List of resources.

        Returns
        -------
        TestResource
            The test resource.
        """
        print("Performing energy detector test...")

        # Extract all inputs
        r_config_in = self.get_resource_from_name(self.n_setup_in) if self.n_setup_in is not None else None
        r_planet_params_in = self.get_resource_from_name(
            self.n_planet_params_in) if self.n_planet_params_in is not None else None
        data_in = self.get_resource_from_name(self.n_data_in).get_data()

        r_data_out = DataResource(self.n_data_out)

        times = r_config_in.phringe.get_time_steps().cpu().numpy()
        wavelengths = r_config_in.phringe.get_wavelength_bin_centers().cpu().numpy()
        wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths().cpu().numpy()

        # TODO: handle mutiple planets
        flux = r_planet_params_in.params[0].sed.cpu().numpy()  # * 1e6  # convert to per um

        # TODO: Handle orbital motion
        posx = r_planet_params_in.params[0].pos_x
        posy = r_planet_params_in.params[0].pos_y

        model = r_config_in.phringe.get_model_counts(
            kernels=True,
            spectral_energy_distribution=flux,
            x_position=posx,
            y_position=posy,
        )
        model = torch.tensor(
            model,
            device=data_in.device,
            dtype=data_in.dtype
        )

        data_out = copy(data_in) - model

        # # Bootsrap
        # mean = np.zeros(data_out.shape[1])
        # cov_w = torch.cov(data_out[0]).cpu().numpy()
        # samples = np.random.multivariate_normal(mean, cov_w, size=data_out[0].shape[1]).T
        # samples2 = np.expand_dims(samples, axis=0)
        # counts_boot = torch.tensor(samples2, dtype=data_in.dtype, device=data_in.device)
        # # counts_boot += model
        # plt.imshow(counts_boot.cpu().numpy()[0], cmap='viridis')
        # plt.colorbar()
        # plt.title('0')
        # plt.show()

        for i in range(data_in.shape[2]):
            if i % 2 == 0:
                data_out[:, :, i] = data_out[:, :, 0]
            else:
                data_out[:, :, i] = data_out[:, :, 1]

        # plt.imshow(np.cov(counts_boot[0].cpu().numpy()), cmap='viridis', aspect='auto')
        # plt.colorbar()
        # plt.title('1')
        # plt.show()
        #
        # plt.imshow(data_in[0].cpu().numpy(), cmap='viridis', aspect='auto')
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(model[0].cpu().numpy(), cmap='viridis', aspect='auto')
        # plt.colorbar()
        # plt.show()
        #
        plt.imshow(data_out[0].cpu().numpy(), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.show()

        r_data_out.set_data(data_out + model)

        print('Done')
        return r_data_out
