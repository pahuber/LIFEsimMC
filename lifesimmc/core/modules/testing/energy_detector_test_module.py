import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import ncx2

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.test_resource import TestResource, TestResourceCollection


class EnergyDetectorTestModule(BaseModule):
    """Class representation of an energy detector test module.

    :param n_data_in: Name of the input data resource.
    :param n_test_out: Name of the output test resource.
    :param pfa: Probability of false alarm.
    """

    def __init__(
            self,
            n_data_in: str,
            n_test_out: str,
            pfa: float,
            n_config_in: str = None,
            n_cov_in: str = None,
            n_coordinate_in: str = None,
            n_flux_in: str = None,
            use_theoretical: bool = True
    ):
        """Constructor method.

        :param n_data_in: Name of the input data resource.
        :param n_test_out: Name of the output test resource.
        :param pfa: Probability of false alarm.
        :param n_config_in: Name of the input configuration resource.
        :param n_cov_in: Name of the input covariance resource.
        :param n_coordinate_in: Name of the input coordinate resource.
        :param n_flux_in: Name of the input flux resource.
        :param use_theoretical: Whether to calculate the theoretical or empirical test statistic.
        """
        self.n_data_in = n_data_in
        self.n_test_out = n_test_out
        self.pfa = pfa
        self.n_config_in = n_config_in
        self.n_cov_in = n_cov_in
        self.n_coordinate_in = n_coordinate_in
        self.n_flux_in = n_flux_in
        self.use_theoretical = use_theoretical

    def apply(self, resources: list[BaseResource]) -> TestResourceCollection:
        """Apply the energy detector test.

        :param resources: List of resources.
        :return: Test resource collection.
        """
        print("Performing energy detector test...")

        if not self.use_theoretical:
            assert self.n_config_in is not None, "Configuration resource is required for theoretical test statistic."
            assert self.n_cov_in is not None, "Covariance resource is required for theoretical test statistic."
            assert self.n_coordinate_in is not None, "Coordinate resource is required for theoretical test statistic."
            assert self.n_flux_in is not None, "Flux resource is required for theoretical test statistic."

        rc_test_out = TestResourceCollection(
            self.n_test_out,
            'Collection of TestResources, one for each differential output'
        )

        # Extract all inputs
        r_config_in = self.get_resource_from_name(self.n_config_in) if self.n_config_in is not None else None
        r_cov_in = self.get_resource_from_name(self.n_cov_in) if self.n_cov_in is not None else None
        r_coordinate_in = self.get_resource_from_name(
            self.n_coordinate_in
        ) if self.n_coordinate_in is not None else None

        data = self.get_resource_from_name(self.n_data_in).get_data()
        num_of_diff_outputs = len(data)
        time = r_config_in.phringe.get_time_steps(as_numpy=True)
        self.wavelengths = r_config_in.phringe.get_wavelength_bin_centers(as_numpy=True)
        self.wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths(as_numpy=True)

        # Generate a purely diagonal covariance matrix if none is given
        if not self.use_theoretical:
            i_cov_sqrt = r_cov_in.i_cov_sqrt.cpu().numpy()
            flux_in = r_config_in.scene.planets[0].spectral_flux_density.cpu().numpy()
            x_pos = r_config_in.scene.planets[0].angular_separation_from_star_x.cpu().numpy()[0]
            y_pos = r_config_in.scene.planets[0].angular_separation_from_star_y.cpu().numpy()[0]

        else:
            i_cov_sqrt = torch.diag(
                torch.ones(data.shape[1], device=r_config_in.phringe._director._device)
            ).unsqueeze(0).repeat(data.shape[0], 1, 1).cpu().numpy()
            x_pos = np.array(r_coordinate_in.x)
            y_pos = np.array(r_coordinate_in.y)
            # TODO: Handle multiple planets
            flux_in = self.get_resource_from_name(self.n_flux_in).collection[0].spectral_irradiance.cpu().numpy() \
                if self.n_flux_in is not None else None

        for i in range(num_of_diff_outputs):
            dataf = data[i].flatten()
            ndim = dataf.numel()

            icov2i = i_cov_sqrt[i]
            model = (icov2i @ r_config_in.phringe.get_template_numpy(
                time,
                self.wavelengths,
                self.wavelength_bin_widths,
                x_pos,
                y_pos,
                flux_in
            )[i, :, :, 0, 0]).flatten()

            # Calculate test statistic
            if self.use_theoretical:
                test = (dataf @ dataf)
            else:
                test = (model @ model)

            # Calculate threshold
            xtx = (model @ model)
            xsi = ncx2.ppf(1 - self.pfa, df=ndim, nc=0)

            r_test_out = TestResource(
                name='',
                test_statistic=test,
                xsi=xsi,
                xtx=xtx,
                ndim=ndim
            )
            rc_test_out.collection.append(r_test_out)

            # Plotting test
            z = np.linspace(0.9 * xsi, 1.1 * xsi, 1000)
            zdet = z[z > xsi]
            zndet = z[z < xsi]
            fig = plt.figure(dpi=150)
            plt.plot(z, ncx2.pdf(z, df=ndim, nc=0), label=f"PDF($T_{{E}} | \mathcal{{H}}_0$)")
            plt.fill_between(zdet, ncx2.pdf(zdet, df=ndim, nc=0), alpha=0.3, label=f"$P_{{FA}}$")  # , hatch="//"
            plt.plot(z, ncx2.pdf(z, df=ndim, nc=xtx), label=f"PDF($T_{{E}}| \mathcal{{H}}_1$)")
            plt.fill_between(zdet, ncx2.pdf(zdet, df=ndim, nc=xtx), alpha=0.3, label=f"$P_{{Det}}$")
            plt.axvline(xsi, color="gray", linestyle="--", label=f"$\\xi(P_{{FA}}={self.pfa})$")
            plt.xlabel(f"$T_{{E}}$")
            plt.ylabel(f"$PDF(T_{{E}})$")
            plt.legend()
            plt.show()

        print('Done')
        return rc_test_out
