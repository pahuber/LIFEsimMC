import numpy as np
import torch
from scipy.stats import norm

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.test_resource import TestResourceCollection, TestResource


class NeymanPearsonTestModule(BaseModule):
    """Class representation of a Neyman-Pearson test module.

    :param n_config_in: Name of the input configuration resource.
    :param n_cov_in: Name of the input covariance resource.
    :param n_data_in: Name of the input data resource.
    :param n_flux_in: Name of the input spectrum resource.
    :param n_test_out: Name of the output test resource.
    :param pfa: Probability of false alarm.
    """

    def __init__(
            self,
            n_config_in: str,
            n_data_in: str,
            n_cov_in: str,
            n_test_out: str,
            pfa: float,
            n_coordinate_in: str = None,
            n_flux_in: str = None,
    ):
        """Constructor method.

        :param n_config_in: Name of the input configuration resource.
        :param n_cov_in: Name of the input covariance resource.
        :param n_coordinate_in: Name of the input coordinate resource.
        :param n_data_in: Name of the input data resource.
        :param n_flux_in: Name of the input spectrum resource.
        :param n_test_out: Name of the output test resource.
        :param pfa: Probability of false alarm.
        """
        self.n_config_in = n_config_in
        self.n_cov_in = n_cov_in
        self.n_coordinate_in = n_coordinate_in
        self.n_data_in = n_data_in
        self.n_flux_in = n_flux_in
        self.n_test_out = n_test_out
        self.pfa = pfa

    def apply(self, resources: list[BaseResource]) -> TestResourceCollection:
        """Apply the Neyman-Pearson test.

        :param resources: List of resources.
        :return: Test resource collection.
        """
        print("Performing Neyman-Pearson test...")

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_cov_in = self.get_resource_from_name(self.n_cov_in) if self.n_cov_in is not None else None
        # r_coordinate_in = self.get_resource_from_name(self.n_coordinate_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        # flux_in = self.get_resource_from_name(self.n_flux_in).collection[0].spectral_irradiance.cpu().numpy()

        num_of_diff_outputs = len(data_in)
        i_cov_sqrt = r_cov_in.i_cov_sqrt
        i_cov_sqrt = i_cov_sqrt.cpu().numpy()
        # x_pos = np.array(r_coordinate_in.x)
        # y_pos = np.array(r_coordinate_in.y)
        sky_brightness_distribution = r_config_in.phringe._director._planets[
            0].sky_brightness_distribution  # TODO: Handel multiple planets

        # if orbital motion is modeled, just use the initial position
        if sky_brightness_distribution.ndim == 4:
            sky_brightness_distribution = sky_brightness_distribution[0]

        # Get indices of only pixel that is not zero
        index_x, index_y = torch.nonzero(sky_brightness_distribution[0], as_tuple=True)
        # index_x, index_y = index_x[0].item(), index_y[0].item()
        x_pos = r_config_in.phringe._director._planets[0].sky_coordinates[
            0, index_x[0].item(), index_y[0].item()].cpu().numpy()
        y_pos = r_config_in.phringe._director._planets[0].sky_coordinates[
            1, index_x[0].item(), index_y[0].item()].cpu().numpy()
        time = r_config_in.phringe.get_time_steps(as_numpy=True)
        self.wavelengths = r_config_in.phringe.get_wavelength_bin_centers(as_numpy=True)
        self.wavelength_bin_widths = r_config_in.phringe.get_wavelength_bin_widths(as_numpy=True)
        self.fovs = r_config_in.phringe.get_field_of_view(as_numpy=True)

        rc_test_out = TestResourceCollection(
            self.n_test_out,
            'Collection of test resources, one for each differential output'
        )

        for i in range(num_of_diff_outputs):
            dataf = data_in[i].flatten()
            ndim = dataf.numel()
            icov2i = i_cov_sqrt[i]
            model = (icov2i @ r_config_in.phringe.get_template_numpy(
                time,
                self.wavelengths,
                self.wavelength_bin_widths,
                x_pos,
                y_pos,
                r_config_in.scene.params[0].spectral_flux_density.cpu().numpy()  # flux_in
            )[i, :, :, 0, 0]).flatten()

            # print(np.linalg.norm(model))
            #
            # model /= np.linalg.norm(model)
            # print(np.linalg.norm(model))

            test = (dataf @ model)

            data_h0 = dataf - model
            test_h0 = (data_h0 @ model)

            xtx = model @ model
            # test = model @ model
            # xtx = dataf @ model
            # print(xtx)
            xsi = np.sqrt(xtx) * norm.ppf(1 - self.pfa)
            P_Det = 1 - norm.cdf((xsi - xtx) / np.sqrt(xtx))

            # print(f"SNR: {test / xsi}")
            # print(f"Significance: {test / np.sqrt(xtx)}")

            r_test_out = TestResource(
                name='',
                test_statistic=test,
                xsi=xsi,
                xtx=test_h0,
                ndim=ndim,
                p_det=P_Det
            )
            rc_test_out.collection.append(r_test_out)

            # Plot the test statistic
            # z = np.linspace(-0.5 * xtx, 15 * xsi, 1000)
            # zdet = z[z > xsi]
            # zndet = z[z < xsi]
            # fig = plt.figure(dpi=150)
            # plt.plot(z, norm.pdf(z, loc=0, scale=np.sqrt(xtx)), label=f"Pdf($T_{{NP}} | \mathcal{{H}}_0$)")
            # plt.fill_between(zdet, norm.pdf(zdet, loc=0, scale=np.sqrt(xtx)), alpha=0.3,
            #                  label=f"$P_{{FA}}$")  # , hatch="//"
            # # plt.fill_between(z[], )
            # plt.plot(z, norm.pdf(z, loc=xtx, scale=np.sqrt(xtx)), label=f"Pdf($T_{{NP}}| \mathcal{{H}}_1$)")
            # plt.fill_between(zdet, norm.pdf(zdet, loc=xtx, scale=np.sqrt(xtx)), alpha=0.3, label=f"$P_{{Det}}$")
            # plt.axvline(xsi, color="gray", linestyle="--", label=f"$\\xi(P_{{FA}}={self.pfa})$")
            # plt.xlabel(f"$T_{{NP}}$")
            # plt.ylabel(f"$PDF(T_{{NP}})$")
            # plt.legend()
            # plt.show()

            # calculate p det the detection probability as the are beneath the curve
            # p_det = norm.cdf(xsi, loc=xtx, scale=np.sqrt(xtx))
            # print(f"Detection probability: {1 - p_det}")

            # print(f"Detection Probability: {P_Det:.4f}")

        print('Done')
        return rc_test_out
