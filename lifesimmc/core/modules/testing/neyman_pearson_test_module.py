import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.test_resource import TestResource


class NeymanPearsonTestModule(BaseModule):
    def __init__(self, r_config_in: str, r_data_in: str, r_spectrum_in: str, r_test_out: str,
                 pfa: float, r_cov_in: str = None):
        self.config_in = r_config_in
        self.cov_in = r_cov_in
        self.data_in = r_data_in
        self.spectrum_in = r_spectrum_in
        self.test_out = TestResource(r_test_out)
        self.pfa = pfa

    def apply(self, resources: list[BaseResource]) -> TestResource:
        print("Performing Neyman-Pearson test...")

        config = self.get_resource_from_name(self.config_in)
        cov = self.get_resource_from_name(self.cov_in) if self.cov_in is not None else None
        data = self.get_resource_from_name(self.data_in).get_data()
        spectrum = self.get_resource_from_name(self.spectrum_in).spectrum[0].spectral_flux_density
        num_of_diff_outputs = len(data)

        if cov is not None:
            icov2 = cov.icov2
        else:
            icov2 = torch.diag(torch.ones(data.shape[1], device=config.phringe._director._device)).unsqueeze(0).repeat(
                data.shape[0], 1, 1)

        # flux = torch.tensor(flux).to(config.phringe._director._device)
        x_pos = -3e-7
        y_pos = 3e-7
        flux = spectrum
        time = config.phringe.get_time_steps().to(config.phringe._director._device)
        self.wavelengths = config.phringe.get_wavelength_bin_centers().to(config.phringe._director._device)
        self.fovs = config.phringe.get_field_of_view().cpu().numpy()
        icov2 = icov2.cpu().numpy()

        self.test_out.test_statistic = []
        self.test_out.xsi = []

        for i in range(num_of_diff_outputs):
            dataf = data[i].flatten()
            ndim = dataf.numel()
            icov2i = icov2[i]
            model = (icov2i @ config.phringe.get_template(time, self.wavelengths, x_pos, y_pos,
                                                          flux).cpu().numpy()).flatten()
            xtx = (model.T.dot(model)) / ndim
            # pfa = 0.0001
            xsi = np.sqrt(xtx) * norm.ppf(1 - self.pfa)
            test = (dataf @ model) / ndim

            self.test_out.test_statistic.append(test)
            self.test_out.xsi.append(xsi)
            print(f"Test statistic: {test}, Xsi: {xsi}")

            z = np.linspace(-0.5 * xtx, 4 * xsi, 1000)
            zdet = z[z > xsi]
            zndet = z[z < xsi]
            fig = plt.figure(dpi=150)
            plt.plot(z, norm.pdf(z, loc=0, scale=np.sqrt(xtx)), label=f"Pdf($T_{{NP}} | \mathcal{{H}}_0$)")
            plt.fill_between(zdet, norm.pdf(zdet, loc=0, scale=np.sqrt(xtx)), alpha=0.3,
                             label=f"$P_{{FA}}$")  # , hatch="//"
            # plt.fill_between(z[], )
            plt.plot(z, norm.pdf(z, loc=xtx, scale=np.sqrt(xtx)), label=f"Pdf($T_{{NP}}| \mathcal{{H}}_1$)")
            plt.fill_between(zdet, norm.pdf(zdet, loc=xtx, scale=np.sqrt(xtx)), alpha=0.3, label=f"$P_{{Det}}$")
            plt.axvline(xsi, color="gray", linestyle="--", label=f"$\\xi(P_{{FA}}={self.pfa})$")
            plt.xlabel(f"$T_{{NP}}$")
            plt.ylabel(f"$PDF(T_{{NP}})$")
            plt.legend()
            plt.show()

        print('Done')
        return self.test_out
