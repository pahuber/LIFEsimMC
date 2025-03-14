import numpy as np
import torch
from tqdm.contrib.itertools import product

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.image_resource import ImageResource


class RGBImageModule(BaseModule):
    def __init__(
            self,
            n_config_in: str,
            n_data_in: str,
            n_template_in: str,
            n_cov_in: str,
            n_image_out: str
    ):
        self.n_config_in = n_config_in
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_cov_in = n_cov_in
        self.n_image_out = n_image_out

    def apply(self, resources: list[BaseResource]) -> ImageResource:
        """Perform analytical MLE on a grid of templates to crate a cost function map/image. For each grid point
        estimate the flux and return the flux of the grid point with the maximum of the cost function.

        :param resources: The resources to apply the module to
        :return: The resource
        """
        print('Generating RGB image...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        templates_in = self.get_resource_from_name(self.n_template_in).collection
        r_cov_in = self.get_resource_from_name(self.n_cov_in)
        i_cov_sqrt = r_cov_in.i_cov_sqrt.cpu().numpy()
        image = np.zeros(
            (
                3,
                len(r_config_in.instrument.differential_outputs),
                r_config_in.simulation.grid_size,
                r_config_in.simulation.grid_size
            )
        )
        wl = r_config_in.instrument.wavelength_bin_centers
        wl_min = wl[0]
        wl_max = wl[-1]
        wl_1_3 = wl_min + (wl_max - wl_min) / 3
        wl_2_3 = wl_min + 2 * (wl_max - wl_min) / 3
        i_min = 0
        i_max = len(wl) - 1
        i_1_3 = min(range(len(wl)), key=lambda i: abs(wl[i] - wl_1_3))
        i_2_3 = min(range(len(wl)), key=lambda i: abs(wl[i] - wl_2_3))

        def func(sigma):
            return sigma  # norm.ppf(norm.cdf(sigma))

            cdf_val = mp.ncdf(sigma)
            return mp.nppf(cdf_val)

        for index_x, index_y in product(
                range(r_config_in.simulation.grid_size),
                range(r_config_in.simulation.grid_size)
        ):
            template = \
                [template for template in templates_in if template.x_index == index_x and template.y_index == index_y][
                    0]
            template_data = template.get_data().to(r_config_in.phringe._director._device)[:, :, :, 0, 0]

            for i in range(len(r_config_in.phringe._director._differential_outputs)):
                model_b = (i_cov_sqrt[i, i_min:i_1_3, i_min:i_1_3] @ template_data[i,
                                                                     i_min:i_1_3].cpu().numpy()).flatten()
                model_g = (i_cov_sqrt[i, i_1_3:i_2_3, i_1_3:i_2_3] @ template_data[i,
                                                                     i_1_3:i_2_3].cpu().numpy()).flatten()
                model_r = (i_cov_sqrt[i, i_2_3:i_max, i_2_3:i_max] @ template_data[i,
                                                                     i_2_3:i_max].cpu().numpy()).flatten()

                # model = (i_cov_sqrt[i] @ template_data[i].cpu().numpy()).flatten()
                # xtx = model @ model

                xtx_b = model_b @ model_b
                xtx_g = model_g @ model_g
                xtx_r = model_r @ model_r

                metric_b = data_in[i, i_min:i_1_3].cpu().numpy().flatten() @ model_b / np.sqrt(xtx_b)
                metric_g = data_in[i, i_1_3:i_2_3].cpu().numpy().flatten() @ model_g / np.sqrt(xtx_g)
                metric_r = data_in[i, i_2_3:i_max].cpu().numpy().flatten() @ model_r / np.sqrt(xtx_r)

                # metric = data_in[i].cpu().numpy().flatten() @ model / np.sqrt(xtx)

                # result = fsolve(lambda x: func(x) - metric, x0=np.array(5))

                # r = metric_r
                # g = metric_g
                # b = metric_b

                # r = (r - r.min()) / (r.max() - r.min())
                # g = (g - g.min()) / (g.max() - g.min())
                # b = (b - b.min()) / (b.max() - b.min())

                # import cv2
                # rgb = cv2.merge([b, g, r])

                image[0, i, index_x, index_y] = metric_r
                image[1, i, index_x, index_y] = metric_g
                image[2, i, index_x, index_y] = metric_b

        # Normalize each channel of the image from -1 to 1
        for i in range(3):
            for j in range(len(r_config_in.instrument.differential_outputs)):
                # image[i] = (image[i] - image[i].min()) / (image[i].max() - image[i].min())
                # image[i, j] = (image[i, j] - image[i, j].min()) / (image[i, j].max() - image[i, j].min())
                image[i, j] /= image[i, j].max()

        r_image_out = ImageResource(self.n_image_out)
        r_image_out.set_image(torch.tensor(image))

        print('Done')
        return r_image_out
