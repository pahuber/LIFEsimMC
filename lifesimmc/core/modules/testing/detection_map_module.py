import numpy as np
import torch
from tqdm.contrib.itertools import product

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.image_resource import ImageResource


class DetetionMapModule(BaseModule):
    def __init__(
            self,
            n_config_in: str,
            n_data_in: str,
            n_template_in: str,
            n_image_out: str,
            n_cov_in: str = None,
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
        print('Calculating matched filters...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        templates_in = self.get_resource_from_name(self.n_template_in).collection
        r_cov_in = self.get_resource_from_name(self.n_cov_in) if self.n_cov_in is not None else None
        i_cov_sqrt = r_cov_in.i_cov_sqrt.cpu().numpy() if r_cov_in is not None else None
        image = np.zeros(
            (
                len(r_config_in.instrument.differential_outputs),
                r_config_in.simulation.grid_size,
                r_config_in.simulation.grid_size
            )
        )
        times = r_config_in.phringe._director.simulation_time_steps

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
                if i_cov_sqrt is not None:
                    model = (i_cov_sqrt[i] @ template_data[
                        i].cpu().numpy()).flatten()  # TODO: remove cov argument here as input templates are already white
                else:
                    model = template_data[i].cpu().numpy().flatten()
                xtx = model @ model

                # metric = data_in[i].cpu().numpy().flatten() @ model / np.sqrt(xtx)

                y = data_in[i].cpu().numpy().flatten()
                x = model

                metric = np.corrcoef(x, y)[0, 1]

                # metric = y.T @ x / np.sqrt(xtx)
                #
                # metric = np.sqrt(xtx)

                # sigma2 = np.var(y - x)

                # print(sigma2)

                # metric = len(y) / 2 * np.log(2 * np.pi * sigma2) + 0.5 * (x - y).T @ (x - y) / sigma2
                # metric = 0.5 * (x - y).T @ (x - y)

                # print(xtx)

                # result = fsolve(lambda x: func(x) - metric, x0=np.array(5))

                image[i, index_x, index_y] = metric

        r_image_out = ImageResource(self.n_image_out)
        r_image_out.set_image(torch.tensor(image))

        print('Done')
        return r_image_out
