import numpy as np
from tqdm.contrib.itertools import product

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.coordinate_resource import CoordinateResource
from lifesimmc.core.resources.flux_resource import FluxResourceCollection
from lifesimmc.core.resources.image_resource import ImageResource


class MatchedFilterModule(BaseModule):
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

    def apply(self, resources: list[BaseResource]) -> tuple[
        FluxResourceCollection,
        ImageResource,
        CoordinateResource
    ]:
        """Perform analytical MLE on a grid of templates to crate a cost function map/image. For each grid point
        estimate the flux and return the flux of the grid point with the maximum of the cost function.

        :param resources: The resources to apply the module to
        :return: The resource
        """
        print('Calculating matched filters...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        templates_in = self.get_resource_from_name(self.n_template_in).collection
        r_cov_in = self.get_resource_from_name(self.n_cov_in)
        i_cov_sqrt = r_cov_in.i_cov_sqrt.cpu().numpy()
        image = np.zeros(
            (
                len(r_config_in.instrument.differential_outputs),
                r_config_in.simulation.grid_size,
                r_config_in.simulation.grid_size
            )
        )

        if self.field_of_view is not None:
            fovs = np.linspace(0, self.field_of_view / 206264806.71915, r_config_in.simulation.grid_size)

        for index_x, index_y in product(
                range(r_config_in.simulation.grid_size),
                range(r_config_in.simulation.grid_size)
        ):
            template = \
                [template for template in templates_in if template.x_index == index_x and template.y_index == index_y][
                    0]
            template_data = template.get_data().to(r_config_in.phringe._director._device)[:, :, :, 0, 0]

            for i in range(len(r_config_in.phringe._director._differential_outputs)):
                model = (i_cov_sqrt[i] @ template_data[i].cpu().numpy()).flatten()
                image[i, index_x, index_y] = data_in[i].cpu().numpy().flatten() @ model / np.sqrt((model @ model))

        r_image_out = ImageResource(self.n_image_out)
        r_image_out.image = image

        print('Done')
