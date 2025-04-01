import torch

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.image_resource import ImageResource


class CorrelationMapModule(BaseModule):
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
        print('Calculating correlation map...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        data_in = self.get_resource_from_name(self.n_data_in).get_data()
        template_counts_in = self.get_resource_from_name(self.n_template_in).get_data()

        y = data_in.reshape(data_in.shape[0], -1)
        x = template_counts_in.reshape(
            template_counts_in.shape[0],
            -1,
            template_counts_in.shape[-1],
            template_counts_in.shape[-1]
        )

        image = (
                torch.einsum('ij,ijkl->ikl', y, x)
                / torch.sqrt(torch.einsum('ij, ij->', y, y))
                / torch.sqrt(torch.einsum('ijkl,ijkl->', x, x))
        )

        r_image_out = ImageResource(self.n_image_out)
        r_image_out.set_image(image)

        print('Done')
        return r_image_out
