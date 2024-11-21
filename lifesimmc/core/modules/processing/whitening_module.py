from typing import Union

from tqdm import tqdm

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.template_resource import TemplateResourceCollection, TemplateResource


class WhiteningModule(BaseModule):
    """Class representation of the whitening module.

    :param n_cov_in: The name of the covariance matrix resource
    :param n_data_in: The name of the data resource
    :param n_template_in: The name of the template resource
    :param n_data_out: The name of the output data resource
    :param n_template_out: The name of the output template resource
    """

    def __init__(
            self,
            n_cov_in: str,
            n_config_in: str,
            n_data_in: str = None,
            n_template_in: str = None,
            n_data_out: str = None,
            n_template_out: str = None,
            overwrite: bool = False
    ):
        """Constructor method.

        :param n_cov_in: The name of the covariance matrix resource
        :param n_data_in: The name of the data resource
        :param n_template_in: The name of the template resource
        :param n_config_in: The name of the configuration resource
        :param n_data_out: The name of the output data resource
        :param n_template_out: The name of the output template resource
        :param overwrite: Whether to overwrite/delete the existing resources
        """
        self.n_cov_in = n_cov_in
        self.n_data_in = n_data_in
        self.n_template_in = n_template_in
        self.n_config_in = n_config_in
        self.n_data_out = n_data_out
        self.n_template_out = n_template_out
        self.overwrite = overwrite

    def apply(self, resources: list[BaseResource]) -> Union[None, BaseResource, tuple]:
        """Whiten the data using the covariance matrix.

        :param resources: The resources to apply the module to
        :return: The resource
        """
        print('Whitening data and/or templates...')

        r_config_in = self.get_resource_from_name(self.n_config_in)
        r_cov_in = self.get_resource_from_name(self.n_cov_in)
        # i_cov_sqrt = r_cov_in.i_cov_sqrt
        data_in = self.get_resource_from_name(self.n_data_in).get_data() if self.n_data_in is not None else None
        templates_in = self.get_resource_from_name(
            self.n_template_in).collection if self.n_template_in is not None else None

        r_data_out = DataResource(self.n_data_out) if self.n_data_out is not None else None
        rc_template_out = TemplateResourceCollection(
            self.n_template_out,
            'Collection of TemplateResources, one for each point in the grid'
        ) if self.n_template_out is not None else None

        i_cov_sqrt = r_cov_in.i_cov_sqrt

        if data_in is not None:
            for i in range(data_in.shape[0]):
                data_in[i] = i_cov_sqrt[i] @ data_in[i]
            r_data_out.set_data(data_in)

        if rc_template_out is not None:
            rc_template_out._templates = []
            for it in tqdm(range(len(templates_in))):
                template = templates_in[it]
                i_cov_sqrt = i_cov_sqrt.to(r_config_in.phringe._director._device)
                template_data = template.get_data().to(r_config_in.phringe._director._device)
                for i in range(len(template_data)):
                    template_data[i] = (i_cov_sqrt[i] @ template_data[i, :, :, 0, 0].float())[:, :, None, None]
                rc_template_out.collection.append(
                    TemplateResource(
                        name='',
                        x_coord=template.x_coord,
                        y_coord=template.y_coord,
                        x_index=template.x_index,
                        y_index=template.y_index,
                        _data=template_data
                    )
                )
                if self.overwrite:
                    templates_in[it] = None

        print('Done')
        if self.n_data_out is not None and self.n_template_out is not None:
            return r_data_out, rc_template_out
        elif self.n_data_out is not None:
            return r_data_out
        elif self.n_template_out is not None:
            return rc_template_out
        else:
            return None
