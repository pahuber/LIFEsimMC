from typing import Union

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.template_resource import TemplateResource
from lifesimmc.util.helpers import Template


class WhiteningModule(BaseModule):
    """Class representation of the whitening module."""

    def __init__(
            self,
            r_cov_in: str,
            r_config_in: str,
            r_data_in: str = None,
            r_template_in: str = None,
            r_data_out: str = None,
            r_template_out: str = None
    ):
        """Constructor method."""
        self.cov_in = r_cov_in
        self.data_in = r_data_in
        self.template_in = r_template_in
        self.config_in = r_config_in
        self.data_out = DataResource(r_data_out) if r_data_out is not None else None
        self.template_out = TemplateResource(r_template_out) if r_template_out is not None else None

    def apply(self, resources: list[BaseResource]) -> Union[None, BaseResource, tuple]:
        """Whiten the data using the covariance matrix.
        """
        print('Whitening data and/or templates...')

        cov = self.get_resource_from_name(self.cov_in)
        config = self.get_resource_from_name(self.config_in)
        data = self.get_resource_from_name(self.data_in).get_data() if self.data_in is not None else None
        templates = self.get_resource_from_name(
            self.template_in).get_templates() if self.template_in is not None else None
        icov2 = cov.icov2

        # For all differential outputs
        # for i in range(data.shape[0]):
        #     icov2[i] = icov2

        if data is not None:
            for i in range(data.shape[0]):
                data[i] = icov2[i] @ data[i]
            self.data_out.set_data(data)

        if templates is not None:
            for template in templates:
                icov2 = icov2.to(config.phringe._director._device)
                template_data = template.data.to(config.phringe._director._device)
                for i in range(len(template_data)):
                    template_data[i] = icov2[i] @ template_data[i]
                template = Template(ix=template.ix, iy=template.iy, data=template_data, x=template.x, y=template.y)
                self.template_out.add_template(template)

        print('Done')
        if self.data_out is not None and self.template_out is not None:
            return self.data_out, self.template_out
        elif self.data_out is not None:
            return self.data_out
        elif self.template_out is not None:
            return self.template_out
        else:
            return None
