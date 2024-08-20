from typing import Union

import numpy as np
import torch

from lifesimmc.core.modules.base_module import BaseModule
from lifesimmc.core.resources.base_resource import BaseResource
from lifesimmc.core.resources.data_resource import DataResource
from lifesimmc.core.resources.template_resource import TemplateResource
from lifesimmc.util.helpers import Template


class WhiteningModule(BaseModule):
    """Class representation of the whitening module."""

    def __init__(
            self,
            cov_in: str,
            data_in: str = None,
            template_in: str = None,
            data_out: str = None,
            template_out: str = None
    ):
        """Constructor method."""
        self.cov_in = cov_in
        self.data_in = data_in
        self.template_in = template_in
        self.data_out = DataResource(data_out) if data_out is not None else None
        self.template_out = TemplateResource(template_out) if template_out is not None else None

    def apply(self, resources: list[BaseResource]) -> Union[None, BaseResource, tuple]:
        """Whiten the data using the covariance matrix.
        """
        print('Whitening data and/or templates...')

        cov = self.get_resource_from_name(self.cov_in)
        data = self.get_resource_from_name(self.data_in).data if self.data_in is not None else None
        templates = self.get_resource_from_name(self.template_in).templates if self.template_in is not None else None
        icov2 = torch.zeros(cov.matrix.shape)

        # For all differential outputs
        for i in range(data.shape[0]):
            icov2[i] = torch.tensor(np.linalg.inv(np.sqrt(cov.matrix[i].cpu().numpy())))

        if data is not None:
            for i in range(data.shape[0]):
                data[i] = icov2 @ data[i]
            self.data_out.data = data

        if templates is not None:
            for template in templates:
                template_data = template.data
                for i in range(len(template_data)):
                    template_data[i] = icov2 @ template_data[i]
                template = Template(template.x, template.y, template_data, template.planet_name)
                self.template_out.templates.append(template)

        print('Done')
        if self.data_out is not None and self.template_out is not None:
            return self.data_out, self.template_out
        elif self.data_out is not None:
            return self.data_out
        elif self.template_out is not None:
            return self.template_out
        else:
            return None
