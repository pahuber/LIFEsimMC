import numpy as np
import torch

from lifesimmc.core.base_module import BaseModule
from lifesimmc.util.helpers import Template


class WhiteningModule(BaseModule):
    """Class representation of the whitening module."""

    def __init__(self, name: str, cov_in: str, data_in: str = None, template_in: str = None):
        """Constructor method."""
        super().__init__(name)
        self.cov_in = cov_in
        self.data_in = data_in
        self.template_in = template_in
        self.data = None
        self.templates = None

    def apply(self):
        """Whiten the data using the covariance matrix.
        """
        print('Whitening data and/or templates...')

        cov_module = self.get_module_from_name(self.cov_in)
        data_module = self.get_module_from_name(self.data_in) if self.data_in is not None else None
        template_module = self.get_module_from_name(self.template_in) if self.template_in is not None else None

        data = data_module.data if data_module is not None else None
        templates = template_module.templates if template_module is not None else None

        # For all differential outputs
        icov2 = torch.zeros(cov_module.covariance_matrix.shape)

        for i in range(data.shape[0]):
            icov2[i] = torch.tensor(np.linalg.inv(np.sqrt(cov_module.covariance_matrix[i].cpu().numpy())))

            if data is not None:
                data[i] = icov2 @ data[i]

        if templates is not None:
            self.templates = []
            for template in templates:
                template_data = template.data
                for i in range(len(template_data)):
                    template_data[i] = icov2 @ template_data[i]
                template = Template(template.x, template.y, template_data, template.planet_name)
                self.templates.append(template)

        self.data = data

        print('Done')
