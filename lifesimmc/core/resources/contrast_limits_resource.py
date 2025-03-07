from dataclasses import dataclass

import torch
from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class ContrastLimitsResource(BaseResource):
    """Class representation of the contrast limits resource.

    :param name: The name of the resource
    """
    magnitude_map: Tensor = None
    radius_map: Tensor = None

    def get_contrast_curves(self):
        size = self.magnitude_map.shape[1]
        center_index = size // 2
        contrast_curves = torch.zeros((self.magnitude_map.shape[0], size - center_index))
        for i in range(size - center_index):
            contrast_curves[:, i] = self.magnitude_map[:, center_index + i, center_index + i]
        return contrast_curves
