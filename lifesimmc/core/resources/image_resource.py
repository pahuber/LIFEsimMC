from dataclasses import dataclass
from typing import Union

import numpy as np
from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class ImageResource(BaseResource):
    """Class representation of the image resource.

    Parameters
    ----------
    image : Tensor
        The image tensor.
    """
    _image: Tensor = None

    def get_image(self, as_numpy: bool) -> Union[Tensor, np.array]:
        """Get the image tensor.

        Parameters
        ----------
        as_numpy : bool
            If True, return the image as a NumPy array.

        Returns
        -------
        Tensor or np.ndarray
            The image tensor or NumPy array.
        """
        if as_numpy:
            return self._image.cpu().numpy()
        return self._image

    def set_image(self, image: Tensor):
        """Set the image tensor.

        Parameters
        ----------
        image : Tensor
            The image tensor to set.
        """
        self._image = image
