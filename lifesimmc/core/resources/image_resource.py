from dataclasses import dataclass

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

    def get_image(self, as_numpy: bool):
        if as_numpy:
            return self._image.cpu().numpy()
        return self._image

    def set_image(self, image: Tensor):
        self._image = image
