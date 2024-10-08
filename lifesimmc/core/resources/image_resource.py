from dataclasses import dataclass, field

from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource, BaseResourceCollection


@dataclass
class ImageResource(BaseResource):
    """Class representation of the image resource.

    :param _image: The image to be stored.
    """
    _image: Tensor = None

    def get_image(self, as_numpy: bool):
        if as_numpy:
            return self._image.cpu().numpy()
        return self._image

    def set_image(self, image: Tensor):
        self._image = image


@dataclass
class ImageResourceCollection(BaseResourceCollection):
    """Class representation of the image resource collection.
    """
    collection: list[ImageResource] = field(default_factory=list)
