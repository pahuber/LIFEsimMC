from dataclasses import dataclass
from typing import Union

import numpy as np
from phringe.core.sources.planet import Planet
from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class PlanetResource(BaseResource):
    """Class representation of a planet parameter.

    Attributes
    ----------

    sed : Union[np.ndarray, Tensor]
        SED of the planet.
    std : Union[np.ndarray, Tensor]
        Standard deviation of the planet SED.
    cov : Union[np.ndarray, Tensor]
        Covariance matrix of the planet SED.
    """
    planet: Union[Planet, None] = None
    sed: Union[np.ndarray, Tensor] = None
    std: Union[np.ndarray, Tensor] = None
    cov: Union[np.ndarray, Tensor] = None
    pos_x: Union[float, None] = None
    pos_y: Union[float, None] = None
