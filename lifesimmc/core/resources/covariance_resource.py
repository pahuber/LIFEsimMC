from dataclasses import dataclass

from torch import Tensor

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class CovarianceResource(BaseResource):
    """Class representation of a covariance resource.
    
    :param cov: The covariance matrix
    :param i_cov_sqrt: The inverse square-root of the covariance matrix
    """
    cov: Tensor = None
    i_cov_sqrt: Tensor = None
