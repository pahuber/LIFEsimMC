from typing import Tuple

import torch
from torch import Tensor


def get_indices_of_maximum_of_2d_array(array: Tensor) -> Tuple[int, int]:
    """Return the indices of the maximum of a 2D array.

    :param array: The array
    :return: The indices of the maximum
    """
    index = torch.where(array == torch.max(array))
    return index[0][0], index[1][0]
