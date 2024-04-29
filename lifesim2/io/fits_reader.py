from pathlib import Path

import numpy as np
import torch
from astropy.io import fits


class FITSReader():
    """Class representation of the FITS reader.
    """

    def read(self, path_to_fits_file: Path):
        """Read the data from a FITS file.

        :param path_to_fits_file: The path to the FITS file
        """
        hdul = fits.open(path_to_fits_file)[1:]

        data_of_all_outputs = []
        for data_per_output in hdul:
            data_of_all_outputs.append(torch.from_numpy(data_per_output.data.astype(np.float32)))
        data = torch.stack(data_of_all_outputs)
        hdul.close()
        return data
