from collections import namedtuple

Template = namedtuple('Template', 'x y data')

Spectrum = namedtuple('Spectrum', 'spectral_flux_density err_low err_high wavelengths bins')

Extraction = namedtuple('Extraction', 'flux flux_err_low flux_err_high wavelengths cost_function')
