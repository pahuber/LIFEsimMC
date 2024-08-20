from collections import namedtuple

Template = namedtuple('Template', 'x y data')

Extraction = namedtuple('Extraction', 'flux flux_err_low flux_err_high wavelengths cost_function')
