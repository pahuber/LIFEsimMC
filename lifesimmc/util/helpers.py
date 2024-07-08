from collections import namedtuple

Template = namedtuple('Template', 'x y data')

Extraction = namedtuple('Extraction', 'flux flux_err wavelengths cost_function')
