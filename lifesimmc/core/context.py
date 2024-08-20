class Context:
    """Class representation of the context."""

    def __init__(self):
        self._phringe = None
        self.config_file_path = None
        self.simulation = None
        self.observation_mode = None
        self.instrument = None
        self.scene = None
        self.spectrum_files = None
        self.data = None
        self.templates = None  # List of template objects
        self.templates_subtracted = None  # List of template objects after polyfit subtraction
        self.extractions = []
        self.polyfits = None
        self.bb_max = None
        self.bb_min = None
        self.bb_best = None
        self.icov2 = None
