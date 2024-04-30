class Context:
    """Class representation of the context."""

    def __init__(self):
        self.config_file_path = None
        self.exoplanetary_system_file_path = None
        self.settings = None
        self.observation = None
        self.observatory = None
        self.scene = None
        self.spectrum_files = None
        self.data = None
        self.templates = None  # List of template objects
        self.extractions = []
