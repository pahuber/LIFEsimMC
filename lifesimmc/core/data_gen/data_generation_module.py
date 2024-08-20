from lifesimmc.core.base_module import BaseModule


class DataGenerationModule(BaseModule):
    """Class representation of the data generation module."""

    def __init__(self, name: str, config_module: str, write_to_fits: bool = True, create_copy: bool = True):
        """Constructor method."""
        super().__init__(name)
        self.config_module = config_module
        self.write_to_fits = write_to_fits
        self.create_copy = create_copy
        self.data = None

    def apply(self):
        """Use PHRINGE to generate synthetic data.
        """
        print('Generating synthetic data...')

        config_module = self.get_module_from_name(self.config_module)

        config_module.phringe.run(
            config_file_path=config_module.config_file_path,
            simulation=config_module.simulation,
            observation_mode=config_module.observation_mode,
            instrument=config_module.instrument,
            scene=config_module.scene,
            gpu=self.gpu,
            write_fits=self.write_to_fits,
            create_copy=self.create_copy
        )

        self.data = config_module.phringe.get_data()

        print('Done')
