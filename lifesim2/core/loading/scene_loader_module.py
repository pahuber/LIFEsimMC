from pathlib import Path
from typing import overload

from phringe.core.entities.scene import Scene
from phringe.io.utils import get_dict_from_path

from lifesim2.core.base_module import BaseModule
from lifesim2.core.context import Context


class SceneLoaderModule(BaseModule):
    """Class representation of the scene loader module."""

    @overload
    def __init__(self, exoplanetary_system_file_path: Path, spectrum_files: Path = None):
        ...

    @overload
    def __init__(self, scene: Scene, spectrum_files: Path = None):
        ...

    def __init__(self, exoplanetary_system_file_path: Path = None, scene: Scene = None, spectrum_files: Path = None):
        """Constructor method.

        :param exoplanetary_system_file_path: The path to the exoplanetary system file
        :param scene: The scene object
        """
        self.exoplanetary_system_file_path = exoplanetary_system_file_path
        self.scene = scene
        self.spectrum_files = spectrum_files

    def apply(self, context: Context) -> Context:
        """Load the scene file.

        :param context: The context object of the pipeline
        :return: The (updated) context object
        """
        system_dict = get_dict_from_path(
            self.exoplanetary_system_file_path) if self.exoplanetary_system_file_path else None

        scene = Scene(**system_dict) if not self.scene else self.scene

        context.exoplanetary_system_file_path = self.exoplanetary_system_file_path
        context.scene = scene
        context.spectrum_files = self.spectrum_files
        return context
