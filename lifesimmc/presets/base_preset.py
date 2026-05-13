from abc import ABC, abstractmethod


class BasePreset(ABC):
    """Base class for all presets.

    Attributes
    ----------

    version : str
        The version of the preset.
    _IS_FACTORY : bool
        Whether the preset is a factory.
    """
    _IS_FACTORY = False

    @classmethod
    def _get_preset_mappings(cls):
        return {}

    def __new__(cls, version: str = "latest", **kwargs):
        if not cls.__dict__.get("_IS_FACTORY", False):
            return super().__new__(cls)

        preset_mappings = cls._get_preset_mappings()

        try:
            preset_cls = preset_mappings[version]
        except KeyError:
            valid_versions = ", ".join(preset_mappings)
            raise ValueError(
                f"Unknown {cls.__name__} version {version!r}. "
                f"Available versions are: {valid_versions}."
            )

        return preset_cls(**kwargs)

    @abstractmethod
    def run(self):
        """Execute the preset pipeline."""
        pass
