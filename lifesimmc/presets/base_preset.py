from abc import ABC, abstractmethod


class BasePreset(ABC):
    """Base class for all presets."""

    _IS_FACTORY = False
    _LATEST_VERSION = None

    @classmethod
    def _get_preset_mappings(cls) -> dict[str, type]:
        """Get the mapping of preset versions to classes.

        Returns
        -------
        dict[str, type]
            The mapping of preset versions to classes.
        """
        return {}

    @classmethod
    def _get_latest_version(cls) -> str:
        """Get the latest version of the preset.

        Returns
        -------
        str
            The latest version of the preset.
        """
        if cls._LATEST_VERSION is None:
            raise ValueError(
                f"{cls.__name__} defines no latest version. "
                "Set _LATEST_VERSION on the preset factory class."
            )
        return cls._LATEST_VERSION

    def __new__(cls, version: str = "latest", **kwargs):
        """Create a new instance of the preset."""
        if not cls.__dict__.get("_IS_FACTORY", False):
            return super().__new__(cls)

        preset_mappings = cls._get_preset_mappings()

        requested_version = version
        resolved_version = (
            cls._get_latest_version()
            if version == "latest"
            else version
        )

        try:
            preset_cls = preset_mappings[resolved_version]
        except KeyError:
            valid_versions = ", ".join(["latest", *preset_mappings])
            raise ValueError(
                f"Unknown {cls.__name__} version {requested_version!r}. "
                f"Available versions are: {valid_versions}."
            )

        instance = preset_cls(**kwargs)

        instance.version = resolved_version
        instance.requested_version = requested_version
        instance.preset_name = cls.__name__

        return instance

    @abstractmethod
    def run(self):
        """Execute the preset pipeline."""
        pass
