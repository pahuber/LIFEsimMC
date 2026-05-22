from typing import Union

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from phringe.io.sed_loader import SEDLoader


def convert_spectral_units(sed: np.ndarray, wavelengths: np.ndarray, units_in: Union[str, Quantity],
                           units_out: Union[str, Quantity], wavelength_units: Union[str, Quantity]) -> np.ndarray:
    units_in = u.Unit(units_in) if isinstance(units_in, str) else units_in
    units_out = u.Unit(units_out) if isinstance(units_out, str) else units_out
    wavelength_units = u.Unit(wavelength_units) if isinstance(wavelength_units, str) else wavelength_units

    wavelengths = wavelengths * wavelength_units

    is_sed = SEDLoader._is_sed_unit(units_in)
    is_cov = SEDLoader._is_sed_unit(units_in ** 0.5)

    if not (is_sed or is_cov):
        raise ValueError(f"The provided SED units '{units_in}' are not valid SED units.")

    if is_cov:
        factor = (
            (np.ones_like(wavelengths.value) * (units_in ** 0.5))
            .to(units_out ** 0.5, equivalencies=u.spectral_density(wavelengths))
            .value
        )
        return sed * np.outer(factor, factor)

    return (
        (sed * units_in)
        .to(units_out, equivalencies=u.spectral_density(wavelengths))
        .value
    )


def convert_wavelength_units(
        wavelengths: np.ndarray,
        units_in: Union[str, u.UnitBase],
        units_out: Union[str, u.UnitBase],
) -> np.ndarray:
    units_in = u.Unit(units_in) if isinstance(units_in, str) else units_in
    units_out = u.Unit(units_out) if isinstance(units_out, str) else units_out

    q = wavelengths * units_in

    if units_in.is_equivalent(u.m):
        return q.to(units_out).value

    if units_in.is_equivalent(u.Hz):
        return q.to(units_out, equivalencies=u.spectral()).value

    raise ValueError(f"Invalid wavelength units '{units_in}'. Must be length or frequency.")
