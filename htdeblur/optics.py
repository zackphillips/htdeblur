"""Pupil and OTF models. Ported from comptic.imaging."""

import numpy as np
from . import _util as ut
from ._util import Ft, iFt


def pupil(shape, camera_pixel_size=6.5e-6, objective_magnification=10, system_magnification=1.0,
          illumination_wavelength=0.53e-6, objective_numerical_aperture=0.25, center=True,
          dtype=None, backend=None):
    """Binary circular pupil sampled on the sensor-plane frequency grid.

    Args:
        shape: (ny, nx) of the sensor in pixels.
        camera_pixel_size: physical camera pixel size.
        objective_magnification, system_magnification: used for the effective pixel size.
        illumination_wavelength: wavelength, same units as camera_pixel_size.
        objective_numerical_aperture: detection NA.
        center: if False, return the ifftshifted pupil.
    """
    assert len(shape) == 2, "pupil must be 2D"

    dtype = dtype if dtype is not None else ut.DEFAULT_DTYPE
    backend = backend if backend is not None else 'numpy'

    effective_pixel_size = camera_pixel_size / system_magnification / objective_magnification
    fylin, fxlin = ut.grid(shape, 1 / effective_pixel_size / np.asarray(shape))

    pupil_radius = objective_numerical_aperture / illumination_wavelength
    p = np.asarray((fxlin ** 2 + fylin ** 2) <= pupil_radius ** 2).astype(float)
    p = ut.cast(p, dtype)

    return p if center else np.fft.ifftshift(p)


def otf(shape, pixel_size, illumination_wavelength, objective_numerical_aperture,
        center=True, dtype=None, backend=None):
    """Incoherent OTF (pupil autocorrelation), normalized to a peak of 1.

    Args:
        pixel_size: effective (sample-plane) pixel size, same units as wavelength.

    The comptic original passed these to `pupil` positionally, so the wavelength
    landed in `objective_magnification` and `center` in `illumination_wavelength`.
    Only `Reconstruction(use_psf=True)` reached it; that path was off by default.
    """
    p = pupil(shape,
              camera_pixel_size=pixel_size,
              objective_magnification=1.0,
              system_magnification=1.0,
              illumination_wavelength=illumination_wavelength,
              objective_numerical_aperture=objective_numerical_aperture,
              center=True, dtype=dtype, backend=backend)

    result = iFt(Ft(p) * np.conj(Ft(p)))
    result /= ut.max(np.abs(result))

    return result if center else np.fft.ifftshift(result)
