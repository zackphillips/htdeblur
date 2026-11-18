"""SNR model used by fig_fom_2019 (FOM 2019 abstract).

Ported verbatim from libwallerlab.projects.motiondeblur.bluranalysis (Feb 2019),
the version that notebook was written against, with two fixes
(marked below) for cases where it raised. It differs from htdeblur.analysis
(the journal-paper version): frameRateToExposure takes no photon rate, and
getOptimalDnf searches random kernels instead of using the power-law estimate.
"""
import math

import numpy as np

from htdeblur import _util as ut
from htdeblur import blurkernel


def genBlurVector(kernel_length, beta=0.5, n_pulses=None, n_tests=10, metric='dnf', padding_size=0, method='coded'):
    '''
    This is a helper function for solving for a blur vector in terms of it's condition #
    '''
    kernel_list = []

    if n_pulses is None:
        n_pulses = math.floor(beta * kernel_length)
    else:
        # note: n_pulses overrides beta
        beta = n_pulses / kernel_length

    if padding_size is None: padding_size = 0

    # Guard from the later htdeblur.analysis: more pulses than kernel positions is
    # infeasible (libwallerlab raised in np.random.choice). Infinite DNF -> zero SNR.
    if n_pulses > kernel_length:
        return np.inf, None

    for _ in range(n_tests):
        # if padding_size <= 0:
        #     kernel = np.zeros(kernel_length+padding_size)
        # else:
        #     kernel = np.zeros(kernel_length)
        # indicies = np.random.choice(kernel_length, size=n_pulses, replace=False)
        # kernel = np.zeros(kernel_length)
        # kernel[indicies] = 1.0
        kernel = blurkernel.generate((kernel_length+padding_size,), kernel_length, method=method,
                 blur_illumination_fraction=beta, position='center', normalize=False)
        kernel_list.append(kernel)

    if metric == 'cond':
        # Determine kernel with best condition #
        metric_best = 1e10
        kernel_best = []
        for kernel in kernel_list:
            spectra = np.abs(np.fft.fft(kernel))
            kappa = np.max(spectra) / np.min(spectra)
            if kappa < metric_best:
                kernel_best = kernel
                metric_best = kappa
    elif metric == 'dnf':
        # Determine kernel with best dnf
        metric_best = 1e10
        kernel_best = []
        for kernel in kernel_list:
            dnf = calcDnfFromKernel(kernel)
            if dnf < metric_best:
                kernel_best = kernel
                metric_best = dnf
    else:
        raise ValueError

    return (metric_best, kernel_best)


def calcDnfFromKernel(x):
    if len(x) == 0:
        return 0
    else:
        # Normalize
        x = x / ut.scalar(ut.sum(x))

        # Take fourier transform intensity
        x_fft = ut.Ft(x)
        sigma_x = np.abs(x_fft) ** 2

        # Calculate DNF
        return np.sqrt(1 / len(x) * np.sum(1 / sigma_x))


def getOptimalDnf(kernel_size, beta=0.5, n_pulses=None, n_tests=10, metric='dnf', upperbound=False, padding_size=0, method='coded'):
    if kernel_size == 0:
        return 0
    elif kernel_size == 1:
        return 1
    else:
        dnf, x = genBlurVector(kernel_size, beta=beta, n_pulses=n_pulses, n_tests=n_tests, metric=metric, padding_size=padding_size, method=method)
        return dnf


def illuminanceToPhotonPixelRate(illuminance,
                                 numerical_aperture=1.0,
                                 pixel_size=6.5e-6,
                                 magnification=1,
                                 **kwargs):

    """
    Function which converts source illuminance and microscope parameters to
    photons / px / s.

    Based heavily on the publication:
    "When Does Computational Imaging Improve Performance?,"
    O. Cossairt, M. Gupta and S.K. Nayar,
    IEEE Transactions on Image Processing,
    Vol. 22, No. 2, pp. 447–458, Aug. 2012.

    However, this function implements the same result for
    microscopy, replacing f/# with NA, removing reflectance,
    and including magnification.

    Args:
     exposure_time: Integration time, s
     source_illuminance: Photometric source illuminance, lux
     numerical_aperture: System numerical aperture
     pixel_size: Pixel size of detector, um
     magnification: Magnification of imaging system

    Returns:
      Photon counts at the camera.
    """

    # Conversion factor from radiometric to photometric cordinates
    # https://www.thorlabs.de/catalogPages/506.pdf
    K = 1 / 680

    # Planck's constant
    # h_bar = 6.626176e-34
    h_bar = 1.054572e-34

    # Speed of light
    c = 2.9979e8

    # Average wavelength
    wavelength = 0.55e-6

    # Constant term
    const = K * wavelength / h_bar / c

    # Calculate number of counts
    counts = const * (numerical_aperture ** 2) * illuminance * (pixel_size / magnification) ** 2

    # Return
    return counts


def photonPixelRateToSnr(photon_pixel_rate, exposure_time,
                         dnf=1,
                         camera_quantum_efficency=0.6,
                         camera_dark_current=0.9,
                         camera_background_counts=0,
                         camera_ad_conversion=0.46,
                         pulse_time=None,
                         camera_readout_noise=2.5,
                         camera_max_counts=65535, debug=False, **kwargs):
    """
    Function which converts deconvolution noise factor to signal to noise ratio.
    Uses equations from https://www.photometrics.com/resources/learningzone/signaltonoiseratio.php and the dnf from the Agrawal and Raskar 2009 CVPR paper found here: http://ieeexplore.ieee.org/document/5206546/
    Default values are for the PCO.edge 5.5 sCMOS camera (https://www.pco.de/fileadmin/user_upload/pco-product_sheets/pco.edge_55_data_sheet.pdf)

    Args:
        photon_pixel_rate: Number of photons per pixel per second
        exposure_time: Integration time, s
        dnf: Deconvolution noise factor as specified in Agrawal et. al.
        camera_quantum_efficency: QE of camera, in [0,1]
        camera_dark_current: Dark current from datasheet, electrons / s
        pulse_time: Amount of time the illumination is pulsed, s
        camera_readout_noise: Readout noise from datasheet, electrons
        camera_max_counts: Maximum discrete bins of camera, usually 255 or 65535.

    Returns:
        The Estimated SNR.
    """

    # Call component function
    result = photonPixelRateToNoiseComponents(photon_pixel_rate, exposure_time,
                                              dnf=dnf,
                                              camera_quantum_efficency=camera_quantum_efficency,
                                              camera_dark_current=camera_dark_current,
                                              pulse_time=pulse_time,
                                              camera_readout_noise=camera_readout_noise,
                                              camera_ad_conversion=camera_ad_conversion,
                                              camera_background_counts=camera_background_counts,
                                              camera_max_counts=camera_max_counts, **kwargs)
    signal, noise_independent, noise_dependent = result

    if debug:
        print('DNF: %g, signal: %g, independent noise: %g, dependent noise: %g' % tuple([dnf] + list(result)))

    # Return SNR
    if dnf * noise_independent * noise_dependent == 0:
        return 0
    else:
        return signal / (dnf * np.sqrt(noise_independent ** 2 + noise_dependent ** 2))


def photonPixelRateToNoiseComponents(photon_pixel_rate,
                                     exposure_time,
                                     dnf=1,
                                     camera_quantum_efficency=0.6,
                                     camera_dark_current=0.9,
                                     camera_ad_conversion=0.46,
                                     pulse_time=None,
                                     camera_readout_noise=2.5,
                                     camera_max_counts=65535,
                                     camera_background_counts=0,
                                     **kwargs):

    """
    Function which calculates the variance of signal dependent noise and signal
    independent noise components.

    Args:
        photon_pixel_rate: Number of photons per pixel per second
        exposure_time: Integration time, s
        dnf: Deconvolution noise factor as specified in Agrawal et. al.
        camera_quantum_efficency: QE of camera, in [0,1]
        camera_dark_current: Dark current from datasheet, electrons / s
        camera_readout_noise: Readout noise from datasheet, electrons
        camera_max_counts: Maximum discrete bins of camera, usually 255 or 65535.

    Returns:
        (noise_var_dependent, noise_var_independent)
    """

    # Return zero if either photon_pixel_rate or exposure_time are zero
    if photon_pixel_rate * exposure_time == 0:
        return 0, 0, 0

    # Signal term
    if pulse_time is None:
        signal_mean = (photon_pixel_rate * camera_quantum_efficency * exposure_time) / camera_ad_conversion
    else:
        signal_mean = (photon_pixel_rate * camera_quantum_efficency * pulse_time) / camera_ad_conversion

    # Ensure the camera isnt saturating
    if signal_mean > camera_max_counts * camera_ad_conversion:
        return 0, 0, 0

    # Signal-independent noise term
    noise_independent = (camera_readout_noise / camera_ad_conversion)

    # Signal-dependent noise term
    noise_dependent = np.sqrt((signal_mean + camera_background_counts) + camera_dark_current * exposure_time / camera_ad_conversion)

    # Return
    return signal_mean, noise_dependent, noise_independent


def exposureTimeToNoiseComponents(exposure_time, **system_parameters):

    """
    Helper function which converts illuminance to noise components using a
    system parameters dictionary.

    Args:
        exposure_time: Exposure time, seconds
        system_parameters: dictionary of system parameters

    Returns:
        Imaging SNR
    """

    # Calculate photon pixel rate (photons / px / s)
    photon_pixel_rate = illuminanceToPhotonPixelRate(**system_parameters)

    # Calculate noise components using photon pixel rate (photons / px / s) and
    # exposure time (s)
    return photonPixelRateToNoiseComponents(photon_pixel_rate, exposure_time, **system_parameters)


def exposureTimeToSnr(exposure_time, debug=False, **system_parameters):

    """
    Helper function which calculates ratio of exposure

    Args:
        exposure_time: Exposure time, seconds
        system_parameters: dictionary of system parameters

    Returns:
        Imaging SNR
    """

    # Calculate photon pixel rate (photons / px / s)
    photon_pixel_rate = illuminanceToPhotonPixelRate(**system_parameters)

    # Calculate SNR using photon pixel rate (photons / px / s) and exposure time (s)
    return photonPixelRateToSnr(photon_pixel_rate, exposure_time, debug=debug, **system_parameters)


def acquisitionParameters(camera_frame_rate=10, debug=True,
                          camera_readout_time=0.016, camera_max_counts=65535,
                          camera_quantum_efficency=1.0, pulse_count=None,
                          motion_settle_time=0.1, motion_acceleration=1e4,
                          motion_velocity=None, motion_velocity_max=40e-3,
                          pixel_count=(2580, 2180), motion_axis=1,
                          illumination_beta=0.5, illumination_min_pulse_time=4e-6,
                          illuminance=1, numerical_aperture=1.0, pixel_size=6.5e-6,
                          magnification=1, **kwargs):
    """ This function calculates relevent acquisition parameters given system parameters. """

    # Calculate effective pixel size
    system_pixel_size = pixel_size / magnification

    # Calculate FOV in meters
    fov = [p * system_pixel_size for p in pixel_count]

    # Calculate frame time
    frame_time = 1 / camera_frame_rate

    # Calculate camera exposure time
    exposure_time_max = frame_time - camera_readout_time

    # Ensure exposure time isn't too short
    if exposure_time_max <= 0:
        exposure_time_max = 0

    # Calculate photon pixel rate (photons / px / s)
    photon_pixel_rate = illuminanceToPhotonPixelRate(illuminance,
                                                     numerical_aperture=numerical_aperture,
                                                     pixel_size=pixel_size,
                                                     magnification=magnification)

    # Calculate camera measured counts rate (counts / px / s)
    exposure_rate = photon_pixel_rate * camera_quantum_efficency

    # Determine maximum number of exposure units which would saturate the camera
    exposure_saturation_time = camera_max_counts / exposure_rate

    # Calculate velocity if not provided
    if motion_velocity is None:
        motion_velocity = fov[motion_axis] / frame_time

    # Calculate required LED array update speed
    pulse_time = min(system_pixel_size / motion_velocity, exposure_time_max)

    # Check if motion velocity is too low to capture all frames
    if motion_velocity > motion_velocity_max:
        if debug:
            print('Necessary velocity (%g m/s) exceeds maximum velocity (%g m/s))' % (motion_velocity, motion_velocity_max))
        return (0, 0, 0, 0, 0, (0, 0), 0)

    # Ensure strobe time isn't too fast for hardware
    if pulse_time < illumination_min_pulse_time:
        if debug:
            print('Necessary pulse time (%g s) exceeds min pulse time (%g s))' % (pulse_time, illumination_min_pulse_time))
        return (0, 0, 0, 0, 0, (0, 0), 0)

    return pulse_time, motion_velocity, exposure_rate, exposure_saturation_time, system_pixel_size, fov, frame_time


def frameRateToExposure(camera_frame_rate, acquisition_strategy, n_tests=10,
                        camera_readout_time=0.016, camera_max_counts=65535,
                        camera_quantum_efficency=1.0, pulse_count=None,
                        motion_settle_time=0.1, motion_acceleration=1e4,
                        motion_velocity=None, motion_velocity_max=40e-3,
                        pixel_count=(2580, 2180), motion_axis=1, dnf=None,
                        illumination_beta=0.5, illumination_min_pulse_time=4e-6,
                        illuminance=1, numerical_aperture=1.0, pixel_size=6.5e-6,
                        magnification=1, debug=False, frame_overlap=1.0, **kwargs):

    # Convert mm/s to m/s
    motion_velocity_max /= 1000
    motion_acceleration /= 1000
    if motion_velocity:
        motion_velocity /= 1000

    # Calculate derived parameters
    params = acquisitionParameters(camera_frame_rate=camera_frame_rate,
                                   pixel_size=pixel_size,
                                   camera_readout_time=camera_readout_time,
                                   camera_max_counts=camera_max_counts,
                                   camera_quantum_efficency=camera_quantum_efficency,
                                   pulse_count=pulse_count,
                                   motion_velocity=motion_velocity,
                                   motion_velocity_max=motion_velocity_max,
                                   frame_overlap=1.0,
                                   pixel_count=pixel_count,
                                   motion_axis=motion_axis,
                                   illumination_min_pulse_time=illumination_min_pulse_time,
                                   illuminance=illuminance,
                                   numerical_aperture=numerical_aperture,
                                   magnification=magnification,
                                   debug=debug)

    pulse_time, motion_velocity_continuous, exposure_count_rate, exposure_saturation_time, system_pixel_size, fov, frame_time = params
    # Check if any values are zero
    # libwallerlab multiplied by motion_velocity here, which is None unless passed explicitly
    # (TypeError). Use the computed continuous velocity for this zero check only.
    _velocity = motion_velocity if motion_velocity is not None else motion_velocity_continuous
    if frame_time * pulse_time * _velocity * exposure_count_rate * exposure_saturation_time * fov[0] * frame_time == 0:
        return (0, 1)

    if 'stop_and_stare' in acquisition_strategy:

        if motion_velocity is not None:
            rapid_velocity = motion_velocity
        else:
            rapid_velocity = motion_velocity_max

        # Calculate the time to start and stop
        t_start_stop = rapid_velocity / motion_acceleration

        # Calculate the distance to start and stop
        d_start_stop = 0.5 * motion_acceleration * t_start_stop ** 2

        # Calculate movement time (constant velocity)
        t_move = (fov[motion_axis] - d_start_stop) / rapid_velocity

        # Calculate exposure time (frame time - (the maximum of readout amd movement))
        exposure_time_effective = frame_time - max(t_move + 2 * t_start_stop + motion_settle_time, camera_readout_time)

        # Ensure exposure is non-negative
        exposure_time_effective = max(exposure_time_effective, 0)

        # We can reduce exposure time to avoid saturation if necessary
        if debug and exposure_time_effective > exposure_saturation_time:
            print('Clamping exposure!')
        exposure_time_effective = min(exposure_time_effective, exposure_saturation_time)

        # No deconvolution here
        dnf = 1

    else:
        # # Check motion velocity to see if we're moving too fast to scan frames
        # if motion_velocity_continuous > frame_overlap * fov[motion_axis] / frame_time:
        #     if debug:
        #         print('Motion velocity (%g m/s) is faster than FOV-limited velocity (%g m/s)' % (motion_velocity_continuous, fov[motion_axis] / frame_time))
        #     return (0, 1)  # Zero signal, dnf = 1

        if 'strobe' in acquisition_strategy:

            # Calculate signal exposure
            exposure_time_effective = pulse_time

            # No deconvolution here
            dnf = 1

        elif 'code' in acquisition_strategy:

            # Calculate maximum kernel length
            if pulse_time == 0:
                pulse_time = 1

            kernel_length_px_max = int(round(exposure_saturation_time / pulse_time))
            kernel_length_px_max = min(kernel_length_px_max, int(round(fov[motion_axis] / system_pixel_size)))

            # Set kernel length to be a fraction of length it woult take to saturate camera
            if pulse_count is None:
                kernel_length_px = int(math.floor(exposure_saturation_time / pulse_time))
            else:
                kernel_length_px = int(pulse_count / illumination_beta)

            # Ensure kernel length is not greater than maximum
            if kernel_length_px > kernel_length_px_max:
                kernel_length_px = kernel_length_px_max

            # Calculate DNF
            if kernel_length_px > 0:
                # Calculate DNF
                if dnf is None:
                    dnf = getOptimalDnf(kernel_length_px,
                                        n_pulses=pulse_count,
                                        beta=illumination_beta,
                                        n_tests=n_tests)

                # Determine exposure
                exposure_time_effective = kernel_length_px * pulse_time * illumination_beta

            else:
                if debug:
                    print('Invalid kernel length %d' % kernel_length_px)
                return (0, 1)  # Zero signal, dnf = 1

    # Filter exposure to saturation
    if exposure_time_effective > exposure_saturation_time:
        if debug:
            print('Effective exposure time (%g s) is greater than saturation exposure time (%g s)' % (exposure_time_effective, exposure_saturation_time))
        return (0, 1)
        # exposure_time_effective = min(exposure_time_effective, signal_exposure_saturate)

    return (exposure_time_effective, dnf)
