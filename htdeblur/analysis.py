import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import llops as yp
from . import blurkernel

# Default illumination parameters
illumination_power_dict = {'LED': 4781.42,
                           'Halogen Lamp': 6830.60,
                           'Metal Halide': 392759.57,
                           'xenon': 360655.74,
                           'mercury': 224043.72}

# Define system parameters for our system
_system_params_default = {
                             'pixel_count': (2580, 2180),
                             'numerical_aperture': 0.25,
                             'magnification': 10.0,
                             'pixel_size': 6.5e-6,
                             'illumination_rep_rate': 250e3,  # Hz
                             'motion_settle_time': 0.25,  # seconds
                             'motion_acceleration': 1e4,  # mm / s / s
                             'motion_velocity_max': 40,  # mm / s
                             'motion_velocity': 25,  # mm / s
                             'illumination_beta': 0.5,
                             'motion_axis': 1,
                             'illuminance': 450,  # lux
                             'n_tests': 25,
                             'frame_overlap': 0.8,
                             'sample_quantum_yield': 1.0,
                             'camera_is_color': False,
                             'camera_readout_time': 0.032,  # seconds
                             'camera_quantum_efficency': 0.55,
                             'camera_max_counts': 65535,
                             'camera_dark_current': 0.9,
                             'camera_readout_noise': 3.85,
                             'sample_fill_fraction': 0.05
                         }

def getDefaultSystemParams(**kwargs):
    """Returns a list of default system parameters."""
    params = copy.deepcopy(_system_params_default)
    for key in kwargs:
        if key in params:
            params[key] = kwargs[key]

    return params

def genBlurVector(kernel_length, beta=0.5, n_pulses=None, n_tests=10,
                  metric='dnf', padding_size=0, kernel_generation_method='coded'):
    '''
    This is a helper function for solving for a blur vector in terms of it's condition #
    '''
    kernel_list = []

    if n_pulses is None:
        n_pulses = math.floor(beta * kernel_length)
    else:
        # note: n_pulses overrides beta
        beta = n_pulses / kernel_length

    if padding_size is None:
        padding_size = 0

    if n_pulses > kernel_length:
        return np.inf, None

    for _ in range(n_tests):
        kernel = blurkernel.generate((kernel_length+padding_size,), kernel_length, method=kernel_generation_method,
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
        x = x / yp.scalar(yp.sum(x))

        # Take fourier transform intensity
        x_fft = yp.Ft(x)
        sigma_x = yp.abs(x_fft) ** 2

        # Calculate DNF
        return np.sqrt(1 / len(x) * np.sum(1 / sigma_x))


def calcCondNumFromKernel(x):
    if len(x) == 0:
        return 0
    else:
        # x = x / np.sum(x)
        x_fft = np.fft.fft(x)
        sigma_x = np.abs(x_fft)
        return np.max(sigma_x) / np.min(sigma_x)


# def getOptimalDnf(kernel_size, beta=0.5, n_pulses=None, n_tests=10, metric='dnf', upperbound=False, padding_size=0, method='coded'):
    ''' note: n_pulses overrides beta'''
def getOptimalDnf(kernel_size, pulse_count=10, beta=0.5, n_tests=10, metric='dnf',
                  method='estimate', padding_size=0, kernel_generation_method='coded'):
    if kernel_size == 0:
        return 0
    elif kernel_size == 1:
        return 1
    else:
        if method == 'estimate':
            return 1.11653804 * pulse_count ** 0.64141481
        else:
            dnf, x = genBlurVector(kernel_size, beta=beta, n_pulses=pulse_count,
                                   n_tests=n_tests, metric=metric,
                                   padding_size=padding_size,
                                   kernel_generation_method=kernel_generation_method)
            return dnf

    # if kernel_size == 0:
    #     return 0
    # elif kernel_size == 1:
    #     return 1
    # elif n_pulses is not None and n_pulses > kernel_size:
    #     return 0
    # else:
    #     if method == 'estimate':
    #         return 1.12 * pulse_count ** 0.64
    #     else:
    #         dnf, x = genBlurVector(kernel_size, beta=beta, n_pulses=pulse_count,
    #                                n_tests=n_tests, metric=metric,
    #                                padding_size=padding_size,
    #                                kernel_generation_method=kernel_generation_method)
    #         return dnf


def getOptimalPulseCountFromStrobe(signal_strobe, camera_readout_noise=3.85, camera_ad_conversion=0.46):
    """Returns the optimal pulse count given a strobed signal in counts.

       Calculating derivative of SNR: https://www.wolframalpha.com/input/?i=d%2Fdx+(s+*+x)+%2F+(1.12+*+x+%5E+0.64+*+sqrt(s+*+x+%2B+r+%5E+2))

       Calculating this equation: https://www.wolframalpha.com/input/?i=(s+(0.321429+r%5E2+%2B+0.321429+s+x+-+0.446429+s+x%5E1.))%2F(x%5E0.64+(r%5E2+%2B+s+x)%5E(3%2F2))+%3D+0"""
    # TODO: should update with more precise powerlaw
    # Also should deal with integer maximization
    camera_readout_noise = camera_readout_noise / camera_ad_conversion
    return 2.57143 * camera_readout_noise ** 2 / signal_strobe


def countsToIlluminance(signal_counts, total_throughput, camera_background_counts=385,
                        camera_quantum_efficency=0.6, numerical_aperture=0.25,
                        magnification=10, pixel_size=6.5e-6):
    """Converts pixel counts to illuminance."""
    # Number of photons collected
    photon_count = (signal_counts - camera_background_counts) / camera_quantum_efficency

    # Photon throughput (photons / px / s)
    photon_pixel_rate = photon_count / total_throughput

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

    # Calculate illuminance
    illuminance = photon_pixel_rate / (const * (numerical_aperture ** 2) * (pixel_size / magnification) ** 2)

    # Return
    return illuminance


def illuminanceToPhotonPixelRate(illuminance,
                                 numerical_aperture=1.0,
                                 pixel_size=6.5e-6,
                                 magnification=1,
                                 sample_quantum_yield=1.,
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

    # Calculate photon_pixel_rate
    photon_pixel_rate = sample_quantum_yield * const * (numerical_aperture ** 2) * illuminance * (pixel_size / magnification) ** 2

    # Return
    return photon_pixel_rate


def photonPixelRateToSnr(photon_pixel_rate, exposure_time,
                         dnf=1,
                         camera_quantum_efficency=0.6,
                         camera_dark_current=0.9,
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
        signal_mean_counts = (photon_pixel_rate * camera_quantum_efficency * exposure_time)
    else:
        signal_mean_counts = (photon_pixel_rate * camera_quantum_efficency * pulse_time)

    # Ensure the camera isnt saturating
    if signal_mean_counts > camera_max_counts * camera_ad_conversion:
        return 0, 0, 0

    # Signal-independent noise term
    noise_independent_e = (camera_readout_noise / camera_ad_conversion)

    # Signal-dependent noise term
    noise_dependent_e = np.sqrt(signal_mean_counts + camera_dark_current * exposure_time / camera_ad_conversion)

    # Return
    return signal_mean_counts, noise_dependent_e, noise_independent_e


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


def acquisitionParameters(acquisition_strategy, camera_frame_rate=10, debug=True,
                          camera_readout_time=0.016, camera_max_counts=65535,
                          camera_quantum_efficency=1.0, pulse_count=None,
                          motion_settle_time=0.1, motion_acceleration=1e4,
                          motion_velocity=None, motion_velocity_max=40e-3,
                          pixel_count=(2580, 2180), motion_axis=1,
                          illumination_beta=0.5, illumination_rep_rate=250e3,
                          illuminance=1, numerical_aperture=1.0, pixel_size=6.5e-6,
                          magnification=1, dnf_calculation_method='estimate', **kwargs):
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
    if pulse_time < 1 / illumination_rep_rate and acquisition_strategy is not 'stop_and_stare':
        if debug:
            print('Necessary pulse time (%g s) exceeds min pulse time (%g s))' % (pulse_time, 1 / illumination_rep_rate))
        return (0, 0, 0, 0, 0, (0, 0), 0)

    return pulse_time, motion_velocity, exposure_rate, exposure_saturation_time, system_pixel_size, fov, frame_time


def frameRateToExposure(camera_frame_rate, photon_pixel_rate, acquisition_strategy,
                        camera_readout_time=0.016, camera_max_counts=65535,
                        camera_quantum_efficency=1.0, pulse_count=None, n_tests=10,
                        dnf_calculation_method='estimate',
                        motion_settle_time=0.1, motion_acceleration=1e4,
                        motion_velocity=None, motion_velocity_max=40e-3,
                        camera_readout_noise=3.85, camera_ad_conversion=0.46,
                        pixel_count=(2580, 2180), motion_axis=1, dnf=None,
                        illumination_beta=0.5, illumination_rep_rate=250e3,
                        illuminance=1, numerical_aperture=1.0, pixel_size=6.5e-6,
                        magnification=1, debug=False, frame_overlap=1.0, powerlaw_DNF=None, **kwargs):

    # Convert mm/s to m/s
    motion_velocity_max /= 1000
    motion_acceleration /= 1000
    if motion_velocity:
        motion_velocity /= 1000

    # Calculate derived parameters
    params = acquisitionParameters(acquisition_strategy,
                                   camera_frame_rate=camera_frame_rate,
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
                                   illumination_rep_rate=illumination_rep_rate,
                                   illuminance=illuminance,
                                   numerical_aperture=numerical_aperture,
                                   magnification=magnification,
                                   debug=debug)

    pulse_time, motion_velocity_continuous, exposure_count_rate, exposure_saturation_time, system_pixel_size, fov, frame_time = params
    # Check if any values are zero
    if frame_time * pulse_time * motion_velocity * exposure_count_rate * exposure_saturation_time * fov[0] * frame_time == 0:
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
        total_throughput = frame_time - max(t_move + 2 * t_start_stop + motion_settle_time, camera_readout_time)

        # Ensure exposure is non-negative
        total_throughput = max(total_throughput, 0)

        # We can reduce exposure time to avoid saturation if necessary
        if debug and total_throughput > exposure_saturation_time:
            print('Clamping exposure!')
        total_throughput = min(total_throughput, exposure_saturation_time)

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
            total_throughput = pulse_time

            # No deconvolution here
            dnf = 1

        elif 'code' in acquisition_strategy:

            # Calculate maximum kernel length
            if pulse_time == 0:
                pulse_time = 1
                print('WARNING: Pulse time is zero!')

            pulse_count_max = int(round(exposure_saturation_time / pulse_time))
            pulse_count_max = min(pulse_count_max, int(round(fov[motion_axis] / system_pixel_size)))

            # Set kernel length to be a fraction of length it would take to saturate camera
            if pulse_count is None:
                strobe_pixel_counts = photon_pixel_rate * pulse_time * camera_quantum_efficency
                # kernel_length_px = int(math.floor(exposure_saturation_time / pulse_time))
                pulse_count = getOptimalPulseCountFromStrobe(strobe_pixel_counts,
                                                             camera_readout_noise=camera_readout_noise,
                                                             camera_ad_conversion=camera_ad_conversion)
            # Invalid as coded since this is actually strobed
            if np.round(pulse_count) <= 1.:
                pulse_count = 2

            # Ensure kernel length is not greater than maximum
            if pulse_count > pulse_count_max:
                pulse_count = pulse_count_max
                print('WARNING: Saturating pulse count')

            # Calculate DNF
            if pulse_count > 0:
                # Calculate DNF
                if dnf is None:
                    # if powerlaw_DNF is not None and kernel_length_px >= pulse_count / illumination_beta:
                    #     data = np.load(powerlaw_DNF)
                    #     dnf = data['coeffs'][1] * pulse_count ** data['coeffs'][0]
                    # else:
                    #     print('not precomputing!', kernel_length_px, pulse_count / illumination_beta)
                    #     dnf = getOptimalDnf(kernel_length_px,
                    #                     n_pulses=pulse_count,
                    dnf = getOptimalDnf(pulse_count * 2,
                                        method=dnf_calculation_method,
                                        pulse_count=pulse_count,
                                        beta=illumination_beta,
                                        n_tests=n_tests)

                # Determine exposure
                total_throughput = pulse_count * pulse_time

            else:
                if debug:
                    print('Invalid kernel length %d' % pulse_count)
                return (0, 1)  # Zero signal, dnf = 1

    # Filter exposure to saturation
    if total_throughput > exposure_saturation_time:
        if debug:
            print('Effective exposure time (%g s) is greater than saturation exposure time (%g s)' % (total_throughput, exposure_saturation_time))
        return (0, 1)
        # total_throughput = min(total_throughput, signal_exposure_saturate)

    return (total_throughput, dnf)


def parameterSweep(parameter_list, parameter_sweep_list, system_params, debug=False, powerlaw_DNF=None):
    # Perform a deep copy of system parameters
    system_params = copy.deepcopy(system_params)

    snr_coded_list, snr_sns_list, snr_strobe_list = [], [], []

    # Loop over frame rates
    for index, parameter_value_outer in enumerate(parameter_sweep_list[0]):

        # Initialize local lists
        snr_sns_sublist, snr_coded_sublist, snr_strobe_sublist = [], [], []

        # Set outer parameter
        system_params[parameter_list[0]] = parameter_value_outer

        # Loop over parameter values -- possibly paired
        for parameter_value in parameter_sweep_list[1]:

            # Set parameter(s)
            if hasattr(parameter_value, '__iter__'):
                for parameter, parameter_val in zip(parameter_list[1], parameter_value):
                    system_params[parameter] = parameter_val
            else:
                system_params[parameter_list[1]] = parameter_value

            #
            photon_pixel_rate = illuminanceToPhotonPixelRate(**system_params)

            # SNS
            t_sns, dnf_sns = frameRateToExposure(system_params['frame_rate'], photon_pixel_rate, 'stop_and_stare', **system_params, debug=debug)
            snr_sns = exposureTimeToSnr(t_sns, dnf=dnf_sns, **system_params, debug=debug)

            # Strobed
            t_strobe, dnf_strobe = frameRateToExposure(system_params['frame_rate'], photon_pixel_rate, 'strobe', **system_params, debug=debug)
            snr_strobe = exposureTimeToSnr(t_strobe, dnf=dnf_strobe, **system_params, debug=debug)

            # Coded
            t_coded, dnf_coded = frameRateToExposure(system_params['frame_rate'], photon_pixel_rate, 'code', **system_params, debug=debug, powerlaw_DNF=powerlaw_DNF)
            snr_coded = exposureTimeToSnr(t_coded, dnf=dnf_coded, **system_params, debug=debug)

            # Append
            snr_sns_sublist.append(snr_sns)
            snr_coded_sublist.append(snr_coded)
            snr_strobe_sublist.append(snr_strobe)

        # Append
        snr_coded_list.append(snr_coded_sublist)
        snr_sns_list.append(snr_sns_sublist)
        snr_strobe_list.append(snr_strobe_sublist)
    return snr_coded_list, snr_sns_list, snr_strobe_list


def plotParameterListSweep(parameter_list,
                           parameter_sweep_list,
                           system_params,
                           axis_values_list=None,
                           plot_type='rgb',
                           use_db=False,
                           cmap='viridis',
                           context='',
                           clim=None,
                           title=None,
                           figure_aspect_ratio=1.3,
                           parameter_is_log_scaled=False,
                           param_units=None,
                           divider_color='w',
                           show_hatches=False,
                           include_comparison=False,
                           show_colorbar=False,
                           normalize_pixels=True,
                           show_labels_y=True,
                           show_labels_x=True,
                           show_legend=True,
                           precomputed=None,
                           abbrev_label=False):

    if precomputed is None:
        snr_coded_list, snr_sns_list, snr_strobe_list = parameterSweep(parameter_list, parameter_sweep_list, system_params)
    else:
        try:
            data = np.load(precomputed)
            snr_coded_list = data['snr_coded_list']
            snr_sns_list = data['snr_sns_list']
            snr_strobe_list = data['snr_strobe_list']
        except FileNotFoundError:
            snr_coded_list, snr_sns_list, snr_strobe_list = parameterSweep(parameter_list, parameter_sweep_list, system_params)
            np.savez(precomputed, snr_coded_list=snr_coded_list,
                                       snr_sns_list=snr_sns_list, snr_strobe_list=snr_strobe_list)

    # Get axis values
    if axis_values_list is None:
        axis_values_list = parameter_sweep_list

    # Define axis label
    labels = []
    for parameter in parameter_list:
        label_string = str(parameter)
        for elt in ['(', ')', '\'']:
            label_string = label_string.replace(elt, '')
        label_string = label_string.replace(',', ' /')
        labels.append(label_string.replace('_', ' ').title())

    make2DPlots(snr_coded_list,
                snr_sns_list,
                snr_strobe_list,
                axis_values_list,
                labels,
                plot_type=plot_type,
                use_db=use_db,
                cmap=cmap,
                context=context,
                clim=clim,
                figure_aspect_ratio=figure_aspect_ratio,
                show_hatches=show_hatches,
                include_comparison=include_comparison,
                divider_color=divider_color,
                parameter_is_log_scaled=parameter_is_log_scaled,
                show_colorbar=show_colorbar,
                normalize_pixels=normalize_pixels,
                show_labels_y=show_labels_y,
                show_labels_x=show_labels_x,
                show_legend=show_legend,
                debug=False,
                abbrev_label=abbrev_label)
    return snr_coded_list, snr_sns_list, snr_strobe_list


def plotParameterSweep(parameter_name,
                       parameter_sweep,
                       dependent_variable,
                       dependent_variable_sweep,
                       system_parameters,
                       precomputed=None,
                       axis_values_list=None,
                       debug=False,
                       powerlaw_DNF=None,
                       **kwargs):
    if precomputed is None:
        snr_coded_list, snr_sns_list, snr_strobe_list = parameterSweep([dependent_variable, parameter_name],
                                                                       [dependent_variable_sweep, parameter_sweep],
                                                                       system_parameters, debug=debug, powerlaw_DNF=powerlaw_DNF)
    else:
        try:
            data = np.load(precomputed)
            snr_coded_list = data['snr_coded_list']
            snr_sns_list = data['snr_sns_list']
            snr_strobe_list = data['snr_strobe_list']
        except FileNotFoundError:
            snr_coded_list, snr_sns_list, snr_strobe_list = parameterSweep([dependent_variable, parameter_name],
                                                                           [dependent_variable_sweep, parameter_sweep],
                                                                           system_parameters, debug=debug, powerlaw_DNF=powerlaw_DNF)
            np.savez(precomputed,
                     snr_coded_list=snr_coded_list,
                     snr_sns_list=snr_sns_list,
                     snr_strobe_list=snr_strobe_list)

    # Get axis values
    if axis_values_list is None:
        if False: # isinstance(parameter_sweep[0], tuple):
             axis_values_list = dependent_variable_sweep, np.array([param[0] for param in parameter_sweep])
        else:
            axis_values_list = dependent_variable_sweep, parameter_sweep

    # Define axis label
    labels = []
    for parameter in [dependent_variable, parameter_name]:
        label_string = str(parameter)
        for elt in ['(', ')', '\'']:
            label_string = label_string.replace(elt, '')
        label_string = label_string.replace(',', ' /')
        label_string = label_string.replace('_', ' ').title()
        if label_string == 'Pulse Count':
            label_string = 'Total Throughput'
        labels.append(label_string)

    # Generate plots
    im = make2DPlots(snr_coded_list,
                snr_sns_list,
                snr_strobe_list,
                axis_values_list,
                labels,
                **kwargs)

    return snr_coded_list, snr_sns_list, snr_strobe_list, im


def make2DPlots(snr_coded_list,
                snr_sns_list,
                snr_strobe_list,
                axis_values_list,
                labels,
                ax=None,
                plot_type='rgb',
                use_db=False,
                cmap='viridis',
                context='',
                title=None,
                clim=None,
                units=None,
                debug=False,
                show_hatches=False,
                show_divider=True,
                divider_color='w',
                include_comparison=False,
                figure_aspect_ratio=1.3,
                parameter_is_log_scaled=False,
                x_is_log_scaled=False,
                contour_pixel_offset=0,
                interpolation='gaussian',
                show_colorbar=False,
                normalize_pixels=True,
                show_labels_y=True,
                show_labels_x=True,
                show_legend=True,
                abbrev_label=False,
                log_colors=True,
                text_list=None):


    # Initialize lists of SNRS
    snr_list = [np.asarray(snr_sns_list).T, np.asarray(snr_strobe_list).T, np.asarray(snr_coded_list).T]
    snr_list_sns = [np.asarray(snr_sns_list).T]
    snr_list_cont = [np.asarray(snr_strobe_list).T, np.asarray(snr_coded_list).T]
    argmax_array = np.argmax(snr_list, axis=0)
    max_array = np.maximum.reduce(snr_list)

    # Reduce Arrays if Abbreviating Labels
    if abbrev_label == 'optical':
        argmax_array[argmax_array < 1] = 0
        argmax_array = argmax_array > 1
        max_array = np.maximum.reduce(snr_list_cont)
    elif abbrev_label == 'mechanical':
        argmax_array = argmax_array > 0
        max_array = np.maximum.reduce(snr_list_sns + [np.maximum.reduce(snr_list_cont)])

    # Initialize arrays to plot
    imshow_array = None
    to_contour = None
    im = None

    if 'bar' in plot_type:
        if isinstance(axis_values_list[1][0], tuple):
            # redo y labels
            yticklabels = []
            for pair in axis_values_list[1]:
                yticklabels.append(str(pair[0])+'x / '+str(np.round(pair[1],2)))
        else:
            yticklabels = axis_values_list[1]
        # redo y positions
        new_y = np.arange(0,len(axis_values_list[1])+2)
        axis_values_list = (axis_values_list[0],new_y)


    # Make figure

    if plot_type in ['rgb', 'cmy', 'rgb_max', 'cmy_max']:
        # Create RGB image with the correct dimensions
        rgb_array = np.asarray(snr_list)
        rgb_array = np.transpose(rgb_array, (1,2,0))
        if 'max' in plot_type:
            max_ind = np.argmax(rgb_array, 2)
            max_array = np.max(rgb_array, 2)
            max_array = max_array
            rgb_array = np.asarray([np.zeros_like(max_array)] * 3).transpose(1, 2, 0)
            for ind in range(3):
                rgb_array[:, :, ind][max_ind == ind] = max_array[max_ind == ind]
            rgb_array[rgb_array == 0.0] = np.min(rgb_array)
            rgb_array -= np.min(rgb_array)
            rgb_array /= np.max(rgb_array)
            for ind in range(3):
                rgb_array[:, :, ind] /= np.mean(rgb_array[:, :, ind], 1)[:, np.newaxis]
        elif normalize_pixels:
            rgb_array /= np.max(rgb_array, 2)[:, :, np.newaxis]
            rgb_array = rgb_array
        if plot_type == 'cmy':
            rgb_array = 1 - rgb_array

        # parameters for figure
        show_colorbar = False
        imshow_array = rgb_array
        to_contour = None

    if 'combined' in plot_type:

        imshow_array = max_array

    elif 'regions' in plot_type:

        imshow_array = argmax_array.astype(np.float)
        imshow_array += (max_array > 0).astype(np.float)
        log_colors = False

    if 'regions' in plot_type:

        to_contour = argmax_array
        contour_x = axis_values_list[0].flatten() - contour_pixel_offset
        contour_y = axis_values_list[1].flatten() - contour_pixel_offset

    if to_contour is not None:
        if abbrev_label == 'mechanical':
            contour_lim = [0.5, 1.5]
            hatches = ['/', '-']
        elif abbrev_label == 'optical':
            contour_lim = [0.5, 1.5]
            hatches = ['/', '\\']
        else:
            contour_lim = [0.5, 1.5, 2.5]
            hatches = ['-', '/', '\\']

    if ax is None:
        fig, ax = plt.subplots()

    # color image background
    if imshow_array is not None:
        from matplotlib.colors import LogNorm, NoNorm
        if log_colors:
            cnorm = LogNorm()
        else:
            cnorm = NoNorm()

        if 'bar' not in plot_type:

            meshgrid = np.meshgrid(axis_values_list[0], axis_values_list[1])
            im = ax.scatter(meshgrid[0], meshgrid[1], c=imshow_array, norm=None, cmap=cmap,
                            edgecolors='none', marker='s',s=5, rasterized=True)

        else:
            meshgrid = np.meshgrid(axis_values_list[0], axis_values_list[1][1:-1])
            im = ax.scatter(meshgrid[0], meshgrid[1], c=imshow_array, norm=cnorm, cmap=cmap,
                    edgecolors='none', marker='s', s=5, rasterized=True)
            from matplotlib.patches import Rectangle

            # require that cmap was actual cmap argument
            get_rgba = lambda x: cmap(cnorm(x))

            # Add rectangles
            meshgrid_to_diff = np.meshgrid(np.hstack([axis_values_list[0],axis_values_list[0][-1]]), axis_values_list[1][1:-1])
            widths = np.diff(meshgrid_to_diff[0],axis=1).flatten()

            h = 0.5 #np.diff(meshgrid[1],axis=0).flatten()

            # (np.amax(axis_values_list[1]) - np.amin(axis_values_list[1])) / len(axis_values_list[1])
            for x,y,c,w in zip(meshgrid[0].flatten(), meshgrid[1].flatten(), imshow_array.flatten(), widths):
                ax.add_patch(Rectangle(xy=(x-w/2, y-h/2),
                             width=w*1.5, height=h, linewidth=0,
                             color=get_rgba(c), fill=True, rasterized=True))
            if yticklabels is None:
                yticklabels = axis_values_list[1][1:-1]

            if 'regions' in plot_type:

                for i,row in enumerate(argmax_array):
                    # finding x position
                    boundary = np.where(np.hstack([0, np.diff(row)]))[0]
                    y = axis_values_list[1][1+i]
                    w = 10; h = 1
                    for i in boundary:
                        x = axis_values_list[0][i]
                        w = 0.1 * x
                        ax.add_patch(Rectangle(xy=(x-w/2, y-h/2),
                             width=w, height=h, linewidth=0,
                             color='black', fill=True))
                to_contour = None

        if parameter_is_log_scaled:
            ax.set_yscale("log")
        if x_is_log_scaled:
            ax.set_xscale("log")
        ax.autoscale(enable=True, axis='both', tight=True)
        if clim is not None:
            im.set_clim(clim)
        if show_colorbar:
            plt.colorbar(im)

    ax.set_ylim(min(axis_values_list[1]), max(axis_values_list[1]))
    ax.set_xlim(min(axis_values_list[0]), max(axis_values_list[0]))

    # draw boundary for image regions
    if to_contour is not None:
        ax.contour(contour_x, contour_y, to_contour, contour_lim, colors=divider_color, linewidths=2)

    # Show hatches, if desired
    if show_hatches:
        ax.contourf(contour_x, contour_y, to_contour, contour_lim,
                    hatches=hatches, extend='both', alpha=0)

    if title is not None:
        ax.set_title(title)

    if show_labels_y:
        if units:
            ax.set_ylabel(labels[1] + ' (%s)' % units)
        else:
            ax.set_ylabel(labels[1])
    if show_labels_x:
        ax.set_xlabel(labels[0])

    # Add legend
    if show_legend:
        if not abbrev_label or abbrev_label == 'mechanical':
            ax.scatter([-1], [-1], marker='s', s=100, facecolor='white', hatch=3*hatches[1], label='SNS')
        if not abbrev_label or abbrev_label == 'optical':
            ax.scatter([-1], [-1], marker='s', s=100, facecolor='white', hatch=3*hatches[0], label='Strobed')
        if not abbrev_label or abbrev_label == 'optical':
            ax.scatter([-1], [-1], marker='s', s=100, facecolor='white', hatch=3*hatches[1], label='Coded')
        if abbrev_label == 'mechanical':
            ax.scatter([-1], [-1], marker='s', s=100, facecolor='white', hatch=3*hatches[0], label='Continuous')
        ax.set_ylim((min(axis_values_list[1]), max(axis_values_list[1])))
        ax.set_xlim((min(axis_values_list[0]), max(axis_values_list[0])))
        ax.legend()

    if 'bar' in plot_type:
        ax.set_yticks(axis_values_list[1][1:-1])
        ax.set_yticklabels(yticklabels)

    if text_list is not None:
        for text, pos in text_list:
            ax.text(pos[0], pos[1], text, color='black', size=18)

    return im

def add_colors_legend(ax, labels, colors, legend_pos, off_axes_pos=(-1, -1)):
    # adding legend
    from matplotlib.patches import Rectangle
    ps = []
    for color,label in zip(colors, labels):
        ps.append(ax.add_patch(Rectangle(xy=off_axes_pos,
                                         width=0.1, height=0.1, linewidth=0,
                                         color=color, fill=True, label=label)))
    ax.legend()


def exposureToCounts(exposure_time_s, camera_exposure_counts_per_unit=1, **kwargs):
    return int(round(exposure_time_s * camera_exposure_counts_per_unit))

def countsToSNR(signal_counts, dnf=1):
    # Use shot-noise calculation of counts to SNR
    return signal_counts / (dnf * np.sqrt(signal_counts))
