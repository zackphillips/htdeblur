"""
Copyright 2017 Zack Phillips, Sarah Dean

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

from . import solver
from . import project_simplex_box
from . import pgd

import llops as yp
import llops.operators as ops

from llops.solvers import iterative, objectivefunctions
from llops import iFt, Ft
from llops.config import default_backend, default_dtype

eps = 1e-13

def dnf(x):
    if len(x) == 0:
        return 0
    else:
        # x = x / np.sum(x)
        x_fft = np.fft.fft(x)
        sigma_x = np.abs(x_fft) ** 2
        return np.sqrt(1 / len(x) * np.sum(np.max(sigma_x) / sigma_x))


def cond(x):
    if len(x) == 0:
        return 0
    else:
        # x = x / np.sum(x)
        x_fft = np.fft.fft(x)
        sigma_x = np.abs(x_fft)
        return np.max(sigma_x) / np.min(sigma_x)

def vector(pulse_count, kernel_length=None,
           method='random_phase', n_tests=100, metric='dnf', dtype=None, backend=None):
    """
    This is a helper function for solving for a blur vector in terms of it's condition #
    """

    # Parse dtype and backend
    dtype = dtype if dtype is not None else yp.config.default_dtype
    backend = backend if backend is not None else yp.config.default_backend

    # Calculate kernel length if not provided
    if kernel_length is None:
        kernel_length = 2 * pulse_count

    # Compute many kernels
    kernel_list = []
    for _ in range(n_tests):
        # Generate blur kernel
        if method == 'random_phase':
            # Ensure first and last time point are illuminated
            indicies = np.random.choice(kernel_length, size=(pulse_count - 2), replace=False)
            illum = np.zeros(kernel_length)
            illum[indicies] = 1.0
            illum[0], illum[-1] = 1.0, 1.0
        elif method == 'random':
            illum = np.random.uniform(size=kernel_length)
        else:
            raise ValueError('Invalid kernel generation method %s' % method)

        # Append kernel to list
        kernel_list.append(illum)

    ## Choose best kernel
    if metric == 'cond':
        # Determine kernel with best condition #
        metric_best = 1e10
        kernel_best = []
        for kernel in kernel_list:
            kappa = cond(kernel)
            if kappa < metric_best:
                kernel_best = kernel
                metric_best = kappa

    elif metric == 'dnf':
        # Determine kernel with best dnf
        metric_best = 1e10
        kernel_best = []
        for kernel in kernel_list:
            _dnf = dnf(kernel)
            if _dnf < metric_best:
                kernel_best = kernel
                metric_best = _dnf
    else:
        raise ValueError

    # Normalize kernel
    kernel_best /= np.sum(kernel_best)

    # Cast
    kernel_best = yp.cast(kernel_best, dtype, backend)

    return (kernel_best, metric_best)


def kernel(shape, pulse_count, kernel_length=None, method='random_phase',
           n_tests=100, metric='dnf', axis=1, position='center'):

            # Generate blur vector
            blur_vector, _ = vector(pulse_count,
                                kernel_length=kernel_length,
                                method=method,
                                n_tests=n_tests,
                                metric=metric)

            # Generate kernel from vector
            return fromVector(blur_vector, shape=shape, axis=axis, position=position)


def generate(shape, blur_kernel_length, method='random_phase', axis=1,
             blur_illumination_fraction=0.5, position='center',normalize=True):
    # Generate blur kernel
    if method == 'constant':
        illum = yp.ones(blur_kernel_length) * blur_illumination_fraction
    elif method == 'random_phase' or method == 'coded':
        illum, _ = genRandInitialization(blur_kernel_length, blur_illumination_fraction)
    elif method == 'random' or  method == 'uniform':
        illum = np.random.uniform(size=blur_kernel_length)
    else:
        assert False, "method " + method + " unrecognized"

    # Generate kernel
    kernel = fromVector(illum, shape, axis, position, normalize=normalize)

    # Return kernel
    return kernel


def fromVector(blur_vector, shape, axis=1, position='center',
               normalize=True, reverse=False, interpolation_factor=1.0):
    """Converts a blur vector to a blur kernel."""

    # Get length of kernel
    blur_kernel_length = yp.size(blur_vector)

    # Find correct dimension
    ndims = len(shape)

    # Expand illum to 2D and ensure it's in the correct direction
    blur_vector = yp.expandDims(blur_vector, ndims)

    # Reverse blur vector if requested
    if reverse:
        blur_vector = yp.flip(blur_vector)

    # Ensure blur vector is 1D
    blur_vector = yp.vec(blur_vector)

    # Apply interpolation
    if interpolation_factor != 1.0:
        interpolated_length = int(np.round(interpolation_factor * len(blur_vector)))
        blur_vector = yp.real(yp.iFt(yp.pad(yp.Ft(blur_vector), interpolated_length, center=True)))

    # Ensure blur kernel has the correct dimensions
    blur_vector = yp.expandDims(blur_vector, ndims)

    # Rotate if necessary
    if axis == 1:
        blur_vector = blur_vector.T

    # Position kernel in image
    if position == 'center':
        kernel = yp.pad(blur_vector, shape, center=True)
    elif position == 'center_left':
        roll_amount = [0, 0]
        roll_amount[axis] = -blur_kernel_length // 2
        kernel = yp.roll(yp.pad(blur_vector, shape, center=True), roll_amount)
    elif position == 'center_right':
        roll_amount = [0, 0]
        roll_amount[axis] = blur_kernel_length // 2
        kernel = yp.roll(yp.pad(blur_vector, shape, center=True), roll_amount)
    elif position == 'origin':
        kernel = yp.pad(blur_vector, shape, crop_start=(0, 0))
    else:
        raise ValueError('Invalid position %s' % position)

    # Center kernel after pad. This is a hack.
    roll_values = [1] * yp.ndim(kernel)
    kernel = yp.roll(kernel, roll_values)

    # Normalize kernel
    if normalize:
        kernel /= yp.scalar(yp.sum(kernel))

    return kernel


######################################################################################################
################################ UTILITIES FOR READING FROM DATA #####################################
######################################################################################################

def blurVectorsFromDataset(dataset, dtype=None, backend=None, debug=False,
                           use_phase_ramp=False, corrections={}):
    """
    This function generates the object size, image size, and blur kernels from
    a comptic dataset object.

        Args:
            dataset: An io.Dataset object
            dtype [np.float32]: Which datatype to use for kernel generation (All numpy datatypes supported)
        Returns:
            object_size: The object size this dataset can recover
            image_size: The computed image size of the dataset
            blur_kernel_list: A dictionary of blur kernels lists, one key per color channel.

    """

    dtype = dtype if dtype is not None else yp.config.default_dtype
    backend = backend if backend is not None else yp.config.default_backend

    # Calculate effective pixel size if necessaey
    if dataset.metadata.system.eff_pixel_size_um is None:
        dataset.metadata.system.eff_pixel_size_um = dataset.metadata.camera.pixel_size_um / \
            (dataset.metadata.objective.mag * dataset.metadata.system.mag)

    # Recover and store position and illumination list
    blur_vector_roi_list = []
    position_list, illumination_list = [], []
    frame_segment_map = []

    for frame_index in range(len(dataset.frame_list)):
        frame_state = dataset.frame_state_list[frame_index]

        # Store which segment this measurement uses
        frame_segment_map.append(frame_state['position']['common']['linear_segment_index'])

        # Extract list of illumination values for each time point
        if 'illumination' in frame_state:
            illumination_list_frame = []
            for time_point in frame_state['illumination']['states']:
                illumination_list_time_point = []
                for illumination in time_point:
                    illumination_list_time_point.append(
                        {'index': illumination['index'], 'value': illumination['value']})
                illumination_list_frame.append(illumination_list_time_point)

        else:
            raise ValueError('Frame %d does not contain illumination information' % frame_index)


        # Extract list of positions for each time point
        if 'position' in frame_state:
            position_list_frame = []
            for time_point in frame_state['position']['states']:
                position_list_time_point = []
                for position in time_point:
                    if 'units' in position['value']:
                        if position['value']['units'] == 'mm':
                            ps_um = dataset.metadata.system.eff_pixel_size_um
                            position_list_time_point.append(
                                [1000 * position['value']['y'] / ps_um, 1000 * position['value']['x'] / ps_um])
                        elif position['value']['units'] == 'um':
                            position_list_time_point.append(
                                [position['value']['y'] / ps_um, position['value']['x'] / ps_um])
                        elif position['value']['units'] == 'pixels':
                            position_list_time_point.append([position['value']['y'], position['value']['x']])
                        else:
                            raise ValueError('Invalid units %s for position in frame %d' %
                                             (position['value']['units'], frame_index))
                    else:
                        # print('WARNING: Could not find posiiton units in metadata, assuming mm')
                        ps_um = dataset.metadata.system.eff_pixel_size_um
                        position_list_time_point.append(
                            [1000 * position['value']['y'] / ps_um, 1000 * position['value']['x'] / ps_um])

                position_list_frame.append(position_list_time_point[0])  # Assuming single time point for now.

            # Define positions and position indicies used
            positions_used, position_indicies_used = [], []
            for index, pos in enumerate(position_list_frame):
                for color in illumination_list_frame[index][0]['value']:
                    if any([illumination_list_frame[index][0]['value'][color] > 0 for color in illumination_list_frame[index][0]['value']]):
                        position_indicies_used.append(index)
                        positions_used.append(pos)

            # Generate ROI for this blur vector
            blur_vector_roi = getPositionListBoundingBox(positions_used)

            # Append to list
            blur_vector_roi_list.append(blur_vector_roi)

            # Crop illumination list to values within the support used
            illumination_list.append([illumination_list_frame[index] for index in range(min(position_indicies_used), max(position_indicies_used) + 1)])
            # Store corresponding positions
            position_list.append(positions_used)

    # Apply kernel scaling or compression if necessary
    if 'scale' in corrections:
        for index in range(len(position_list)):
            _positions = np.asarray(position_list[index])
            for ax in range(yp.shape(_positions)[1]):
                _positions[:, ax] = ((_positions[:, ax] - yp.min(_positions[:, ax])) * corrections['scale'] + yp.min(_positions[:, ax]))

            position_list[index] = _positions.tolist()

            blur_vector_roi_list[index].shape = [corrections['scale'] * sh for sh in blur_vector_roi_list[index].shape]

    # Synthesize blur vectors
    blur_vector_list = []
    for frame_index in range(len(dataset.frame_list)):
        #  Generate blur vectors
        if use_phase_ramp:
            kernel_shape = [yp.fft.next_fast_len(max(sh, 1)) for sh in blur_vector_roi_list[frame_index].shape]
            offset = yp.cast([sh // 2 + st for (sh, st) in zip(kernel_shape, blur_vector_roi_list[frame_index].start)], 'complex32')

            # Create phase ramp and calculate offset
            R = ops.PhaseRamp(kernel_shape, dtype='complex32')

            # Generate blur vector
            blur_vector = yp.zeros(R.M)
            for pos, illum in zip(position_list[frame_index], illumination_list[frame_index]):
                blur_vector += (R * (yp.cast(pos, 'complex32') - offset))

            # Take inverse Fourier Transform
            blur_vector = yp.abs(yp.cast(yp.iFt(blur_vector)), 0.0)

        else:
            blur_vector = yp.asarray([illum[0]['value']['w'] for illum in illumination_list[frame_index]],
                                     dtype=dtype, backend=backend)

        # Normalize illuminaiton vectors
        blur_vector /= yp.scalar(yp.sum(blur_vector))

        # Append to list
        blur_vector_list.append(blur_vector)

    # Subtract mininum of frame_segment_map
    frame_segment_map = [segment - min(frame_segment_map) for segment in frame_segment_map]

    # Return
    return blur_vector_list, blur_vector_roi_list, frame_segment_map, position_list, illumination_list


def blurKernelRecoveryFromStatic(blurred, static, solver='iterative', reg=None, iteration_count=10, system_otf=None, threshold=0.2):
    static_mean = np.mean(static)
    if static_mean > 1e-4:
        static = (static.copy() - static_mean) / static_mean

    blurred_mean = np.mean(blurred)
    if blurred_mean > 1e-4:
        blurred = (blurred.copy() - blurred_mean) / blurred_mean

    # if system_otf is not None:
    #     static = iFt(Ft(static) * system_otf)

    if solver == 'iterative':
        A = ops.Convolution(blurred.shape, static, mode='windowed')
        y = blurred.reshape(-1).astype(np.complex64)

        # Initialization: choosing a "good" coefficient value will help in convergence
        initialization = np.ones(y.shape, y.dtype)

        # Define cost function
        objective = objectivefunctions.L2(A, y, l2_reg=reg) #, reg=5e-3)


        # Gradient descent implementation
        kernel_recovered = iterative.GradientDescent(objective).solve(initialization=initialization,
                                                                  step_size=1e-3,
                                                                  nesterov_enabled=True,
                                                                  iteration_count=iteration_count,
                                                                  display_type='text',
                                                                  display_iteration_delta=max((iteration_count // 10),1))
    else:
        if reg is None:
            reg = 0
        kernel_recovered = iFt((np.conj(Ft(static)) * Ft(blurred)) / (np.abs(Ft(static)) ** 2 + reg))

    # Take real part
    kernel_recovered = np.real(kernel_recovered).reshape(static.shape)

    # Subtract low-frequency information
    kernel_recovered -= scipy.ndimage.filters.gaussian_filter(np.real(kernel_recovered.reshape(blurred.shape)), 10)

    # Filter by OTF support, threshold
    if system_otf is not None:
        kernel_recovered = np.real(iFt(Ft(kernel_recovered.reshape(blurred.shape)) * system_otf))
        kernel_recovered *= (kernel_recovered > threshold * np.max(kernel_recovered))

    return(kernel_recovered)

def registerDatasetImages(dataset, roi=None):
    from comptic.registration import registerImage
    shift_list = []
    image_list = []
    for index in range(1, len(dataset.frame_list)):
        if roi is not None:
            shift_list.append(registerImage(dataset.frame_list[index - 1][roi.slice],
                                            dataset.frame_list[index][roi.slice]))

            image_list.append((dataset.frame_list[index - 1][roi.slice], dataset.frame_list[index][roi.slice]))
        else:
            shift_list.append(registerImage(dataset.frame_list[index - 1], dataset.frame_list[index]))
        print(shift_list)
        print("Registered image %d of %d, shift was (%d, %d) pixels" %
              (index, len(dataset.frame_list), shift_list[-1][0], shift_list[-1]))
    return(shift_list, image_list)

def cropAndCenterKernel(kernel_recovered, kernel_size):
    # Center maximum value in blur kernel
    max_pos = np.unravel_index(np.argmax(kernel_recovered), kernel_recovered.shape)
    kernel_centered = np.roll(kernel_recovered, -np.asarray(max_pos) + np.asarray(kernel_recovered.shape) //2)

    # Crop to 2x blur kernel fov
    kernel_zeroed = np.zeros(kernel_centered.shape, dtype=kernel_centered.dtype)
    kernel_zeroed[kernel_centered.shape[0] // 2 - kernel_size[0]:kernel_centered.shape[0] // 2 + kernel_size[0],
                  kernel_centered.shape[1] // 2 - kernel_size[1]:kernel_centered.shape[1] // 2 + kernel_size[1]] = \
                    kernel_centered[kernel_centered.shape[0] // 2 - kernel_size[0]:kernel_centered.shape[0] // 2 + kernel_size[0],
                                  kernel_centered.shape[1] // 2 - kernel_size[1]:kernel_centered.shape[1] // 2 + kernel_size[1]]

    # Center at middle of blur kernel
    p = np.where(kernel_zeroed > 0)
    kernel_centered = np.roll(kernel_zeroed, -np.round(np.asarray((np.mean(p[0]), np.mean(p[1]))) + np.asarray(kernel_zeroed.shape) // 2).astype(np.int))

    kernel_size_small = kernel_size //2

    # Zero everything outside a resonable shift range
    kernel_zeroed_crop = np.zeros(kernel_centered.shape, dtype=kernel_centered.dtype)
    kernel_zeroed_crop[kernel_centered.shape[0] // 2 - kernel_size_small[0]:kernel_centered.shape[0] // 2 + kernel_size_small[0],
                  kernel_centered.shape[1] // 2 - kernel_size_small[1]:kernel_centered.shape[1] // 2 + kernel_size_small[1]] = \
                    kernel_centered[kernel_centered.shape[0] // 2 - kernel_size_small[0]:kernel_centered.shape[0] // 2 + kernel_size_small[0],
                                  kernel_centered.shape[1] // 2 - kernel_size_small[1]:kernel_centered.shape[1] // 2 + kernel_size_small[1]]
    return(kernel_zeroed_crop)


def plotBlurKernelList(blur_kernel_list, max_count_to_show=5, measurement_list=None, figsize=None):
    """ Plots a list of blur kernels and (optionally) corresponding measurements """

    count_to_show = min(max_count_to_show, len(blur_kernel_list))
    if figsize is None:
        plt.figure(figsize=(count_to_show * 2.5, 4 * (1 + int(measurement_list is not None))))
    else:
        plt.figure(figsize=figsize)

    for i in range(count_to_show):
        plt.subplot(1 + int(measurement_list is not None), count_to_show, i + 1)
        plt.imshow(blur_kernel_list[i], interpolation='bilinear')
        plt.title('Blur Kernel ' + str(i))


def illustrateMultiFrameKernel(blur_kernel_list, filename):
    """ Function which illustrates a multi-frame blur kernel and saves it to the disk"""
    image_c = np.zeros((blur_kernel_list[0].shape[0], blur_kernel_list[0].shape[1], 3))
    color_list = ['r', 'g', 'c', 'm', 'w', 'y']
    for index, blur_kernel in enumerate(blur_kernel_list):
        rgb = matplotlib.colors.to_rgb(color_list[index])
        image_c[:, :, 0] += blur_kernel * rgb[0]
        image_c[:, :, 1] += blur_kernel * rgb[1]
        image_c[:, :, 2] += blur_kernel * rgb[2]
    image_c /= np.amax(image_c)

    plt.figure()
    plt.imshow(image_c, interpolation='bilinear')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    plt.savefig(filename, transparent=True)


def genSamplingComb(object_size, image_size, dtype=np.complex64):
    """ Generates a comb function corresponding with seperation defined by
        image_size, centered at the center of object_size """
    sampling = np.floor(((np.asarray(object_size) / 2) / np.asarray(image_size)))
    sampling_comb = np.zeros(object_size, dtype=dtype)
    yy, xx = np.meshgrid(np.arange(-sampling[0], sampling[0] + 1), np.arange(-sampling[1], sampling[1] + 1))
    positions_0 = np.hstack((yy.ravel()[:, np.newaxis], xx.ravel()[:, np.newaxis])).astype(np.int)
    positions = np.zeros(positions_0.shape, dtype=positions_0.dtype)
    positions[:, 0] = object_size[0] // 2 + positions_0[:, 0] * image_size[0]
    positions[:, 1] = object_size[1] // 2 + positions_0[:, 1] * image_size[1]
    for position in positions:
        sampling_comb[position[0], position[1]] = 1

    positions -= np.asarray(object_size) // 2
    return((sampling_comb, positions))


def genConvolutionSupportList(blur_kernel_list, image_size, threshold=0.05):
    """
    This function generates a list of images defining the support of a windowed convolution operation.
    """
    object_size = blur_kernel_list[0].shape
    W = ops.Crop(object_size, image_size)
    kernel_support_mask = []
    object_support_mask = []
    print(W.dtype)
    window_mask = np.abs(W.H * W * np.ones(W.shape[1], dtype=np.complex64)).reshape(object_size)
    for blur_kernel in blur_kernel_list:
        C = ops.Convolution((blur_kernel > threshold).astype(np.complex64), mode='windowed',
                            pad_value=0, pad_size=int(object_size[0] / 2))
        kernel_support_mask += [((C * (window_mask.reshape(-1).astype(np.complex64))).reshape(object_size) > threshold)]
        object_support_mask.append(kernel_support_mask[-1])
        for dim in range(kernel_support_mask[-1].ndim):
            object_support_mask[-1] = np.flip(object_support_mask[-1], dim)

    return (kernel_support_mask, object_support_mask)


def blurKernelFromPositions(object_size, position_list, illum_list, flip_kernels=False, use_phase_ramp=False,
                            pos_perturbation=None, dtype=default_dtype, backend=default_backend):
    """
    This function generates a single blur kernel from a list of positions and illuminations. not multiframe.
    """

    # Initialize blur kernels
    blur_kernel = np.zeros(object_size, dtype=np.complex64)

    for position_index, position in enumerate(position_list):
        y = position[0]
        x = position[1]

        if pos_perturbation is not None:
            y = y + pos_perturbation[position_index, 0]
            x = x + pos_perturbation[position_index, 1]

        if not use_phase_ramp:
            x = int(round(x))
            y = int(round(y))

        # Assign illumination values
        if illum_list[position_index] > 0:
            if not use_phase_ramp:
                blur_kernel[y, x] += illum_list[position_index]
            else:
                R = ops.PhaseRamp(blur_kernel.shape, dtype=dtype, backend=backend)
                x_ = yp.astype(np.asarray((y - object_size[0] // 2, x - object_size[1] // 2)), R.dtype)
                ramp = yp.reshape(R * x_, blur_kernel.shape)
                blur_kernel += (ramp * illum_list[position_index])

    if use_phase_ramp:
        blur_kernel = iFt(blur_kernel)
        blur_kernel[blur_kernel < 1e-8] = 0.0

    if flip_kernels:
        blur_kernel = np.fliplr(blur_kernel)
    if np.sum(blur_kernel) > 0:
        blur_kernel /= np.sum(blur_kernel)

    return blur_kernel

def positionListToBlurKernelMap(kernel_size, position_list, return_fourier=True):
    """Function which converts a list of positions in a blur kernel to a full (non-sparse) blur kernel map.

    Args:
        kernel_size: Size of first two dimensions in blur_kernel_map
        position_list: List of x,y tuples which are the locaitons of each position in the blur kernel.
        return_fourier: Optional, enables return of blur kernels in frequency (Fourier) domain.

    Returns:
        A 2D blur_kernel_map, which has dimensions (kernel_size[0], kernel_size[1], size(position_list,1))
    """
    # TODO redundant
    print("can this be replaced with blurKernelFromPositions?")
    n_positions = np.size(position_list, 0)
    blur_kernel_map = np.zeros((n_positions, kernel_size[0], kernel_size[1]))

    for pos in np.arange(0, n_positions):
        blur_kernel_map[pos, position_list[pos, 0], position_list[pos, 1]] = 1

    if return_fourier:
        blur_kernel_map = Ft(blur_kernel_map.astype(np.complex64))

    return(blur_kernel_map)

def pointListToBlurKernel(kernel_size, position_list, illumination_vector):
    """Converts point list and illuminaiton vector to blur kernel"""
    # TODO redundant
    print("can this be replaced with blurKernelFromPositions?")
    position_count = np.size(position_list, 0)
    blur_kernel = np.zeros((kernel_size[0], kernel_size[1]))
    assert position_count == len(illumination_vector)

    for index, position in enumerate(position_list):
        blur_kernel[position[0], position[1]] = illumination_vector[index]

    return(blur_kernel)



def colorBlurKernelsToMonochrome(blur_kernel_list_color):
    """
    This function converts a list of color blur kernels to monochrome, assuming no optical effects.
        Args:
            blur_kernel_list_color: A dictionary of blur kernel lists, where each key indicates the illumination color channel of that kernel.
        Returns:
            A list of blur kernels which is the sum of the lists of each key in blur_kernel_list_color
    """
    blur_kernel_list = []
    for index, blur_kernel in enumerate(blur_kernel_list_color):
        first_channel = list(blur_kernel.keys())[0]
        new_kernel = np.zeros(blur_kernel[first_channel].shape, dtype=blur_kernel[first_channel].dtype)
        for channel in blur_kernel:
            new_kernel += blur_kernel[channel]
        blur_kernel_list.append(new_kernel)
    return(blur_kernel_list)


def getPositionListBoundingBox(kernel_position_list, use_mean=False):
    """
    This function returns the bounding box of a single blur kernel or list of blur kernels, defined as a list of positions
        Args:
            kernel_position_list: list of points (y,x)
        Returns:
            A list of the extreme values in the blur kernel in the format [y_min, y_max, x_min, x_max]
    """
    bounding_box = [1e10, -1e10, 1e10, -1e10]
    assert type(kernel_position_list) in [list, np.ndarray]

    # Make a single kernel_position_list a list with one element
    if type(kernel_position_list[0][0]) not in [list, np.ndarray, tuple]:
        kernel_position_list = [kernel_position_list]

    for position in kernel_position_list:
        if type(position[0][0]) in [np.ndarray, list, tuple]:
            # TODO: This will break if we blur by more than one pixel during each pixel motion
            if not use_mean:
                max_y, max_x = np.max(np.asarray(position), axis=0)[0]
                min_y, min_x = np.min(np.asarray(position), axis=0)[0]
            else:
                mean_y, mean_x = np.mean(np.asarray(position), axis=0)[0]
        else:
            if not use_mean:
                max_y, max_x = np.max(np.asarray(position), axis=0)
                min_y, min_x = np.min(np.asarray(position), axis=0)
            else:
                mean_y, mean_x = np.mean(np.asarray(position), axis=0)

        if not use_mean:
            bounding_box = [min(min_y, bounding_box[0]),
                            max(max_y, bounding_box[1]),
                            min(min_x, bounding_box[2]),
                            max(max_x, bounding_box[3])]
        else:
            bounding_box = [min(mean_y, bounding_box[0]),
                            max(mean_y, bounding_box[1]),
                            min(mean_x, bounding_box[2]),
                            max(mean_x, bounding_box[3])]
    # Create ROI object
    kernel_support_roi = yp.Roi(start=(int(round(bounding_box[0])), int(round(bounding_box[2]))),
                                end=(int(round(bounding_box[1])), int(round(bounding_box[3]))))

    return(kernel_support_roi)

######################################################################################################
##################################### AUTOCALIBRATION ################################################
######################################################################################################


class BsplineND():
    # from http://pythology.blogspot.com/2017/07/nd-b-spline-basis-functions-with-scipy.html
    def __init__(self, knots, degree=3, periodic=False):
        """
        :param knots: a list of the spline knots with ndim = len(knots)

        TODO (sarah) incorporate 2d aspect?
        """
        self.ndim = len(knots)
        self.splines = []
        self.knots = knots
        self.degree = degree
        for idim, knots1d in enumerate(knots):
            nknots1d = len(knots1d)
            y_dummy = np.zeros(nknots1d)
            knots1d, coeffs, degree = sp.interpolate.splrep(knots1d, y_dummy, k=degree,
                                                per=periodic)
            self.splines.append((knots1d, coeffs, degree))
        self.ncoeffs = [len(coeffs) for knots, coeffs, degree in self.splines]

    def evaluate_independent(self, position):
        """
        :param position: a numpy array with size [ndim, npoints]
        :returns: a numpy array with size [nspl1, nspl2, ..., nsplN, npts]
                  with the spline basis evaluated at the input points
        """
        ndim, npts = position.shape

        values_shape = self.ncoeffs + [npts]
        values = np.empty(values_shape)
        ranges = [range(icoeffs) for icoeffs in self.ncoeffs]
        for icoeffs in itertools.product(*ranges):
            values_dim = np.empty((ndim, npts))
            for idim, icoeff in enumerate(icoeffs):
                coeffs = [1.0 if ispl == icoeff else 0.0 for ispl in
                          range(self.ncoeffs[idim])]
                values_dim[idim] = sp.interpolate.splev(
                        position[idim],
                        (self.splines[idim][0], coeffs, self.degree))

            values[icoeffs] = np.product(values_dim, axis=0)
        return values

    def evaluate(self, position):
        assert self.weights is not None, "Must specify coefficients with set_coeffs()"
        values = self.evaluate_independent(position)
        return self.weights.dot(values)

    def set_weights(self, weights):
        assert len(weights) == self.ncoeffs[0], "must input correct number of weights"
        self.weights = weights

def get_basis_splines(extent, num_basis_fn):
    knotsx = np.linspace(0,extent-1,num_basis_fn)
    bspline = BsplineND([knotsx])
    pointsx1d = np.linspace(knotsx[0], knotsx[-1], extent)
    basis_matrix = bspline.evaluate_independent(pointsx1d[None, :])
    return basis_matrix[:num_basis_fn].T


def constructAlternatingMin(illuminations, shifts, image_size, n_frames, y):
    assert False, "DEPRECIATED, try getAutocalibrationFns"

def positions_to_splines(spl_basis, pos):
    # TODO (sarah) can be implemented as matrix inversion
    return np.linalg.pinv(spl_basis.T.dot(spl_basis)).dot(spl_basis.T.dot(pos))

    # def gradw(w):
    #     return spl_basis.T.dot( pos - spl_basis.dot(w))
    # for i in range(100):
    #     w = w + 0.1 * gradw(w)
    # return w

def get_monotone_projection(spline_basis):
    A = spline_basis[:-1] - spline_basis[1:]
    return lambda x: project_inequality(A, x)

def project_inequality(A, x):
    # projection onto Ax <= 0
    # TODO assumes A is real
    # TODO check that this is actually working: seems that resulting x is in full A nullspace...
    # assert len(x.shape==2)
    A_active = A[np.where(A.dot(x) > 0)[0]]
    if A_active.shape[0] == 0:
        return x
    if len(A_active.shape) < 2:
        A_active = np.expand_dims(A_active,1)
    AAinv = np.linalg.pinv(A_active.dot(A_active.T))
    P = A_active.T.dot(AAinv).dot(A_active)
    Px = P.dot(x)
    return x - Px

def getAutocalibrationFns(y, object_size, illums, basis_y, basis_x, weights_initial, object_initial,
                         dtype=None, backend=None, verbose=False):
    if verbose: print("generating BlurKernelBasis operator")
    Hsum = ops.BlurKernelBasis(object_size, (basis_y, basis_x), illums,
                            dtype=dtype, backend=backend, verbose=verbose)
    F = ops.FourierTransform(object_size, dtype=dtype, backend=backend, normalize=False)

    if verbose: print("defining diagonalized components")
    D_weights = ops.Diagonalize((Hsum * weights_initial).reshape(object_size), \
                                label='D_{shift}', dtype=dtype, backend=backend)
    # TODO (sarah) use windowing here
    D_object = ops.Diagonalize(F * object_initial, \
                               label='D_{object}', dtype=dtype, backend=backend)

    # Forward models
    if verbose: print("defining forward models")
    A_object = F.H * D_weights * F
    A_weights = F.H * D_object * Hsum

    # Objectives
    if verbose: print("defining objectives")
    L2 = ops.L2Norm(object_size, dtype=dtype, backend=backend)
    objective_object = L2 * (A_object - y)
    objective_weights = L2 * (A_weights - y)

    np_dtype = yp.getNativeDatatype(F.dtype, F.backend)

    objective_object_set_weights = lambda weights: objective_object.setArgument('D_{shift}', \
                                                                    Hsum * np.asarray(weights).astype(np_dtype))
    objective_weights_set_object = lambda object_update: objective_weights.setArgument('D_{object}', \
                                                            F * np.asarray(object_update).astype(np_dtype))
    objectives = [objective_object, objective_weights]
    update_fns = [objective_object_set_weights, objective_weights_set_object]
    return objectives, update_fns


def tand(x):
    return np.float(np.tan(x * np.pi / 180))


def sind(x):
    return np.float(np.sin(x * np.pi / 180))


def cosd(x):
    return np.float(np.cos(x * np.pi / 180))


def dnf2snr(dnf, exposureUnits, exposureCountsPerUnit=6553, darkCurrentE=0.9, patternNoiseE=3.9, readoutNoiseE=2.5, cameraBits=16, fullWellCapacity=30000):
    """Function which converts deconvolution noise factor to signal to noise ratio.
    Uses equations from https://www.photometrics.com/resources/learningzone/signaltonoiseratio.php and the dnf from the Agrawal and Raskar 2009 CVPR paper found here: http://ieeexplore.ieee.org/document/5206546/
    Default values are for the PCO.edge 5.5 sCMOS camera (https://www.pco.de/fileadmin/user_upload/pco-product_sheets/pco.edge_55_data_sheet.pdf)

    Args:
        dnf: Deconvolution noise factor as specified in Agrawal et. al.
        exposureUnits: exposure time, time units (normally ms)
        exposureCountsPerUnit: Average number of raw image counts for a 1 unit of exposureUnits exposure time
        darkCurrentE: Dark current from datasheet, units electrons
        patternNoiseE: Pattern noise from datasheet, units electrons
        readoutNoiseE: Readout noise from datasheet, units electrons
        cameraBits: Number of bits in camera

    Returns:
        A 2D numpy array which indicates the support of the optical system in the frequency domain.
    """
    countsToE = fullWellCapacity / (2 ** cameraBits - 1)
    return countsToE * exposureUnits * exposureCountsPerUnit \
        / (dnf * math.sqrt((countsToE * exposureCountsPerUnit + readoutNoiseE) * exposureUnits + (darkCurrentE + patternNoiseE)))


def genRandInitialization(n, beta, bounds=[0.0, 1.0], remainder=False):
    blurVec = np.zeros(n)

    # # Make a random assortment of columns 1
    # randSeed = (np.random.rand(n) * [bounds[1] - bounds[0]]) - bounds[0]
    # randSort = np.argsort(randSeed)
    # mask = randSort <= np.floor(beta * n)
    n_pulses = int(np.floor(beta * n))
    mask = np.random.choice(n, size=n_pulses, replace=False)

    # # Set these values to max
    blurVec[mask] = bounds[1]

    # # Assign the remainder to a value which is zero
    if remainder:
        zeroIdx = np.argsort(blurVec)
        blurVec[zeroIdx[0]] = (beta * n) % 1

    # Compute Condition Number
    condNum = max(abs(fftpack.fft(mask))) / min(abs(fftpack.fft(mask)))

    return blurVec, condNum


def condLowerBound(N, beta, bounds=[0, 1]):
    """Function which generates a lower bound on condition number using the PSD description of the optimal solution.
    Args:
        N: Number of positions on blur kernel
        beta: throughoput coefficient in range [0,1]
        bounds: bounds of resulting kernel, usually set to [0,1]

    Returns:
        Scalar condition number lower bound
    """

    # Theoretical maximum bound on sum(x^2)
    ps_max = np.floor(N * beta) + np.mod(N * beta, 1) ** 2

    # Convert power spectrum to frequency domain power specturm using Parseval's theorem
    ps_f = ps_max * N

    # This is the DC term in real space
    dc = N * beta

    # Compute mininum value of power spectra using parseval's theorem and the DC
    minPs = (ps_f - dc ** 2) / (N - 1)

    # Compute Condition Number
    kappa = dc / np.sqrt(minPs)

    return kappa


def dnfUpperBound(N, beta, bounds=[0, 1]):
    """Function which generates a upper bound on deconvolution noise factor (DNF) using the PSD description of the optimal solution.
    DNF is described in the Agrawal and Raskar 2009 CVPR paper found here: http://ieeexplore.ieee.org/document/5206546/

    Args:
        N: Number of positions on blur kernel
        beta: throughoput coefficient in range [0,1]
        bounds: bounds of resulting kernel, usually set to [0,1]

    Returns:
        Scalar dnf lower bound
    """

    # Theoretical maximum bound on sum(x^2)
    ps_max = np.floor(N * beta) + np.mod(N * beta, 1) ** 2

    # Convert power spectrum to frequency domain power specturm using Parseval's theorem
    ps_f = ps_max * N

    # This is the DC term in real space
    dc = N * beta

    # Compute mininum value of power spectra using parseval's theorem and the DC
    if N > 1:
        minPs = (ps_f - dc ** 2) / (N - 1)

        # Compute DNF
        dnf = (N - 1) * 1 / np.sqrt(minPs) + 1 / (dc)
    else:
        dnf = 1

    return dnf


def genKernelMapCol(colIdx, innerKernelMap, outerKernelMap, kernelSupport=None, supportThreshold=-1):
    """Function which generates a single column in a kernel map from an inner and outer base kernelMap.

    Args:
        innerKernelMap: kernelMap to pattern on inner dimension, as in the minor stride in the final kernel. Should be of same (x,y) size as outerKernelMap (2nd and 3rd dim)
        outerKernelMap: kernelMap to pattern on inner dimension, as in the major stride in the final kernel. Should be of same (x,y) size as innerKernelMap (2nd and 3rd dim)
        kernelSupport: Binary array for 2D support in both inner and outer kernelMaps
        supportThreshold: A threshold for the final kernelMap magnitude. Only used if kernelSupport = None (default)

    Returns:
        A 1D column from the kernelMap formed by innerKernelMap and outerKernelMap. Size in the first dimension will be the product of the first two dimensions in both innerKernelMap and outerKernelMap by default, or can be less if kernelSupport or supportThreshold are passed.
    """

    innerSize = np.size(innerKernelMap, 2)
    outerSize = np.size(outerKernelMap, 2)

    assert colIdx < innerSize * outerSize, "colIdx should be less than the product of the third dimensions in innerKernelMap and outerKernelMap"

    innerIdx = int(colIdx % innerSize)
    outerIdx = int(np.floor(colIdx / innerSize))

    kernelMapCol = (innerKernelMap[:, :, innerIdx] * outerKernelMap[:, :, outerIdx]).reshape(-1)

    if (np.any(kernelSupport) == None):
        if supportThreshold > 0:
            support = np.abs(kernelMapCol) > supportThreshold
        else:
            support = ones(innerKernelMap[:, :, 0].reshape(-1).shape)
    else:
        support = kernelSupport
    kernelMapCol = kernelMapCol[support.reshape(-1) > 0]

    return kernelMapCol


def genKernelMapSupport(innerKernelMap, outerKernelMap, supportThreshold=-1):
    """Function which generates a 2D support plot given both inner and outer kernelMaps.

    Args:
        innerKernelMap: kernelMap to pattern on inner dimension, as in the minor stride in the final kernel. Should be of same (x,y) size as outerKernelMap (2nd and 3rd dim)
        outerKernelMap: kernelMap to pattern on inner dimension, as in the major stride in the final kernel. Should be of same (x,y) size as innerKernelMap (2nd and 3rd dim)
        supportThreshold: A threshold for the final kernelMap magnitude.

    Returns:
        A 2D complete kernelMap with size equal to the first two dimensions in innerKernelMap and outerKernelMap. This is a binary array which indicates whether the magnitude of all combinations of inner and outerKernelMap will be greater than the supportThreshold at every position. By default, returns an array which is all True, unless supportThreshold is passed.
    """
    return (np.sum(np.abs(innerKernelMap), 2) / np.mean(np.sum(np.abs(innerKernelMap), 2)) * np.sum(np.abs(outerKernelMap)) / np.mean(np.sum(np.abs(outerKernelMap)))) > supportThreshold


def genKernelMap(innerKernelMap, outerKernelMap, kernelSupport=None, supportThreshold=-1):
    """Function which generates an kernel map from an inner and outer base kernelMap.

    Args:
        innerKernelMap: kernelMap to pattern on inner dimension, as in the minor stride in the final kernel. Should be of same (x,y) size as outerKernelMap (2nd and 3rd dim)
        outerKernelMap: kernelMap to pattern on inner dimension, as in the major stride in the final kernel. Should be of same (x,y) size as innerKernelMap (2nd and 3rd dim)
        kernelSupport: Binary array for 2D support in both inner and outer kernelMaps
        supportThreshold: A threshold for the final kernelMap magnitude. Only used if kernelSupport = None (default)

    Returns:
        A 2D complete kernelMap with second dimension equal to the product of the last dimensions in innerKernelMap and outerKernelMap. Size in the first dimension will be the product of the first two dimensions in both innerKernelMap and outerKernelMap by default, or can be less if kernelSupport or supportThreshold are passed.
    """
    # Number of columns to build
    if type(outerKernelMap) is type(None):
        outerKernelMap = np.ones((np.size(innerKernelMap, 0), np.size(innerKernelMap, 1), 1))

    if np.ndim(innerKernelMap) == 2:
        innerKernelMap = np.expand_dims(innerKernelMap, 2)

    if np.ndim(outerKernelMap) == 2:
        outerKernelMap = np.expand_dims(outerKernelMap, 2)

    outerSize = np.size(outerKernelMap, 2)
    innerSize = np.size(innerKernelMap, 2)
    nColumns = innerSize * outerSize

    # Generate support based on power spectra of each kernel
    if (kernelSupport) == None:
        if supportThreshold > 0:
            support = genKernelMapSupport(innerKernelMap, outerKernelMap, supportThreshold=supportThreshold)
        else:
            support = np.ones(innerKernelMap[:, :, 0].reshape(-1).shape)
    else:
        support = kernelSupport
    kernelMap = np.zeros((np.sum(support > 0), nColumns), dtype=np.complex64)
    for colIdx in np.arange(0, innerSize * outerSize):
        kernelMap[:, colIdx] = genKernelMapCol(colIdx, innerKernelMap, outerKernelMap, kernelSupport=support)

    return(kernelMap)



######################################################################################################
##################################### PATHWAY GENERATION #############################################
######################################################################################################


def genMotionPathIncrimentPlot(blur_kernel_map):
    """Function which generates a 2D kernel map for illustration where every sequential position is it's index in blur_kernel_map

    Args:
        blur_kernel_map: A blur kernel map, usually a bunch of delta functions. Should be a 3D ndarray

    Returns:
        A 2D ndarray where each value is it's index in the input blur_kernel_map
    """
    kernel_map_full = np.zeros((np.size(blur_kernel_map, 0), np.size(blur_kernel_map, 1)))
    for idx in np.arange(np.size(blur_kernel_map, 2)):
        kernel_map_full = kernel_map_full + blur_kernel_map[:, :, idx] * idx
    return kernel_map_full


def genRasterMotionPathwayOld(object_size, image_size, full_object_multi_pass=0, measurement_redundancy=1):
    """Function which generates a list of points which make up a complete raster scan of a given FOV, given a small capture FOV

    Args:
        image_size: Capture frame size in (y,x)
        object_size: Sample size in (y,x), should be larger than image_size
        full_object_multi_pass: (0) Flag to force kernel generation to scan full object multiple times instead of dividing it into segments. If between 0 and 1, scans the object in halves.
        measurement_redundancy: redundancy in x, increases number of measurements by this factor

    Returns:
        A 2D ndarray where each value is it's index in the input blur_kernel_map
    """

    # Determine major axis
    major_axis = np.argmax(np.asarray(object_size))
    if object_size[0] == object_size[1]:
        major_axis = 1

    if major_axis == 0:
        object_size = np.flip(object_size, 0)
        image_size = np.flip(image_size, 0)

    measurement_count = np.ceil(np.asarray(object_size) / np.asarray(image_size)
                                ).astype(np.int)  # two components in x and y

    assert np.any(measurement_count > 1), "image_size must be smaller than object_size!"
    print("Image size requires %d x %d images" % (measurement_count[0], measurement_count[1]))

    measurement_count[1] = int(measurement_redundancy * measurement_count[1])

    raster_segments = np.zeros((measurement_count[0] * 2, 2), dtype=np.int)

    y_stride = object_size[0] / measurement_count[0]
    x_stride = object_size[1] / measurement_count[1]

    minor_stride_side = 'r'

    raster_point_list = []
    for row in np.arange(measurement_count[0]):

        # Place the vertical upright
        if minor_stride_side == 'r':
            raster_segments[(2 * row), :] = [np.ceil(image_size[0] * (row + 0.5)).astype(int),
                                             np.ceil(image_size[1] * 0.5).astype(int)]
            raster_segments[(2 * row) + 1, :] = [np.ceil(image_size[0] * (row + 0.5)).astype(int),
                                                 np.ceil(object_size[1] - image_size[1] * 0.5).astype(int)]
            minor_stride_side = 'l'
        else:
            raster_segments[(2 * row), :] = [np.ceil(image_size[0] * (row + 0.5)).astype(int),
                                             np.ceil(object_size[1] - image_size[1] * 0.5).astype(int)]
            raster_segments[(2 * row) + 1, :] = [np.ceil(image_size[0] * (row + 0.5)
                                                         ).astype(int), np.ceil(image_size[1] * 0.5).astype(int)]
            minor_stride_side = 'r'

        # Determine movement direction of this row in X
        if raster_segments[(2 * row), 1] < raster_segments[(2 * row) + 1, 1]:
            move_direction_x = 1
        else:
            move_direction_x = -1

        # always move down one
        move_direction_y = 1

        # Determine points to use for horizontal scan
        if row == 0:
            if measurement_count[0] == 1:
                x_position_list = np.arange(0, object_size[1], move_direction_x)
            else:
                x_position_list = np.arange(0, raster_segments[(2 * row) + 1, 1], move_direction_x)

        elif row == measurement_count[0] - 1:
            x_position_list = np.arange(raster_segments[(2 * row), 1], object_size[1], move_direction_x)
        else:
            x_position_list = np.arange(raster_segments[(2 * row), 1],
                                        raster_segments[(2 * row) + 1, 1], move_direction_x)

        for position_x in x_position_list:
            raster_point_list.append([raster_segments[(2 * row), 0], position_x])

        # Vertical scan
        if np.ceil(image_size[0] * (row + 1.5)) < object_size[0]:
            for position_y in np.arange(np.ceil(image_size[0] * (row + 0.5)), np.ceil(image_size[0] * (row + 1.5))).astype(int):
                raster_point_list.append([position_y.astype(int), raster_segments[(2 * row) + 1, 1]])

    raster_point_list = np.asarray(raster_point_list)

    # Determine number of points per image
    points_per_image = np.floor(raster_point_list.shape[0] / np.prod(measurement_count))
    measurement_indicies = np.arange(raster_point_list.shape[0])
    measurement_indicies = np.floor(measurement_indicies / points_per_image)

    # If full_object_multi_pass flag is specified, we want to scan the object backwards
    # and forwards multiple times instead of dividing it up into segments.
    raster_point_list_segmented = []
    for measurement_index in range(np.prod(measurement_count)):
        if not full_object_multi_pass:
            raster_point_list_segmented.append(raster_point_list[measurement_indicies == measurement_index, :])
        elif full_object_multi_pass < 1:
            midpoint = int(np.ceil(raster_point_list.shape[0] / 2))
            if measurement_index % 2:
                if measurement_index % 3:
                    raster_point_list_segmented.append(raster_point_list[midpoint:, :])
                else:
                    raster_point_list_segmented.append(np.flip(raster_point_list[0:midpoint, :], axis=0))
            else:
                if measurement_index % 4:
                    raster_point_list_segmented.append(np.flip(raster_point_list[midpoint:, :], axis=0))
                else:
                    raster_point_list_segmented.append(raster_point_list[0:midpoint, :])
        else:
            if measurement_index % 2:
                raster_point_list_segmented.append(raster_point_list)
            else:
                raster_point_list_segmented.append(np.flip(raster_point_list, axis=0))

    # Transpose points if user desires
    if major_axis == 0:
        return(np.flip(raster_point_list_segmented, axis=2))
    else:
        return(raster_point_list_segmented)


def gen90Corner(image_size, orientation='ru'):
    position_list = []
    for x_position in np.arange(0, np.ceil(image_size[1] * 0.5).astype(int)):
        position_list.append([np.ceil(image_size[0] * 0.5).astype(int), x_position])
    for y_position in np.arange(np.ceil(image_size[0] * 0.5).astype(int), image_size[0]):
        position_list.append([y_position, np.ceil(image_size[1] * 0.5).astype(int)])
    if orientation == 'rl':
        return flip_pts(position_list, image_size, ['ud', 'reverse'])
    elif orientation == 'lu':
        return flip_pts(position_list, image_size, ['lr'])
    elif orientation == 'll':
        return flip_pts(position_list, image_size, ['lr', 'ud', 'reverse'])
    else:
        return position_list


def genRasterMotionPathway(object_size, image_size, corner_gen_fn=gen90Corner, full_object_multi_pass=0, measurement_redundancy=1):
    """Function which generates a list of points which make up a complete raster scan of a given FOV, given a small capture FOV

    Args:
        image_size: Capture frame size in (y,x)
        object_size: Sample size in (y,x), should be larger than image_size
        corner_gen_fn: function that generates corner shapes with given image size, offset, and orientation
        full_object_multi_pass: (0) Flag to force kernel generation to scan full object multiple times instead of dividing it into segments. If between 0 and 1, scans the object in halves.
        measurement_redundancy: redundancy in x, increases number of measurements by this factor

    Returns:
        A 2D ndarray where each value is it's index in the input blur_kernel_map
    """

    # Determine major axis
    major_axis = np.argmax(np.asarray(object_size))
    if object_size[0] == object_size[1]:
        major_axis = 1

    if major_axis == 0:
        object_size = np.flip(object_size, 0)
        image_size = np.flip(image_size, 0)

    measurement_count = np.ceil(np.asarray(object_size) / np.asarray(image_size)
                                ).astype(np.int)  # two components in x and y

    assert np.any(measurement_count > 1), "image_size must be smaller than object_size!"
    print("Image size requires %d x %d images" % (measurement_count[0], measurement_count[1]))

    measurement_count[1] = int(measurement_redundancy * measurement_count[1])

    raster_segments = np.zeros((measurement_count[0] * 2, 2), dtype=np.int)

    y_stride = object_size[0] / measurement_count[0]
    x_stride = object_size[1] / measurement_count[1]

    minor_stride_side = 'r'

    raster_point_list = []
    for row in np.arange(measurement_count[0]):

        if row == 0:
            if measurement_count[0] == 1:  # if final row
                # straight line only
                x_position_list = np.arange(0, object_size[1])
                y_position = np.ceil(image_size[0] * 0.5).astype(int)
                for position_x in x_position_list:
                    raster_point_list.append([y_position, position_x])
            else:
                # straight line
                x_position_list = np.arange(0, object_size[1] - image_size[1])
                y_position = np.ceil(image_size[0] * 0.5).astype(int)
                for position_x in x_position_list:
                    raster_point_list.append([y_position, position_x])
                # plus corner
                for position_x, position_y in corner_gen_fn(image_size, orientation='ru'):
                    raster_point_list.append([position_y, position_x + object_size[1] - image_size[1]])
        elif row % 2:  # odd
            for position_x, position_y in corner_gen_fn(image_size, orientation='rl'):
                raster_point_list.append([position_y + row * image_size[0],
                                          position_x + object_size[1] - image_size[1]])
            if measurement_count[0] == row + 1:  # final row: straight line
                x_position_list = np.arange(0, object_size[1] - image_size[1], -1)
                y_position = np.ceil(image_size[0] * 0.5 + row * image_size[0]).astype(int)
                for position_x in x_position_list:
                    raster_point_list.append([y_position, position_x])
            else:
                # straight portion
                x_position_list = np.arange(image_size[1], object_size[1] - image_size[1], -1)
                y_position = np.ceil(image_size[0] * 0.5 + row * image_size[0]).astype(int)
                for position_x in x_position_list:
                    raster_point_list.append([y_position, position_x])
                # corner
                for position_x, position_y in corner_gen_fn(image_size, orientation='lu'):
                    raster_point_list.append([position_y + row * image_size[0], position_x])
        else:  # even
            for position_x, position_y in corner_gen_fn(image_size, orientation='ll'):
                raster_point_list.append([position_y + row * image_size[0], position_x])
            if measurement_count[0] == row + 1:  # final row: straight line
                x_position_list = np.arange(image_size[1], object_size[1])
                y_position = np.ceil(image_size[0] * 0.5 + row * image_size[0]).astype(int)
                for position_x in x_position_list:
                    raster_point_list.append([y_position, position_x])
            else:
                # straight portion
                x_position_list = np.arange(image_size[1], object_size[1] - image_size[1])
                y_position = np.ceil(image_size[0] * 0.5 + row * image_size[0]).astype(int)
                for position_x in x_position_list:
                    raster_point_list.append([y_position, position_x])
                # corner
                for position_x, position_y in corner_gen_fn(image_size, orientation='ru'):
                    raster_point_list.append([position_y + row * image_size[0],
                                              position_x + object_size[1] - image_size[1]])
    raster_point_list = np.asarray(raster_point_list)

    # Determine number of points per image
    points_per_image = np.floor(raster_point_list.shape[0] / np.prod(measurement_count))
    measurement_indicies = np.arange(raster_point_list.shape[0])
    measurement_indicies = np.floor(measurement_indicies / points_per_image)

    # If full_object_multi_pass flag is specified, we want to scan the object backwards
    # and forwards multiple times instead of dividing it up into segments.
    raster_point_list_segmented = []
    for measurement_index in range(np.prod(measurement_count)):
        if not full_object_multi_pass:
            raster_point_list_segmented.append(raster_point_list[measurement_indicies == measurement_index, :])
        elif full_object_multi_pass < 1:
            midpoint = int(np.ceil(raster_point_list.shape[0] / 2))
            if measurement_index % 2:
                if measurement_index % 3:
                    raster_point_list_segmented.append(raster_point_list[midpoint:, :])
                else:
                    raster_point_list_segmented.append(np.flip(raster_point_list[0:midpoint, :], axis=0))
            else:
                if measurement_index % 4:
                    raster_point_list_segmented.append(np.flip(raster_point_list[midpoint:, :], axis=0))
                else:
                    raster_point_list_segmented.append(raster_point_list[0:midpoint, :])
        else:
            if measurement_index % 2:
                raster_point_list_segmented.append(raster_point_list)
            else:
                raster_point_list_segmented.append(np.flip(raster_point_list, axis=0))

    # Transpose points if user desires
    if major_axis == 0:
        return(np.flip(raster_point_list_segmented, axis=2))
    else:
        return(raster_point_list_segmented)


def flip_pts(points, image_size, orientations):
    new_points = points
    for orientation in orientations:
        if orientation == 'reverse':
            new_points = np.flipud(new_points)
        else:
            if orientation == 'ud':
                def point_op(point): return (point[0], image_size[0] - point[1])
            elif orientation == 'lr':
                def point_op(point): return (image_size[1] - point[0], point[1])
            else:
                assert 0, 'unrecognized orientation'
            new_points = [point_op(point) for point in new_points]
    return new_points


# messy separate version for now, eventually merge custom corner logic with everything else to subsume this case
def genLinearRasterMotionPathway(object_size, image_size, full_object_multi_pass=0, measurement_redundancy=1):
    """Function which generates a list of points which make up a complete raster scan of a given FOV, given a small capture FOV

    Args:
        image_size: Capture frame size in (y,x)
        object_size: Sample size in (y,x), should be larger than image_size
        full_object_multi_pass: (0) Flag to force kernel generation to scan full object multiple times instead of dividing it into segments. If between 0 and 1, scans the object in halves.
        measurement_redundancy: redundancy in x, increases number of measurements by this factor

    Returns:
        A 2D ndarray where each value is it's index in the input blur_kernel_map
    """

    # Determine major axis
    major_axis = np.argmax(np.asarray(object_size))
    if object_size[0] == object_size[1]:
        major_axis = 1
    if major_axis == 0:
        object_size = np.flip(object_size, 0)
        image_size = np.flip(image_size, 0)

    measurement_count = np.ceil(object_size / image_size).astype(np.int)  # two components in x and y

    assert np.any(measurement_count > 1), "image_size must be smaller than object_size!"
    print("Image size requires %d x %d images" % (measurement_count[0], measurement_count[1]))

    measurement_count[1] = int(measurement_redundancy * measurement_count[1])

    raster_segments = np.zeros((measurement_count[0] * 2, 2), dtype=np.int)

    y_stride = object_size[0] / measurement_count[0]
    x_stride = object_size[1] / measurement_count[1]

    raster_point_list = []
    for row in np.arange(measurement_count[0]):

        # Place the vertical upright
        raster_segments[(2 * row), :] = [np.ceil(image_size[0] * (row + 0.5)).astype(int),
                                         np.ceil(image_size[1] * 0.5).astype(int)]
        raster_segments[(2 * row) + 1, :] = [np.ceil(image_size[0] * (row + 0.5)).astype(int),
                                             np.ceil(object_size[1] - image_size[1] * 0.5).astype(int)]

        # Determine points to use for horizontal scan
        x_position_list = np.arange(0, object_size[1], 1)

        for position_x in x_position_list:
            raster_point_list.append([raster_segments[(2 * row), 0], position_x])

    raster_point_list = np.asarray(raster_point_list)

    # Determine number of points per image
    points_per_image = np.floor(raster_point_list.shape[0] / np.prod(measurement_count))
    measurement_indicies = np.arange(raster_point_list.shape[0])
    measurement_indicies = np.floor(measurement_indicies / points_per_image)

    # If full_object_multi_pass flag is specified, we want to scan the object backwards
    # and forwards multiple times instead of dividing it up into segments.
    raster_point_list_segmented = []
    for measurement_index in range(np.prod(measurement_count)):
        if not full_object_multi_pass:
            raster_point_list_segmented.append(raster_point_list[measurement_indicies == measurement_index, :])
        elif full_object_multi_pass < 1:
            midpoint = int(np.ceil(raster_point_list.shape[0] / 2))
            if measurement_index % 2:
                if measurement_index % 3:
                    raster_point_list_segmented.append(raster_point_list[midpoint:, :])
                else:
                    raster_point_list_segmented.append(np.flip(raster_point_list[0:midpoint, :], axis=0))
            else:
                if measurement_index % 4:
                    raster_point_list_segmented.append(np.flip(raster_point_list[midpoint:, :], axis=0))
                else:
                    raster_point_list_segmented.append(raster_point_list[0:midpoint, :])
        else:
            if measurement_index % 2:
                raster_point_list_segmented.append(raster_point_list)
            else:
                raster_point_list_segmented.append(np.flip(raster_point_list, axis=0))

    # Transpose points if user desires
    if major_axis == 0:
        return(np.flip(raster_point_list_segmented, axis=2))
    else:
        return(raster_point_list_segmented)


def genCustomRasterPathway(image_size, object_size, corner_fn, measurement_redundancy=1):
    # very rough function, to be improved later
    assert (object_size / image_size == [3, 3]).all(), 'only design for 3x3 grid'
    line_list = []
    line = genTwoPointLineBlurposition_list((0, int(image_size[0] / 2)), (image_size[1], int(image_size[0] / 2)))
    line_list.append(line)
    line = genTwoPointLineBlurposition_list(
        (image_size[1], int(image_size[0] / 2)), (2 * image_size[1], int(image_size[0] / 2)))
    line_list.append(line)
    line = corner_fn(image_size, offset=(2 * image_size[1], 0))
    line_list.append(line)
    line = corner_fn(image_size, offset=(2 * image_size[1], image_size[0]), orientations=['ud', 'reverse'])
    line_list.append(line)
    line = genTwoPointLineBlurposition_list(
        (2 * image_size[0], image_size[1] + int(image_size[1] / 2)), (image_size[0], image_size[1] + int(image_size[1] / 2)))
    line_list.append(line)
    line = corner_fn(image_size, offset=(0, image_size[0]), orientations=['lr'])
    line_list.append(line)
    line = corner_fn(image_size, offset=(0, 2 * image_size[0]), orientations=['lr', 'ud', 'reverse'])
    line_list.append(line)
    line = genTwoPointLineBlurposition_list(
        (image_size[1], 2 * image_size[0] + int(image_size[0] / 2)), (2 * image_size[1], 2 * image_size[0] + int(image_size[0] / 2)))
    line_list.append(line)
    line = genTwoPointLineBlurposition_list(
        (2 * image_size[1], 2 * image_size[0] + int(image_size[0] / 2)), (3 * image_size[1], 2 * image_size[0] + int(image_size[0] / 2)))
    line_list.append(line)

    point_list_segmented = []
    for line in line_list:
        if measurement_redundancy == 2:
            middle = int(np.floor((len(line) - 1) / 2))
            point_list_segmented.append(np.asarray(line[:middle]))
            point_list_segmented.append(np.asarray(line[middle:-1]))
        else:
            point_list_segmented.append(np.asarray(line[:-1]))
    # for points in point_list_segmented:
    #    points[np.where(points >= 3*image_size)] = 3*image_size[0]-1
    return point_list_segmented


def generate_open_corner(image_size, offset=(0, 0), orientations=[]):
    midside = (int(image_size[0] / 2), int(image_size[1] / 2))
    diamond_points = [(0, midside[0]), (int(image_size[1] / 3), int(image_size[0] / 6)),
                      (image_size[1], image_size[0])]
    diamond_points = flip_pts(diamond_points, image_size, orientations)
    line_list = []
    for i in range(len(diamond_points) - 1):
        p1 = tuple(p + q for p, q in zip(diamond_points[i], offset))
        p2 = tuple(p + q for p, q in zip(diamond_points[i + 1], offset))
        line = genTwoPointLineBlurposition_list(p1, p2)
        line_list.append(line)
    return np.concatenate(line_list)


def generate_diamond_corner(image_size, offset=(0, 0), orientations=[]):
    midside = (int(image_size[0] / 2), int(image_size[1] / 2))
    diamond_points = [(0, midside[0]), (midside[1], int(image_size[0] / 5)), (int(4 * image_size[1] / 5), midside[0]),
                      (midside[1], image_size[0])]
    diamond_points = flip_pts(diamond_points, image_size, orientations)
    line_list = []
    for i in range(len(diamond_points) - 1):
        p1 = tuple(p + q for p, q in zip(diamond_points[i], offset))
        p2 = tuple(p + q for p, q in zip(diamond_points[i + 1], offset))
        line = genTwoPointLineBlurposition_list(p1, p2)
        line_list.append(line)
    return np.concatenate(line_list)


def genRasterMotionPathway_fallback(object_size, image_size, full_object_multi_pass=False):
    """Function which generates a list of points which make up a complete raster scan of a given FOV, given a small capture FOV
    Args:
        image_size: Capture frame size in (y,x)
        object_size: Sample size in (y,x), should be larger than image_size
        full_object_multi_pass: (False) Flag to force kernel generation to scan full object multiple times instead of dividing it into segments
    Returns:
        A 2D ndarray where each value is it's index in the input blur_kernel_map
    """

    measurement_count = np.ceil(object_size / image_size).astype(np.int)  # two components in x and y

    assert np.any(measurement_count > 1), "image_size must be smaller than object_size!"
    print("Image size requires %d x %d images" % (measurement_count[0], measurement_count[1]))

    raster_segments = np.zeros((measurement_count[0] * 2, 2), dtype=np.int)

    y_stride = object_size[0] / measurement_count[0]
    x_stride = object_size[1] / measurement_count[1]

    minor_stride_side = 'r'

    raster_point_list = []
    for row in np.arange(measurement_count[0]):

        # Place the vertical upright
        if minor_stride_side == 'r':
            raster_segments[(2 * row), :] = [np.ceil(image_size[0] * (row + 0.5)).astype(int),
                                             np.ceil(image_size[1] * 0.5).astype(int)]
            raster_segments[(2 * row) + 1, :] = [np.ceil(image_size[0] * (row + 0.5)).astype(int),
                                                 np.ceil(object_size[1] - image_size[1] * 0.5).astype(int)]
            minor_stride_side = 'l'
        else:
            raster_segments[(2 * row), :] = [np.ceil(image_size[0] * (row + 0.5)).astype(int),
                                             np.ceil(object_size[1] - image_size[1] * 0.5).astype(int)]
            raster_segments[(2 * row) + 1, :] = [np.ceil(image_size[0] * (row + 0.5)
                                                         ).astype(int), np.ceil(image_size[1] * 0.5).astype(int)]
            minor_stride_side = 'r'

        # Determine movement direction of this row in X
        if raster_segments[(2 * row), 1] < raster_segments[(2 * row) + 1, 1]:
            move_direction_x = 1
        else:
            move_direction_x = -1

        # always move down one
        move_direction_y = 1

        # Determine points to use for horizontal scan
        if row == 0:
            if measurement_count[0] == 1:
                x_position_list = np.arange(0, object_size[1], move_direction_x)
            else:
                x_position_list = np.arange(0, raster_segments[(2 * row) + 1, 1], move_direction_x)

        elif row == measurement_count[0] - 1:
            x_position_list = np.arange(raster_segments[(2 * row), 1], object_size[1], move_direction_x)
        else:
            x_position_list = np.arange(raster_segments[(2 * row), 1],
                                        raster_segments[(2 * row) + 1, 1], move_direction_x)

        for position_x in x_position_list:
            raster_point_list.append([raster_segments[(2 * row), 0], position_x])

        # Vertical scan
        if np.ceil(image_size[0] * (row + 1.5)) < object_size[0]:
            for position_y in np.arange(np.ceil(image_size[0] * (row + 0.5)), np.ceil(image_size[0] * (row + 1.5))).astype(int):
                raster_point_list.append([position_y.astype(int), raster_segments[(2 * row) + 1, 1]])

    raster_point_list = np.asarray(raster_point_list)

    # Determine number of points per image
    points_per_image = np.floor(raster_point_list.shape[0] / np.prod(measurement_count))
    measurement_indicies = np.arange(raster_point_list.shape[0])
    measurement_indicies = np.floor(measurement_indicies / points_per_image)

    # If full_object_multi_pass flag is specified, we want to scan the object backwards
    # and forwards multiple times instead of dividing it up into segments.
    raster_point_list_segmented = []
    for measurement_index in range(np.prod(measurement_count)):
        if not full_object_multi_pass:
            raster_point_list_segmented.append(raster_point_list[measurement_indicies == measurement_index, :])
        else:
            if measurement_index % 2:
                raster_point_list_segmented.append(raster_point_list)
            else:
                raster_point_list_segmented.append(np.flip(raster_point_list, axis=0))

    return(raster_point_list_segmented)


def genTwoPointLineBlurposition_list(startPos, endPos):
    """Function which generates a blur kernel map which make a linear pathway between two points

    Args:
        kernel_size: Tuple with size in x and y, should be integer
        startPos: Start position, should be tuple of integers
        endPos: End position, should be tuple of integers

    Returns:
        A list of x,y positions to generate a blur kernel map
    """
    from skimage.draw import line

    rr, cc = line(startPos[1], startPos[0], endPos[1], endPos[0])

    # Convert list to array (TODO: Make this work with lists instead of arrays)
    return np.asarray([rr, cc]).T


# Generate linear blur kernel map as a list of positions
def genLinearBlurKernelMapPositionList(kernel_size, n_positions, point_seperation=0, centered=True, centerOffset=(0, 0)):
    """Function which generates an example blur kernel map which make a linear pathway.

    Args:
        kernel_size: Tuple with size in x and y, should be integer
        n_positions: Length of kernel, should be integer
        point_seperation: Seperation between points, should be integer

    Returns:
        A list of x,y positions to generate a blur kernel map
    """

    startPos = round(kernel_size[1] / 2 - (point_seperation + 1) * (n_positions / 2))
    height = round(kernel_size[0] / 2)

    position_list = np.zeros((n_positions, 2))
    position_list[:, 1] = startPos + np.arange(0, (point_seperation + 1) *
                                               n_positions, (point_seperation + 1)) - centerOffset[1]
    position_list[:, 0] = height - centerOffset[0]

    position_list = position_list.astype(np.int16)

    if not centered:
        position_list[:, 0] = position_list[:, 0] - np.ceil(kernel_size[0] * 0.5).astype(int)
        position_list[:, 1] = position_list[:, 1] - \
            np.ceil(kernel_size[1] * 0.5).astype(int) + (point_seperation + 1) * (n_positions / 2)

    return(position_list)


def genCircularBlurKernelMapposition_list(kernel_size, radius, sweepAngle, startAngle=0, center=(0, 0)):
    """Function which generates an example blur kernel map which make an arc pathway. Uses sweep angle as an input.

    Args:
        kernel_size: Tuple with size in x and y, should be integer
        radius: Desired Radius of arc, can be double or integer
        sweepAngle: Desired angle of arc in degrees
        startAngle: Angle from which to start arc, degrees
        center: Center of Arc (x,y), integer

    Returns:
        A list of x,y positions to generate a blur kernel map
    """

    # Generate circle
    x = np.arange(-(kernel_size[1] - np.mod(kernel_size[1], 2)) / 2, (kernel_size[1] -
                                                                      np.mod(kernel_size[1], 2)) / 2 - (np.mod(kernel_size[1], 2) == 1)) - center[1]
    y = np.arange(-(kernel_size[0] - np.mod(kernel_size[0], 2)) / 2, (kernel_size[0] -
                                                                      np.mod(kernel_size[0], 2)) / 2 - (np.mod(kernel_size[0], 2) == 1)) - center[0]
    [xx, yy] = np.meshgrid(x, y)
    fullCircle = np.abs((xx ** 2 + yy ** 2) - radius ** 2) < 40

    M2 = ((-sind(startAngle + 180 + startAngle)) * xx) <= ((cosd(startAngle + sweepAngle + 180)) * yy)
    M3 = (-sind(startAngle + 180) * xx > cosd(startAngle + 180) * yy)

    fullCircle = fullCircle * M2 * M3
    n_positions = np.sum(fullCircle)
    print("generated %d positions" % n_positions)

    position_list = np.zeros((n_positions, 2))

    # [zp] I'm sure there is a faster way to do this...
    it = np.nditer(fullCircle, flags=['multi_index'])
    sIdx = 0
    posIdx = 0
    while not it.finished:
        if it[0] > 0:
            position_list[posIdx, 1] = it.multi_index[0]
            position_list[posIdx, 0] = it.multi_index[1]
            posIdx = posIdx + 1
        it.iternext()

    position_list = position_list.astype(np.int16)
    return(position_list)

# Generate circular blur kernel position list by n_positions


def genCircularBlurKernelMapposition_listN(kernel_size, radius, n_positions, startAngle=0):
    """Function which generates an example blur kernel map which make an arc pathway. Uses number of positions as an input.

    Args:
        kernel_size: Tuple with size in x and y, should be integer
        radius: Desired Radius of arc, can be double or integer
        n_positions: Length of kernel, should be integer
        startAngle: Angle from which to start arc, degrees

    Returns:
        A list of x,y positions to generate a blur kernel map
    """

    # Generate circle
    x = np.arange(-(kernel_size[1] - mod(kernel_size[1], 2)) / 2, (kernel_size[1] -
                                                                   mod(kernel_size[1], 2)) / 2 - (mod(kernel_size[1], 2) == 1))
    y = np.arange(-(kernel_size[0] - mod(kernel_size[0], 2)) / 2, (kernel_size[0] -
                                                                   mod(kernel_size[0], 2)) / 2 - (mod(kernel_size[0], 2) == 1))
    [xx, yy] = np.meshgrid(x, y)
    fullCircle = np.abs((xx**2 + yy**2) - radius**2) < 40

    position_list = np.zeros((n_positions, 2))

    # [zp] I'm sure there is a faster way to do this...
    it = np.nditer(fullCircle, flags=['multi_index'])
    sIdx = 0
    posIdx = 0
    while not it.finished and posIdx < n_positions:
        if it[0] > 0:
            position_list[posIdx, 1] = it.multi_index[0]
            position_list[posIdx, 0] = it.multi_index[1]
            posIdx = posIdx + 1
        it.iternext()

    position_list = position_list.astype(np.int16)
    return(position_list)

# Generate diagonal blur kernel position list


def genDiagonalBlurKernelMapposition_list(kernel_size, n_positions, point_seperation, slope=-1):
    """Function which generates an example blur kernel map which has a diagonal pathway (slope of -1)
    Args:
        kernel_size: Tuple with size in x and y, should be integer
        n_positions: Length of kernel, should be integer
        point_seperation: Seperation between points, should be integer
    Returns:
        A list of x,y positions to generate a blur kernel map
    """

    startPos = int(np.round(kernel_size[1] / 2 + slope * (point_seperation + 1) * (n_positions / 2)))

    position_list = np.zeros((n_positions, 2))
    position_list[:, 1] = startPos - slope * np.arange(0, n_positions)
    position_list[:, 0] = slope * position_list[:, 1]

    position_list = position_list.astype(np.int16)
    return(position_list)



######################################################################################################
##################################### ILLUMINATION GENERATION ########################################
######################################################################################################


def genIllum_pseudoRandom_len(kernel_length, beta=0.5, n_tests=10, led_count=1):
    """
    This is a helper function for solving for a blur vector in terms of it's condition #
    """
    kernel_list = []
    for test in range(n_tests):
        n_elements_max = math.floor(beta * kernel_length * led_count)
        kernel = np.zeros(kernel_length * led_count)
        indicies = np.arange(kernel_length * led_count)
        for index in range(n_elements_max):
            rand_index = np.random.randint(0, high=np.size(indicies) - 1, size=1)
            kernel[indicies[rand_index]] = 1.
            indicies = np.delete(indicies, rand_index)

        rand_index = np.random.randint(0, high=np.size(indicies), size=1)

        kernel[rand_index] = beta * kernel_length * led_count - np.sum(kernel)
        assert beta * kernel_length * led_count - np.sum(kernel) <= 1
        kernel_list.append(kernel)

    # Determine kernel with best conditioon #
    kappa_best = 1e10
    kernel_best = []
    for kernel in kernel_list:
        spectra = np.abs(np.fft.fft(kernel))
        kappa = np.max(spectra) / max(np.min(spectra), eps)
        if kappa < kappa_best:
            kernel_best = kernel
            kappa_best = kappa

    kernel_best = kernel_best.reshape((kernel_length, led_count))
    return (kappa_best, kernel_best)

# crude random search method using point lists


def genIllum_randomSearch(point_list, object_size_0, maxiter=100, throughputCoeff=0.5):
    best_sv = 0
    illum, _ = genRandInitialization(len(point_list), throughputCoeff, bounds=[0, 1])
    best_illum = illum

    for i in range(maxiter):
        blur_kernel = np.zeros(object_size_0)
        for position_index, position in enumerate(point_list):
            blur_kernel[position[0], position[1]] = illum[position_index]
        blurKernelF = Ft(blur_kernel)
        minSv = np.amin(np.abs(blurKernelF))
        if minSv > best_sv:
            best_sv = minSv
            best_illum = illum
        illum, _ = genRandInitialization(len(point_list), throughputCoeff, bounds=[0, 1])
    return best_illum, best_sv


def genIllum_pseudoRandom(blurMapCol, p, maxiter=1000, throughputCoeff=0.5, resultType='final', verbose=False):
    # note: this funtion now takes in the blurmapCol function rather than the map itself
    obj = solver.kernel_objectives(blurMapCol, 1)

    result = {}
    result['history'] = {}
    result['history']['f'] = [-1] * (maxiter + 1)
    result['history']['x'] = [np.empty(p)] * (maxiter + 1)

    # initialize first value, for compatability with other genIllums
    blurVec, k = genRandInitialization(p, throughputCoeff, bounds=[0, 1])
    result['history']['f'][0] = obj.conditionNumber(blurVec)
    result['history']['x'][0] = blurVec

    # Set up verbose printing
    if verbose == True:
        print("  Iter  |  Value ")

    # Iteration Loop
    for itr in np.arange(1, maxiter + 1):
        blurVec, _ = genRandInitialization(p, throughputCoeff, bounds=[0, 1])
        k = obj.conditionNumber(blurVec)
        if verbose == True:
            print("   %02d   |  %0.02f " % k)

        # Return full or partial result depending on user imput
        result['history']['f'][itr] = k
        result['history']['x'][itr] = blurVec

    # Store best result
    from operator import itemgetter
    index, kmin = min(enumerate(result['history']['f']), key=itemgetter(1))
    result['fopt'] = kmin
    result['xopt'] = result['history']['x'][index]
    result['it'] = maxiter  # change if we change stopping criterion
    return result


def genIllum_GS(realSpaceSupport, fourierSpaceSupport, throughputCoeff=0.5,
                resultType='final', verbose=False, usePureRandInit=False, maxiter=50):
    """Function which generates blur kernel using a modified Gerchberg-Saxton Algorithm

    Args:
        kernelMap_blur: A kernelMap which contains path information in real space
        opticalSupport: A mask which defines the optical support in the frequency domain
        throughputCoeff: A scalar coefficient in [0,1], usually set to 0.5
        resultType: How to return the result dictionary. Options are "final" which returns only the final result, and "full", which returns kernels from all iterations.
        usePureRandInit: Whether to use a pure random initialization or the output of the genRandInitialization function as an initialization
        maxiter: the maximum number of iterations

    Returns:
        A dictionary with two fields: "condNum" contains the condition numbers in all iterations, and "illumVector" contains the LED intensities which lead to this condition number. If resultType is "final" (default), the best condition number can be accessed using result['condNum'][-1].

    """

    def Ft(x): return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x, axes=(0, 1)), axes=(0, 1), norm='ortho'), axes=(0, 1))

    def iFt(x): return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(
        x, axes=(0, 1)), axes=(0, 1), norm='ortho'), axes=(0, 1))

    image_size = fourierSpaceSupport.shape
    p = np.sum(realSpaceSupport)

    # DC Term in Fourier Space
    dc = np.zeros(image_size, dtype=np.double)
    dc[np.ceil(image_size[0] / 2).astype(np.int), np.ceil(image_size[1] / 2).astype(np.int)] = (throughputCoeff * p) ** 2

    maxPSD = p * (np.floor(throughputCoeff * p) + np.remainder(throughputCoeff * p, 1.0)
                  ** 2)  # max power spectrum in fourier domain
    minPS = (maxPSD - np.sum(dc)) / np.sum(fourierSpaceSupport)

    powerSpectrum = minPS * fourierSpaceSupport + dc

    # Check Power spectrum integrity
    assert np.any(np.abs(np.sum(powerSpectrum) - maxPSD) <= 1e-3), "Power spectrum does not match PSD criterion."

    # initialize result dictionary
    result = {}
    if resultType == 'full':
        result['history'] = {}
        result['history']['f'] = [-1] * (maxiter + 1)
        result['history']['x'] = [np.empty(p)] * (maxiter + 1)

    # Initialize with random initialization
    blurKernel = np.zeros(image_size, dtype=np.complex64)
    if usePureRandInit:
        blurKernel[realSpaceSupport] = np.random.rand(p)
        # blurKernel[realSpaceSupport] = np.ones(p)
    else:
        blurKernel[realSpaceSupport], k = genRandInitialization(p, throughputCoeff)

    # Project initilization into valid simplex
    blurKernel[realSpaceSupport] = project_simplex_box.project(
        np.real(blurKernel[realSpaceSupport]), throughputCoeff * p, alg='is')

    # Add initialization to results
    result['init'] = {}
    result['init']['f'] = np.amax(np.abs(Ft(blurKernel))[fourierSpaceSupport]) / \
        np.amin(np.abs(Ft(blurKernel))[fourierSpaceSupport])
    result['init']['x'] = np.abs(blurKernel[realSpaceSupport])

    # Store initialization in history variable
    if (resultType == "full"):
        result['history']['f'][0] = np.amax(np.abs(Ft(blurKernel))[fourierSpaceSupport]) / \
            np.amin(np.abs(Ft(blurKernel))[fourierSpaceSupport])
        result['history']['x'][0] = np.abs(blurKernel[realSpaceSupport])

    # Set up verbose printing
    if verbose == True:
        print("  Iter  |  Value ")

    # Iteration Loop
    for itr in np.arange(1, maxiter + 1):
        # Enforce power spectrum in frequency domain
        blurKernel_ft = np.sqrt(powerSpectrum).astype(np.complex64) * \
            np.exp(1j * fourierSpaceSupport * np.angle(Ft(blurKernel)))

        # Enforce support in real domain
        blurKernel = iFt(blurKernel_ft)
        blurKernel = blurKernel * realSpaceSupport

        # Perform projection onto simplex box
        blurKernel[realSpaceSupport] = project_simplex_box.project(
            np.real(blurKernel[realSpaceSupport]), throughputCoeff * p, alg='is')

        if verbose == True:
            print("   %02d   |  %0.02f " % (itr, np.amax(np.abs(Ft(blurKernel))[
                  fourierSpaceSupport]) / np.amin(abs(Ft(blurKernel))[fourierSpaceSupport])))

        # Return full or partial result depending on user imput
        if (resultType == 'full'):
            result['history']['f'][itr] = np.amax(
                np.abs(Ft(blurKernel))[fourierSpaceSupport]) / np.amin(np.abs(Ft(blurKernel))[fourierSpaceSupport])
            result['history']['x'][itr] = np.abs(blurKernel[realSpaceSupport])

    # Store result
    result['fopt'] = np.amax(np.abs(Ft(blurKernel))[fourierSpaceSupport]) / \
        np.amin(np.abs(Ft(blurKernel))[fourierSpaceSupport])
    result['xopt'] = np.abs(blurKernel[realSpaceSupport])
    result['it'] = maxiter  # change if we change stopping criterion

    return result


def genIllum(blurMapCol, nColumns, throughputCoeff=0.5, DNF=False, verbose=False,
             usePureRandInit=False, maxiter=500, resultType='final', init=None):
    """Function which generates blur kernel using a projected gradient algorithm

    Args:
        blurMapCol: A function which returns columns of the blur map
        nColumns: number of columns in the blur map
        throughputCoeff: A scalar coefficient in [0,1], usually set to 0.5
        DNF: flag to indicate use of the DNF objective function (otherwise, condition number is default)
        verbose: flag for printing each iteratation of the gradent algorithm
        usePureRandInit: Whether to use a pure random initialization or the output of the genRandInitialization function as an initialization
        maxiter: the maximum number of iterations
        resultType: How to return the result dictionary. Options are "final" which returns only the final result, and "full", which returns kernels from all iterations.
                    note that the full type runs to the full input maxiter rather than using stopping criteria to stop after convergence.


    Returns:
        A dictionary with fields xopt, it, and history. If resultType is full, history contains all iterates and function values, x and f.
        note that the returned f's are smoothed versions of the objective functions. To compute actual values of objective functions, the iterates must be used directly.

    """
    # initialize result dictionary
    result = {}

    n = nColumns
    beta = throughputCoeff

    obj = solver.kernel_objectives(blurMapCol, 1)

    # Define Peojection
    def projis(v):
        return project_simplex_box.project(v, beta * n, alg='is')

    # Define Objective Gradient and Function
    if DNF:
        f = obj.svSquaredReciprocalSumSmooth
        f_true = obj.svSquaredReciprocalSum
        f_true2 = obj.svSquaredReciprocalSum
        grad = obj.gradSvSquaredReciprocalSum
    else:
        f = obj.minSvSquaredSmooth
        f_true = obj.conditionNumber
        f_true2 = obj.minSvSquared
        grad = obj.gradMinSvSquared

    # Define Initialization
    if init is None:
        if usePureRandInit:
            x0 = np.random.rand(n)
        else:
            x0, kappa = genRandInitialization(n, beta)
    else:
        x0 = init

    # True objective function

    x0 = projis(x0)
    while True:
        try:
            if (resultType == 'full'):
                x, fval = pgd.projectedIterativeMax_developement(
                    x0, obj, f, grad, projis, pgd.backtrackingstep, pgd.smoothing_pow, maxiter, verbose=verbose)
            else:
                xstar, it = pgd.projectedIterativeMax(
                    x0, obj, f, grad, projis, pgd.backtrackingstep, pgd.smoothing_pow, maxiter, verbose=verbose)
            break
        except ArithmeticError:
            print('no projection convergence, restarting')
            x0 = np.random.rand(n)
            x0 = projis(x0)

    # Store result
    if resultType == 'full':
        result['history'] = {}
        result['history']['f_smooth'] = fval
        result['history']['f'] = []
        for itr in np.arange(maxiter):
            result['history']['f'].append(f_true(x[:, itr]))
        result['history']['x'] = x
        result['it'] = maxiter
        result['xopt'] = x[:, maxiter - 1]
        result['fopt'] = f_true(x[:, maxiter - 1])
        result['fopt2'] = f_true2(x[:, maxiter - 1])
    else:
        result['xopt'] = xstar
        result['fopt'] = f_true(xstar)
        result['fopt2'] = f_true2(xstar)
        result['it'] = it

    return result

# Development verison, may be unstable


def genIllumDev(blurMapCol, nColumns, throughputCoeff, sumobj=False, verbose=False,
                usePureRandInit=False, maxiter=500, resultType='final'):
    n = nColumns
    beta = throughputCoeff
    obj = solver.kernel_objectives(blurMapCol, 1)

    # Define Peojection
    def projis(v):
        return project_simplex_box.project(v, beta * n, alg='is')

    # Define Objective Gradient and Function
    if sumobj:
        f = obj.svSquaredReciprocalSumSmooth
        grad = obj.gradSvSquaredReciprocalSum
    else:
        f = obj.minSvSquaredSmooth
        grad = obj.gradMinSvSquared

    # Define Initialization
    if usePureRandInit:
        x0 = np.random.rand(n)
    else:
        x0, kappa = genRandInitialization(n, beta)

    # Project x0 onto cardinal simplex
    x0 = projis(x0)

    # Set up results dictionary
    result = {}
    # result['history']['f'] = [-1] * (maxiter + 1) # include initialization as first element
    # result['history']['x'] =[np.empty(p)] * (maxiter + 1) # include initialization as first element

    # Store initialization as first variable in history
    # result['history']['x'][0] = f(x0)
    # result['history']['x'][0] = x0

    while True:
        try:
            xstar, it = optimize_pgd.projectedIterativeMax(
                x0, obj, f, grad, projis, optimize_pgd.backtrackingstep, optimize_pgd.smoothing_pow, maxiter, verbose=verbose)

            break
        except ArithmeticError:
            print('no projection convergence, restarting')
            x0 = np.random.rand(n)
            x0 = projis(x0)

    # Store results inside result dictionary
    result['xopt'] = xstar
    result['fopt'] = obj.conditionNumber(xstar)
    result['it'] = it
    return result



def plotIllumMap(illumVector, ledPointListNa, markerSz=50, plot_colormap="Grays"):
    from plotly.offline import init_notebook_mode, iplot
    init_notebook_mode(connected=True)
    n_positions = illumVector.shape[0]
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }

    # Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis

    # fill in most of layout
    maxR = 1.5 * max(abs(ledPointListNa).reshape(-1))

    figure['layout']['paper_bgcolor'] = 'rgba(0,1,0,1)'
    figure['layout']['plot_bgcolor'] = 'rgba(0,1,0,1)'
    figure['layout']['xaxis'] = {'range': [-maxR, maxR], 'title': 'NA (x)'}
    figure['layout']['yaxis'] = {'range': [-maxR, maxR], 'title': 'NA (y)'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['height'] = 560
    figure['layout']['width'] = 500
    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 200, 'redraw': False},
                                    'fromcurrent': True, 'transition': {'duration': 100, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                                      'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Position:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # make data for first value
    position = 0
    ledVals = illumVector[position, :]
    data_dict = {
        'x': list(ledPointListNa[:, 0]),
        'y': list(ledPointListNa[:, 1]),
        'mode': 'markers',
        'text': list(arange(0, size(ledPointListNa[:, 1]))),
        'marker': {
            'sizemode': 'area',
            'sizeref': 200000,
            'size': markerSz,
            'color': ledVals,
            'colorscale': plot_colormap
        },
        'name': " "
    }
    figure['data'].append(data_dict)

    # make frames
    for position in arange(0, n_positions):
        ledVals = illumVector[position, :]
        frame = {'data': [], 'name': str(position)}

        data_dict = {
            'x': list(ledPointListNa[:, 0]),
            'y': list(ledPointListNa[:, 1]),
            'mode': 'markers',
            'text': list(arange(0, size(ledPointListNa[:, 1]))),
            'marker': {
                'sizemode': 'area',
                'sizeref': 200000,
                'size': markerSz,
                'color': ledVals,
                'colorscale': plot_colormap
            },
            'name': " "
        }
        frame['data'].append(data_dict)

        figure['frames'].append(frame)
        slider_step = {'args': [
            [position],
            {'frame': {'duration': 100, 'redraw': False},
             'mode': 'immediate',
             'transition': {'duration': 100}}
        ],
            'label': position,
            'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    figure['layout']['sliders'] = [sliders_dict]
    iplot(figure)


class BlurKernelBasis(ops.Operator):
    # inputs are weights for singular_vectors functions
    # operator first translates weights to a position,
    # then returns the resulting PhaseRamp

    # TODO in progress

    def __init__(self, N, basis, illums, verbose=False, dtype=None, backend=None, label='R'):

        # Configure backend and datatype
        backend = backend if backend is not None else yp.config.default_backend
        dtype = dtype if dtype is not None else yp.config.default_dtype

        x = np.arange(-N[1] / 2, N[1] / 2, 1.0) / N[1]
        y = np.arange(-N[0] / 2, N[0] / 2, 1.0) / N[0]
        xx, yy = np.meshgrid(x, y)
        ry = -2 * np.pi * yy
        rx = -2 * np.pi * xx

        self.verbose = verbose

        # Convert to the correct dtype and backend: TODO why through numpy?
        dtype_np = yp.getNativeDatatype(dtype, 'numpy')
        self.rx = yp.changeBackend(rx.astype(dtype_np), backend)
        self.ry = yp.changeBackend(ry.astype(dtype_np), backend)
        basis_y, basis_x = basis
        self.basis_y = yp.changeBackend(basis_y.astype(dtype_np), backend)
        self.basis_x = yp.changeBackend(basis_x.astype(dtype_np), backend)
        self.ndim_y = yp.shape(self.basis_y)[1]
        self.ndim_x = yp.shape(self.basis_x)[1]
        self.illums = illums # TODO backend and dtype?

        # TODO remove this
        np.seterr(all='warn')

        super(self.__class__, self).__init__((N, (self.ndim_y + self.ndim_x, 1)), dtype, backend, smooth=True, label=label,
                                             forward=self._forward, gradient=self._gradient,
                                             convex=False,  repr_latex=self._latex)

    def _latex(self, latex_input=None):
        if latex_input is not None:
            return 'e^{-i2\\pi (\\vec{k} \\cdot S \\cdot' + latex_input + ')}'
        else:
            return 'e^{-i2\\pi (\\vec{k} \\cdot S \\cdot [\\cdot ])}'

    def _forward(self, x, y):
        # TODO does this indexing work when not numpy
        weight_y = x[:self.ndim_y]
        weight_x = x[self.ndim_y:]
        rys = yp.matmul(self.basis_y, weight_y)
        rxs = yp.matmul(self.basis_x, weight_x)
        print('forward with position:', np.amax(rys), np.amax(rxs))
        # TODO: need to set this as zero otherwise weird results
        y[:] = yp.zeros(self.M, dtype=self.dtype, backend=self.backend)
        for t in range(len(self.illums)):
            y[:] += self._single_forward(self.illums[t], [rys[t], rxs[t]])
            # self.illums[t] * exp(self.ry * scalar(rys[t]) + self.rx * scalar(rxs[t]))

    def _single_forward(self, illum, r):
        # using euler's formula instead of
        # exp(self.ry * scalar(r[0]) + self.rx * scalar(r[1]))
        inner = self.ry * yp.scalar(r[0]) + self.rx * yp.scalar(r[1])
        # if self.verbose: print(np.amax(np.abs(inner)), r)
        try:
            result = illum * (yp.cos(inner) + 1j * yp.sin(inner))
        except RuntimeWarning as e:
            print(e, illum, r)
        return result

    def _gradient(self, x=None, inside_operator=None):
        from .stack import Vstack
        weight_y = x[:self.ndim_y]
        weight_x = x[self.ndim_y:]
        rys = yp.matmul(self.basis_y, weight_y)
        rxs = yp.matmul(self.basis_x, weight_x)
        print(np.amax(np.abs(rys)), np.amax(np.abs(rxs)), np.amax(self.illums))

        # sum across t
        if self.verbose: print('computing over t')
        sum_exp_y = yp.zeros([self.ndim_y, self.M[0], self.M[1]])
        sum_exp_x = yp.zeros([self.ndim_x, self.M[0], self.M[1]])
        for t in range(len(self.illums)):
            if self.verbose: print(t, end=' ')
            forward_t = self._single_forward(self.illums[t], [rys[t], rxs[t]])
            for i in range(max(self.ndim_y, self.ndim_x)):
                if i < self.ndim_y: sum_exp_y[i,:,:] += self.basis_y[t, i] * forward_t
                if i < self.ndim_x: sum_exp_x[i,:,:] += self.basis_x[t, i] * forward_t
        S = ops.Sum(self.M, self.dtype, self.backend)
        if self.verbose: print('\nconstructing columns')
        column_list_y = []; column_list_x = []
        for i in range(max(self.ndim_y, self.ndim_x)):
            if self.verbose: print(i, end=' ')
            if i < self.ndim_y: column_list_y.append(S * ops.Diagonalize(conj(1j * self.ry) * yp.conj(sum_exp_y[i])))
            if i < self.ndim_x: column_list_x.append(S * ops.Diagonalize(conj(1j * self.rx) * yp.conj(sum_exp_x[i])))
        print('\n')
        G = ops.Vstack(column_list_y + column_list_x)
        return ops._GradientOperator(G)

    def _gradient_lowmem(self, x=None, inside_operator=None):
        from .stack import Vstack
        weight_y = x[:self.ndim_y]
        weight_x = x[self.ndim_y:]
        rys = yp.matmul(self.basis_y, weight_y)
        rxs = yp.matmul(self.basis_x, weight_x)
        print(np.amax(np.abs(rys)), np.amax(np.abs(rxs)))

        S = ops.Sum(self.M, self.dtype, self.backend)
        column_list_y = []; column_list_x = []
        for i in range(max(self.ndim_y, self.ndim_y)):
            if self.verbose:
                print('computing for column', i)
            forward_0 = self._single_forward(self.illums[0], [rys[0], rxs[0]])
            if i < self.ndim_y:
                sum_exp_y = self.basis_y[0, i] * forward_0
            if i < self.ndim_x:
                sum_exp_x = self.basis_x[0, i] * forward_0
            for t in range(1, len(self.illums)):
                forward_t = self._single_forward(self.illums[t], [rys[t], rxs[t]])
                if i < self.ndim_y:
                    sum_exp_y += self.basis_y[t, i] * forward_t
                if i < self.ndim_x: sum_exp_x += self.basis_x[t, i] * forward_t
            if i < self.ndim_y: column_list_y.append(S * ops.Diagonalize(conj(1j * self.ry) * yp.conj(sum_exp_y)))
            if i < self.ndim_x: column_list_x.append(S * ops.Diagonalize(conj(1j * self.rx) * yp.conj(sum_exp_x)))

        G = ops.Vstack(column_list_y + column_list_x)
        return ops._GradientOperator(G)
