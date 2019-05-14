# General imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import skimage
import os

# Comptic imports
from comptic.imaging import pupil, otf
from comptic import registration, containers

from . import blurkernel

# Llops imports
import llops.operators as ops
import llops as yp
from llops.solvers import iterative, objectivefunctions, regularizers

class Reconstruction():
    """
    Reconstruct class contains methods relevant for reconstructing large FOV
    images, both from blurred and unblurred measurements.
    """

    def __init__(self, dataset,
                 preprocess=False,
                 color_channel=0,
                 normalize=False,
                 pad_value=0,
                 blur_axis=1,
                 estimate_background_poly=False,
                 background_estimation_method=None,
                 use_phase_ramp_for_blur_vectors=False,
                 interpolation_factor=1,
                 use_psf=False,
                 kernel_corrections={},
                 use_mean_background=True,
                 pad_with_adjacent_measurements=False,
                 alpha_blend_distance=1000,
                 estimated_background=None,
                 dtype=None, backend=None, **kwargs):
        """
            Args:
                dataset: A dataset object contains
            Optional Args:
                color_channel: color channel to process
        """

        self.verbose = kwargs.pop('verbose', False)

        # Store dtype and backend
        self.dtype = dtype if dtype is not None else yp.config.default_dtype
        self.backend = backend if backend is not None else yp.config.default_backend

        # Store dataset
        self.dataset = dataset

        # Preprocess for MD if requested by user
        if preprocess:
            preprocess(self.dataset)

        # Get and store pad mode
        self.pad_value = pad_value

        # Whether to use phase ramp for blur vectors (can be False if illuminaiton pulses have pixel-spacing)
        self.use_phase_ramp_for_blur_vectors = use_phase_ramp_for_blur_vectors

        # Get alpha blend distance
        self.alpha_blend_distance = alpha_blend_distance

        # Measurement normalization scalars
        self.measurement_normalization = None

        # Background estimation method
        self.background_estimation_method = background_estimation_method

        # Whether to flatten background
        self.estimate_background_poly = estimate_background_poly

        # Whether to use per-frame or average background estimation
        self.use_mean_background = use_mean_background

        # Whether to perform normalization of frames
        self.normalize = normalize

        # Store interpolation factor
        self.interpolation_factor = interpolation_factor

        # Estimated backgroundx
        self.estimated_background = estimated_background

        # Whether to use adjacent measurements for padding
        self.pad_with_adjacent_measurements = pad_with_adjacent_measurements

        # Get frame_segment_list
        self.frame_segment_list = self.dataset.frame_segment_list

        # If the dataset uses a template illumination sequence, use this
        if self.dataset.use_first_illumination_sequence_as_template:
            self.dataset.expandFrameStateList()

        # Read blur kernel info
        self.blur_vector_list, self.blur_vector_roi_list = self.dataset.blur_vectors(corrections=kernel_corrections,
                                                                                     use_phase_ramp=self.use_phase_ramp_for_blur_vectors)

        # Define object_true
        self.object_true = None

        # Define reg_types
        self.reg_types_recon = None

        # Define system otf
        if use_psf:
            self._genSystemPsf()
        else:
            self.psf = None

        # Determine segment directions
        self.frame_segment_direction_list = [d[1] > 0 for d in dataset.frame_segment_direction_list]

        # Generate ROI objects
        self._genRoiObjects()

        # Generate blur kernels and roi lists
        if self.verbose: print('Generating blur kernels')
        self._genBlurKernels(axis=blur_axis)

        # Calculate object shape and valid object support
        self._calculateObjectSize()

        # Align ROI objects to origin
        self._alignRoiObjectsToOrigin()

        # Generate forward models
        if self.verbose: print('Generating forward operators')
        self._gen_forward_operators()

        # Generate padded measurements
        if self.verbose: print('Generating measurements')
        self._gen_measurements()

    def _getSegmentDirections(self, blur_axis=1):
        """Calculate directions of each linear segment in the dataset."""
        self.frame_segment_direction_list = []
        self.segments_in_dataset = yp.unique(self.frame_segment_list)
        for segment_index in self.segments_in_dataset:
            frames_in_segment = [frame_index for frame_index in range(self.dataset.shape[0]) if self.frame_segment_list[frame_index] == segment_index]

            # Get first and last ROI in segment
            first_roi, last_roi = self.blur_vector_roi_list[frames_in_segment[0]], self.blur_vector_roi_list[frames_in_segment[-1]]

            # Append to list
            self.frame_segment_direction_list.append(last_roi.start[blur_axis] > first_roi.start[blur_axis])

    def _genRoiObjects(self, centered_blur=True):
        self.roi_list_convolution = []
        self.roi_list_measurement_support = []
        self.roi_list = []
        for index, blur_roi in enumerate(self.blur_vector_roi_list):

            # Get segment direction
            segment_direction = self.frame_segment_direction_list[index]

            # Determine blur kernel shape
            blur_kernel_shape = [(yp.next_fast_even_number(sh_roi + sh)) for (sh_roi, sh) in zip(blur_roi.shape, self.dataset.frame_shape)]

            # Determine convolution support
            convolution_support_start = [kernel_start - sh for (kernel_start, sh) in zip(blur_roi.start, self.dataset.frame_shape)]
            self.roi_list_convolution.append(yp.Roi(start=convolution_support_start,
                                                    shape=blur_kernel_shape))

            # Determine measurement support within convolution
            if not centered_blur:
                if segment_direction:  # True if ->
                    measurement_support_start = (0, 0)
                else:   # True if <----
                    measurement_support_start = blur_roi.shape
            else:
                measurement_support_start = [s // 2 for s in blur_roi.shape]

            # Assign measurement support
            self.roi_list_measurement_support.append(yp.Roi(start=measurement_support_start,
                                                            shape=self.dataset.frame_shape,
                                                            input_shape=blur_kernel_shape))

            # Combine these into a ROI which contains only measurement support
            self.roi_list.append(self.roi_list_convolution[-1] * self.roi_list_measurement_support[-1])

            # Apply registration offset if provided
            if 'registration' in self.dataset.metadata.calibration:
                offset = self.dataset.metadata.calibration['registration']['frame_offsets'][self.dataset.frame_mask[index]]
                self.roi_list_convolution[-1] += offset
                self.roi_list_measurement_support[-1] += offset
                self.roi_list[-1] += offset

        # Store global coordinates for object and convolution support ROIs
        self.roi_start_global = sum(self.roi_list).start
        self.roi_start_global_convolution = sum(self.roi_list_convolution).start

        # Align all ROI lists to origin
        alignRoiListToOrigin(self.roi_list_convolution)
        alignRoiListToOrigin(self.roi_list_measurement_support)
        alignRoiListToOrigin(self.roi_list)

    def _genBlurKernels(self, centered_blur=True, axis=1):
        """Generate blur kernels."""
        from htdeblur import blurkernel
        # Determine blur kernel shape
        self.blur_kernel_list = []
        for index, (blur_vector, blur_roi) in enumerate(zip(self.blur_vector_list, self.blur_vector_roi_list)):

            # Determine blur kernel shape
            blur_kernel_shape = [yp.next_fast_even_number(sh_roi + sh) for (sh_roi, sh) in zip(blur_roi.shape, self.dataset.frame_shape)]

            # Generate blur kernel

            blur_kernel = blurkernel.fromVector(blur_vector,
                                                blur_kernel_shape,
                                                axis=axis,
                                                reverse=False,
                                                position='center')

            # Optionally filter to OTF support
            if self.psf is not None:
                self.psf /= yp.scalar(yp.sum(self.psf))
                blur_kernel = yp.convolve(blur_kernel, self.psf)

            # Normalize
            blur_kernel /= yp.scalar(yp.sum(blur_kernel))

            # Append to list
            self.blur_kernel_list.append(blur_kernel)

    def _calculateObjectSize(self):
        """Generate object size and object_valid_roi."""

        # Calculate ROI object for full support of all measurements (including convolution support)
        roi_sum = sum(self.roi_list_convolution)

        # Determine object shape
        self.object_shape = roi_sum.shape

        # Calculate valid roi by taking the sum of all measurement supports to determine valid ROI
        self.object_valid_roi = sum(self.roi_list)
        self.object_valid_roi.input_shape = sum(self.roi_list_convolution).shape

        # Assign correct input sizes to measurement sizes
        for roi in self.roi_list_convolution:
            roi.input_shape = self.object_shape

    def _alignRoiObjectsToOrigin(self):
        """Align all roi objects to origin"""

        # Determine mininum start position of roi objects
        min_start_position = (1e10, 1e10)
        for roi in self.roi_list_convolution:
            min_start_position = [min(st, min_dim) for (st, min_dim) in zip(roi.start, min_start_position)]

        # Align roi objects to origin
        for roi in self.roi_list_convolution:
            roi -= min_start_position

        # Align object valid support roi to origin
        self._calculateObjectSize()

    def _gen_measurements(self):
        """Generate padded measurements."""
        # Clear existing measurements (if any)
        self.y_list = []

        # Generate measurements
        mean_values = []
        for frame_index, frame in enumerate(self.dataset.frame_list):

            # Perform deep copy of measurements
            frame = yp.dcopy(frame)

            # Store mean
            mean_values.append(yp.scalar(yp.mean(frame)))

            # Convert to floating point
            frame = yp.cast(frame, self.dtype, self.backend)

            # Apply normalization if scale values are not none
            if self.measurement_normalization is not None:
                frame *= self.measurement_normalization[frame_index]

            # Apply normalizaiton vectors if they are not nont
            if 'normalization' in self.dataset.metadata.calibration:
                normalization_y = yp.cast(self.dataset.metadata.calibration['normalization']['frame_normalization_y'][self.dataset.frame_mask[frame_index]], self.dtype, self.backend)
                normalization_x = yp.cast(self.dataset.metadata.calibration['normalization']['frame_normalization_x'][self.dataset.frame_mask[frame_index]], self.dtype, self.backend)
                frame[:] = frame * yp.outer(normalization_y, normalization_x)

            # Pad frame using measurement support ROI and append to list
            self.y_list.append(frame)

        # Calculate and normalize by background
        if self.estimate_background_poly:
            bg = estimateBackgroundList(self.y_list)
            self.y_list = [y / bg for y in self.y_list]

        # Calculate mean value and normalize
        if self.normalize:
            mean = yp.mean(mean_values)
            self.y_list = [y / mean for y in self.y_list]
        else:
            # Normalize all measurements to have the same mean as before
            self.y_list = [y / yp.mean(y) * mean for (y, mean) in zip(self.y_list, mean_values)]

    def _gen_forward_operators(self):
        """Generate forward operators to be used by _gen_optimizer functions."""
        # Read sizes
        blur_kernel_size = self.roi_list_convolution[0].size

        # Generate convolution and convolution support crop operators
        C_list, R_list = [], []
        for blur_kernel, measurement_roi in zip(self.blur_kernel_list, self.roi_list_measurement_support):

            # Generate and append convolution operator
            C_list.append(ops.Convolution(blur_kernel,
                                          # mode='circular',
                                          pad_value=self.pad_value,
                                          dtype=self.dtype,
                                          backend=self.backend))

            # Generate and append convolution support crop operator
            R_list.append(ops.Crop(roi=measurement_roi,
                                   N=blur_kernel_size,
                                   pad_value=0,
                                   dtype=self.dtype,
                                   backend=self.backend))

        # Generate full convolution operator
        self.C = ops.Dstack(C_list)

        # Generate full convolution support crop operator
        self.R = ops.Dstack(R_list)

        # Create segmentation operator (convolution support)
        self.G = ops.Segmentation(self.roi_list_convolution,
                                  alpha_blend_size=self.alpha_blend_distance,
                                  dtype=self.dtype, backend=self.backend)

        # Create segmentation operator (measurement support)
        self.G_meas = ops.Segmentation(self.roi_list,
                                       alpha_blend_size=self.alpha_blend_distance,
                                       dtype=self.dtype, backend=self.backend)

    def applyFrameDependentOffset(self, offset=26, axis=1):
        """Applys a direction-dependant position offset to ROIs."""

        # Loop over convolution ROIs
        for index, roi in enumerate(self.roi_list_convolution):

            # Determine the direction of this segment
            frame_segment_number = self.frame_segment_list[index]
            left_to_right_directio5n = self.frame_segment_direction_list[frame_segment_number - min(self.frame_segment_list)]

            # number of positions before this segment
            segment_index = len([segment for segment in self.frame_segment_list[:index] if segment == self.frame_segment_list[index]])
            delta = [0, 0]
            delta[axis] = segment_index * offset
            if not left_to_right_direction:
                delta[axis] *= -1
            roi -= delta

        # Re-align to origin in case any ROI objects now have zero-indicies
        self._alignRoiObjectsToOrigin()

        # Update forward operators
        self._gen_forward_operators()

    def applySegmentDependentOffset(self, offset=550, axis=1):
        """Applys a frame-independent but segment-dependent position offset to ROIs."""

        # Only apply this offset if there are more than one segment in this dataset
        for index, roi in enumerate(self.roi_list_convolution):

            # Determine the direction of this segment
            frame_segment_number = self.frame_segment_list[index]
            left_to_right_direction = self.frame_segment_direction_list[frame_segment_number - min(self.frame_segment_list)]

            # number of positions before this segment
            delta = [0, 0]
            delta[axis] = offset
            if not left_to_right_direction:
                delta[axis] *= -1
            roi -= delta

        # Ensure ROI objects are aligned to origin
        self._alignRoiObjectsToOrigin()

        # Update forward operators
        self._gen_forward_operators()

    def _invert(self, inverse_regularizer=1e-6):
        """Invert a forward operator."""
        # Check if forward model is invertable before inverting
        if self.A.invertable:
            # Perform analytic inverse
            self.object_recovered = self.A.inverse(inverse_regularizer=inverse_regularizer) * self.y
        elif self.A.inner_operators.invertable:
            # Apply adjoint of forward operator to measurement before inverting
            _A = self.A.inner_operators
            _y = self.A.outer_operator.H * self.y

            # Perform analytic inverse
            self.object_recovered = _A.inverse(inverse_regularizer=inverse_regularizer) * _y
        else:
            raise ValueError('Forward model is not invertable.')

    def reconstruct(self, iteration_count, reg_types={},
                    mode='sequential', initialization=None,
                    frame_number=0, step_size=0.1, display_type='text', **kwargs):
        """
            Args:
                iteration_count: number of iterations to run. If <0, direct inversion
                initialization (optional): initial value
                reg_types: dictionary whose keys determine type of regularization, and whose
                           values determine the coefficient
                frame_number: None means full multiframe,
                              otherwise index for singleframe
                              TODO: subset of frames?

            TODO: allow passing in other kwargs to solve()
        """

        # Generate different forward models based on mode keyword
        if mode == 'single':
            # Single measurement
            self.optimizer, self.post_opimize_function = self._get_optimizer_single(reg_types, frame_number)

        elif mode == 'single_fourier':
            # Single measurement
            self.optimizer, self.post_opimize_function = self._get_optimizer_single_fourier(reg_types, frame_number)

        elif mode == 'global':
            # Stitch all measurements
            self.optimizer, self.post_opimize_function = self._get_optimizer_global(reg_types)

        elif mode == 'global_fourier':
            # Stitch all measurements
            self.optimizer, self.post_opimize_function = self._get_optimizer_global_fourier(reg_types)

        elif mode == 'sequential':
            # Deconvolve measurements individually and place in frame
            self.optimizer, self.post_opimize_function = self._get_optimizer_sequential(reg_types)

        elif mode == 'static':
            # Deconvolve measurements individually and place in frame
            self.optimizer, self.post_opimize_function = self._get_optimizer_static(reg_types)

        # Use custom initialization if desired
        if initialization is not None:
            self.initialization = self._gen_initialization()
        else:
            self.initialization = initialization

        # A negative iteration count means direct inverion
        if iteration_count < 0:
            self._invert(inverse_regularizer=reg_types.get('l2', 1e-6))
        else:

            # remark: call reconstruction(initialization=recon.object_recovered) to run more
            # steps of recovery after the fact.
            self.object_recovered = self.optimizer.solve(initialization=self.initialization,
                                                         step_size=step_size,
                                                         nesterov_restart_enabled=True,
                                                         use_nesterov_acceleration=True,
                                                         let_diverge=True,
                                                         iteration_count=iteration_count,
                                                         display_type=display_type,
                                                         display_iteration_delta=max((iteration_count // 10), 1))

            # Apply post_opimize_function:
            if self.post_opimize_function is not None:
                self.object_recovered = self.post_opimize_function(self.object_recovered)

            if mode in ['sequential', 'global']:
                self.object_recovered = self.object_recovered[self.object_valid_roi.slice]

        # Set reconstruction pareameters
        self.reconstruction_parameters = {'step_size': kwargs.get('step_size', 0.1),
                                          'iteration_count': iteration_count}

    def _gen_initialization(self):
        """Generate initialization."""
        assert self.A is not None, "Generate forward model before calling _gen_initialization."

        # Generate initialization
        return yp.ones(self.A.N, dtype=self.dtype, backend=self.backend) * yp.mean(self.y)

    def _get_optimizer_static(self, reg_types={}):
        """
            Args:
                reg_types: dictionary whose keys determine type of regularization, and whose
                           values determine the coefficient
        """
        # Store regularization types
        self.reg_types_recon = reg_types

        # Generate measurement vector and forward operator
        self.A = self.G_meas
        self.y = ops.VecStack(self.y_list)

        # Create data fidelity objective
        data_fidelity_term = objectivefunctions.L2(self.A, self.y)

        # Generate cost function (with regularization)
        cost_function = get_cost_function(data_fidelity_term, reg_types)

        # Generate optimizer
        optimizer = iterative.Fista(cost_function)

        return optimizer, None


    def _get_optimizer_sequential(self, reg_types={}):
        """
            Args:
                reg_types: dictionary whose keys determine type of regularization, and whose
                           values determine the coefficient
        """
        # Store regularization types
        self.reg_types_recon = reg_types

        # Get measurement size
        A_list = []
        for frame_number in range(len(self.y_list)):

            # Get convolution operator
            C = self.C[frame_number]

            # Get measurement support crop
            R = ops.Crop(self.C.stack_operators[frame_number].M, yp.shape(self.y_list[frame_number]),
                         center=True, pad_value='mean')

            # Generate forward operator and measurement
            A_list.append(R * C)

        # Generate new convolution operators which are the same size as G_meas
        self.A = ops.Dstack(A_list) * self.G
        self.y = ops.VecStack(self.y_list)

        # Create data fidelity objective
        data_fidelity_term = objectivefunctions.L2(self.A, self.y)

        # Generate cost function (with regularization)
        cost_function = get_cost_function(data_fidelity_term, reg_types)

        # Generate optimizer
        optimizer = iterative.Fista(cost_function)

        return optimizer, None

    def _get_optimizer_global(self, reg_types={}, frame_number=None):
        """
            Args:
                reg_types: dictionary whose keys determine type of regularization, and whose
                           values determine the coefficient
        """

        assert len(yp.unique(self.frame_segment_list)) == 1, "Only single-segment reconstructions are supported!"

        # Store regularization types
        self.reg_types_recon = reg_types

        # Pad y for convolution support
        padded_size = [yp.next_fast_even_number(int(sz + blur_sz)) for (sz, blur_sz) in zip(self.G_meas.N, self.blur_vector_roi_list[0].shape)]

        # Generate blur kernel with this size
        blur_kernel = blurkernel.fromVector(self.blur_vector_list[0],
                                            padded_size,
                                            reverse=True,
                                            axis=1,
                                            position='center')

        # Stitch y
        self.y = self.G_meas.inv * ops.VecStack(self.y_list)
        self.y[yp.isnan(self.y)] = 0.0

        # Generate convolutional forward model
        C = ops.Convolution(blur_kernel, mode='circular')

        # Create pad operator
        CR = ops.Crop(padded_size, self.G_meas.N, center=True, pad_value='mean')

        # Create forward operator
        self.A = CR * C

        # Create data fidelity objective
        data_fidelity_term = objectivefunctions.L2(self.A, self.y)

        # Generate cost function (with regularization)
        cost_function = get_cost_function(data_fidelity_term, reg_types)

        # Generate Optimizer
        optimizer = iterative.Fista(cost_function)

        return optimizer, None


    def _get_optimizer_global_fourier(self, reg_types={}, frame_number=None):
        """
            Args:
                reg_types: dictionary whose keys determine type of regularization, and whose
                           values determine the coefficient
        """

        assert len(yp.unique(self.frame_segment_list)) == 1, "Only single-segment reconstructions are supported!"

        # Store regularization types
        self.reg_types_recon = reg_types

        # Pad y for convolution support
        padded_size = [yp.next_fast_even_number(sz + blur_sz) for (sz, blur_sz) in zip(self.G_meas.N, self.blur_vector_roi_list[0].shape)]

        # Generate blur kernel with this size
        blur_kernel = blurkernel.fromVector(self.blur_vector_list[0],
                                            padded_size,
                                            reverse=False,
                                            axis=1,
                                            position='center')

        # Fourier Transform Blur Kernel
        blur_kernel_f = yp.Ft(blur_kernel)

        # Generate convolutional forward model
        K = ops.Diagonalize(blur_kernel_f, dtype='complex32')

        # Create pad operator
        CR = ops.Crop(padded_size, self.G_meas.N, center=True, pad_value='mean')

        # Create forward operator
        self.A = K

        # Stitch y
        self.y = CR.H*self.G_meas.inv * ops.VecStack(self.y_list)
        self.y[yp.isnan(self.y)] = 0.0
        self.y = yp.Ft(self.y)

        # Create data fidelity objective
        data_fidelity_term = objectivefunctions.L2(self.A, self.y)

        # Generate cost function (with regularization)
        cost_function = get_cost_function(data_fidelity_term, reg_types, dtype='complex32')

        # Generate Optimizer
        optimizer = iterative.Fista(cost_function)

        # Define post-solver function
        post_opimize_function = lambda x: yp.real(yp.iFt(x))

        return optimizer, post_opimize_function

    def _get_optimizer_single(self, reg_types={}, frame_number=0):
        """
            Args:
                reg_types: dictionary whose keys determine type of regularization, and whose
                           values determine the coefficient
        """

        # Store regularization types
        self.reg_types_recon = reg_types

        # Generate crop operator
        CR = ops.Crop(self.C.stack_operators[frame_number].M, yp.shape(self.y_list[frame_number]),
                      center=True, pad_value='mean')

        # Generate forward operator and measurement
        self.A = CR * self.C[frame_number]
        self.y = self.y_list[frame_number]

        # Create data fidelity objective
        data_fidelity_term = objectivefunctions.L2(self.A, self.y)

        # Generate cost function (with regularization)
        cost_function = get_cost_function(data_fidelity_term, reg_types)

        # Create optimizer
        optimizer = iterative.Fista(cost_function)

        # Post-optimization function
        def post_opimize_function(x):
            return CR * x

        return optimizer, post_opimize_function

    def _get_optimizer_single_fourier(self, reg_types={}, frame_number=0):
        """
            Args:
                reg_types: dictionary whose keys determine type of regularization, and whose
                           values determine the coefficient
        """

        # Store regularization types
        self.reg_types_recon = reg_types

        # Generate crop operator
        CR = ops.Crop(self.C.stack_operators[frame_number].M, yp.shape(self.y_list[frame_number]),
                      center=True, pad_value='mean')

        # Generate forward operator and measurement
        F = ops.FourierTransform(CR.M, dtype='complex32')

        self.A = ops.Diagonalize(yp.conj(F * yp.astype((CR * self.blur_kernel_list[frame_number]), 'complex32')))
        self.y = F * yp.astype(self.y_list[frame_number], 'complex32')

        # Create data fidelity objective
        data_fidelity_term = objectivefunctions.L2(self.A, self.y)

        # Generate cost function (with regularization)
        cost_function = get_cost_function(data_fidelity_term, reg_types)

        # Create optimizer
        optimizer = iterative.Fista(cost_function)

        # Post-optimization function
        def post_opimize_function(x):
            return yp.real(yp.iFt(x))

        return optimizer, post_opimize_function



    def showRaw(self, clim=None, figsize=None, colorbar=False):
        yp.display.listPlotFlat(self.y_list, figsize=figsize, clim=clim, colorbar=colorbar)

    def showResult(self, clim=None, ax=None, colorbar=False, fig=None):
        if ax is None:
            if fig is None:
                fig = plt.figure(kwargs.get(figsize), (10, 10 * yp.aspectRatio()))
            ax = plt.gca()
        plot_object = np.abs(yp.changeBackend(self.object_recovered, 'numpy'))
        if adjust_contrast:
            plot_object = skimage.filters.gaussian(plot_object,sigma=1)
            plot_object = skimage.exposure.equalize_hist(plot_object)
        ax.imshow(plot_object, cmap='gray')

    def showBackground(self):
        if type(self.estimated_background) is list:
            yp.display.listPlotFlat(self.estimated_background[:10])
        else:
            plt.figure()
            plt.imshow(yp.real(self.estimated_background))
            plt.colorbar()
            plt.axis('off')
            plt.tight_layout()

    def show(self, clim=None, figsize=(10,5), iFt=False, show_raw=False, adjust_contrast=False):

        if show_raw:
            plt.figure(figsize=figsize)
            plt.subplot(121)

            if iFt:
                new_y = []
                for i, y in enumerate(self.y_list):
                    new_y.append(self.R0_list[i].H * y)
                i = plt.imshow(np.abs(yp.changeBackend(ops.VecStack(new_y), 'numpy')), cmap='gray')
            else:
                i = plt.imshow((np.real(yp.changeBackend(self.y, 'numpy'))), cmap='gray')
            plt.title('Raw Data')
            plt.colorbar()
            plt.subplot(122)
        else:
            aspect_ratio = yp.shape(self.object_recovered)[0] / yp.shape(self.object_recovered)[1] + 0.1
            plt.figure(figsize=(figsize[0], figsize[0] * aspect_ratio))

        plot_object = np.abs(yp.changeBackend(self.object_recovered, 'numpy'))
        if adjust_contrast:
            plot_object = skimage.filters.gaussian(plot_object, sigma=1)
            plot_object = skimage.exposure.equalize_hist(plot_object)
        plt.imshow(plot_object, cmap='gray')
        plt.title('Reconstruction')
        plt.colorbar()
        # print(i.get_clim())
        if clim is None and show_raw:
            plt.clim(i.get_clim())
        elif clim is not False:
            plt.clim(clim)
        if self.object_true is not None and self.object_true.shape == self.object_recovered.shape:
            plt.figure(figsize=(12, 5))
            plt.subplot(221)
            plot_object = np.abs(yp.changeBackend(self.object_true, 'numpy'))
            plt.imshow(plot_object, cmap='gray')
            plt.title('True')
            plt.subplot(223)
            plot_object = np.abs(yp.changeBackend(self.object_recovered, 'numpy'))
            plt.imshow(plot_object, cmap='gray')
            plt.title('Reconstruction')
            plt.subplot(222)
            plot_object = np.abs(yp.changeBackend(self.object_recovered - self.object_true, 'numpy'))
            plt.imshow(plot_object, cmap='viridis')
            plt.title('Errors')
            plt.colorbar()

    def saveDataset(self):
        """Save Dataset object with reconstruction result."""
        if self.object_recovered is not None:

            # Update dataset reconstruction class
            self.dataset.reconstruction = io.Reconstruction()
            self.dataset.reconstruction.object = self.object_recovered
            self.dataset.reconstruction.parameters = self.reconstruction_parameters

            # Save Dataset
            self.dataset.save()

    def save(self, filepath, filename=None, formats=['npz'], downsample=1, save_raw=False):
        # Get filename
        if filename is None:
            filename = self.dataset.metadata.file_header
        # filename += '_recovered_segment=' + str(self.segments_in_dataset)

        if self.reg_types_recon is not None:
            regstr = '_regularize=['
            for regul in self.reg_types_recon.keys():
                regstr += regul + '{:.1e}'.format(self.reg_types_recon[regul])
            filename += regstr + ']'

        # Generate full output filename
        output_filename_full = os.path.join(filepath, filename)

        if 'npz' in formats:
            np.savez(output_filename_full,
                     object_recovered=self.object_recovered,
                     y=self.y,
                     segment_index=self.dataset.frame_segment_list[0],
                     roi=(sum(self.roi_list) + self.roi_start_global).__dict__)

            print('Saved .npz file to %s.npz' % output_filename_full)

        if 'tiff8' in formats:
            import tifffile
            data = yp.changeBackend(self.object_recovered, 'numpy')
            tifffile.imsave(output_filename_full + '.tiff', np.uint8(np.round(data / np.max(data) * 256.)), compress=False)

            if save_raw:
                data = yp.changeBackend(self.y, 'numpy')
                tifffile.imsave(output_filename_full + '_measurement.tiff', np.uint8(np.round(data / np.max(data) * 256.)), compress=False)

        if 'tiff' in formats:
            import tifffile
            data = yp.changeBackend(self.object_recovered, 'numpy')
            tifffile.imsave(output_filename_full + '.tiff', np.uint16(np.round(data / np.max(data) * 65535.0)), compress=False)

            if save_raw:
                data = yp.changeBackend(self.y, 'numpy')
                tifffile.imsave(output_filename_full + '_measurement.tiff', np.uint16(np.round(data / np.max(data) * 65535.0)), compress=False)

        if 'png' in formats:
            if downsample is not None:
                new_size = [int(round(sz / downsample)) for sz in yp.shape(self.object_recovered)]
                plt.imsave(output_filename_full + '.png',
                           skimage.transform.resize(np.abs(yp.changeBackend(self.object_recovered, 'numpy')), new_size),
                           cmap='gray')

                if save_raw:
                    new_size_raw = [int(round(sz / downsample)) for sz in yp.shape(self.y)]
                    plt.imsave(output_filename_full + '_measurement.png',
                               skimage.transform.resize(np.abs(yp.changeBackend(self.y, 'numpy')), new_size_raw),
                               cmap='gray')
            else:
                plt.imsave(output_filename_full + '.png',
                           np.abs(yp.changeBackend(self.object_recovered, 'numpy')),
                           cmap='gray')

                if save_raw:
                    plt.imsave(output_filename_full + '_measurement.png',
                               np.abs(yp.changeBackend(self.y, 'numpy')),
                               cmap='gray')

            print('Saved reconstruction .png file to %s.png' % output_filename_full)
            if save_raw:
                print('Saved measurement .png file to %s.png' % output_filename_full)

    def simulate_measurements(self, filename, adjust_by=None, crop_offset=(0, 0), show=False):
        """
            Loads an object from filename, crops it to the correct size, and then
            uses the loaded forward model to replace y with simulated
            measurements.

            Adjust_by changes the model to induce model mismatch
        """
        data = np.load(filename)
        y_idx = (crop_offset[0], crop_offset[0] + self.G.N[0])
        x_idx = (crop_offset[1], crop_offset[1] + self.G.N[1])
        self.object_true = data['object_recovered'][y_idx[0]:y_idx[1], x_idx[0]:x_idx[1]]

        self.y_list = []
        self.object_true = yp.pad(yp.changeBackend(yp.astype(self.object_true, yp.config.default_dtype), yp.config.default_backend), self.G.N, pad_value='mean')
        self.y_list = ops.VecSplit(self.R * self.C * self.G * self.object_true, len(self.C.stack_operators))

        # Modifying forward model to induce mismatch
        if adjust_by is not None:
            self.update_forward_model(adjust_by=adjust_by)

        if show:
            plt.figure(figsize=(9, 3))
            plt.subplot(122)
            plt.imshow(np.abs(yp.changeBackend((ops.VecStack(self.y_list)), 'numpy')), cmap='gray')
            plt.title('y')
            plt.subplot(121)
            plt.imshow(np.abs(yp.changeBackend(self.object_true, 'numpy')), cmap='gray')
            plt.title('true object')

    def normalize_measurements_single_strip(self, debug=False, use_overlap_region=True,
                               method='relative',
                               wrap_final_value=False,
                               wrap_coefficient=None,
                               reverse=False,
                               write_results=True, high_pass_filter=False):
        """
        Normalize the intensity of individual measurements
        """

        # Generate Roi list for measurement support ONLY
        roi_list_measurement_support = [roi * roi_meas for (roi, roi_meas) in zip(self.roi_list_convolution, self.roi_list_measurement_support)]

        # Perform normalization
        scale_values, normalization_parameters = normalize_roi_list(self.y_list,
                                                                    roi_list_measurement_support,
                                                                    use_overlap_region=use_overlap_region,
                                                                    wrap_final_value=wrap_final_value,
                                                                    wrap_coefficient=wrap_coefficient,
                                                                    movement_is_positive=not self.frame_segment_direction_list[0],
                                                                    high_pass_filter=high_pass_filter,
                                                                    reverse=reverse,
                                                                    method=method,
                                                                    debug=debug)

        # Store normalization_parameters
        if write_results:
            self.measurement_normalization = scale_values

            # Re-generate measurements
            self._gen_measurements()

        # Return scale values for debugging
        return scale_values

    def register_measurements(self,
                              use_overlap_region=True,
                              debug=False,
                              align_to_origin=False,
                              use_mean_offset=False,
                              replace_untrusted=True,
                              axis=None,
                              method='orb',
                              preprocess_methods=['normalize'],
                              force_2d=True,
                              write_results=True,
                              tolerance=(50, 50),
                              energy_ratio_threshold=100.0):
        """
        Assumes that the images are in a strip
        """

        # Generate Roi list for measurement support ONLY
        roi_list_measurement_support = [roi * roi_meas for (roi, roi_meas) in zip(self.roi_list_convolution, self.roi_list_measurement_support)]

        # Perform actual registration
        offsets = register_roi_list(self.y_list, roi_list_measurement_support,
                                    use_overlap_region=use_overlap_region,
                                    preprocess_methods=preprocess_methods,
                                    axis=axis,
                                    use_mean_offset=use_mean_offset,
                                    replace_untrusted=replace_untrusted,
                                    method=method,
                                    force_2d=force_2d,
                                    debug=debug,
                                    tolerance=tolerance,
                                    energy_ratio_threshold=energy_ratio_threshold)

        # Save parameters
        self.registration_parameters = {'use_overlap_region': use_overlap_region,
                                        'energy_ratio_threshold': energy_ratio_threshold,
                                        'offset_list': offsets}

        # Align measurements to origin, if requested
        if align_to_origin:
            offsets = np.asarray(offsets) - np.max(np.asarray(offsets), 0)

        # Write the results to the forward operator
        if write_results:

            # Also write to ROI operators
            for roi, shift in zip(self.roi_list_convolution, offsets):
                roi += np.asarray(shift).ravel().tolist()

            # Re-align to origin in case any ROI objects now have zero-indicies
            self._alignRoiObjectsToOrigin()

            # Update forward operators
            self._gen_forward_operators()

        # Return the recovered offsets
        return offsets

    def field_flatten_measurements(self, method='ramp', max_ramp_value=1.056):
        if method == 'ramp':
            # Apply ramp correction
            ramp = yp.ramp(self.dataset.frame_shape, axis=1, min_value=1, max_value=max_ramp_value, reverse=False)
            for measurement in self.y_list:
                measurement /= ramp
        elif method == 'gaussian':
            raise NotImplementedError
        else:
            raise ValueError('Invalid field flattening method %s' % method)

    def reset(self):
        """
            resets everything except for initialization variables
            todo: fill this in once state variables are finalized
        """
        pass

    def _genSystemPsf(self):
        na = self.dataset.metadata.objective.na
        pixel_size = self.dataset.metadata.camera.pixel_size_um / (self.dataset.metadata.objective.mag * self.dataset.metadata.system.mag)
        shape = self.dataset.frame_shape
        wavelength = self.dataset.metadata.illumination.spectrum.center['w']

        self.psf = yp.iFt(otf(shape, pixel_size, wavelength, na, dtype=self.dtype, backend=self.backend))
        self.psf /= yp.abs(yp.scalar(yp.sum(self.psf)))
        self.psf = yp.cast(self.psf, self.dtype, self.backend)

    def estimateBackground(self, method='svd', decimation_factor=16,
                           batch_size=None, debug=False,
                           pca_threshold=3, use_mean_bg=False):
        assert method in ('svd', 'gaussian', 'polynomial')
        # Determine which frames to use
        frame_count = len(self.dataset._frame_list)

        if batch_size is None:
            batch_size = frame_count
        else:
            use_mean_bg = True

        frame_indicies_to_process = np.random.randint(0, frame_count, batch_size)

        # Get matrix size and generatematrix
        decimated_size = yp.shape(yp.filter.decimate(self.dataset.getSingleFrame(0), decimation_factor))

        # Generate frame matrix
        frame_matrix = yp.zeros((yp.prod(decimated_size), frame_count))

        # Generate backgorund list for output
        bg_list = []

        # Load decimated frames
        for frame_index in yp.display.progressBar(frame_indicies_to_process, name='Frames Loaded for Background Subtraction'):
            frame_matrix[:, frame_index] = yp.vec(yp.filter.decimate(self.dataset.getSingleFrame(frame_index), decimation_factor))
            frame_matrix[:, frame_index] -= yp.scalar(yp.mean(frame_matrix[:, frame_index]))

        # Generate SVD of measurement
        if method == 'svd':

            # Compute SVD
            u, s, v = yp.linalg.svd(np.asarray(frame_matrix), full_matrices=False)

            # Zero out up to pca threshold
            s[pca_threshold:] = 0.0

            # Re-combine
            _bg = (u @ np.diag(s) @ v).T
            bg_list = [yp.reshape(yp.cast(b, self.dtype, self.backend), decimated_size) for b in _bg]

            # Store debug elemenbts
            debug_elements = [u, s, v]

        # Take gaussian blurred measurement
        elif method == 'gaussian':
            for frame_index in frame_indicies_to_process:
                frame = yp.reshape(frame_matrix[:, frame_index], decimated_size)

                kernel_size = [d // 4 for d in decimated_size]
                bg_list.append(yp.filter.gaussian(frame, sigma=kernel_size))

            debug_elements = []
        elif method == 'polynomial':
            for frame_index in frame_indicies_to_process:
                frame = yp.reshape(frame_matrix[:, frame_index], decimated_size)
                bg_list.append(flattenFrameBackground(frame, polynomial_order=2))

        # Take mean if requested
        if use_mean_bg:
            bg_list = sum(bg_list) / len(bg_list)
            bg_list = yp.resize(bg_list, self.dataset.frame_shape)
        else:
            # Get backgrounds of current frames used and resize to correct size
            bg_list = [yp.resize(bg_list[index], self.dataset.frame_shape) for index in self.dataset.frame_mask]

        if debug:
            print(([frame_matrix, decimated_size] + [debug_elements]))
            return bg_list, tuple([frame_matrix, decimated_size] + [debug_elements])
        else:
            return bg_list


def get_cost_function(objective, reg_types, dtype=None):

    # Get object size
    object_shape = objective.N

    # Get object datatype
    dtype = objective.dtype

    # Parsing regularizers
    regularizer_list = []
    if 'l2' in reg_types.keys():
        regularizer_list.append(reg_types['l2'] * ops.L2Norm(object_shape, dtype=dtype))
    if 'l1' in reg_types.keys():
        regularizer_list.append(reg_types['l1'] * ops.L1Norm(object_shape, dtype=dtype))
    if 'l1F' in reg_types.keys():
        regularizer_list.append(reg_types['l1F'] * ops.L1Norm(object_shape, dtype=dtype)
                                                 * ops.FourierTransform(object_shape, dtype=dtype, pad=True))
    if 'wavelet' in reg_types.keys():
        regularizer_list.append(reg_types['wavelet'] * regularizers.WaveletSparsity(object_shape, dtype=dtype, wavelet_type='db4',
                                                                                    extention_mode='symmetric', level=None,
                                                                                    use_cycle_spinning=True, axes=None))

    if 'tv' in reg_types.keys():
        regularizer_list.append(reg_types['tv'] * regularizers.TV(object_shape, dtype=dtype))

    if 'bilateral' in reg_types.keys():
        regularizer_list.append(reg_types['bilateral'] * regularizers.RegDenoiser(object_shape, dtype=dtype, denoise_type='bilateral'))

    if 'median' in reg_types.keys():
        regularizer_list.append(reg_types['median'] * regularizers.RegDenoiser(object_shape, dtype=dtype, denoise_type='median'))

    if 'tv_wavelet' in reg_types.keys():
        regularizer_list.append(reg_types['tv_wavelet'] * regularizers.RegDenoiser(object_shape, dtype=dtype, denoise_type='tv_bregman', weight=1.0))

    # Defining Cost Function
    cost_function = objective
    for regul in regularizer_list:
        cost_function += regul
    return cost_function


def subtract_lowpass(frame_list, sigma=100, show=False):
    for i in range(len(frame_list)):
        plot_object = np.abs(frame_list[i])
        plot_object = plot_object.astype(np.float)
        plot_object = skimage.exposure.adjust_log(plot_object)

        plot_object -= skimage.filters.gaussian(plot_object, sigma=100)
        plot_object = skimage.filters.gaussian(plot_object, sigma=3)

        if show:
            plt.figure()
            im = plt.imshow(plot_object, cmap='gray')
            print(im.get_clim())
        frame_list[i] = plot_object


def register_roi_list(measurement_list, roi_list, axis=None,
                      use_overlap_region=True, debug=False,
                      preprocess_methods=['highpass', 'normalize'],
                      use_mean_offset=False, replace_untrusted=True,
                      tolerance=(200, 200), force_2d=False,
                      energy_ratio_threshold=1.5, method='xc'):
    """
    Assumes that the images are in a strip
    """

    # Loop over frame indicies
    offsets = []
    trust_mask = []

    # Parse and set up axis definition
    if axis is not None and force_2d:
        _axis = None
    else:
        _axis = axis

    # Loop over frames
    rois_used = []
    for frame_index in range(len(measurement_list)):

        # Get ROIs
        roi_current = roi_list[frame_index]
        frame_current = measurement_list[frame_index]

        # Determine which rois overlap
        overlapping_rois = [(index, roi) for (index, roi) in enumerate(rois_used) if roi.overlaps(roi_current)]

        # Loop over overlapping ROIs
        if len(overlapping_rois) > 0:

            local_offset_list = []
            for index, overlap_roi in overlapping_rois:
                # Get overlap regions
                overlap_current, overlap_prev = yp.roi.getOverlapRegion((frame_current, measurement_list[index]),
                                                                        (roi_current, roi_list[index]))

                # Perform registration
                _local_offset, _trust_metric = registration.registerImage(overlap_current,
                                                                          overlap_prev,
                                                                          axis=_axis,
                                                                          method=method,
                                                                          preprocess_methods=preprocess_methods,
                                                                          pad_factor=1.5,
                                                                          pad_type=0,
                                                                          energy_ratio_threshold=energy_ratio_threshold,
                                                                          sigma=0.1,
                                                                          debug=False)

                # Deal with axis definitions
                if axis is not None and force_2d:
                    local_offset = [0] * len(_local_offset)
                    local_offset[axis] = _local_offset[axis]
                else:
                    local_offset = _local_offset

                # Filter to tolerance
                for ax in range(len(local_offset)):
                    if abs(local_offset[ax]) > tolerance[ax]:
                        local_offset[ax] = 0
                # local_offset = np.asarray([int(min(local_offset[i], tolerance[i])) for i in range(len(local_offset))])
                # local_offset = np.asarray([int(max(local_offset[i], -tolerance[i])) for i in range(len(local_offset))])

                # Append offset to list
                if _trust_metric > 1.0:
                    local_offset_list.append(local_offset)
                    if debug:
                        print('Registered with trust ratio %g' % _trust_metric)
                else:
                    if debug:
                        print('Did not register with trust ratio %g' % _trust_metric)

            # Append offset to list
            if len(local_offset_list) > 0:
                offsets.append(tuple((np.round(yp.mean(np.asarray(local_offset_list), axis=0)[0]).tolist())))
                trust_mask.append(True)
            else:
                offsets.append((0, 0))
                trust_mask.append(False)

        else:
            offsets.append((0, 0))
            trust_mask.append(True)

        # Store thir ROI in rois_used
        rois_used.append(roi_current)

    # Convert offsets to array and reverse diretion
    offsets = -1 * np.array(offsets)

    if not any(trust_mask):
        print('WARNING: Did not find any good registration values! Returning zero offset.')
        offsets = [np.asarray([0, 0])] * len(offsets)
    else:
        # Take mean of offsets if desired
        if use_mean_offset:
            # This flag sets all measurements to the mean of trusted registration
            offsets = np.asarray(offsets)
            trust_mask = np.asarray(trust_mask)
            offsets[:, 0] = np.mean(offsets[trust_mask, 0])
            offsets[:, 1] = np.mean(offsets[trust_mask, 1])
            offsets = offsets.tolist()
        elif replace_untrusted:
            # This flag replaces untrusted measurements with the mean of all trusted registrations
            offsets = np.asarray(offsets)
            trust_mask = np.asarray(trust_mask)
            trust_mask_inv = np.invert(trust_mask)
            offsets[trust_mask_inv, 0] = np.round(np.mean(offsets[trust_mask, 0]))
            offsets[trust_mask_inv, 1] = np.round(np.mean(offsets[trust_mask, 1]))
            offsets = offsets.tolist()

    # Convert to numpy array
    offsets = np.asarray(offsets)

    # Determine aggrigate offsets
    aggrigate_offsets = [offsets[0]]
    for offset_index in range(len(offsets) - 1):
        aggrigate_offsets.append(sum(offsets[slice(0, offset_index + 2)]).astype(np.int).tolist())

    # Return the recovered offsets
    return aggrigate_offsets


def normalize_roi_list(measurement_list,
                       roi_list,
                       method='relative',
                       use_overlap_region=True,
                       debug=False,
                       reverse=False,
                       wrap_final_value=False,
                       wrap_coefficient=False,
                       movement_is_positive=True,
                       high_pass_filter=False):
    """
    Assumes that the images are in a strip
    """

    # Store parameters for metadata
    normalization_parameters = {'scale_list': [], 'use_overlap_region': use_overlap_region}

    # Get axis of motion
    axis = yp.argmin(yp.asbackend([last - first for (last, first) in zip(roi_list[-1].start, roi_list[0].start)], 'numpy'))[0]

    # Get direction along axis
    motion_is_forward = [last - first for (last, first) in zip(roi_list[-1].start, roi_list[0].start)][axis] > 0

    # Store parameters for metadata
    normalization_parameters = {'scale_list': [], 'use_overlap_region': use_overlap_region}

    # Loop over frame indicies
    mean_list_first, mean_list_second, scale_values, rois_used = [], [], [], []

    # Loop over frames
    for frame_index in range(len(measurement_list)):

        # Get ROIs
        roi_current = roi_list[frame_index]
        frame_current = measurement_list[frame_index]

        # Determine which rois overlap
        overlapping_rois = [(index, roi) for (index, roi) in enumerate(rois_used) if roi.overlaps(roi_current)]

        # Loop over overlapping ROIs
        if len(overlapping_rois) > 0:
            # Get ratios of overlap areas
            ratio_list = []
            ratio_dimensions = 0
            for index, overlap_roi in overlapping_rois:
                overlap_current, overlap_prev = yp.roi.getOverlapRegion((frame_current, measurement_list[index]), (roi_current, roi_list[index]))

                ratio_list.append(scale_values[index] * yp.mean(yp.abs(overlap_prev)) / yp.mean(yp.abs(overlap_current)) * yp.prod(yp.shape(overlap_prev)))
                ratio_dimensions += yp.prod(yp.shape(overlap_prev))

            # Calculate average amount to scale
            scale = yp.scalar(yp.sum([ratio / ratio_dimensions for ratio in ratio_list]))
        else:
            scale = 1.0

        # Indicate that we have "used" this ROI
        rois_used.append(roi_current)

        # Store scale value
        scale_values.append(yp.scalar(scale))

    # Wrap final value around
    if wrap_final_value or wrap_coefficient:
        # This should be equal to overlap size - extract from roi overlap
        wrap_size = yp.min([sh - abs(st_1 - st_0) for (sh, st_1, st_0) in zip(roi_list[1].shape, roi_list[1].start, roi_list[0].start)])

        # Determine size of ROI to measure
        wrap_roi_size = list(roi_list[0].shape)
        wrap_roi_size[axis] = wrap_size

        if not motion_is_forward:
            first_start = (0, 0)
            last_start = [0, 0]
            last_start[axis] = roi_list[-1].shape[axis] - wrap_size
        else:
            last_start = (0, 0)
            first_start = [0, 0]
            first_start[axis] = roi_list[0].shape[axis] - wrap_size

        # Create ROI objects
        roi_first = yp.Roi(start=first_start, shape=wrap_roi_size)
        roi_last = yp.Roi(start=last_start, shape=wrap_roi_size)

        # Append mean values to list
        first_region = measurement_list[0][roi_first.slice]
        first_region = first_region[first_region != 0.0]

        second_region = measurement_list[-1][roi_last.slice]
        second_region = second_region[second_region != 0.0]

        # Note that the mis-match is not a bug - the cycle loops
        wrapped_mean_second = yp.mean(yp.real(first_region[~yp.isnan(first_region)]))
        wrapped_mean_first = yp.mean(yp.real(second_region[~yp.isnan(second_region)]))

    # Store normalization parameters
    normalization_parameters['scale_list'].append({'scale': scale_values})

    # Scale values so that the final scale is one
    if wrap_final_value or wrap_coefficient:

        # Calculate necessary mean ratio
        if wrap_coefficient is None:
            wrap_coefficient = yp.scalar((scale_values[0] * wrapped_mean_second) / (scale_values[-1] * wrapped_mean_first)) / len(measurement_list)
            print('Calculated wrapping coefficient %g' % wrap_coefficient)
        else:
            print('Using wrapping coefficient %g' % wrap_coefficient)

        # Reverse if movement is not positive
        if not movement_is_positive:
            # TODO: Why multiply by two?
            wrap_coefficient /= 2

        # Scale wrap coefficient by number of measurements
        wrap_coefficient *= len(measurement_list)

        # Weigh scale values by wrap coefficient
        scale_list_wrap = [(1 + (wrap_coefficient - 1) * (index / (len(measurement_list)))) for index in range(len(measurement_list))]
        for scale_value_index in range(len(scale_values)):
            scale_values[scale_value_index] *= scale_list_wrap[scale_value_index]

    # High-pass filter results
    if high_pass_filter:
        filtered = sp.ndimage.filters.gaussian_filter1d(np.asarray(scale_values), 10)
        filtered /= filtered[0]
        scale_values = (np.asarray(scale_values) / filtered).tolist()

    # Debug
    if debug:
        # Generate difference in overlapping regions
        P = np.diagflat(scale_values[1:])
        diff = np.asarray(mean_list_first) - P.dot(np.asarray(mean_list_second))

        # Append additional value
        if wrap_final_value:
            diff = np.append(diff, (scale_values[0] * wrapped_mean_first) - (scale_values[-1] * wrapped_mean_second))

        print('L2 difference: %g' % (yp.norm(diff)))

    # Return
    return scale_values, normalization_parameters


def flattenFrameBackground(frame, polynomial_order=2, debug=False):
    """Given a frame, estimate the background using a polynomial fit."""

    # Convert to numpy backend
    original_backend = yp.getBackend(frame)
    frame = yp.changeBackend(frame, 'numpy')

    # Define data as projections onto each axis
    data_x = np.sum(frame, axis=0)
    data_x /= yp.max(data_x)

    data_y = np.sum(frame, axis=1)
    data_y /= yp.max(data_y)

    # Calculate pixel coordinates
    coords_x = np.arange(frame.shape[1])
    coords_y = np.arange(frame.shape[0])

    # Perform fit
    fit_func_x = np.poly1d(np.polyfit(coords_x, data_x, polynomial_order))
    fit_func_y = np.poly1d(np.polyfit(coords_y, data_y, polynomial_order))

    # Calculate background
    bg = (fit_func_x(coords_x)[np.newaxis, :] * fit_func_y(coords_y)[:, np.newaxis])

    # Convert back to origional backend
    bg = yp.changeBackend(bg, original_backend)

    # Print debugging information
    if debug:
        plt.figure()
        plt.subplot(121)
        plt.plot(data_y)
        plt.plot(fit_func_y(coords_y))
        plt.subplot(122)
        plt.plot(data_x)
        plt.plot(fit_func_x(coords_x))

    # Return
    return bg


def normalize_segments(roi_list, measurement_list, write_results=True, debug=False, polynomial_order=1):

    # Initialize variables
    rois_used, overlap_pair_list, overlap_roi_list = [], [], []

    # Determine object shape
    object_shape = sum(roi_list).shape

    # Loop over frames
    for frame_index in range(len(measurement_list)):

        # Get ROIs
        roi_current = roi_list[frame_index]
        frame_current = measurement_list[frame_index]

        # Determine which rois overlap
        overlapping_rois = [(index, roi) for (index, roi) in enumerate(rois_used) if roi.overlaps(roi_current)]

        # Ensure we have only one overlap position (or none, as is the case for the first frame)
        assert len(overlapping_rois) < 2

        # Loop over overlapping ROIs
        if len(overlapping_rois) > 0:

            # Get exact ROI overlap
            roi_overlap = overlapping_rois[0][1] & roi_current
            overlap_roi_list.append(roi_overlap)

            # Get overlap regions
            index, overlap_roi = overlapping_rois[0]
            overlap_current, overlap_prev = yp.roi.getOverlapRegion((frame_current, measurement_list[index]), (roi_current, roi_list[index]))
            overlap_pair_list.append((overlap_current, overlap_prev))

        # Indicate that we have "used" this ROI
        rois_used.append(roi_current)

    # Filter overlap pairs into even and odd groups and weigh these group by scale_values
    normalization_vector_list, fit_coefficients_list, x_list, y_list = [], [], [], []
    for (index, pair) in enumerate(overlap_pair_list):

        # Compute ratio of second pair to first
        overlap_ratio_meas = yp.cast(pair[1] / pair[0], backend='numpy')

        # Vectorize
        x_single = np.arange(overlap_roi_list[index].start[1], overlap_roi_list[index].end[1])
        x = np.tile(x_single, overlap_ratio_meas.shape[0])
        y = overlap_ratio_meas.ravel()

        # # Filter outliers
        # mask = (y < 2) & (y > 0.25)
        # x, y = x[mask], y[mask]

        # Fit
        fit_coefficients = np.polyfit(x, y, polynomial_order)

        # Calculate normalization vector
        fit_fn = np.poly1d(fit_coefficients)

        # Store lists
        x_list.append(x)
        y_list.append(y)
        fit_coefficients_list.append(fit_coefficients)
        normalization_vector_list.append(fit_fn(np.arange(object_shape[1])))

    # Show weights if debug flag is set
    if debug:
        for index in range(len(fit_coefficients_list)):
            fit_fn = np.poly1d(fit_coefficients_list[index])
            x = x_list[index]
            y = y_list[index]
            plt.figure()
            plt.plot(x, y, '.')
            plt.plot(x[:overlap_ratio_meas.shape[1]], fit_fn(x[:overlap_ratio_meas.shape[1]]))
            plt.xlabel('Pixel Coordinate')
            plt.ylabel('Normalization coefficient')
            plt.ylim((0, 3))
            plt.title(str(fit_coefficients_list[index]))

    # Seralize normalization filters
    normalization_vector_list_serial = [yp.prod(normalization_vector_list[:index + 1]) for index in range(len(normalization_vector_list))]

    # Append ones to the first vector
    normalization_vector_list_serial = [yp.ones(normalization_vector_list_serial[0].shape)] + normalization_vector_list_serial

    # # Apply weighting if desired
    # if write_results:
    #     for index in range(len(measurement_list)):
    #         normalization = normalization_vector_list_serial[index]
    #         measurement = measurement_list[index]
    #
    #         # Crop normalization to correct FOV
    #         normalization = normalization[roi_list[index].start[1]:roi_list[index].start[1] + yp.shape(measurement)[1]]
    #         # normalization_vector_list_serial[index] = normalization
    #
    #         # Broadcast along last dimension
    #         measurement[:] = yp.bcast(measurement, yp.changeBackend(normalization, yp.getBackend(measurement)))

    # Return weighting functions
    return normalization_vector_list_serial


def alignRoiListToOrigin(roi_list):
    """Align ROI list to origin."""
    # Calculate mininum position in ROIs
    min_start_position = (1e10, 1e10)
    for roi in roi_list:
        min_start_position = [min(st, min_dim) for (st, min_dim) in zip(roi.start, min_start_position)]

    # Align roi objects to origin and assign input_shape
    input_shape = sum(roi_list).shape
    for roi in roi_list:
        roi -= min_start_position
        roi.input_shape = input_shape


def estimateBackgroundList(frame_list, polynomial_order=2, downsample=32, debug=False):
    """Estimate the commpon background of a frame list."""
    # Estimate background of first frame
    bg = flattenFrameBackground(frame_list[0], polynomial_order=polynomial_order)
    # Estimate remaining frames
    for frame in frame_list[1:]:
        bg = bg + flattenFrameBackground(frame, polynomial_order=polynomial_order)

    # Normalize
    bg = bg / len(frame_list)
    bg = bg / yp.max(bg)

    # Return
    return bg


def estimateBackgroundPolynomial(frame, polynomial_order=1, downsample=32, debug=False):
    """Calculates the background of a frame using polynomail fit."""
    # Downsample frame
    frame_downsampled = yp.filter.downsample(yp.changeBackend(frame, 'numpy'), downsample)

    # Fit pixels to line
    points_y, points_x = yp.grid(yp.shape(frame_downsampled), backend='numpy')
    mask = frame_downsampled != 0
    fit_coefficients_y = np.polyfit(points_y[mask].ravel(), frame_downsampled[mask].ravel(), polynomial_order)
    fit_coefficients_x = np.polyfit(points_x[mask].ravel(), frame_downsampled[mask].ravel(), polynomial_order)

    # Apply fit function
    fit_fn_y = np.poly1d(fit_coefficients_y)
    fit_fn_x = np.poly1d(fit_coefficients_x)

    # Create flat-field vectors
    fit_vector_y = fit_fn_y(np.arange(yp.shape(frame_downsampled)[0]))
    fit_vector_x = fit_fn_x(np.arange(yp.shape(frame_downsampled)[1]))

    # Normalize flat-field vectors
    fit_vector_y /= yp.mean(fit_vector_y)
    fit_vector_x /= yp.mean(fit_vector_x)

    # Expand and upsample
    flat_field_y = yp.squeeze(yp.resize(fit_vector_y[:, np.newaxis], (yp.shape(frame)[0], 1)))
    flat_field_x = yp.squeeze(yp.resize(fit_vector_x[np.newaxis, :], (1, yp.shape(frame)[1])))

    # Cast to correct dtype
    flat_field = yp.changeBackend(flat_field_y, yp.getBackend(frame)), yp.changeBackend(flat_field_x, yp.getBackend(frame))

    # Show debuging informaiton if desired
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        plt.imshow(yp.outer(flat_field[0], flat_field[1]))
        plt.subplot(122)
        plt.imshow(frame)

    # Return
    return flat_field

# TODO move this into mddataset!
def normalize_measurements(dataset, blur_axis=1, decimation_factor=8, flatten_field=False,
                           debug=False, polynomial_order=1):
    stitched_segment_list, stitched_segment_roi_list, edge_normalization_factors_list = [], [], []
    segment_frame_roi_list = []
    for segment_index in dataset.frame_segment_list_full:
        # Set segment index
        dataset.frame_segment_list = [segment_index]

        # Load measurement list
        measurement_list = [yp.filter.decimate(dataset.frame_list[index], decimation_factor) for index in range(len(dataset.frame_mask))]

        # Get ROI list from dataset metadata
        roi_list_measurements = [roi.copy() for roi in dataset.roi_list]

        # Apply registration if provided
        if 'registration' in dataset.metadata.calibration:
            for index, roi in enumerate(roi_list_measurements):
                offset = dataset.metadata.calibration['registration']['frame_offsets'][dataset.frame_mask[index]]
                roi += offset

        # Decimate ROIs
        roi_list_measurements = [roi.decimate(decimation_factor) for roi in roi_list_measurements]

        # Calculate support of this segment
        segment_support_roi = sum(roi_list_measurements)

        # Align local ROI to origin
        alignRoiListToOrigin(roi_list_measurements)

        # Store measurement rois
        segment_frame_roi_list.append([roi.copy() for roi in roi_list_measurements])

        # Assign input shape
        input_shape = sum(roi_list_measurements).shape
        for roi in roi_list_measurements:
            roi.input_shape = input_shape

        # Perform edge normalization
        edge_normalization_factors = normalize_roi_list(measurement_list, roi_list_measurements)[0]
        for (factor, measurement) in zip(edge_normalization_factors, measurement_list):
            measurement[:] = measurement * factor

        # Store edge normalization factors
        edge_normalization_factors_list.append(edge_normalization_factors)

        # Stitch segment
        G = ops.Segmentation(roi_list_measurements)
        y = ops.VecStack(measurement_list)

        # Append segment and segment roi to list
        stitched_segment_list.append(G.inv * y)
        stitched_segment_roi_list.append(segment_support_roi)

    # Center stitched_segment_roi_list
    alignRoiListToOrigin(stitched_segment_roi_list)

    if debug:
        yp.listPlotFlat(stitched_segment_list)

    if len(stitched_segment_roi_list) > 1:
        # Perform segment normalization
        segment_normalization_list = normalize_segments(stitched_segment_roi_list,
                                                        stitched_segment_list,
                                                        write_results=False,
                                                        polynomial_order=polynomial_order,
                                                        debug=debug)

    else:
        segment_roi = stitched_segment_roi_list[0]
        segment_normalization_list = [yp.ones(segment_roi.end[1], dataset.dtype, dataset.backend)]

    # Stitch measurements and compute flat-field
    G = ops.Segmentation(stitched_segment_roi_list, alpha_blend_size=0)

    # Concatenate measurements
    y = ops.VecStack(stitched_segment_list)

    # Check that all ROI objects are of the correct dimensions
    for segment, roi in zip(stitched_segment_list, stitched_segment_roi_list):
        assert all([yp.shape(segment)[i] == roi.shape[i] for i in range(len(roi.shape))])
        assert yp.shape(segment)[0] == stitched_segment_roi_list[0].shape[0], "All segments must have same height!"

    # Stitch downsampled recon
    stitched = G.inv * y

    # Compute flat field
    flat_field_y, flat_field_x = estimateBackgroundPolynomial(stitched, polynomial_order=polynomial_order, debug=debug)

    # Convert all variables back to original datatype
    segment_normalization_list = [yp.changeBackend(segment, dataset.backend) for segment in segment_normalization_list]

    # Combine both segment and frame normalization factors into a single, frame-based normalization factor list
    frame_normalization_list_y, frame_normalization_list_x = [], []
    for (segment_normalization, frame_roi_list, segment_roi, edge_normalization_factors) in zip(segment_normalization_list, segment_frame_roi_list, stitched_segment_roi_list, edge_normalization_factors_list):

        # Crop segment normalization to correct FOV
        _segment_normalization = segment_normalization[segment_roi.start[1]:segment_roi.end[1]]

        # Generate frame normalization list in x (includes flat field, segment normalization, and single-frame edge matching)
        flat_field_x_filtered = flat_field_x[segment_roi.start[1]:segment_roi.end[1]] if flat_field_x is not None else yp.ones(segment_roi.shape[1])
        frame_normalization_list_x += [yp.filter.upsample(((1 / flat_field_x_filtered) * _segment_normalization)[frame_roi.start[1]: frame_roi.end[1]], decimation_factor) * factor for (frame_roi, factor) in zip(frame_roi_list, edge_normalization_factors)]

        # Generate frame normalization list in y (includes only flat-field)
        flat_field_y_filtered = flat_field_y[segment_roi.start[0]:segment_roi.end[0]] if flat_field_y is not None else yp.ones(segment_roi.shape[0])
        frame_normalization_list_y += [yp.filter.upsample((1 / flat_field_y_filtered)[frame_roi.start[0]: frame_roi.end[0]], decimation_factor) for frame_roi in frame_roi_list]

    return (frame_normalization_list_y, frame_normalization_list_x)
