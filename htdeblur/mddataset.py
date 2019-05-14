from comptic.containers import Dataset
import llops as yp
import numpy as np

class MotionDeblurDataset(Dataset):
    """ A subclass of the comptic dataset with motion deblur specific functions"""

    def __init__(self, *args, **kwargs):
        """This class implements motion-deblur specific changes and fixes."""
        super(self.__class__, self).__init__(*args, **kwargs)

        # Assign type
        self.type = 'motion_deblur'

        # Instantiate variables
        self._frame_segment_list = None


    def fixOldMdDatasets(self):
        # Expand frame_state_list
        self.expandFrameStateList(self.frame_state_list, position_units='mm')

        # Fix frame state list
        self.fixFrameStateList(self.frame_state_list)

        # Flip position coordinates
        self.flipPositionCoordinates(x=True)

        # Flip illumination sequences
        self.flipIlluminationSequence()

    @property
    def frame_segment_map(self):
        """Calculates and returns position segment indicies of each frame."""
        frame_segment_list = []
        for frame_state in self.frame_state_list:
            frame_segment_list.append(frame_state['position']['common']['linear_segment_index'])

        return frame_segment_list

    @property
    def frame_segment_direction_list(self):
        """Calculates and returns position segment indicies of each frame."""

        # Determine the direction of individual segments
        segment_direction_list = []
        segment_list = sorted(yp.unique(self.frame_segment_map))

        # Store previous position segment indicies
        frame_mask_old = self.frame_mask

        # Loop over segments
        for segment_index in segment_list:

            # Set positon segment index
            self.frame_segment_list = [segment_index]

            # Get start position of first frame in segment
            x_start = self.frame_state_list[0]['position']['states'][0][0]['value']['x']
            y_start = self.frame_state_list[0]['position']['states'][0][0]['value']['y']

            # Get start position of last frame in segment
            x_end = self.frame_state_list[-1]['position']['states'][-1][0]['value']['x']
            y_end = self.frame_state_list[-1]['position']['states'][-1][0]['value']['y']

            vector = np.asarray(((y_end - y_start), (x_end - x_start)))
            vector /= np.linalg.norm(vector)

            # Append segment direction vector to list
            segment_direction_list.append(vector.tolist())

        # Reset position segment indicies
        self.frame_mask = frame_mask_old

        # Expand to frame basis
        frame_segment_direction_list = []
        for frame_index in range(self.shape[0]):
            # Get segment index
            segment_index = self.frame_segment_map[frame_index] - min(self.frame_segment_map)

            # Get segment direction
            segment_direction = segment_direction_list[segment_index]

            # Append to list
            frame_segment_direction_list.append(segment_direction)

        return frame_segment_direction_list

    def expandFrameStateList(self, frame_state_list, position_units='mm'):
        """ This function expands redundant information in the frame_state_list of a dataset (specific to motion deblur datasets for now)"""

        # Store first frame as a datum
        frame_state_0 = frame_state_list[0]

        # Loop over frame states
        for frame_state in frame_state_list:

            # Fill in illumination and position if this frame state is compressed
            if type(frame_state['illumination']) is str:
                frame_state['illumination'] = copy.deepcopy(frame_state_0['illumination'])

                # Deterime direction of scan
                dx0 = frame_state['position']['states'][-1][0]['value']['x'] - frame_state['position']['states'][0][0]['value']['x']
                dy0 = frame_state['position']['states'][-1][0]['value']['y'] - frame_state['position']['states'][0][0]['value']['y']
                direction = np.asarray((dy0, dx0))
                direction /= np.linalg.norm(direction)

                # Get frame spacing
                spacing = frame_state['position']['common']['velocity'] * frame_state['position']['common']['led_update_rate_us'] / 1e6
                dy = direction[0] * spacing
                dx = direction[1] * spacing

                # Assign new positions in state_list
                states_new = []
                for state_index in range(len(frame_state_0['position']['states'])):
                    state = copy.deepcopy(frame_state['position']['states'][0])
                    state[0]['time_index'] = state_index
                    state[0]['value']['x'] += dx * state_index
                    state[0]['value']['y'] += dy * state_index
                    state[0]['value']['units'] = position_units
                    states_new.append(state)

                frame_state['position']['states'] = states_new

    def fixFrameStateList(self, frame_state_list, major_axis='y'):
        """Catch-all function for various hacks and dataset incompatabilities."""
        axis_coordinate_list = []
        # Loop over frame states
        for frame_state in frame_state_list:
            # Check if this state is a shallow copy of the first frame
            if id(frame_state['illumination']) is not id(self._frame_state_list[0]['illumination']):

                # Check if position is in the correct format (some stop and stare datasets will break this)
                if type(frame_state['position']) is list:

                    # Fix up positions
                    frame_state['position'] = {'states': frame_state['position']}
                    frame_state['position']['common'] = {'linear_segment_index': 0}

                    # Fix up illumination
                    frame_state['illumination'] = {'states': frame_state['illumination']}

                # Add linear segment indicies if these do not already exist
                frame_axis_coordinate = frame_state['position']['states'][0][0]['value'][major_axis]
                if frame_axis_coordinate not in axis_coordinate_list:
                    axis_coordinate_list.append(frame_axis_coordinate)
                    position_segment_index = len(axis_coordinate_list) - 1
                else:
                    position_segment_index = axis_coordinate_list.index(frame_axis_coordinate)

                # frame_segment_list.append(position_segment_index)
                frame_state['position']['common']['linear_segment_index'] = position_segment_index
            else:
                print('Ignoring state.')

    def flipPositionCoordinates(self, x=False, y=False):
        for frame_state in self._frame_state_list:
            # Check if this state is a shallow copy of the first frame
            if id(frame_state) is not id(self._frame_state_list[0]):
                for state in frame_state['position']['states']:
                    for substate in state:
                        if x:
                            substate['value']['x'] *= -1
                        if y:
                            substate['value']['y'] *= -1
            else:
                print('Ignoring state.')

    def flipIlluminationSequence(self):
        for frame_state in self._frame_state_list:
            frame_state['illumination']['states'] = list(reversed(frame_state['illumination']['states']))

    def blur_vectors(self, dtype=None, backend=None, debug=False,
                     use_phase_ramp=False, corrections={}):
        """
        This function generates the object size, image size, and blur kernels from
        a libwallerlab dataset object.

            Args:
                dataset: An io.Dataset object
                dtype [np.float32]: Which datatype to use for kernel generation (All numpy datatypes supported)
            Returns:
                object_size: The object size this dataset can recover
                image_size: The computed image size of the dataset
                blur_kernel_list: A dictionary of blur kernels lists, one key per color channel.

        """
        # Assign dataset
        dataset = self

        # Get corrections from metadata
        if len(corrections) is 0 and 'blur_vector' in self.metadata.calibration:
            corrections = dataset.metadata.calibration['blur_vector']

        # Get datatype and backends
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

        for frame_index in range(dataset.shape[0]):
            frame_state = dataset.frame_state_list[frame_index]

            # Store which segment this measurement uses
            frame_segment_map.append(frame_state['position']['common']['linear_segment_index'])

            # Extract list of illumination values for each time point
            if 'illumination' in frame_state:
                illumination_list_frame = []
                if type(frame_state['illumination']) is str:
                    illum_state_list = self._frame_state_list[0]['illumination']['states']
                else:
                    illum_state_list = frame_state['illumination']['states']
                for time_point in illum_state_list:
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
                from htdeblur.blurkernel import getPositionListBoundingBox
                blur_vector_roi = getPositionListBoundingBox(positions_used)

                # Append to list
                blur_vector_roi_list.append(blur_vector_roi)

                # Crop illumination list to values within the support used
                illumination_list.append([illumination_list_frame[index] for index in range(min(position_indicies_used), max(position_indicies_used) + 1)])

                # Store corresponding positions
                position_list.append(positions_used)

        # Apply kernel scaling or compression if necessary
        if 'scale' in corrections:

            # We need to use phase-ramp based kernel generation if we modify the positions
            use_phase_ramp = True

            # Modify position list
            for index in range(len(position_list)):
                _positions = np.asarray(position_list[index])
                for scale_correction in corrections['scale']:
                    factor, axis = corrections['scale']['factor'], corrections['scale']['axis']
                    _positions[:, axis] = ((_positions[:, axis] - yp.min(_positions[:, axis])) * factor + yp.min(_positions[:, axis]))
                position_list[index] = _positions.tolist()

        # Synthesize blur vectors
        blur_vector_list = []
        for frame_index in range(dataset.shape[0]):
            #  Generate blur vectors
            if use_phase_ramp:
                import ndoperators as ops
                kernel_shape = [yp.fft.next_fast_len(max(sh, 1)) for sh in blur_vector_roi_list[frame_index].shape]
                offset = yp.cast([sh // 2 + st for (sh, st) in zip(kernel_shape, blur_vector_roi_list[frame_index].start)], 'complex32', dataset.backend)

                # Create phase ramp and calculate offset
                R = ops.PhaseRamp(kernel_shape, dtype='complex32', backend=dataset.backend)

                # Generate blur vector
                blur_vector = yp.zeros(R.M, dtype='complex32', backend=dataset.backend)
                for pos, illum in zip(position_list[frame_index], illumination_list[frame_index]):
                    pos = yp.cast(pos, dtype=dataset.dtype, backend=dataset.backend)
                    blur_vector += (R * (yp.cast(pos - offset, 'complex32')))

                # Take inverse Fourier Transform
                blur_vector = yp.abs(yp.cast(yp.iFt(blur_vector)), 0.0)

                if position_list[frame_index][0][-1] > position_list[frame_index][0][0]:
                    blur_vector = yp.flip(blur_vector)

            else:
                blur_vector = yp.asarray([illum[0]['value']['w'] for illum in illumination_list[frame_index]],
                                         dtype=dtype, backend=backend)

            # Normalize illuminaiton vectors
            blur_vector /= yp.scalar(yp.sum(blur_vector))

            # Append to list
            blur_vector_list.append(blur_vector)

        # Return
        return blur_vector_list, blur_vector_roi_list

    @property
    def roi_list(self):

        # Get blur vectors and ROIs
        blur_vector_list, blur_vector_roi_list = self.blur_vectors()

        # Generate measurement ROIs
        roi_list = []
        for index, (blur_vector, blur_roi) in enumerate(zip(blur_vector_list, blur_vector_roi_list)):
            # Determine ROI start from blur vector ROI
            convolution_support_start = [kernel_center - sh // 2 for (kernel_center, sh) in zip(blur_roi.center, self.frame_shape)]

            # Generate ROI
            roi_list.append(yp.Roi(start=convolution_support_start, shape=self.frame_shape))

        return roi_list

    @property
    def frame_segment_list(self):
        """Returns all segment indicies which are currently used (one per segment, NOT one per frame)."""
        if self._frame_segment_list:
            # Return saved frame_segment_list
            return self._frame_segment_list
        else:
            # Determine which position segment indicies are in this dataset
            frame_segment_list = []
            for frame_state in self.frame_state_list:
                if frame_state['position']['common']['linear_segment_index'] not in frame_segment_list:
                    frame_segment_list.append(frame_state['position']['common']['linear_segment_index'])

            # Return these indicies
            return frame_segment_list

    @frame_segment_list.setter
    def frame_segment_list(self, new_frame_segment_list):

        # Ensure the input is of resonable type
        assert type(new_frame_segment_list) in (tuple, list)

        # Check that all elements are within the number of frames
        assert all([index in self.frame_segment_list_full for index in new_frame_segment_list])

        # Set new position_segment_index
        self._frame_segment_list = new_frame_segment_list

        # Update frame_mask to reflect this list
        frame_subset = []
        for position_segment_index in new_frame_segment_list:
            for index, frame_state in enumerate(self._frame_state_list):
                if frame_state['position']['common']['linear_segment_index'] == position_segment_index:
                    frame_subset.append(index)
        self.frame_mask = frame_subset

    @property
    def frame_frame_segment_list(self):
        """Returns a list of segment indicies for each frame."""
        _frame_frame_segment_list = []
        for frame_state in self._frame_state_list:
            _frame_frame_segment_list.append(frame_state['position']['common']['linear_segment_index'])
        return _frame_frame_segment_list

    @property
    def frame_segment_list_full(self):
        """Returns all segment indicies which are in the dataset (one per segment, NOT one per frame)."""
        _frame_segment_list_full = []
        for frame_state in self._frame_state_list:
            if frame_state['position']['common']['linear_segment_index'] not in _frame_segment_list_full:
                _frame_segment_list_full.append(frame_state['position']['common']['linear_segment_index'])
        return _frame_segment_list_full

    def normalize(self, force=False):
        if 'normalization' not in self.metadata.calibration or force:
            # Calculation normalization vectors
            from htdeblur.recon import normalize_measurements
            (frame_normalization_list_y, frame_normalization_list_x) = normalize_measurements(self, debug=False)

            # Convert to numpy for saving
            _frame_normalization_list_y = [yp.changeBackend(norm, 'numpy').tolist() for norm in frame_normalization_list_y]
            _frame_normalization_list_x = [yp.changeBackend(norm, 'numpy').tolist() for norm in frame_normalization_list_x]

            # Save in metadata
            self.metadata.calibration['normalization'] = {'frame_normalization_x': _frame_normalization_list_x,
                                                                  'frame_normalization_y': _frame_normalization_list_y}
            # Save calibration file
            self.saveCalibration()

    def register(self, force=False, segment_offset=(0, 550), frame_offset=-26,
                 blur_axis=1, frame_registration_mode=None, debug=False,
                 segment_registration_mode=None, write_file=True):

        if 'registration' not in self.metadata.calibration or force:

            # Assign all segments
            self.frame_segment_list = self.frame_segment_list_full

            # Pre-compute indicies for speed
            frame_segment_map = self.frame_segment_map
            frame_segment_direction_list = self.frame_segment_direction_list

            # Apply pre-computed offset
            frame_offset_list = []
            for frame_index in range(len(self.frame_mask)):

                # Get segment index and direction
                segment_direction_is_left_right = frame_segment_direction_list[frame_index][1] > 0

                # Get index of frame in this segment
                frame_segment_index = len([segment for segment in frame_segment_map[:frame_index] if segment == frame_segment_map[frame_index]])

                # Get index of current segment
                segment_index = frame_segment_map[frame_index]

                # Apply frame dependent offset
                _offset_frame = [0, 0]
                _offset_frame[blur_axis] = frame_segment_index * frame_offset
                if not segment_direction_is_left_right:
                    _offset_frame[blur_axis] *= -1

                # Apply segment dependent offset
                _offset_segment = list(segment_offset)
                if segment_direction_is_left_right:
                    for ax in range(len(_offset_segment)):
                        if ax is blur_axis:
                            _offset_segment[ax] *= -1
                        else:
                            _offset_segment[ax] *= segment_index

                # Combine offsets
                offset = [_offset_frame[i] + _offset_segment[i] for i in range(2)]

                # Append to list
                frame_offset_list.append(offset)

            # Apply registration
            if frame_registration_mode is not None:

                # Register frames within segments
                for segment_index in self.frame_segment_list_full:
                    self.frame_segment_list = [segment_index]

                    # Get frame ROI list
                    roi_list = self.roi_list

                    # Get offsets for this segment
                    frame_offset_list_segment = [frame_offset_list[index] for index in self.frame_mask]

                    # Apply frame offsets from previous steps
                    for roi, offset in zip(roi_list, frame_offset_list_segment):
                        roi += offset

                    # Perform registration
                    from comptic.registration import register_roi_list
                    frame_offset_list_segment = register_roi_list(self.frame_list,
                                                                  roi_list,
                                                                  debug=debug,
                                                                  tolerance=(1000, 1000),
                                                                  method=frame_registration_mode,
                                                                  force_2d=False,
                                                                  axis=1)

                    # Apply correction to frame list
                    for index, frame_index in enumerate(self.frame_mask):
                        for i in range(len(frame_offset_list[frame_index])):
                            frame_offset_list[frame_index][i] += frame_offset_list_segment[index][i]

            if segment_registration_mode is not None:
                from llops.operators import VecStack, Segmentation
                from htdeblur.recon import alignRoiListToOrigin, register_roi_list
                stitched_segment_list, stitched_segment_roi_list = [], []
                # Stitch segments
                for segment_index in self.frame_segment_list_full:
                    self.frame_segment_list = [segment_index]

                    # Get frame ROI list
                    roi_list = self.roi_list

                    # Get offsets for this segment
                    frame_offset_list_segment = [frame_offset_list[index] for index in self.frame_mask]

                    # Apply frame offsets from previous steps
                    for roi, offset in zip(roi_list, frame_offset_list_segment):
                        roi += offset

                    # Determine segment ROI
                    stitched_segment_roi_list.append(sum(roi_list))

                    # Align ROI list to origin
                    alignRoiListToOrigin(roi_list)

                    # Create segmentation operator
                    G = Segmentation(roi_list)

                    # Get measurement list
                    y = yp.astype(VecStack(self.frame_list), G.dtype)

                    # Append to list
                    stitched_segment_list.append(G.inv * y)

                # Register stitched segments
                frame_offset_list_segment = register_roi_list(stitched_segment_list,
                                                              stitched_segment_roi_list,
                                                              debug=debug,
                                                              tolerance=(200, 200),
                                                              method=segment_registration_mode)

                # Apply registration to all frames
                self.frame_segment_list = self.frame_segment_list_full

                # Apply offset to frames
                for frame_index in range(self.shape[0]):

                    # Get segment index
                    segment_index = self.frame_segment_map[frame_index]

                    # Apply offset
                    for i in range(len(frame_offset_list[frame_index])):
                        frame_offset_list[frame_index][i] += frame_offset_list_segment[segment_index][i]

            # Set updated values in metadata
            self.metadata.calibration['registration'] = {'frame_offsets': frame_offset_list,
                                                         'segment_offset': segment_offset,  # For debugging
                                                         'frame_offset': frame_offset}      # For debugging

            # Save calibration file
            if write_file:
                self.saveCalibration()
