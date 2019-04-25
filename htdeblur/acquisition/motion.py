# Copyright 2017 Regents of the University of California
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with # the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os, sys, time, copy, collections, math, json
import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
import llops as yp

# Custom scale bar object
from matplotlib_scalebar.scalebar import ScaleBar

# Libwallerlab imports
from llops import display
from llops import Roi


class StopAndStareAcquisition():
    # Initialization
    def __init__(self, hardware_controller_list, system_metadata,
                 illumination_type='bf',
                 illumination_sequence=None,
                 frame_spacing_mm=1,
                 object_size_mm=(0.5, 0.5),
                 reuse_illumination_sequence=True,
                 max_exposure_time_s=2,
                 exposure_time_pad_s=0.0,
                 velocity_mm_s=None,
                 exposure_time_s=None,
                 debug=False,
                 trigger_mode='software',
                 motion_acceleration_mm_s_2=1e3,
                 flip_pathway=False,
                 acquisition_timeout_s=3,
                 illumination_na_pad=0.03,
                 illumination_color={'w': 127},
                 settle_time_s=0):


        # Parse options
        self.illumination_type = illumination_type
        self.settle_time_s = settle_time_s
        self.object_size_mm = object_size_mm
        self.frame_spacing_mm = frame_spacing_mm
        self.flip_pathway = flip_pathway
        self.exposure_time_pad_s = exposure_time_pad_s
        self.debug = debug
        self.motion_acceleration_mm_s_2 = motion_acceleration_mm_s_2
        self.velocity_mm_s = velocity_mm_s
        self.max_exposure_time_s = max_exposure_time_s
        self.illumination_na_pad = illumination_na_pad
        self.illumination_color = illumination_color
        self.acquisition_timeout_s = acquisition_timeout_s

        # Define controller objects, which act as hardware interfaces.
        # These should be in an ordered dictionary because the order which they
        # are initialized matters when using a mix of hardware and software triggering.
        self.hardware_controller_list = collections.OrderedDict()

        # First add hardware triggered elements so they perform their set-up before we trigger software elements
        for controller in hardware_controller_list:
            if controller.trigger_mode is 'hardware':
                self.hardware_controller_list[controller.type] = controller
                controller.reset()
                controller.seq_clear()

        # Then, add software triggered elements
        for controller in hardware_controller_list:
            if controller.trigger_mode is 'software':
                self.hardware_controller_list[controller.type] = controller
                controller.reset()
                controller.seq_clear()

        # Check to be sure a sequence acquisition is not running
        assert 'camera' in self.hardware_controller_list, 'Did not find camera controller!'

        # Store metadata object
        self.metadata = system_metadata

        # Ensure we have all necessary metadata for basic acquisition
        assert self.metadata.objective.na is not None, 'Missing objective.na in metadata.'
        assert self.metadata.objective.mag is not None, 'Missing objective.mag in metadata.'
        assert self.metadata.camera.pixel_size_um is not None, 'Missing pixel size in metadata.'

        # Update effective pixel size (for scale bar)
        self.metadata.system.eff_pixel_size_um = self.metadata.camera.pixel_size_um / (self.metadata.objective.mag * self.metadata.system.mag)

        # Trigger Constants
        self.TRIG_MODE_EVERY_PATTERN = 1
        self.TRIG_MODE_ITERATION = -1
        self.TRIG_MODE_START = -2

        # Frame state time sequence, will default to a sequence of one exposure time per frame if left as None
        self.time_sequence_s = None
        self.exposure_time_s = None
        self.hardware_sequence_timing = None

        # Turn off fast sequencing for illumination by default since this is only avaolable with certain LED arrays
        if 'illumination' in self.hardware_controller_list:
            self.hardware_controller_list['illumination'].use_fast_sequence = False

        # print(type(self.))
        self.metadata.type = 'stop and stare'

        assert 'illumination' in self.hardware_controller_list,  'Stop and Stare acquisition requires programmable light source'
        assert 'position' in self.hardware_controller_list,  'Stop and Stare acquisition requires programmable positioning device'

        # Generate motion pathway
        self.hardware_controller_list['position'].state_sequence = self.genStopAndStarePathwayRaster(
            self.object_size_mm, self.frame_spacing_mm)

        # Generate illumination sequence
        illuminaiton_pattern_sequence = [self.illumination_type] * \
            len(self.hardware_controller_list['position'].state_sequence)
        self.hardware_controller_list['illumination'].state_sequence = self.genMultiContrastSequence(
            illuminaiton_pattern_sequence)

        # Tell device not to use feedback
        self.hardware_controller_list['illumination'].trigger_wait_flag = False
        self.hardware_controller_list['illumination'].command('trs.0.500.0')
        self.hardware_controller_list['illumination'].command('trs.1.500.0')

        self.hardware_controller_list['position'].goToPosition((0,0))
        self.hardware_controller_list['position'].command('ENCODER X 1')
        self.hardware_controller_list['position'].command('ENCODER Y 1')
        self.hardware_controller_list['position'].command('ENCW X 100')
        self.hardware_controller_list['position'].command('ENCW Y 100')

    def acquire(self, exposure_time_ms=50):
        # Allocate memory for frames
        if self.hardware_controller_list['camera'].isSequenceRunning():
            self.hardware_controller_list['camera'].sequenceStop()
        self.hardware_controller_list['camera'].setBufferSizeMb(
            20 * len(self.hardware_controller_list['position'].state_sequence))

        # Set camera exposure
        self.hardware_controller_list['camera'].setExposure(exposure_time_ms / 1e3)
        self.hardware_controller_list['camera'].setTriggerMode('hardware')

        self.hardware_controller_list['camera'].runSequence()

        self.hardware_controller_list['illumination'].bf()

        # Snap one image to ensure all acquisitons are started
        self.hardware_controller_list['camera'].snap()

        # generate frame_list
        t0 = time.time()
        frames_acquired = 0
        frame_list = []
        for frame in yp.display.progressBar(self.hardware_controller_list['position'].state_sequence, name='Frames Acquired'):
            pos = frame['states']
            x = pos[0][0]['value']['x']
            y = pos[0][0]['value']['y']
            self.hardware_controller_list['position'].goToPosition((x, y), blocking=True)

            time.sleep(self.settle_time_s)
            frame_list.append(self.hardware_controller_list['camera'].snap())

            frames_acquired += 1
            # print('Acquired %d of %d frames' % (frames_acquired, len(self.hardware_controller_list['position'].state_sequence)))

        t_acq_sns = time.time() - t0
        print("Acquisition took %.4f seconds" % (t_acq_sns))

        # Create dataset
        from htdeblur.mddataset import MotionDeblurDataset
        dataset = MotionDeblurDataset()

        # Assign acquisition time
        self.metadata.acquisition_time_s = t_acq_sns

        # Apply simple geometric transformations
        if self.metadata.camera.transpose:
            frame_list = frame_list.transpose(0, 2, 1)
        if self.metadata.camera.flip_x:
            frame_list = np.flip(frame_list, 2)
        if self.metadata.camera.flip_y:
            frame_list = np.flip(frame_list, 1)

        # Assign
        dataset.frame_list = [frame for frame in frame_list]

        # Set frame state list
        self.n_frames = len(self.hardware_controller_list['position'].state_sequence)
        frame_state_list = []
        for frame_index in range(self.n_frames):
            single_frame_state_list = {}

            # Loop over hardware controllers and record their state sequences
            for hardware_controller_name in self.hardware_controller_list:
                hardware_controller = self.hardware_controller_list[hardware_controller_name]
                if hardware_controller.state_sequence is not None:
                    single_frame_state_list[hardware_controller_name] = hardware_controller.state_sequence[frame_index]

            # Record time_sequence_s
            single_frame_state_list['time_sequence_s'] = [0]

            # Add to list of all frames
            frame_state_list.append(single_frame_state_list)

        dataset.metadata = self.metadata
        dataset.type = 'stop_and_stare'
        dataset.frame_state_list = frame_state_list

        return dataset

    def genStopAndStarePathwayRaster(self, object_size_mm, frame_spacing_mm, major_axis=1, include_minor_axis=False):

        # Determine major axis
        if major_axis is None:
            major_axis = np.argmax(np.asarray(object_size_mm))
            if object_size_mm[0] == object_size_mm[1]:
                major_axis = 1

        # Detemine number of measurements
        measurement_count = np.ceil(np.asarray(object_size_mm) / np.asarray(frame_spacing_mm)
                                    ).astype(np.int)  # two components in x and y

        # Determine slightly smaller frame spacing for optimal coverage of object
        frame_spacing_mm = (object_size_mm[0] / measurement_count[0], object_size_mm[1] / measurement_count[1])

        # Error checking
        assert np.any(measurement_count > 1), "image_size must be smaller than object_size!"
        print("Image size requires %d x %d images" % (measurement_count[0], measurement_count[1]))

        # This variable will be populated by the loop below
        raster_segments = np.zeros((measurement_count[0] * 2, 2))

        # Generate raster points
        raster_end_point_list = []
        pathway = []
        linear_segment_index = 0  # This variable keeps track of linear segments, for use with path planning
        for row in np.arange(measurement_count[0]):
            if row % 2 == 0:
                for index, col in enumerate(range(measurement_count[1])):
                    # Add pathway to list
                    pathway.append({'x_start': frame_spacing_mm[1] * col,
                                    'y_start': frame_spacing_mm[0] * row,
                                    'x_end': frame_spacing_mm[1] * col,
                                    'y_end': frame_spacing_mm[0] * row,
                                    'linear_segment_index': linear_segment_index})
            else:
                for index, col in enumerate(reversed(range(measurement_count[1]))):
                    # Add pathway to list
                    frame_spacing_mm[0] * row
                    pathway.append({'x_start': frame_spacing_mm[1] * col,
                                    'y_start': frame_spacing_mm[0] * row,
                                    'x_end': frame_spacing_mm[1] * col,
                                    'y_end': frame_spacing_mm[0] * row,
                                    'linear_segment_index': linear_segment_index})
            linear_segment_index += 1

        # make the center the mean of the pathway
        path_means = []
        for path in pathway:
            path_mean = ((path['y_start']), (path['x_start']))
            path_means.append(path_mean)
        # mean = np.sum(np.asarray(path_means), axis=1) / len(path_means)
        mean = np.sum(np.asarray(path_means), axis=0) / len(path_means)
        for path in pathway:
            path['x_start'] -= mean[1]
            path['x_end'] -= mean[1]
            path['y_start'] -= mean[0]
            path['y_end'] -= mean[0]

        # return pathway
        state_sequence = []
        for path in pathway:

            # Store common information about this frame
            common_state_dict = {}
            common_state_dict['frame_time'] = self.hardware_controller_list['camera'].getExposure()
            common_state_dict['led_update_rate_us'] = None
            common_state_dict['linear_segment_index'] = None
            common_state_dict['frame_distance'] = 0
            common_state_dict['exposure_distance'] = 0
            common_state_dict['velocity'] = self.velocity_mm_s
            common_state_dict['acceleration'] = self.motion_acceleration_mm_s_2
            common_state_dict['n_blur_positions_exposure'] = 1
            common_state_dict['position_delta_x_mm'] = 0
            common_state_dict['position_delta_y_mm'] = 0

            path_dict = {'value': {'time_index' : 0,
                                   'x': path['x_start'],
                                   'y': path['y_start']}}
            state_sequence.append({'states' : [[path_dict]], 'common' :  common_state_dict})
        return(state_sequence)

    def plotPathway(self):
        sequence_list = self.hardware_controller_list['position'].state_sequence
        point_list_start = []
        point_list_end = []
        for sequence in sequence_list:
            start_pos = (sequence['states'][0][0]['value']['x'], sequence['states'][0][0]['value']['y'])
            end_pos = (sequence['states'][-1][0]['value']['x'], sequence['states'][-1][0]['value']['y'])
            point_list_start.append(start_pos)
            point_list_end.append(end_pos)

        point_list_start = np.asarray(point_list_start)
        point_list_end = np.asarray(point_list_end)
        plt.figure()
        for index in range(len(point_list_start)):
            plt.scatter(point_list_start[index, 0], point_list_start[index, 1], c='b')
            plt.scatter(point_list_end[index, 0], point_list_end[index, 1], c='r')

            plt.plot([point_list_start[index, 0], point_list_end[index, 0]],
                     [point_list_start[index, 1], point_list_end[index, 1]], c='y')

        plt.xlabel('Position X (mm)')
        plt.ylabel('Position Y (mm)')
        plt.title('Pathway (b is start, y/o is end)')
        plt.gca().invert_yaxis()


    def genMultiContrastSequence(self, illumination_pattern_sequence, n_acquisitions=1,
                                 darkfield_annulus_width_na=0.1):
        led_list = np.arange(self.metadata.illumination.state_list.design.shape[0])
        bf_mask = self.metadata.illumination.state_list.design[:, 0] ** 2 \
            + self.metadata.illumination.state_list.design[:, 1] ** 2 < (
                self.metadata.objective.na + self.illumination_na_pad) ** 2

        led_list_bf = led_list[bf_mask]
        led_list_df = led_list[~bf_mask]
        led_list_an = led_list[~bf_mask & (self.metadata.illumination.state_list.design[:, 0] ** 2
                                           + self.metadata.illumination.state_list.design[:, 1] ** 2 < (self.metadata.objective.na + darkfield_annulus_width_na) ** 2)]

        illumination_sequence = []
        self.pattern_type_list = []

        pattern_dict = {'dpc.top': np.ndarray.tolist(led_list_bf[self.metadata.illumination.state_list.design[bf_mask, 1] > 0]),
                        'dpc.bottom': np.ndarray.tolist(led_list_bf[self.metadata.illumination.state_list.design[bf_mask, 1] < 0]),
                        'dpc.left': np.ndarray.tolist(led_list_bf[self.metadata.illumination.state_list.design[bf_mask, 0] > 0]),
                        'dpc.right': np.ndarray.tolist(led_list_bf[self.metadata.illumination.state_list.design[bf_mask, 0] < 0]),
                        'single': [0],
                        'bf': np.ndarray.tolist(led_list_bf),
                        'df': np.ndarray.tolist(led_list_df),
                        'an': np.ndarray.tolist(led_list_an),
                        'full': np.ndarray.tolist(led_list)
                        }

        # DPC does not flicker patterns within frames
        n_time_points_per_frame = 1

        illumination_state_list = []

        # Write image sequence to list
        for acquisition_index in range(n_acquisitions):

            # Loop over DPC patterns (frames)
            for frame_index, pattern in enumerate(illumination_pattern_sequence):

                single_frame_state_list_illumination = []

                # Loop over time points (irrelevent for dpc)
                for time_index in range(n_time_points_per_frame):

                    time_point_state_list = []
                    # Loop over DPC patterns (which are themselves frames)
                    for led_idx in pattern_dict[pattern]:

                        values_dict = {}
                        for color_name in self.illumination_color:
                            values_dict[color_name] = self.illumination_color[color_name]

                        led_dict = {
                            'index': int(led_idx),
                            'time_index': 0,
                            'value': values_dict
                        }
                        # Append this to list with elements for each interframe time point
                        time_point_state_list.append(led_dict)

                    # Append to frame_dict
                    single_frame_state_list_illumination.append(time_point_state_list)

                # Define illumination sequence
                illumination_state_list.append({'states' : single_frame_state_list_illumination, 'common' : {}})

        # Define illumination list
        self.state_list = self.metadata.illumination.state_list.design

        return illumination_state_list

class MotionDeblurAcquisition():
    # Initialization
    def __init__(self, hardware_controller_list, system_metadata,
                 illumination_sequence=None,
                 motion_path_type='linear',
                 use_l1_distance_for_motion_calculations=True,
                 blur_vector_method='pseudo_random',
                 kernel_pulse_count=150,
                 saturation_factor=1.0,
                 frame_spacing_mm=1,
                 object_size_mm=(0.5, 0.5),
                 reuse_illumination_sequence=True,
                 max_exposure_time_s=2,
                 max_velocity_mm_s=40.0,
                 max_led_update_rate_us=0.01,
                 exposure_time_pad_s=0.0,
                 velocity_mm_s=None,
                 exposure_time_s=None,
                 debug=False,
                 motion_acceleration_mm_s_2=1e3,
                 extra_run_up_time_s=0,
                 flip_pathway=False,
                 segment_delay_s=0,
                 initial_auto_exposure=False,
                 acquisition_timeout_s=3,
                 illumination_sequence_count=1,
                 illumination_na_pad=0.03,
                 illumination_color={'w': 127},
                 only_store_first_and_last_position=True):

        # Parse options
        self.motion_path_type = motion_path_type
        self.object_size_mm = object_size_mm
        self.frame_spacing_mm = frame_spacing_mm
        self.flip_pathway = flip_pathway
        self.use_l1_distance_for_motion_calculations = use_l1_distance_for_motion_calculations
        self.velocity_mm_s = velocity_mm_s
        self.exposure_time_pad_s = exposure_time_pad_s
        self.debug = debug
        self.motion_acceleration_mm_s_2 = motion_acceleration_mm_s_2
        self.max_led_update_rate_us = max_led_update_rate_us
        self.max_exposure_time_s = max_exposure_time_s
        self.max_velocity_mm_s = max_velocity_mm_s
        self.illumination_na_pad = illumination_na_pad
        self.saturation_factor = saturation_factor
        self.reuse_illumination_sequence = reuse_illumination_sequence
        self.blur_vector_method = blur_vector_method
        self.kernel_pulse_count = kernel_pulse_count
        self.illumination_color = illumination_color
        self.extra_run_up_time_s = extra_run_up_time_s
        self.initial_auto_exposure = initial_auto_exposure
        self.acquisition_timeout_s = acquisition_timeout_s
        self.segment_delay_s = segment_delay_s
        self.only_store_first_and_last_position = only_store_first_and_last_position
        self.illumination_sequence = illumination_sequence
        self.illumination_sequence_count = illumination_sequence_count

        # Define controller objects, which act as hardware interfaces.
        # These should be in an ordered dictionary because the order which they
        # are initialized matters when using a mix of hardware and software triggering.
        self.hardware_controller_list = collections.OrderedDict()

        # First add hardware triggered elements so they perform their set-up before we trigger software elements
        for controller in hardware_controller_list:
            if controller.trigger_mode is 'hardware':
                self.hardware_controller_list[controller.type] = controller
                controller.reset()
                controller.seq_clear()

        # Then, add software triggered elements
        for controller in hardware_controller_list:
            if controller.trigger_mode is 'software':
                self.hardware_controller_list[controller.type] = controller
                controller.reset()
                controller.seq_clear()

        # Check to be sure a sequence acquisition is not running
        assert 'camera' in self.hardware_controller_list, 'Did not find camera controller!'

        # Store metadata object
        self.metadata = system_metadata

        # Ensure we have all necessary metadata for basic acquisition
        assert self.metadata.objective.na is not None, 'Missing objective.na in metadata.'
        assert self.metadata.objective.mag is not None, 'Missing objective.mag in metadata.'
        assert self.metadata.camera.pixel_size_um is not None, 'Missing pixel size in metadata.'

        # Update effective pixel size (for scale bar)
        self.metadata.system.eff_pixel_size_um = self.metadata.camera.pixel_size_um / (self.metadata.objective.mag * self.metadata.system.mag)

        # Trigger Constants
        self.TRIG_MODE_EVERY_PATTERN = 1
        self.TRIG_MODE_ITERATION = -1
        self.TRIG_MODE_START = -2

        # Frame state time sequence, will default to a sequence of one exposure time per frame if left as None
        self.time_sequence_s = None
        self.exposure_time_s = None
        self.hardware_sequence_timing = None

        # Turn off fast sequencing for illumination by default since this is only avaolable with certain LED arrays
        if 'illumination' in self.hardware_controller_list:
            self.hardware_controller_list['illumination'].use_fast_sequence = False

        # Set metadata type
        self.metadata.type = 'motiondeblur'

        assert 'illumination' in self.hardware_controller_list,  'Motion deblur object requires programmable light source'
        assert 'position' in self.hardware_controller_list,  'Motion deblur object requires motion stage'

        # Initialize state_sequence
        self.state_sequence = []

        # Generate position sequence
        self.hardware_controller_list['position'].state_sequence, self.time_sequence_s = self.genMotionPathway(
            pathway_type=self.motion_path_type, frame_spacing_mm=frame_spacing_mm)

        # Generate illumination sequence
        self.hardware_controller_list['illumination'].state_sequence = self.genMotionIlluminationSequenceRandom(illumination_sequence=illumination_sequence,
                                                 sequence_count=self.illumination_sequence_count)

        # Set up subframe captures
        self.subframe_capture_count = len(self.hardware_controller_list['illumination'].state_sequence[0])
        self.force_preload_all_frames = True
        self.hardware_controller_list['position'].continuous_states_between_frames = True

        # Configure illuination to use fast sequence updating if specified in options
        self.hardware_controller_list['illumination'].use_fast_sequence = True

        # Set bit depth
        self.illumination_sequence_bit_depth = 1

        # Set extra options for position controller
        self.hardware_controller_list['position'].extra_run_up_time_s = self.extra_run_up_time_s

        # Calculate effective pixel size if it hasn't already been calculated
        self.metadata.system.eff_pixel_size_um = self.metadata.camera.pixel_size_um / \
            (self.metadata.objective.mag * self.metadata.system.mag)

    def preAcquire(self):
        ''' This method sets up the camera for an acquisition '''
        # Check that the length of motion, illuimination, pupil, and focal sequences are same (or None)
        frame_counts = []
        for hardware_controller_name in list(self.hardware_controller_list):
            # Get controller object from dictionary
            hardware_controller = self.hardware_controller_list[hardware_controller_name]
            if hardware_controller.state_sequence is not None:

                # Reset Controller
                hardware_controller.reset()

                # Get number of frames in sequence. If there is no sequence, remove this element from hw_controller_list
                if hardware_controller.type is not 'camera':
                    if hardware_controller.state_sequence is not None:
                        frame_counts.append(len(hardware_controller.state_sequence))
                    else:
                        self.hardware_controller_list.pop(hardware_controller_name)
            else:
                # Remove this controller from the list
                if hardware_controller_name is not 'camera':
                    del self.hardware_controller_list[hardware_controller_name]

        # Turn on hardware triggering for initialization
        self.hardware_controller_list['camera'].setTriggerMode('hardware')

        # Set illumination parameters
        if 'illumination' in self.hardware_controller_list:
            # self.hardware_controller_list['illumination'].setColor(self.illumination_color)
            self.hardware_controller_list['illumination'].setSequenceBitDepth(
                self.illumination_sequence_bit_depth)

        # Ensure all hardware elements have the same number of frames
        if len(frame_counts) > 0:
            if not np.sum(np.mean(np.asarray(frame_counts)) == np.asarray(frame_counts)) == len(frame_counts):
                raise ValueError('Sequence lengths are not the same (or None).')
            else:
                self.n_frames = frame_counts[0]
        else:
            raise ValueError('No sequence provided!')

        # Initialize frame_list
        self.frame_list = np.zeros((self.n_frames,
                                    self.hardware_controller_list['camera'].getImageHeight(), self.hardware_controller_list['camera'].getImageWidth()), dtype=np.uint16)

        # Apply simple geometric transformations
        if self.metadata.camera.transpose:
            self.frame_list = self.frame_list.transpose(0, 2, 1)
        if self.metadata.camera.flip_x:
            self.frame_list = np.flip(self.frame_list, 2)
        if self.metadata.camera.flip_y:
            self.frame_list = np.flip(self.frame_list, 1)

        # Generate frame_state_list
        frame_state_list = []

        if self.time_sequence_s is None:
            self.time_sequence_s = []
            for _ in range(self.n_frames):
                self.time_sequence_s.append([0])

        # Loop over frames
        for frame_index in range(self.n_frames):
            single_frame_state_list = {}

            # Loop over hardware controllers and record their state sequences
            for hardware_controller_name in self.hardware_controller_list:
                hardware_controller = self.hardware_controller_list[hardware_controller_name]
                if hardware_controller.state_sequence is not None:
                    single_frame_state_list[hardware_controller_name] = hardware_controller.state_sequence[frame_index]

            # Record time_sequence_s
            single_frame_state_list['time_sequence_s'] = self.time_sequence_s[frame_index]

            # Add to list of all frames
            frame_state_list.append(single_frame_state_list)

        self.frame_state_list = frame_state_list

        # Perform auto-exposure if user desires
        if self.initial_auto_exposure:

            # Illuminate with first pattern
            if 'illumination' in self.hardware_controller_list:
                self.hardware_controller_list['illumination'].sequenceReset()
                self.hardware_controller_list['illumination'].time_sequence_s = [[0]]
                self.hardware_controller_list['illumination'].preloadSequence(0)
                self.hardware_controller_list['illumination'].sequenceStep()

            # Small delay to ensure illumination gets updated
            time.sleep(0.1)

            # Run Auto-Exposure
            self.hardware_controller_list['camera'].autoExposure()

        # Set camera memory footprint
        if (self.hardware_controller_list['camera'].getBufferTotalCapacity() < self.frame_list.shape[0]):
            self.frame_size_mb = int(
                np.ceil(float(self.frame_list.shape[0] / 1e6) * float(self.frame_list.shape[1]) * float(self.frame_list.shape[2]) * 2))
            print('Allocating %dmb for frames' % self.frame_size_mb)
            self.hardware_controller_list['camera'].setBufferSizeMb(self.frame_size_mb)

            assert self.hardware_controller_list['camera'].getBufferTotalCapacity(
            ) >= self.frame_list.shape[0], 'Buffer size too small!'

        # Store initial time (acquisition start)
        t0 = time.time()

        # Tell camera to start waiting for frames
        self.hardware_controller_list['camera'].runSequence()

        # Keep track of how many images we have acquired
        self.total_frame_count = 0

    def acquire(self,
                dataset=None,
                reset_devices=False):
        '''
        This is a generic acquisition class, where LEDs are updated according to the sequence variable.
        '''

        # Call preacquire. which initializes hardware and variables
        self.preAcquire()

        # Determine which frames can be preloaded before serial acquisition. If each frame is only one state, we assume that we can preload all frames. But, if the state of any hardware element changes within any frame, we will assume we can't preload the frames

        frame_count = 0
        linear_segment_list = []
        for frame_state in self.hardware_controller_list['position'].state_sequence:
            if frame_state['common']['linear_segment_index'] >= 0:
                frame_count += 1
                if frame_state['common']['linear_segment_index'] not in linear_segment_list:
                    linear_segment_list.append(frame_state['common']['linear_segment_index'])
        print("Found %d segments and %d frames" % (len(linear_segment_list), frame_count))

        t_start = time.time()
        for linear_segment_index in linear_segment_list:
            self.frames_to_acquire = []
            # Determine which linear segments to run
            for frame_index, frame_state in enumerate(self.hardware_controller_list['position'].state_sequence):
                if frame_state['common']['linear_segment_index'] == linear_segment_index:
                    self.frames_to_acquire += [frame_index]
            self.n_frames_to_acquire = len(self.frames_to_acquire)

            x_start = self.hardware_controller_list['position'].state_sequence[self.frames_to_acquire[0]]['states'][0][0]['value']['x']
            y_start = self.hardware_controller_list['position'].state_sequence[self.frames_to_acquire[0]]['states'][0][0]['value']['y']

            x_end = self.hardware_controller_list['position'].state_sequence[self.frames_to_acquire[-1]]['states'][0][0]['value']['x']
            y_end = self.hardware_controller_list['position'].state_sequence[self.frames_to_acquire[-1]]['states'][0][0]['value']['y']

            print('Starting linear segment %d which has %d frames moving from (%.4f, %.4f)mm to (%.4f, %.4f)mm' %
                  (linear_segment_index, self.n_frames_to_acquire, x_start, y_start, x_end, y_end))

            frame_has_multiple_states = []
            for frame_index in self.frames_to_acquire:
                number_of_states_in_current_frame = 0
                for hardware_controller_name in self.hardware_controller_list:
                    if hardware_controller_name is not 'camera' and self.hardware_controller_list[hardware_controller_name].state_sequence is not None:
                        # Check if this frame can be preloaded (if it has more than one state, it can't be preloaded)
                        number_of_states_in_current_frame = max(number_of_states_in_current_frame,  len(
                            self.hardware_controller_list[hardware_controller_name].state_sequence[frame_index]['states']))

                # Check that the length of time_sequence_s matches the max number of state changes within this frame
                if number_of_states_in_current_frame > 1:
                    frame_has_multiple_states.append(True)

                    assert self.time_sequence_s is not None, "time_sequence_s can not be None if any frame has multiple states!"
                    assert len(self.time_sequence_s[frame_index]) == number_of_states_in_current_frame, "time_sequence_s for frame %d is of wrong length!" % len(
                        self.time_sequence_s[frame_index]['states'])
                else:
                    frame_has_multiple_states.append(False)

            # Determine if the entire multi-frame sequence can be preloaded (this will be False if ther eis only one system state (e.g. LED pattern) within each frame)
            all_frames_will_be_preloaded = (not any(frame_has_multiple_states)) or self.force_preload_all_frames

            # Determine optimal exposure time for all frames
            if self.exposure_time_s is not None:
                self.hardware_controller_list['camera'].setExposure(self.exposure_time_s)

            elif self.time_sequence_s is not None and max(self.time_sequence_s[0]) > 0:
                frame_exposures = []
                for frame_index in range(self.n_frames_to_acquire):
                    frame_exposures.append(max(self.time_sequence_s[frame_index]))

                self.exposure_time_s = sum(frame_exposures) / (self.n_frames_to_acquire)
                self.hardware_controller_list['camera'].setExposure(self.exposure_time_s)
            else:
                self.exposure_time_s = self.hardware_controller_list['camera'].getExposure()

            # Check that exposure time is correct
            assert abs(self.exposure_time_s - self.hardware_controller_list['camera'].getExposure(
            )) < 1e-3, "Desired exposure time %.2f is not equal to device exposure %.2f. This is probably a MM issue" % (self.exposure_time_s, self.hardware_controller_list['camera'].getExposure())
            # print('Using exposure time %.2fs (%d ms)' % (self.exposure_time_s, int(self.exposure_time_s * 1000)))

            # Check that time_sequence_s for multiple frames exists if there are inter-frame state changes
            if (not any(frame_has_multiple_states)) or self.time_sequence_s is None:
                self.time_sequence_s = [self.exposure_time_s]

            # Configure hardware triggering
            trigger_output_settings = [0, 0]
            trigger_input_settings = [0, 0]
            for hardware_controller_name in self.hardware_controller_list:
                hardware_controller = self.hardware_controller_list[hardware_controller_name]

                if 'hardware' in hardware_controller.trigger_mode:
                    # Check that trigger pins are configured
                    assert hardware_controller.trigger_pin is not None, 'Trigger pin must be configured for hardware triggering!'

                    # Determine if we're performing preloadable acquisitions or not
                    if self.subframe_capture_count > 1:
                        if self.reuse_illumination_sequence:
                            if hardware_controller_name == 'camera':
                                if self.illumination_sequence_count == 1:
                                    trigger_output_settings[hardware_controller.trigger_pin] = self.TRIG_MODE_ITERATION
                                    trigger_input_settings[hardware_controller.trigger_pin] = self.TRIG_MODE_ITERATION
                                else:
                                    trigger_output_settings[hardware_controller.trigger_pin] = len(self.hardware_controller_list['position'].state_sequence[0]['states']) // self.illumination_sequence_count
                                    trigger_input_settings[hardware_controller.trigger_pin] = len(self.hardware_controller_list['position'].state_sequence[0]['states']) // self.illumination_sequence_count

                            elif hardware_controller_name == 'position':
                                trigger_output_settings[hardware_controller.trigger_pin] = self.TRIG_MODE_START
                                trigger_input_settings[hardware_controller.trigger_pin] = self.TRIG_MODE_START
                        else:
                            if hardware_controller_name == 'camera':
                                trigger_output_settings[hardware_controller.trigger_pin] = self.subframe_capture_count
                                trigger_input_settings[hardware_controller.trigger_pin] = self.subframe_capture_count

                            elif hardware_controller_name == 'position':
                                trigger_output_settings[hardware_controller.trigger_pin] = self.TRIG_MODE_START
                                trigger_input_settings[hardware_controller.trigger_pin] = self.TRIG_MODE_START

                    # Case where there is only one system state wihtin each frame (trigger each frame)
                    elif all_frames_will_be_preloaded:
                        trigger_output_settings[hardware_controller.trigger_pin] = self.TRIG_MODE_EVERY_PATTERN
                        trigger_input_settings[hardware_controller.trigger_pin] = self.TRIG_MODE_EVERY_PATTERN

                    # Case where we only want to trigger on first frame. This is probably not a good default.
                    else:
                        trigger_output_settings[hardware_controller.trigger_pin] = self.TRIG_MODE_ITERATION
                        trigger_input_settings[hardware_controller.trigger_pin] = self.TRIG_MODE_ITERATION

                # Check that this hardware controller is ready for a sequence, if it is sequencable.
                if hardware_controller.state_sequence is not None:
                    # Reset controller sequence to initial state
                    hardware_controller.sequenceReset()
                    time.sleep(0.1)

                    # Wait until initialization is complete
                    initialization_wait_time = 0
                    for hardware_controller_name in self.hardware_controller_list:
                        while not self.hardware_controller_list[hardware_controller_name].isReadyForSequence():
                            time.sleep(0.05)
                            initialization_wait_time += 0.05
                            if initialization_wait_time > self.acquisition_timeout_s:
                                raise ValueError('Pre-acquisiton isReadyForSequence timeout for %s' % hardware_controller_name)

                    # Tell the hardware controller about the acquisition time sequence
                    if len(hardware_controller.state_sequence) == len(self.time_sequence_s):
                        hardware_controller.time_sequence_s = [self.time_sequence_s[i] for i in self.frames_to_acquire]
                    else:
                        hardware_controller.time_sequence_s = [
                            [self.hardware_controller_list['camera'].getExposure()]] * self.n_frames_to_acquire

            # Set up triggering for hardware acquision
            self.hardware_controller_list['illumination'].trigger_output_settings = trigger_output_settings
            self.hardware_controller_list['illumination'].trigger_input_settings = trigger_input_settings

            # Determine which sequences get preloaded
            if all_frames_will_be_preloaded:  # One system state per acquisition
                frame_preload_sequence = [-1]  # Preload all frames at once
            else:
                frame_preload_sequence = range(self.n_frames_to_acquire)  # Preload each frame serially

            # Loop over frames to capture (may only execute once if we're preloading all frames)
            for preload_index in frame_preload_sequence:

                # Loop over hardware controllers, preload, and determine necessary exposure time (if using inter-frame state changes)
                for hardware_controller_name in self.hardware_controller_list:

                    # If we're using the motion stage, calculate the mechanical delay
                    if hardware_controller_name == 'position':
                        # Get velocity and acceleration from state sequence
                        if preload_index == -1:
                            index = 0
                        else:
                            index = preload_index

                        velocity = self.hardware_controller_list[hardware_controller_name].state_sequence[0]['common']['velocity']
                        acceleration = self.hardware_controller_list[hardware_controller_name].getAcceleration()
                        jerk = self.hardware_controller_list[hardware_controller_name].getJerk()

                        # Calculate spin-up time and distance
                        # http://www.wolframalpha.com/input/?i=v+%3D+t+*+(a+%2B+0.5*j+*+t)+solve+for+t
                        # http://www.wolframalpha.com/input/?i=v+%3D+t+*+(a+%2B+(1%2F8)*j+*+t)+solve+for+t

                        # Good reference:
                        # http://www.et.byu.edu/~ered/ME537/Notes/Ch5.pdf

                        # Total period
                        if False:
                            # First period (acceleration of acceleration)
                            t_1 = acceleration / jerk
                            # x_1 = 1/6 * jerk * t_1 ** 3
                            x_1 = acceleration ** 2 / (6 * jerk) * t_1
                            # v_1 = 1/2 * jerk * t_1 ** 2
                            v_1 = acceleration ** 2 / (2 * jerk)

                            # Second period (linear region)
                            dv = velocity - 2 * v_1
                            assert dv > 0
                            t_2 = dv / acceleration
                            x_2 = v_1 * t_2 + 1/2 * acceleration * t_2 ** 2
                            v_2 = velocity - v_1

                            # Third period (decelleration of acceleration)
                            t_3 = acceleration / jerk
                            x_3 = (v_2 + acceleration ** 2 / (3 * jerk)) * t_3
                            v_3 = v_1

                            # Calculate spin-up distance and time
                            spin_up_time_s = t_1 + t_2 + t_3
                            spin_up_distance_mm = x_1 + x_2 + x_3
                            assert (v_1 + v_2 + v_3 - velocity) < 1e-1, "Calculated velocity is %.4f, desired is %.4f" % (v_1 + v_2 + v_3, velocity)

                        else:
                            spin_up_time_s = velocity / acceleration
                            spin_up_distance_mm = 1/2 * acceleration * spin_up_time_s ** 2

                        # Add extra spin_up time
                        spin_up_time_s += self.extra_run_up_time_s
                        spin_up_distance_mm += self.extra_run_up_time_s * velocity

                        # spin_up_distance_mm = 0
                        spin_up_time_s = max(spin_up_time_s, 0.0001)

                        self.hardware_controller_list['illumination'].setupTriggering(self.hardware_controller_list['illumination'].motion_stage_trigger_index, int(
                            self.hardware_controller_list[hardware_controller_name].trigger_pulse_width_us), int(spin_up_time_s * 1e6))  # convert to seconds

                        # Tell motion stage to offset it's positions by these amounts
                        self.hardware_controller_list['position'].preload_run_up_distance_mm = spin_up_distance_mm

                    else:
                        # no delay for other components
                        self.hardware_controller_list[hardware_controller_name].trigger_start_delay_s = 0

                    if hardware_controller_name is not 'camera' and self.hardware_controller_list[hardware_controller_name].state_sequence is not None:
                        if hardware_controller_name is not 'illumination' or linear_segment_index == 0:
                            if hardware_controller_name == 'illumination' and self.reuse_illumination_sequence:
                                self.hardware_controller_list[hardware_controller_name].preloadSequence(0)
                            else:
                                state_sequence_used = [
                                    self.hardware_controller_list[hardware_controller_name].state_sequence[i] for i in self.frames_to_acquire]
                                self.hardware_controller_list[hardware_controller_name].preloadSequence(
                                    preload_index, state_sequence=state_sequence_used)

                if preload_index < 0 or self.reuse_illumination_sequence:
                    frames_to_wait_for = self.n_frames_to_acquire  # wait for all frames
                else:
                    frames_to_wait_for = 1

                # Set trigger frame time based on first pathway TODO: This is a hack
                if 'position' in self.hardware_controller_list:
                    self.hardware_controller_list['illumination'].trigger_frame_time_s[self.hardware_controller_list['illumination']
                                                                                       .camera_trigger_index] = self.hardware_controller_list['position'].state_sequence[0]['common']['frame_time'] * 1e6

                    # Tell stage to start moving
                    self.hardware_controller_list['position'].runSequence()

                if linear_segment_index == 0:
                    t_start = time.time()

                # Tell illumination to start moving
                if self.reuse_illumination_sequence:
                    self.hardware_controller_list['illumination'].runSequence(
                        n_acquisitions=1 * self.n_frames_to_acquire)
                else:
                    self.hardware_controller_list['illumination'].runSequence(n_acquisitions=1)

                # Wait for frames to be captured
                t_frame = time.time()
                frame_count = 0

                while frame_count < frames_to_wait_for:
                    if self.total_frame_count + frame_count == frames_to_wait_for:
                        break
                    else:
                        if self.total_frame_count + frame_count == self.hardware_controller_list['camera'].getBufferSizeFrames():
                            time.sleep(0.01)
                            if (time.time() - t_frame) > self.acquisition_timeout_s:
                                print(self.hardware_controller_list['illumination'].response())
                                raise ValueError('Acquisition timeout (Total frame count: %d, Buffer size: %d, preload index %d, frames to wait for: %d)' % (
                                    self.total_frame_count, self.hardware_controller_list['camera'].getBufferSizeFrames(), preload_index, frames_to_wait_for))

                        else:
                            if ((self.total_frame_count + frame_count) % int((self.n_frames) / min(10, self.n_frames_to_acquire))) == 0:
                                print('Acquired %d of %d frames' % (
                                    self.hardware_controller_list['camera'].getBufferSizeFrames(), self.n_frames_to_acquire))
                            frame_count = self.hardware_controller_list['camera'].getBufferSizeFrames(
                            ) - self.total_frame_count
                            self.total_frame_count = self.hardware_controller_list['camera'].getBufferSizeFrames()
                            t_frame = time.time()

                # Get sequence timing information
                time.sleep(0.1)
                print(self.hardware_controller_list['illumination'].response())

                # Wait for hardware to stop
                for hardware_controller_name in self.hardware_controller_list:
                    while not self.hardware_controller_list[hardware_controller_name].isReadyForSequence():
                        time.sleep(0.05)

                self.sequence_timing_dict = {}

            # Reset sequences
            for hardware_controller_name in self.hardware_controller_list:
                if hardware_controller_name is not 'camera':
                    self.hardware_controller_list[hardware_controller_name].sequenceReset()

            # Let user know we're finished
            print('Finished linear segment %d' % linear_segment_index)

            time.sleep(self.segment_delay_s)

        t_acq = time.time() - t_start
        self.metadata.acquisition_time_s = t_acq
        print("Acquisition took %.4f seconds" % (t_acq))

        # Call post-acquire functions
        dataset = self.postAcquire(dataset=dataset, reset_devices=reset_devices)

        # Return
        return dataset

    def postAcquire(self, dataset=None, reset_devices=True):
        """Post-acquisition steps for resetting hardware and preparing dataset."""
        # Stop acquisition
        # self.hardware_controller_list['camera'].sequenceStop()

        # Parse dataset
        if dataset is None:
            from htdeblur.mddataset import MotionDeblurDataset
            dataset = MotionDeblurDataset()

        # Read frames and timestamps from buffer
        (self.frame_list, elapsed_frame_time_ms) = self.hardware_controller_list['camera'].readFramesFromBuffer()
        # Apply simple geometric transformations
        if self.metadata.camera.transpose:
            self.frame_list = self.frame_list.transpose(0, 2, 1)
        if self.metadata.camera.flip_x:
            self.frame_list = np.flip(self.frame_list, 2)
        if self.metadata.camera.flip_y:
            self.frame_list = np.flip(self.frame_list, 1)

        # Let user know we're finished
        print('Read frames from buffer.')

        # Store camera timing in a standardized timing dict
        self.sequence_timing_dict = {}
        self.sequence_timing_dict['sequence_timing'] = []

        for frame_index, frame_time in enumerate(elapsed_frame_time_ms):
            timing_dict = {'trigger_number' : 0, 'acquisition_number' : frame_index, 'camera_start_time_us' : frame_time * 1000}
            self.sequence_timing_dict['sequence_timing'].append(timing_dict)

        # Reset all hardware elements
        if reset_devices:
            for hardware_controller_name in self.hardware_controller_list:
                self.hardware_controller_list[hardware_controller_name].reset()

        if self.only_store_first_and_last_position:
            for frame_state in self.frame_state_list[1:]:
                frame_state['position']['states'] = [frame_state['position']['states'][0], frame_state['position']['states'][-1]]

        # Remove repeated illumination patterns and time_sequence_s if we used the same illumination for each pulse
        if self.reuse_illumination_sequence:
            for frame_state in self.frame_state_list[1:]:
                frame_state['time_sequence_s'] = 'see_frame_#1'
                frame_state['illumination'] = 'see_frame_#1'

        # Illuminate with brightfield to indicate we're Finished
        self.hardware_controller_list['illumination'].bf()
        self.hardware_controller_list['position'].goToPosition((0,0))

        # Save results to an itoools.Dataset object
        dataset.frame_list = self.frame_list
        dataset.frame_state_list = self.frame_state_list
        dataset.metadata = self.metadata
        dataset.type = 'motion_deblur'

        # Return
        return dataset

    def genMotionPathway(self, n_acquisitions=1, pathway_type='raster', frame_spacing_mm=1.):
        '''
        This function generates a few example motion pathways.
        '''
        if pathway_type is 'raster':
            pathway = self.genMotionPathwayRaster(self.object_size_mm, self.frame_spacing_mm)

        elif (pathway_type is 'linear') or (pathway_type is 'linear_x'):
            # predefine linear y sequence
            n_frames = int(math.ceil(self.object_size_mm[1] / self.frame_spacing_mm[1]))
            pathway = []
            for frame_index in range(n_frames):
                pathway.append({'x_start': frame_index * self.frame_spacing_mm[1],
                                'x_end':  (frame_index + 1) * self.frame_spacing_mm[1],
                                'y_start': 0, 'y_end': 0, 'linear_segment_index': 0})

        elif pathway_type in ['linear_y']:
            # predefine linear y sequence
            n_frames = int(np.ceil(self.object_size_mm[0] / self.frame_spacing_mm[0]))
            pathway = []
            for frame_index in range(n_frames):
                pathway.append({'y_start':  -frame_index * self.frame_spacing_mm[0],
                                'y_end':  -(frame_index + 1) * self.frame_spacing_mm[0],
                                'x_start': 0, 'x_end': 0, 'linear_segment_index': 0})

        elif pathway_type is 'linear_diag':
            # predefine linear y sequence
            n_frames = int(np.ceil(self.object_size_mm[0] / self.frame_spacing_mm[0]))
            pathway = []
            for frame_index in range(n_frames):
                pathway.append({'y_start':  frame_index * self.frame_spacing_mm[0],
                                'y_end':  (frame_index + 1) * self.frame_spacing_mm[0],
                                'x_start':  frame_index * self.frame_spacing_mm[0],
                                'x_end':  (frame_index + 1) * self.frame_spacing_mm[0],
                                'linear_segment_index': 0})
        else:
            raise ValueError('Pathway type %s is not implemented.' % pathway_type)

        # make the center the mean of the pathway
        path_xmin = 1e8
        path_ymin = 1e8
        path_xmax = -1e8
        path_ymax = -1e8
        for path in pathway:
            path_mean = ((path['y_start']), (path['y_start']))
            path_xmin = min(path_xmin, min([path['x_start'], path['x_end']]))
            path_xmax = max(path_xmax, max([path['x_start'], path['x_end']]))
            path_ymin = min(path_ymin, min([path['y_start'], path['y_end']]))
            path_ymax = max(path_ymax, max([path['y_start'], path['y_end']]))

        mean = ((path_ymax + path_ymin) / 2, (path_xmax + path_xmin) / 2)
        for path in pathway:
            path['x_start'] = path['x_start'] - mean[1]
            path['x_end'] = path['x_end'] - mean[1]
            path['y_start'] = path['y_start'] - mean[0]
            path['y_end'] = path['y_end'] - mean[0]

        # Flip pathway if user desired
        if self.flip_pathway:
            for path in pathway:
                path['x_start'] *= -1
                path['x_end'] *= -1
                path['y_start'] *= -1
                path['y_end'] *= -1

        position_state_list = []
        time_sequence_s = []

        # Write image sequence to list
        for acquisition_index in range(n_acquisitions):

            # Loop over DPC patterns (frames)
            for frame_index, position in enumerate(pathway):

                # define distance in terms of l1 or l2 distance
                distance_l2 = float(np.sqrt((position['x_end'] - position['x_start'])
                                            ** 2 + (position['y_end'] - position['y_start']) ** 2))
                distance_l1 = float(abs(position['x_end'] - position['x_start']) +
                                    abs(position['y_end'] - position['y_start']))

                if self.use_l1_distance_for_motion_calculations:
                    position['frame_distance'] = int(round(distance_l1 * 1000)) / 1000  # round to nearest um
                else:
                    position['frame_distance'] = int(round(distance_l2 * 1000)) / 1000  # round to nearest um

                # Determine number of qunatifiable positions in pathway
                position['n_blur_positions_frame'] = int(
                    math.floor(position['frame_distance'] / (self.metadata.system.eff_pixel_size_um / 1000)))

                # Determine necessary velocity
                if self.velocity_mm_s is not None:
                    position['velocity_mm_s'] = self.velocity_mm_s
                else:
                    position['velocity_mm_s'] = self.max_velocity_mm_s   # Use fastest speed possible

                # Calculate time between frames
                position['frame_time_s'] = position['frame_distance'] / position['velocity_mm_s']  # t = x / v

                # Determine camera exposure time for this frame
                position['exposure_time_s'] = int(math.floor((self.hardware_controller_list['camera'].calcExposureTimeFromBusyTime(
                    position['frame_time_s']) - self.exposure_time_pad_s) * 1000)) / 1000  # round to nearest ms

                # Determine LED update rate
                dx_pixel = position['frame_distance'] / position['n_blur_positions_frame']
                dt_pixel_raw = dx_pixel / position['velocity_mm_s']
                position['led_update_rate_us'] = math.ceil(dt_pixel_raw * 1e6) # Round up to integer us

                # Determine new velocity (ps / update rate)
                new_velocity_mm_s = (self.metadata.system.eff_pixel_size_um / 1e3) / (position['led_update_rate_us'] / 1e6)
                if self.debug > 0:
                    print('Reducing velocity to %.4f mm/s from %.4f mm/s to match illumination update rate of %d us' % (new_velocity_mm_s, position['velocity_mm_s'], position['led_update_rate_us']))

                position['velocity_mm_s'] = new_velocity_mm_s

                # Update frame time based on velocity
                position['frame_time_s'] = position['frame_distance'] / position['velocity_mm_s']

                # Determine number of pixels in exposure time
                position['n_blur_positions_exposure'] = math.floor(position['exposure_time_s'] / (position['led_update_rate_us'] / 1e6))

                # Determine the distance traveled during the exposure time
                position['exposure_distance'] = position['n_blur_positions_exposure'] * position['led_update_rate_us'] / 1e6 * position['velocity_mm_s']

                # Store acceleration
                position['acceleration_mm_s_2'] = self.motion_acceleration_mm_s_2

                # Print information about this pattern
                if self.debug > 0:
                    print('Segment %d, index %d will require %d blur positions per frame (%d during exposure), %.2fms exposure time (%.2fms total frame time), scan %.2fmm (%.2fmm with exposure), move at %.2fmm/s, and update speed %dus' %
                          (position['linear_segment_index'], frame_index, position['n_blur_positions_frame'],position['n_blur_positions_exposure'], 1000. * position['exposure_time_s'], 1000. * position['frame_time_s'], position['frame_distance'], position['exposure_distance'], position['velocity_mm_s'], position['led_update_rate_us']))

                # Check that all blur parameters are valid
                assert position['led_update_rate_us'] >= self.max_led_update_rate_us, "LED Array update rate (%dms) < max update rate (%dms)" % (
                    position['led_update_rate_us'], self.max_led_update_rate_us)

                assert position['exposure_time_s'] <= self.max_exposure_time_s, "Exposure time (%.3fs) > max_exposure_time_s (%.3f)" % (
                    position['exposure_time_s'], self.max_exposure_time_s)

                assert position['velocity_mm_s'] <= self.max_velocity_mm_s, "Velocity (%.3fs) > max_velocity_mm_s (%.3f)" % (
                    position['velocity_mm_s'], self.max_velocity_mm_s)

                # List for this positions
                single_frame_state_list_position = []
                single_frame_time_sequence_s = []

                # Determine movement direction
                direction = np.asarray((position['y_end'] - position['y_start'],
                                        position['x_end'] - position['x_start']))
                direction /= np.linalg.norm(direction)

                # Store common information about this frame
                common_state_dict = {}
                common_state_dict['frame_time'] = position['frame_time_s']
                common_state_dict['led_update_rate_us'] = position['led_update_rate_us']
                common_state_dict['linear_segment_index'] = position['linear_segment_index']
                common_state_dict['frame_distance'] = position['frame_distance']
                common_state_dict['exposure_distance'] = position['exposure_distance']
                common_state_dict['velocity'] = position['velocity_mm_s']
                common_state_dict['acceleration'] = position['acceleration_mm_s_2']
                common_state_dict['n_blur_positions_exposure'] = position['n_blur_positions_exposure']
                common_state_dict['position_delta_x_mm'] = direction[1] * position['velocity_mm_s'] * position['led_update_rate_us'] / 1e6
                common_state_dict['position_delta_y_mm'] = direction[0] * position['velocity_mm_s'] * position['led_update_rate_us'] / 1e6

                # Loop over time points (irrelevent for dpc)
                for time_index in range(position['n_blur_positions_exposure']):

                    time_point_state_list = []
                    x = position['x_start'] + direction[1] * abs(common_state_dict['position_delta_x_mm']) * time_index
                    y = position['y_start'] + direction[0] * abs(common_state_dict['position_delta_x_mm']) * time_index

                    # Append this to list with elements for each interframe time point
                    time_point_state_list.append({'time_index': time_index,
                                                  'value': {'x': x, 'y': y}})

                    # Append to frame_dict
                    single_frame_state_list_position.append(time_point_state_list)

                    single_frame_time_sequence_s.append((time_index + 1) * position['led_update_rate_us'] / 1e6)

                # Define illumination sequence
                position_state_list.append({'states' : single_frame_state_list_position, 'common' : common_state_dict})

                # Define time_sequence
                time_sequence_s.append(single_frame_time_sequence_s)

        # for state in position_state_list:
        #     print(state['states'][0][0]['value']['x'] - state['states'][-1][0]['value']['x'])

        return (position_state_list, time_sequence_s)

    def genMotionPathwayRaster(self, object_size_mm, frame_spacing_mm, major_axis=None, include_minor_axis=False):

        # Hard-code major axis since the rest of the code doesn't respect it for now
        _major_axis = 1

        # Detemine number of measurements
        measurement_count = np.ceil(np.asarray(object_size_mm) / np.asarray(frame_spacing_mm)).astype(np.int)  # two components in x and y

        # Error checking
        assert np.any(measurement_count > 1), "image_size must be smaller than object_size!"
        print("Image size requires %d x %d images" % (measurement_count[0], measurement_count[1]))

        # If number of measurements along major axis is odd, center this row
        offset = [0, 0]
        offset[_major_axis] -= frame_spacing_mm[_major_axis] / 2

        # Generate raster points
        raster_end_point_list = []
        pathway = []
        linear_segment_index = 0  # This variable keeps track of linear segments, for use with path planning
        for row in np.arange(measurement_count[0]):

            if row % 2 == 0:
                for index, col in enumerate(range(measurement_count[1])):
                    # Add pathway to list
                    pathway.append({'x_start': frame_spacing_mm[1] * col + offset[1],
                                    'y_start': frame_spacing_mm[0] * row + offset[0],
                                    'x_end': frame_spacing_mm[1] * (col + 1) + offset[1],
                                    'y_end': frame_spacing_mm[0] * row + offset[0],
                                    'linear_segment_index': linear_segment_index})
                # Add minor stride
                if row < (measurement_count[0] - 1) and include_minor_axis:
                    pathway.append({'x_start': frame_spacing_mm[1] * (measurement_count[1] - 1) + offset[1],
                                    'y_start': frame_spacing_mm[0] * row + offset[0],
                                    'x_end': frame_spacing_mm[1] * (measurement_count[1] - 1) + offset[1],
                                    'y_end': frame_spacing_mm[0] * (row + 1) + offset[0],
                                    'linear_segment_index': -1 * (linear_segment_index + 1)})
            else:
                for index, col in enumerate(reversed(range(measurement_count[1]))):
                    # Add pathway to list
                    pathway.append({'x_start': frame_spacing_mm[1] * col - offset[1],
                                    'y_start': frame_spacing_mm[0] * row - offset[0],
                                    'x_end': frame_spacing_mm[1] * (col - 1) - offset[1],
                                    'y_end': frame_spacing_mm[0] * row - offset[0],
                                    'linear_segment_index': linear_segment_index})

                # Add minor stride
                if row < (measurement_count[0] - 1) and include_minor_axis:
                    pathway.append({'x_start': - offset[1],
                                    'y_start': frame_spacing_mm[0] * row - offset[0],
                                    'x_end': 0 - offset[1],
                                    'y_end': frame_spacing_mm[0] * (row + 1) - offset[0],
                                    'linear_segment_index': -1 * (linear_segment_index + 1)})
            linear_segment_index += 1

        print('Generated motion pathway with %d linear segments' % (linear_segment_index))
        return pathway

    def plotPathway(self):
        sequence_list = self.hardware_controller_list['position'].state_sequence
        point_list_start = []
        point_list_end = []
        for sequence in sequence_list:
            start_pos = (sequence['states'][0][0]['value']['x'], sequence['states'][0][0]['value']['y'])
            end_pos = (sequence['states'][-1][0]['value']['x'], sequence['states'][-1][0]['value']['y'])
            point_list_start.append(start_pos)
            point_list_end.append(end_pos)

        point_list_start = np.asarray(point_list_start)
        point_list_end = np.asarray(point_list_end)
        plt.figure()
        for index in range(len(point_list_start)):
            plt.scatter(point_list_start[index, 0], point_list_start[index, 1], c='b')
            plt.scatter(point_list_end[index, 0], point_list_end[index, 1], c='r')

            plt.plot([point_list_start[index, 0], point_list_end[index, 0]],
                     [point_list_start[index, 1], point_list_end[index, 1]], c='y')

        plt.xlabel('Position X (mm)')
        plt.ylabel('Position Y (mm)')
        plt.title('Pathway (b is start, y/o is end)')
        plt.gca().invert_yaxis()

    def genMotionIlluminationSequenceRandom(self, sequence_count=1,
                                            illumination_sequence=None):
        led_list = np.arange(self.metadata.illumination.state_list.design.shape[0])
        bf_mask = self.metadata.illumination.state_list.design[:, 0] ** 2 \
            + self.metadata.illumination.state_list.design[:, 1] ** 2 < (
                self.metadata.objective.na + self.illumination_na_pad) ** 2

        illumination_state_list = []
        linear_segments_processed = {}

        # Loop over DPC patterns (frames)
        for frame_index, frame_position_dict in enumerate(self.hardware_controller_list['position'].state_sequence):
            frame_position_list = frame_position_dict['states']

            # Get number of positions in blur kernel from this frame. Divide into subsequences
            pattern_count = len(frame_position_list) // sequence_count

            # Determine the number of non-zero illumination positions
            pattern_count_used = int(round(pattern_count * self.saturation_factor))

            # Place patterns at the END of the full sequence
            pattern_count_start = 0

            # Get linear segment index
            current_segment_index = frame_position_dict['common']['linear_segment_index']

            if not self.reuse_illumination_sequence or frame_index == 0:

                blur_vector_full = []

                # Generate several blur vectors
                for _ in range(sequence_count):
                    # Use provided illumination seqence if given
                    if illumination_sequence:
                        blur_vector = illumination_sequence
                    else:
                        blur_vector = np.zeros(pattern_count)
                        # Generate blur vector
                        blur_vector = np.zeros(pattern_count)
                        if self.blur_vector_method == 'strobe':
                            blur_vector = np.zeros(pattern_count)
                            blur_vector[pattern_count_start + pattern_count_used // 2] = 1
                        elif self.blur_vector_method == 'center':
                            blur_vector = np.zeros(pattern_count)
                            # Determine distance traveled within this frame (including readout time)
                            frame_pixel_count = round(frame_position_list[0][0]['frame_distance'] / (self.metadata.system.eff_pixel_size_um / 1000))
                            exposure_pixel_count = round(frame_position_list[0][0]['exposure_distance'] / (self.metadata.system.eff_pixel_size_um / 1000))
                            if not frame_pixel_count // 2 < exposure_pixel_count:
                                print("WARNING: Camera will not expose during center flash (%d pixels, %d pixels used of %d pixels total)" % (frame_pixel_count // 2, exposure_pixel_count, pattern_count))
                                blur_vector[pattern_count_used] = 1
                            else:
                                # Set center position to be on
                                blur_vector[frame_pixel_count // 2] = 1

                        elif self.blur_vector_method == 'start_end':
                            blur_vector = np.zeros(pattern_count)
                            blur_vector[pattern_count_start] = 1
                            blur_vector[pattern_count_start + pattern_count_used - 1] = 1
                        elif self.blur_vector_method == 'start_middle_end':
                            blur_vector = np.zeros(pattern_count)
                            blur_vector[pattern_count_start] = 1
                            blur_vector[pattern_count_start + pattern_count_used // 2] = 1
                            blur_vector[pattern_count_start + pattern_count_used - 1] = 1
                        elif self.blur_vector_method == 'tens':
                            blur_vector = np.zeros(pattern_count)
                            blur_vector[pattern_count_start] = 1
                            blur_vector[pattern_count_start + 10] = 1
                            blur_vector[pattern_count_start + 20] = 1
                            blur_vector[pattern_count_start + 30] = 1
                            blur_vector[pattern_count_start + 40] = 1
                        elif self.blur_vector_method == 'twenties':
                            blur_vector = np.zeros(pattern_count)
                            blur_vector[pattern_count_start + 0] = 1
                            blur_vector[pattern_count_start + 20] = 1
                            blur_vector[pattern_count_start + 40] = 1
                            blur_vector[pattern_count_start + 60] = 1
                            blur_vector[pattern_count_start + 80] = 1
                            blur_vector[pattern_count_start + 100] = 1
                            blur_vector[pattern_count_start + 120] = 1
                            blur_vector[pattern_count_start + 140] = 1
                            blur_vector[pattern_count_start + 160] = 1
                            blur_vector[pattern_count_start + 180] = 1
                        elif self.blur_vector_method == 'quarters':
                            blur_vector = np.zeros(pattern_count)
                            blur_vector[pattern_count_start] = 1
                            blur_vector[pattern_count_start + pattern_count_used // 4] = 1
                            blur_vector[pattern_count_start + pattern_count_used // 2] = 1
                            blur_vector[pattern_count_start + pattern_count_used // 2 + pattern_count_used // 4] = 1
                            blur_vector[pattern_count_start + pattern_count_used - 1] = 1
                        elif self.blur_vector_method == 'random':
                            blur_vector[pattern_count_start:pattern_count_start +
                                        pattern_count_used] = np.random.rand(pattern_count_used)
                        elif self.blur_vector_method == 'constant':
                            blur_vector[pattern_count_start:pattern_count_start +
                                        pattern_count_used] = np.ones(pattern_count_used)
                        elif self.blur_vector_method in ['coded', 'pseudo_random']:
                            if self.kernel_pulse_count is not None:
                                pulse_count = self.kernel_pulse_count
                            else:
                                pulse_count = pattern_count_used // 2

                            from htdeblur import blurkernel
                            blur_vector_tmp, kappa = blurkernel.vector(pulse_count, kernel_length=pattern_count_used)
                            blur_vector[pattern_count_start:pattern_count_start + pattern_count_used] = blur_vector_tmp
                        else:
                            raise ValueError('Invalid blur kernel method: %s' % self.blur_vector_method)

                    # Append to blur_vector_full
                    blur_vector_full += list(blur_vector)

                # Ensure the pattern is the correct length
                if len(blur_vector_full) < len(frame_position_list):
                    blur_vector_full += [0] * (len(frame_position_list) - len(blur_vector_full))
                elif len(blur_vector_full) > len(frame_position_list):
                    raise ValueError

                # Assign
                linear_segments_processed[str(frame_index)] = blur_vector_full
            else:
                blur_vector_full = linear_segments_processed['0']

            single_frame_state_list_illumination = []

            # Loop over time points (irrelevent for dpc)
            for time_index, illumination_value in enumerate(blur_vector_full):

                time_point_state_list = []
                # Loop over DPC patterns (which are themselves frames)
                # for led_number in led_list[bf_mask]:
                led_number = -1

                values_dict = {}
                for color_name in self.illumination_color:
                    values_dict[color_name] = self.illumination_color[color_name] * illumination_value

                led_dict = {
                    'index': int(led_number),
                    'time_index': time_index,
                    'value': values_dict
                }

                # Append this to list with elements for each interframe time point
                time_point_state_list.append(led_dict)

                # Append to frame_dict
                single_frame_state_list_illumination.append(time_point_state_list)

            # Define illumination sequence
            illumination_state_list.append({'states' : single_frame_state_list_illumination, 'common' : {}})

        return(illumination_state_list)
