# Copyright 2017 Regents of the University of California
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with # the distribution.
#
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import scipy as sp

import time
import matplotlib
from matplotlib import pyplot as plt
import copy
import os, sys
# from libwallerlab.utilities import io
# from libwallerlab.utilities import display
# from libwallerlab.utilities import roi
import serial
import json
import llops as yp
from llops import Roi
import comptic
from illuminate import IlluminateController

# Custom scale bar object
from matplotlib_scalebar.scalebar import ScaleBar

# Default MM directory
mm_directory = 'C:\\Program Files\\Micro-Manager-2.0beta'

# Global variable for micro-manager interface
mmc = None

def loadMicroManager(mm_directory, system_cfg_file):
    prev_dir = os.getcwd()
    os.chdir(mm_directory) # MUST change to micro-manager directory for method to work
    import MMCorePy
    mmc = MMCorePy.CMMCore()
    mmc.loadSystemConfiguration(system_cfg_file)
    os.chdir(prev_dir)
    print("Micro-manager was loaded sucessfully!")
    return(mmc)

class HardwareController():
    '''
    This is a generic hardware object, used to wrap specific hw controllers
    '''
    def __init__(self, hw_type, device_name):
        self.type = hw_type
        self.state_sequence = None
        self.device_name = device_name
        self.state_index = 0
        self.accepts_preload = False
        self.preload_all_frames = True
        self.combine_sequences = False
        self.continuous_states_between_frames = False

        # State sequences measured during experiment and for preloading
        self.is_sequencable = False
        self.state_sequence_experiment = None
        self.state_sequence_preload = None
        self.state_sequence = None
        self.time_sequence_s = None
        self.time_sequence_s_preload = None

        # Triggering options
        self.trigger_mode = 'software'
        self.trigger_pin = None
        self.does_trigger_camera = False
        self.camera_trigger_command = None
        self.sequence_dt_ms = None

        # Serial things
        self.command_terminator = '\r'
        self.response_terminator = '\r'
        self.command_debug = 0

    def isReadyForSequence(self):
        '''
        Flag which can indicate if the object is ready for a new sequence
        '''
        return(True)

    def unload(self):
        '''
        Unload device so it cna be controller by another object or program.
        '''
        try:
            self.ser.close()
        except:
            pass

    def sequenceReset(self):
        '''
        Reset sequence to initial state (does not clear stored sequence)
        '''
        pass

    def sequenceStep(self):
        '''
        Stop a running sequence
        '''
        pass

    def runSequence(self, function_to_run_each_iteration=None):
        '''
        Run a sequence
        '''
        pass

    def seq_clear(self):
        self.state_sequence = None

    def runSequenceIndex(self, index_to_run):
        pass

    def preloadSequence(self, frame_index=0):
        '''
        Preload a sequence to hardware (if this exists)
        '''
        pass

    def isSequenceRunning(self):
        '''
        Returns True if sequence is running, false if not
        '''
        return(False)

    def plot(self, figsize=(6,6)):
        pass

    def command(self, cmd, wait_for_response=True):
        '''
        Sends a command to serial device and then waits for any response, then returns it
        '''
        # Flush buffer
        try:
            self.flush()
        except:
            pass


        # Send command
        if self.command_debug > 0:
            print(cmd)

        if self.ser.is_open:
            self.ser.write(str.encode(cmd) + str.encode(str(self.command_terminator)))
        else:
            raise ValueError('Serial port not open!')

        # Get response
        if wait_for_response:
            response = self.response()

            # Print the response, for debugging
            if self.command_debug > 1:
                print(response)

            # Return response
            return(response)

    def response(self, timeout_s = 5):
        '''
        Gets response from device
        '''
        if self.ser.is_open:
            response = self.ser.read_until(str.encode(str(self.response_terminator))).decode("utf-8")
            if 'ERROR' in response:
                raise ValueError(response.split('ERROR'))
            else:
                return(response)
        else:
            raise ValueError('Serial port not open!')

    def flush(self):
        '''
        Flushes serial buffer
        '''
        if self.ser.is_open:
            self.ser.read_all()
        else:
            raise ValueError('Serial port not open!')

    def reset(self):
        '''
        Resets hardware element to default state
        '''
        pass

class LedArrayController(IlluminateController):

    def __init__(self, com_port, device_name='led_array', baud_rate=115200):
        # Device objects
        self.device_name = device_name
        self.type = 'illumination'

        # Device configuration
        self.accepts_preload = True
        self.does_trigger_camera = False
        self.state_sequence = None
        self.trigger_mode = 'software'

        super(self.__class__, self).__init__(com_port, baud_rate)

    def isReadyForSequence(self):
        """This device is always ready for sequence."""
        return True

    def seq_clear(self):
        self.state_sequence = None


class PositionController(HardwareController):
    '''
    This is a class for controlling a position sequence
    '''
    def __init__(self, com_port, velocity=25, acceleration=1e3, trigger_mode='software', device_name="xy_stage_serial", baud_rate=38400):
        HardwareController.__init__(self, "position", device_name)

        # Device objects
        self.device_name = device_name
        self.com_port = com_port
        self.baud_rate = baud_rate
        self.trigger_mode = trigger_mode
        self.ser = None

        # Device configuration
        self.accepts_preload = True
        self.does_trigger_camera = False
        self.is_sequencable = True

        # Stage parameters
        self.state_sequence_experiment = []
        self.state_sequence_preload = []

        self._velocity = velocity
        self._acceleration = acceleration
        self._jerk = 11.5
        self.encoder_window = (0,0)
        self.extra_run_up_time_s = 0

        # Preload run-up parameters
        self.preload_run_up_distance_mm = 0
        self.preload_start_position_mm = (0, 0)

        # Speed and homing tolerance for rapid motion
        self.rapid_velocity = 40
        self.rapid_encoder_window = (10,10)

        # Serial command and response terminators
        self.response_terminator = '\r'
        self.command_terminator = '\r'
        self.trigger_pulse_width_us = 2000
        self.device_steps_per_um = 25 # Don't change
        self._steps_per_um = 1 #

        # Load device
        self.reload()

        # Set up stage
        self.stop()
        self.command('BAUD ' + str(self.baud_rate)[0:2], wait_for_response=False) # Set Baud
        self.command('BLSH 0', wait_for_response=True) # Disable backslash
        self.command('ERROR 1', wait_for_response=True) # Human-readable error codes
        self.command('ENCODER Y 0') # Turn off y encoder
        self.command('ENCODER X 0') # turn off x encoder
        self.command('ENCW X 0') # Set encoder window (prevents excessive homing)
        self.command('ENCW Y 0') # Set encoder window (prevents excessive homing)
        self.command('SERVO X 0')  # Turn off servo behavior
        self.command('SERVO Y 0')  # Turn off servo behavior
        self.command('SS ' + str(self._steps_per_um)) # Set encoder window (prevents excessive homing)

        # Hard-code jerk
        self.command('SCS 60000') # jerk = 11.5 for SCS 60000

        # Set paremeters
        self.velocity = self.velocity
        self.acceleration = self.acceleration

    def reload(self):
        # Close device if it is open
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

        # Create new device and set baud ratezaqzq
        self.ser = serial.Serial(self.com_port)
        self.ser.baudrate = self.baud_rate

        # Reset device
        self.reset()

    def sequenceReset(self):
        # Stop current movement
        self.command('I', wait_for_response=True)

        # Reset sequence variables
        self.position_idx = 0
        self.position_list_experiment = []

        # Disable Triggering
        self.command('TTLTRG 0', wait_for_response=False)

    def preloadSequence(self, frame_index, state_sequence=None):

        pathway_debug = False
        # Preload whole sequence
        if state_sequence is None:
            self.state_sequence_preload = self.state_sequence
        else:
            self.state_sequence_preload = state_sequence

        # Get common metadata
        self.state_sequence_preload_common = self.state_sequence_preload[0]['common']

        # Flatten all states in state_sequence
        if type(frame_index) is list:
            tmp = [self.state_sequence_preload[index]['states'] for index in frame_index]
            self.state_sequence_preload = [[item for sublist in tmp for item in sublist]]
        else:
            # Select subset of preload if it's provided
            if frame_index >= 0:
                self.state_sequence_preload = [self.state_sequence_preload[frame_index]['states']]
            else:
                self.state_sequence_preload = [[item for sublist in self.state_sequence_preload for item in sublist['states']]]


        x_start = self.state_sequence_preload[0][0][0]['value']['x']
        y_start = self.state_sequence_preload[0][0][0]['value']['y']

        x_end = self.state_sequence_preload[0][-1][0]['value']['x']
        y_end = self.state_sequence_preload[0][-1][0]['value']['y']

        if pathway_debug:
            print("Pathway design start: (%.4f, %.4f), end: (%.4f, %.4f)" % (x_start, y_start, x_end, y_end))

        # Delete all previous trigger points
        self.command('TTLDEL 1')

        # Get velocity
        velocity = self.state_sequence_preload_common['velocity']

        # Get run-up distance (for acceleration to maximum velocity)
        run_up_distance_mm = self.preload_run_up_distance_mm

        # for state_index, state in enumerate(self.state_sequence_preload):
        state_index = 0
        state = self.state_sequence_preload[state_index]

        # Determine direction of movement (assuming it's linear)
        direction = np.asarray([(y_end - y_start), (x_end - x_start)])
        direction /= np.linalg.norm(direction, 2) # Normalize direction
        offset = direction * run_up_distance_mm

        # Set the start and end positions including offset
        self.preload_start_position_mm = (x_start - offset[1], y_start - offset[0])
        self.preload_end_position_mm = (x_end + offset[1], y_end + offset[0])

        # Calculate relative move
        self.relative_move_mm = [start - end for (end, start) in zip(self.preload_start_position_mm, self.preload_end_position_mm)]

        # Debug pathway stuff
        if pathway_debug:
            print('Initial position: (%.4f, %.4f)' % self.getPosXY())

            # Print start position
            print('Start position (without offset): (%.4f, %.4f)' % (x_start, y_start))
            print('Start position (with offset): (%.4f, %.4f)' % (self.preload_start_position_mm[0], self.preload_start_position_mm[1]))

            # Print start position
            print('End position (without offset): (%.4f, %.4f)' % (x_end, y_end))
            print('End position (with offset): (%.4f, %.4f)' % (self.preload_end_position_mm[0], self.preload_end_position_mm[1]))

            # Print relative move
            print('Relative move (with offset): (%.4f, %.4f)' % (self.relative_move_mm[0], self.relative_move_mm[1]))

            # Print the offset
            print('Offset is (%g, %g)' % (offset[0], offset[1]))

        # Set velocity
        self.velocity = (velocity)
        assert abs(self.velocity - velocity) < 1e-3

        # Set trigger point
        self.command('TTLTP 1 1')

        # Turn on TTL output while moving
        self.command('TTLMOT 1 1')

        # Send movement command (40 is relative motion, 41 is absolute)
        x_move = int(self.device_steps_per_um / self._steps_per_um * int(1000 * self.relative_move_mm[0]))
        y_move = int(self.device_steps_per_um / self._steps_per_um * int(1000 * self.relative_move_mm[1]))
        self.command('TTLACT 1 40 ' + str(x_move) + ' ' + str(y_move)  + ' 0')

        if pathway_debug:
            print("Preloaded relative move of (%.4f, %.4f) " % (x_move / 1000 / self.device_steps_per_um * self._steps_per_um, y_move / 1000 / self.device_steps_per_um * self._steps_per_um))

        # Send to initial position
        self.goToPosition(self.preload_start_position_mm, blocking=True)
        # self.goToPositionRapid(self.preload_start_position_mm, blocking=True)

    def runSequence(self, commandFunc=None):

        # Set velocity
        self.velocity = (self.state_sequence_preload_common['velocity'])

        # If the stage uses software triggering, send commands directly, otherwise just enable triggering
        if self.trigger_mode is 'software':
            # Loop over remaining positions
            for (idx, position) in enumerate(self.state_sequence_preload):

                self.goToPosition((position['sequence'][0]['value']['x_start'], position['sequence'][0]['value']['y_start']), blocking=True)

                time.sleep(0.1)
                while self.isMoving():
                    time.sleep(0.1)

                x_start = self.getPosX()
                y_start = self.getPosY()

                self.command('G ' + str(1000 * position['sequence'][0]['value']['x_end']) + ' ' + str(1000 * position['sequence'][0]['value']['y_end']))

                # Call user command if provided
                if commandFunc is not None:
                    commandFunc()
                else:
                    time.sleep(0.1)

                while self.isMoving():
                    time.sleep(0.1)

                x_end = self.getPosX()
                y_end = self.getPosY()

                self.state_sequence_experiment.append({'sequence': {'x_start' : x_start, "y_start" : y_start, "x_end" : x_end, "y_end" : y_end}})
                print("%.2f x %.2f" % (self.state_sequence_experiment[-1]['sequence']['x_end'], self.state_sequence_experiment[-1]['sequence']['y_end']))
        else:
            # Enable Triggering
            self.command('TTLTRG 1')

    def sequenceStep(self):
        if self.position_idx == len(self.state_sequence_preload):
            self.position_idx = 0
        position = self.state_sequence_preload[self.position_idx][0][0]['value']

        x_start = self.getPosX()
        y_start = self.getPosY()

        self.goToPosition((position['x'], position['y']))

        while self.isMoving():
            time.sleep(0.01)

        x_end = self.getPosX()
        y_end = self.getPosY()

        self.position_list_experiment.append({'x_start' : x_start, "y_start" : y_start, "x_end" : x_end, "y_end" : y_end})

        print("Moved to position %d of %d. Position is (%.2f, %.2f)" % (self.position_idx, len(self.state_sequence_preload), self.position_list_experiment[-1]['x_end'], self.position_list_experiment[-1]['y_end']))
        self.position_idx += 1

    def runSequenceIndex(self, index, command_function=None):
        if index < len(self.position_list) and index >= 0:
            position = self.position_list[index]

            x_start = self.getPosX()
            y_start = self.getPosY()

            self.command('G ' + str(1000 * position['x_end']) + ' ' + str(1000 * position['y_end']))
            if command_function is not None:
                command_function()
            else:
                time.sleep(0.1)

            while self.isMoving():
                time.sleep(0.1)

            x_end = self.getPosX()
            y_end = self.getPosY()

            self.position_list_experiment.append({'x_start' : x_start, "y_start" : y_start, "x_end" : x_end, "y_end" : y_end}) # TODO: this should index into posiiton_list

            print("Moved to position %d of %d. Position is (%.2f, %.2f)" % (index + 1, len(self.position_list), self.position_list_experiment[index]['x_end'], self.position_list_experiment[index]['y_end']))

        else:
            raise ValueError("Error - invalid position_list_index (%d)" % index)

    ### Position Controller Specific commands ###

    @property
    def steps_per_um(self):
        self._steps_per_um = int(self.command('SS').split(self.command_terminator)[0])
        return self._steps_per_um

    @steps_per_um.setter
    def steps_per_um(self, steps_per_um):
        self._steps_per_um = steps_per_um
        self.command('SS ' + str(self._steps_per_um)) # Set encoder window (prevents excessive homing)

    @property
    def position(self):
        """Returns stage posiiton in mm."""

        # Get position
        raw = self.command('P').split(',')
        self._position = ((float(raw[0])/  1000. * self._steps_per_um / self.device_steps_per_um, float(raw[1]) / 1000. * self._steps_per_um / self.device_steps_per_um))

        # Return cached variable
        return self._position

    @position.setter
    def position(self, new_position):
        """Sets stage position in mm."""

        # Store current position
        self._position = new_position

        # Go to position
        self.command('G ' + str(int(position_mm[0] * 1000 * self.device_steps_per_um / self._steps_per_um)) + ' ' + str(int(position_mm[1] * 1000 * self.device_steps_per_um / self._steps_per_um)))

        # Block until movement is done
        while not self.isReadyForSequence():
            time.sleep(0.01)

    @property
    def velocity(self):
        velocity_str = self.command('SMS I')
        self._velocity = int(velocity_str.replace('\r','')) / 1000  / self.device_steps_per_um
        return self._velocity

    @velocity.setter
    def velocity(self, velocity_mm_s):
        self.stop()
        self._velocity = velocity_mm_s
        self.command('SMS ' + str(int(self._velocity * 1000 * self.device_steps_per_um)) + ' I')

    @property
    def acceleration(self):
        acceleration_str = self.command('SAS I')
        self._acceleration = int(acceleration_str.replace('\r','')) / 1000 / self.device_steps_per_um
        return self._acceleration

    @acceleration.setter
    def acceleration(self, acceleration_mm_s_2):
        self.stop()
        self._acceleration = acceleration_mm_s_2
        self.command('SAS ' + str(int(self.acceleration * 1000 * self.device_steps_per_um)) + ' I') # Change acceleration

    @property
    def jerk(self):
        ''' Returns jerk, the rate of change of acceleration in mm/s/s/s '''
        val = float(self.command('SCS I'))
        self._jerk = int(round(7423222950 / val / 1e3)) # convert to mm_s_s_s
        return self._jerk

    @jerk.setter
    def jerk(self, new_jerk_mm_s_s_s):
        ''' Sets jerk, the rate of change of acceleration in mm/s/s/s '''
        self._jerk = new_jerk_mm_s_s_s
        val = int(round(7423222950 / (new_jerk_mm_s_s_s * 1000))) # convert to um_s_s_s
        self.command('SCS ' + str(val) + ' I')

    def setEncoderWindow(self, encoder_window):
        self.stop()
        self.command('ENCW Y ' + str(encoder_window[0]))
        self.command('ENCW X ' + str(encoder_window[1]))

    def setEncoderState(self, state):
        self.stop()
        if type(state) not in [tuple, list, np.ndarray]:
            self.command('ENCODER X ' + str(int(state > 0)))
            self.command('ENCODER Y ' + str(int(state > 0)))
        else:
            assert len(state) == 2
            self.command('ENCODER X ' + str(int(state[1] > 0)))
            self.command('ENCODER Y ' + str(int(state[0] > 0)))

    def goToPositionRapid(self, position_mm, blocking=True):
        v = self.velocity
        self.velocity = (self.rapid_velocity)
        self.command('ENCODER X 1')
        self.command('ENCODER Y 1')
        self.command('ENCW X 1000')
        self.command('ENCW Y 1000')
        self.goToPosition(position_mm, blocking)
        self.command('ENCODER X 0')
        self.command('ENCODER Y 0')
        self.command('ENCW X 10000')
        self.command('ENCW Y 10000')
        self.velocity = v

    def goToPosition(self, position_mm, blocking=False):
        self.command('G ' + str(int(position_mm[0] * 1000 * self.device_steps_per_um / self._steps_per_um)) + ' ' + str(int(position_mm[1] * 1000 * self.device_steps_per_um / self._steps_per_um)))
        if blocking:
            while not self.isReadyForSequence():
                time.sleep(0.01)

    def test(self, v=0.5, d=1, tol=0.25):
        self.velocity = (10)
        self.acceleration = (self.acceleration)
        self.goToPosition((0, 0),blocking=True)
        t0 = time.time()
        self.velocity = (v)
        self.goToPosition((d, 0),blocking=True)
        t = time.time()
        assert (t- t0) - d / v < tol, "Time difference (%.2f) > tolerance (%.2f)" % ((t- t0) - d / v, tol)
        self.velocity = (v)
        self.goToPosition((0, 0),blocking=True)
        assert (t- t0) - 2 * d / v < tol, "Time difference (%.2f) > tolerance (%.2f)" % ((t- t0) - 2 * d / v, tol)
        return(True)

    def plotPositionList(self, figsize=(6,6)):
        plt.figure(figsize=figsize)
        ax = plt.gca()
        for item in self.position_list:
            plt.plot([item['x_start'], item['x_end']], [item['y_start'], item['y_end']], color='r', linestyle='-', linewidth=2, label="Design Movement Paths")
        if len(self.position_list_experiment) > 0 :
            for item in self.position_list_experiment:
                plt.plot([item['x_start'], item['x_end']], [item['y_start'], item['y_end']], color='b', linestyle='-', linewidth=2, label="Actual Movement Paths")

                ax.arrow(item['x_start'], item['y_start'], item['x_end'] -item['x_start'], item['y_end'] - item['y_start'], head_width=0.05, head_length=0.1, fc='k', ec='k')

        plt.xlabel('Position X (mm)')
        plt.ylabel('Position Y (mm)')
        # plt.legend()
        plt.show()

    def runPositionSequence_vel(self, position_list, velocity = 100000, acceleration=1e6):
        self.mmc.setProperty('xy_stage_serial', 'Command', 'I') # Stop Movement
        self.mmc.setProperty('xy_stage_serial', 'Command', 'SMS ' + str(velocity * 25) + ' 0') # Change velocity
        self.mmc.setProperty('xy_stage_serial', 'Command', 'SAS ' + str(acceleration * 25) + ' 0') # Change acceleration

        self.mmc.setProperty('xy_stage_serial', 'Command', 'G ' + str(position_list[0]['x_start']) + ' ' + str(position_list[0]['y_start']))
        time.sleep(2)
        for position in position_list:
            dx = (position['x_end'] - position['x_start'])
            dy = (position['y_end'] - position['y_start'])

            self.mmc.setProperty('xy_stage_serial', 'Command', 'VS ' + str(1000*dx) + ' ' + str(1000*dy)) # Run macro
            time.sleep(max(abs(dx),abs(dy)) / velocity)
            print("Design: position x is %.2f, y is %.2f" % (factor * position['x_end'], factor * position['y_end']))
            print("Actual: position x is %.2f, y is %.2f\n" % (self.getPosX(), self.getPosY()))

        self.mmc.setProperty('xy_stage_serial', 'Command', 'VS 0 0') # Run macro
        self.mmc.setProperty('xy_stage_serial', 'Command', 'I') # Run macro

    def stop(self):
        '''
        Stops the motion stage
        '''
        self.command('I', wait_for_response=False)
        time.sleep(0.05) # sleep to let this command process

    def reset(self):
        '''
        Reset system
        '''
        self.stop()
        # self.goToPosition((0,0))
        self.position_idx = 0
        self.position_list_experiment = []

        # Disable Triggering
        self.command('TTLTRG 0')

    def isMoving(self):
        status = self.command("$", wait_for_response=True)
        return('0' not in status)

    def isReadyForSequence(self):
        status = self.command("$", wait_for_response=True)
        return('0' in status)

    def getPosX(self):
        '''
        Function to get x position of stage in mm
        '''
        return(float(self.command('PX')) /  1000. * self._steps_per_um / self.device_steps_per_um)

    def getPosY(self):
        '''
        Function to get y position of stage in mm
        '''
        return(float(self.command('PY')) / 1000. * self._steps_per_um / self.device_steps_per_um)

    def getPosXY(self):
        '''
        Function to get y position of stage in mm
        '''
        raw = self.command('P').split(',')
        return((float(raw[0])/  1000. * self._steps_per_um / self.device_steps_per_um, float(raw[1]) / 1000. * self._steps_per_um / self.device_steps_per_um))

    def zero(self):
        self.velocity = (self.rapid_velocity)
        self.command('PS 0 0')

    def setJoystickFlip(self,flip_x=False, flip_y=False):
        if flip_x:
            self.command('JXD -1') # Joystick direction
        else:
            self.command('JXD 1') # Joystick direction

        if flip_y:
            self.command('JYD -1') # Joystick direction
        else:
            self.command('JYD 1') # Joystick direction


class MicroManagerHardwareController(HardwareController):

    def __init__(self, mm_directory, cfg_file, hw_type, device_name):
        global mmc
        if mmc is None:
            mmc = loadMicroManager(mm_directory, cfg_file)

        self.mmc = mmc
        HardwareController.__init__(self, hw_type, device_name)

    def MM_printProperties(self):
        '''
        Prints properties of the micro-manager device adapter along with allowed properties
        '''
        assert self.mmc is not None, "Micro-manager interface not provided!"

        for prop_name in self.mmc.getDevicePropertyNames(self.device_name):
            # Get allowed property valies
            values_str = ''
            for value in self.mmc.getAllowedPropertyValues(self.device_name, prop_name):
                values_str += (value + ', ')
            print(prop_name + ": " + str(self.mmc.getProperty(self.device_name, prop_name)) + '   [' + values_str[:-2] +']')\

    def MM_setProperty(self, property_name, property_value):
        assert self.mmc is not None, "Micro-manager interface not provided!"
        self.mmc.setProperty(self.device_name, property_name, property_value)

    def MM_getProperty(self, property_name):
        assert self.mmc is not None, "Micro-manager interface not provided!"
        return(self.mmc.getProperty(self.device_name, property_name))


class CameraController(MicroManagerHardwareController):

    def __init__(self, camera_name, pixel_size_um=None, trigger_mode='software',
                 mm_directory=None, cfg_file=None, bayer_coupling_matrix=None):

        self.device_name = camera_name
        # Initialize metaclass
        if 'pco' in camera_name.lower():
            cfg_file = 'D:\\Hardware\\Micro-Manager\\config\\pco_only.cfg'
            camera_name = 'pco'
            if pixel_size_um is None:
                pixel_size_um = 6.5
        elif camera_name.lower() == 'retiga':
            cfg_file = "D:\\Hardware\\Micro-Manager\\config\\retiga_only.cfg"
            if pixel_size_um is None:
                pixel_size_um = 4.5
        else:
            raise ValueError("Invalid camera name %s" % camera_name)

        if mm_directory is None:
            mm_directory = 'C:\\Program Files\\Micro-Manager-2.0beta'

        MicroManagerHardwareController.__init__(self, mm_directory, cfg_file, 'camera', camera_name)

        # Camera geometric transforms
        self.transpose = False
        self.flip_y = False
        self.flip_x = False

        # Set camera-specific parameters
        self.pixel_size_um = pixel_size_um
        self.trigger_mode = trigger_mode
        self.trigger_pulse_width_us = 300
        self.setTriggerMode(self.trigger_mode)
        self.line_time_us = 16.4
        self.min_exposure_time = 0.01 # Defined by testing (datasheet seems to be wrong...)
        self.min_frame_dt_s = 1 / 50 # Defined by testing (datasheet seems to be wrong...) [0.41 for fast scan, 0.6 for slow scan]
        self.trigger_mode

        # Get color settings
        if bayer_coupling_matrix is not None:
            self.bayer_coupling_matrix = bayer_coupling_matrix
            self.is_color = True
        else:
            self.bayer_coupling_matrix = None
            self.is_color = False

        # Set up pixel rate for pco
        if self.device_name is 'pco':
            # Ensure we're in global shutter mode
            assert 'global' in self.mmc.getProperty('pco', 'CameraType').lower(), 'Camera is not in global shutter mode! Set this in the PCO software (Camera -> Setup menu item in the top bar).'

            # Set Pixel rate to high-speed
            try:
                self.mmc.setProperty('pco','PixelRate','fast scan') # Set to high-speed
            except:
                pass

            # Set camera noise filter to off
            try:
                self.mmc.setProperty('pco','Noisefilter','off')
            except:
                pass

            self.mmc.setProperty(self.device_name,'Signal 1 (Exposure Trigger) Polarity', 'rising')
        elif self.device_name is "optimos":
            # Disable all FPGA processing
            self.mmc.setProperty(self.device_name, "PP  1   ENABLED:", "No")
            self.mmc.setProperty(self.device_name, "PP  2   ENABLED:", "No")
            self.mmc.setProperty(self.device_name, "PP  3   ENABLED:", "No")
            self.mmc.setProperty(self.device_name, "PP  4   ENABLED:", "No")

        # self.mmc.setCameraDevice(self.device_name)

        # Define camera roi
        self.roi = yp.Roi(shape=(self.getImageHeight(), self.getImageWidth()))
        self.shape = (self.getImageHeight(), self.getImageWidth())

    def fov(self, magnification):
        """Returns the FOV in microns given a system magnification."""

        # Determine FOV size
        ps = self.pixel_size_um / magnification

        # Divide pixel size by two if we're using
        if self.is_color:
            ps = ps / 2

        if self.transpose:
            fov = np.asarray((self.getImageHeight() * ps, self.getImageWidth() * ps)) / 1000.
        else:
            fov = np.asarray((self.getImageWidth() * ps, self.getImageHeight() * ps)) / 1000.

        return fov

    def calcExposureTimeFromBusyTime(self, busy_time_s):
        ''' see https://www.pco.de/fileadmin/user_upload/pco-manuals/pco.edge_manual.pdf '''

        t_line = 9.17e-6 # line time
        t_frame = min(self.getImageHeight(), self.getImageWidth()) * t_line / 2
        t_exp = busy_time_s - t_frame - t_line
        assert t_exp > 0, "Frame too short! (%.3f < %.3f)" % (t_exp, t_frame + t_line)
        return t_exp

    def getImageHeight(self):
        return(self.mmc.getImageHeight())

    def getImageWidth(self):
        return(self.mmc.getImageWidth())

    def runSequence(self):
        self.reset()
        self.mmc.startContinuousSequenceAcquisition(0)

    def sequenceStop(self):
        if self.mmc.isSequenceRunning():
            self.mmc.stopSequenceAcquisition()

    def sequenceReset(self):
        self.mmc.clearCircularBuffer()
        if self.mmc.isSequenceRunning():
            self.mmc.stopSequenceAcquisition()

    def getBufferTotalCapacity(self):
        return(self.mmc.getBufferTotalCapacity())

    def setExposure(self, exposure_time_s):
        self.mmc.setProperty(self.device_name, 'Exposure', exposure_time_s * 1000)

    def getExposure(self):
        return float(self.mmc.getProperty(self.device_name, 'Exposure')) / 1000.

    def setBufferSizeMb(self, size_mb):
        if size_mb != self.mmc.getCircularBufferMemoryFootprint():
            try:
                self.mmc.setCircularBufferMemoryFootprint(size_mb)
            except:
                print("Unable to set buffer to size %d" % size_mb)

    def setRoi(self, roi):
        self.mmc.clearROI()
        if hasattr(roi,'y_start'):

            if roi.y_end < 0 or roi.y_end > self.getImageHeight():
                roi.y_end = self.getImageHeight()
                print("Using max ROI in y")
            if roi.x_end < 0 or roi.x_end > self.getImageWidth():
                roi.x_end = self.getImageWidth()
                print("Using max ROI in x")

            roi.x_end += int(np.mod(roi.x_end - roi.x_start, 2))
            roi.y_end += int(np.mod(roi.y_end - roi.y_start, 2))

            self.mmc.setROI(roi.x_start, roi.y_start, roi.x_end - roi.x_start, roi.y_end - roi.y_start)
        else:
            raise NotImplementedError("setRoi only takes objects of type io.Roi at the moment")

    def getBufferSizeFrames(self):
        return(self.mmc.getRemainingImageCount())

    def readFramesFromBuffer(self, img_bg=None):
        sys.path.append(mm_directory)
        import MMCorePy
        md = MMCorePy.Metadata()
        elapsed_frame_time = []

        frame_list = np.zeros((self.getBufferSizeFrames(), self.getImageHeight(), self.getImageWidth()), dtype=np.uint16)
        for img_idx in range(frame_list.shape[0]):
            if img_bg is not None:
                frame_list[img_idx,:,:] = np.max(self.mmc.popNextImageMD(md).copy().astype(np.int) - img_bg).astype(np.uint16)
            else:
                frame_list[img_idx,:,:] = self.mmc.popNextImageMD(md).copy()
            elapsed_frame_time.append(float(md.GetSingleTag("ElapsedTime-ms").GetValue()))

        elapsed_frame_time = np.asarray(elapsed_frame_time)
        elapsed_frame_time = elapsed_frame_time - np.min(elapsed_frame_time)

        return (frame_list, elapsed_frame_time.tolist())

    def liveView(self, exposure_ms=None, roi=None, zoom_factor=None, figsize=None,
                    contrast_tight_factor = 0.4, contrast_type='tight'):

        # Stop existing sequence acquisition
        if self.mmc.isSequenceRunning():
            print('Stopping Existing sequence acquisition')
            self.mmc.stopSequenceAcquisition()

        # Set a small memory footprint
        self.setBufferSizeMb(30)

        # Set up triggering
        self.setTriggerMode('software')

        # Set exposure
        if exposure_ms is not None:
            self.setExposure(exposure_ms / 1000.)

        # Set figsize
        if figsize is None:
            figsize=(8,6)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

        # Turn axis off
        plt.axis('off')

        # Set zoom factor
        if zoom_factor is not None:
            start = [sh // 2 - sh // int(2 * zoom_factor) for sh in (self.roi.shape)]
            shape = [sh // zoom_factor for sh in self.roi.shape]
            roi = yp.Roi(start=start, shape=shape)

        # Snap first image to populate buffer
        img_0 = self.snap(roi=roi)

        if self.is_color:
            img_0 = display.demosaicFrameDeconv(img_0, white_balance=True, bayer_coupling_matrix=self.bayer_coupling_matrix)

        im = ax.imshow(img_0, interpolation='nearest',
                           animated=True, cmap = 'gray', vmin=0, vmax=2**16 - 1)

        fig.canvas.draw()

        # Start Acquisition
        self.mmc.startContinuousSequenceAcquisition(0)

        acq_running = True

        def onclick(event):
            raise KeyboardInterrupt
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # Continuous Acquisition
        t0 = time.time()
        while acq_running:
            try:
                if self.mmc.getRemainingImageCount() > 0 and acq_running:
                    img = self.mmc.getLastImage()
                    if self.transpose:
                        img = img.T
                    if self.flip_x:
                        img = np.fliplr(img)
                    if self.flip_y:
                        img = np.flipud(img)

                    if roi is not None:
                        img = img[roi.slice]
                    if self.is_color:
                        im.set_data(display.demosaicFrameDeconv(img,
                            bayer_coupling_matrix=self.bayer_coupling_matrix))
                    else:
                        im.set_data(img)

                    if contrast_type is "tight":
                        im.set_clim(np.mean(img) * contrast_tight_factor, np.mean(img) * (1/contrast_tight_factor))
                    elif contrast_type is "fit":
                        im.set_clim(np.amin(img), np.amax(img))
                    else:
                        im.set_clim(0,2**16 - 1)

                    fig.canvas.draw()
                    ax.set_title("%.2f FPS - Max value is %d" % ((1/(time.time()-t0), np.max(img))))
                    t0 = time.time()

            except KeyboardInterrupt:
                self.mmc.stopSequenceAcquisition()
                acq_running = False
                break;

    def snap(self, roi=None):

        if self.mmc.isSequenceRunning():
            self.mmc.stopSequenceAcquisition()

        # Snap and return image
        self.mmc.snapImage()
        img = self.mmc.getImage()

        # Apply simple geometric transformations
        if self.transpose:
            img = img.T
        if self.flip_x:
            img = np.fliplr(img)
        if self.flip_y:
            img = np.flipud(img)

        # Apply ROI
        if roi is not None:
            img = img[roi.slice]

        # Reset to hardware triggering
        if self.trigger_mode == 'hardware':
            self.setTriggerMode('hardware')

        return img

    def testTrigger(self, triggering_device_controller):
        t = 0
        t_exp = 0.3
        t_max = 1

        self.setExposure(t_exp)
        self.setTriggerMode('hardware')
        self.runSequence()
        t_0 = time.time()
        while time.time() - t_0 < t_max:
            triggering_device_controller.command('tr')
            time.sleep(0.1)
        n_frames = self.mmc.getBufferTotalCapacity() - self.mmc.getBufferFreeCapacity()
        self.sequenceStop()

        if n_frames > 0:
            return(True)
        else:
            return(False)

    def setTriggerMode(self, mode):
        '''
        This functions sets the trigger mode to be software or hardware. Currently this only works for the PCO camera.
        '''
        # Stop exisitng acquisition if one is running
        self.sequenceStop()

        # Switch based on software or hardware
        if mode == "software":
            if self.device_name == "pco":
                self.mmc.setProperty(self.mmc.getCameraDevice(), 'Acquiremode', 'Internal')
                self.mmc.setProperty(self.mmc.getCameraDevice(), 'Triggermode', 'Internal')
            elif self.device_name == "optimos":
                self.mmc.setProperty(self.mmc.getCameraDevice(), 'TriggerMode', 'Internal Trigger')
            self.trigger_mode = mode
        elif mode == "hardware":
            if self.device_name == "pco":
                self.mmc.setProperty(self.mmc.getCameraDevice(), 'Acquiremode', 'Internal')
                self.mmc.setProperty(self.mmc.getCameraDevice(), 'Triggermode', 'External')
            elif self.device_name == "optimos":
                self.mmc.setProperty(self.mmc.getCameraDevice(), 'TriggerMode', 'Edge Trigger')
            self.trigger_mode = mode
        # elif mode == 'hardware_exposure_control':
        #     if self.device_name == "pco":
        #         self.mmc.setProperty(self.mmc.getCameraDevice(), 'Acquiremode', 'Internal')
        #         self.mmc.setProperty(self.mmc.getCameraDevice(), 'Triggermode', 'External Exp. Ctrl.')
        #     else:
        #         raise ValueError("Only PCO supports full exposure control using the trigger")
        else:
            raise ValueError('Invalid trigger mode: %s' % mode)

    def autoExposure(self, target_max_count = 40000, count_error_threshold=10000, max_exposure_ms = 1000):

        self.sequenceStop()

        self.mmc.setExposure(0.1)
        # Get current exposure and pixel_count
        old_exp = self.mmc.getExposure()
        self.mmc.snapImage()
        img = self.mmc.getImage()
        max_count = np.amax(img)
        # max_count = target_max_count
        new_exp = old_exp

        while np.abs(max_count - target_max_count) > count_error_threshold:
            ratio = max_count / target_max_count
            new_exp = self.mmc.getExposure() * 1 / ratio
            if new_exp > max_exposure_ms:
                new_exp = max_exposure_ms
                print("Exceeding max exposure")
                break
            self.mmc.setExposure(new_exp)
            self.mmc.snapImage()
            time.sleep(new_exp * 2 / 1000)
            img = self.mmc.getImage()
            max_count = np.amax(img)
        self.mmc.setExposure(new_exp)
        self.mmc.snapImage()
        img = self.mmc.getImage()

        print("Auto-exposure finished. Old exposure was %.2fms, new exposure is %.2fms (%d counts)" %(old_exp, self.mmc.getExposure(), np.amax(img)))
        return(new_exp, img)
