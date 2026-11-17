"""On-disk dataset container (multi-page TIFF + JSON metadata).

Ported from comptic.containers / comptic.metadata / comptic.camera, keeping only
what htdeblur uses. Plotting, NA transforms, LED-position loaders and the
reconstruction browser were dropped.
"""

import copy
import datetime
import gc
import glob
import json
import os

import numpy as np
import tifffile
from . import _util as ut
from .roi import Roi

VERSION = 0.3


# ---------------------------------------------------------------------------
# TIFF I/O
# ---------------------------------------------------------------------------

def writeMultiPageTiff(file_name, data, compression=0, bit_depth=16):
    """Write a frame stack to a multi-page TIFF. Nonzero `compression` uses zlib."""
    if bit_depth == 16:
        data = np.uint16(data)
    elif bit_depth == 8:
        data = np.uint8(np.round(data / np.max(data) * 256.))
    else:
        raise ValueError('Invalid bit depth')
    tifffile.imwrite(file_name, data, photometric='minisblack',
                     compression='zlib' if compression else None)


def readMultiPageTiff(file_name, frame_subset=None):
    """Read a multi-page TIFF as a list of 2D frames."""
    with tifffile.TiffFile(str(file_name)) as tif:
        stack = tif.asarray(key=frame_subset)
    return [stack] if np.ndim(stack) != 3 else list(stack)


def isDataset(directory):
    """True if `directory` contains a .tif and a .json file."""
    file_list = os.listdir(directory)
    return any('.tif' in f for f in file_list) and any('.json' in f for f in file_list)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class StateList:
    def __init__(self):
        self.design = None
        self.calibrated = None
        self.grouping = None
        self.units = None


class HardwareElement:
    def __init__(self):
        self.device_name = None
        self.device_type = None
        self.state_list = StateList()


class Camera:
    def __init__(self):
        self.roi = Roi()
        self.transpose = False
        self.flip_x = False
        self.flip_y = False
        self.pixel_size_um = None           # physical pixel size (no magnification)
        self.is_color = None
        self.bayer_coupling_matrix = None
        self.device_name = None
        self.bit_depth = 16
        self.is_demosaiced = False


class Sample:
    def __init__(self):
        self.roi = Roi()
        self.name = None
        self.size_mm = None


class Objective:
    def __init__(self):
        self.mag = None
        self.na = None


class Spectrum:
    def __init__(self):
        self.center = None
        self.full = None
        self.units = "um"


class Illumination(HardwareElement):
    def __init__(self):
        super().__init__()
        self.z_distance_mm = None
        self.bit_depth = 16
        self.is_color = False
        self.spectrum = Spectrum()


class Pupil(HardwareElement):
    pass


class Position(HardwareElement):
    def __init__(self):
        super().__init__()
        self.velocity_mm_s = None
        self.acceleration_mm_s2 = None
        self.units = 'mm'


class Focus(HardwareElement):
    def __init__(self):
        super().__init__()
        self.velocity_mm_s = None
        self.acceleration_mm_s2 = None


class System:
    def __init__(self):
        self.mag = 1.
        self.eff_pixel_size_um = None
        self.image_flip_x = False
        self.image_flip_y = False


class Background:
    def __init__(self):
        self.max_thresh = 1000
        self.roi = Roi()
        self.is_subtracted = False


class GroundTruth:
    def __init__(self):
        self.object = None
        self.parameters = None


class Reconstruction:
    def __init__(self):
        self.object = None
        self.parameters = None


class Metadata:
    def __init__(self, from_dict=None):
        self.focus = Focus()
        self.pupil = Pupil()
        self.position = Position()
        self.illumination = Illumination()
        self.camera = Camera()
        self.objective = Objective()
        self.sample = Sample()
        self.system = System()
        self.background = Background()
        self.groundtruth = GroundTruth()
        self.reconstructions = []
        self.type = None
        self.comments = None
        self.file_header = "None"
        self.directory = "None"
        self.acquisition_time_s = None
        self.calibration = {}

        if from_dict is not None:
            loadDictRecursive(self, from_dict)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset():

    """A TIFF + JSON dataset on disk: frames, per-frame hardware states, and metadata."""

    def __init__(self, dataset_path=None,
                 dtype=None, backend=None,
                 divide_background=False,
                 subtract_mean_dark_current=False,
                 use_median_filter=False,
                 median_filter_threshold=1000,
                 force_type=None,
                 use_calibration=True):

        # Initialize metadata
        self.metadata = Metadata()

        # Store dataset path
        self.directory = dataset_path

        # Store dtype and backend
        self.dtype = dtype if dtype is not None else ut.DEFAULT_DTYPE
        self.backend = backend if backend is not None else 'numpy'

        # Initialize data
        self._frame_list = None
        self._frame_state_list = None
        self.frame_mask = None
        self.channel_mask = None

        # Background, containing artifacts due to system such as illumination
        # and dust on optical elements
        self.background = None
        # Decide whether we need to perform background removal
        self.divide_background = divide_background

        # Dark current, which captures the signal when there is no illumination
        self.dark_current = None
        self.subtract_mean_dark_current = subtract_mean_dark_current

        # Define version
        self.version = VERSION
        
        # Ground truth
        self.ground_truth = None

        # Median filter parameters
        self.median_filter_threshold = median_filter_threshold
        self.use_median_filter = use_median_filter

        # Initialize frame state container
        self.frame_state_container = None

        # Load dataset information if path is provided
        if self.directory is not None:

            # Load dataset metadata
            self.loadMetadata()

            # Force type if requested
            if force_type is not None:
                self.metadata.type = force_type

            # Load background and dark current
            self.loadBackground()

            # Load first frame
            self.loadFrames(frame_mask_override=[0])

            # Load calibration parameters if present in dataset directory
            if use_calibration:
                self.loadCalibration()

    def __getitem__(self, item):
        if self._frame_list[self.frame_mask[item]] is None:
            self.loadFrames(frame_mask_override=[self.frame_mask[item]])

        if item >= len(self.frame_mask):
            raise IndexError("Index out of range")

        # Subsample frame list
        if self.channel_mask is None:
            frame = self._frame_list[item]
        elif len(self.channel_mask) == 1:
            frame = self._frame_list[item][:, :, self.channel_mask[0]]
        else:
            frame = self._frame_list[item][:, :, self.channel_mask]

        return (frame, self.frame_state_list[item])

    def __len__(self):
        if self.frame_mask is not None:
            return(len(self.frame_mask))
        else:
            return len(self._frame_list)

    @property
    def frame_shape(self):
        """Returns dimensions of each frame in dataset."""
        if any([frame is not None for frame in self._frame_list]):
            frame_shape = list(np.shape([frame for frame in self._frame_list if frame is not None][0]))

            if self.channel_mask and len(self.channel_mask) == 1:
                frame_shape = frame_shape[:2]
        else:
            # Load first frame
            self.loadFrames(frame_mask_override=[0])

            # Get shape
            frame_shape = list(np.shape(self._frame_list[0]))

        # Check if channel_mask is used - if so, set the color (third) dimension to be the correct length
        if self.channel_mask is not None and len(self.channel_mask) > 1:
            # Set correct length
            frame_shape.append(len(self.channel_mask))

        return tuple(frame_shape)

    @property
    def frame_mask(self):
        """Returns a list of current frames which are to be used."""
        if self._frame_mask is not None:
            return self._frame_mask
        else:
            return self.frame_mask_full

    @frame_mask.setter
    def frame_mask(self, new_frame_mask):
        """Sets the current frame last. Input should be a list."""
        self._frame_mask = new_frame_mask

    @property
    def frame_mask_full(self):
        """Returns list of all possible frame numbers in the dataset."""
        return list(range(len(self._frame_list)))

    @property
    def shape(self):
        """Returns the shape of the dataset. First dimension is the number of elements in .frame_list, all other elements are the shape of each frame."""
        return tuple([len(self.frame_mask)] + list(self.frame_shape))

    @property
    def frame_count(self):
        """Returns the number of frames in thhe dataset."""
        return len(self)

    @property
    def frame_list(self):
        """Returns a list of frames in the dataset. Also loads each frame as required."""
        # Select subset of frame list if defined

        # Ensure all frames are populated
        if any([self._frame_list[index] is None for index in self.frame_mask]):
            self.loadFrames(frame_mask_override=[index for index in self.frame_mask if self._frame_list[index] is None])

        # Subsample frame list
        if self.channel_mask is None:
            return [self._frame_list[i] for i in self.frame_mask]
        elif len(self.channel_mask) == 1:
            if np.ndim(self._frame_list[0]) == 3:
                return [self._frame_list[i][:, :, self.channel_mask[0]] for i in self.frame_mask]
            else:
                return [self._frame_list[i] for i in self.frame_mask]
        else:
            return [self._frame_list[i][:, :, self.channel_mask] for i in self.frame_mask]

    @frame_list.setter
    def frame_list(self, new_frame_list):
        """Allows assigning frames in the dataset."""

        # Generate frame list if it doesn't already exist
        if not self._frame_list:
            self._frame_list = [None] * len(new_frame_list)
            self._frame_mask_full = list(range(len(new_frame_list)))

        # Assign new frames
        if self._frame_mask is not None:
            for index, frame_index in enumerate(self.frame_mask):
                self._frame_list[frame_index] = np.copy(new_frame_list[index])
        else:
            for i in range(len(new_frame_list)):
                self._frame_list[i] = np.copy(new_frame_list[i])

    @property
    def frame_state_list(self):
        """Returns a list of frame states, which is a free-form variable which is typically over-ridden."""
        if self.frame_mask is not None:
            return [self._frame_state_list[i] for i in self.frame_mask]
        else:
            return self._frame_state_list

    @frame_state_list.setter
    def frame_state_list(self, new_frame_state_list):
        """Returns a list of frame states, which is a free-form variable which is typically over-ridden."""
        if self._frame_state_list:
            if self.frame_mask is not None:
                for i in self.frame_mask:
                    self._frame_state_list[i] = copy.deepcopy(new_frame_state_list[i])
            else:
                for i in range(len(new_frame_state_list)):
                    self._frame_state_list[i] = copy.deepcopy(new_frame_state_list[i])
        else:
            self._frame_state_list = copy.deepcopy(new_frame_state_list)

    @property
    def frame_state_list_full(self):
        """Returns the full frame state list."""
        return self._frame_state_list

    @property
    def frame_list_full(self):
        """Returns the full frame list."""
        return self._frame_list

    def clearMask(self):
        """Clear frame mask (set to default)."""
        self.frame_mask = list(range(len(self._frame_list)))

    def save(self, dataset_path=None, header=None, tif_compression=0,
                   metadata_only=False, precision=6, bit_depth=16):
        """
        This is the workhorse save function for all data types. It will be modified as needed
        """
        json_dict = {}

        # convert structure to metadata
        json_dict['metadata'] = self.metadata.__dict__.copy()
        json_dict['metadata']['camera'] = self.metadata.camera.__dict__.copy()
        json_dict['metadata']['camera']['roi'] = self.metadata.camera.roi.to_dict()
        json_dict['metadata']['sample'] = self.metadata.sample.__dict__.copy()
        json_dict['metadata']['sample']['roi'] = self.metadata.sample.roi.to_dict()
        json_dict['metadata']['pupil'] = self.metadata.pupil.__dict__.copy()
        json_dict['metadata']['pupil']['state_list'] = self.metadata.pupil.state_list.__dict__.copy()
        json_dict['metadata']['focus'] = self.metadata.focus.__dict__.copy()
        json_dict['metadata']['focus']['state_list'] = self.metadata.focus.state_list.__dict__.copy()
        json_dict['metadata']['position'] = self.metadata.position.__dict__.copy()
        json_dict['metadata']['position']['state_list'] = self.metadata.position.state_list.__dict__.copy()
        json_dict['metadata']['illumination'] = self.metadata.illumination.__dict__.copy()
        json_dict['metadata']['illumination']['state_list'] = self.metadata.illumination.state_list.__dict__.copy()
        json_dict['metadata']['illumination']['spectrum'] = self.metadata.illumination.spectrum.__dict__.copy()
        json_dict['metadata']['objective'] = self.metadata.objective.__dict__.copy()
        json_dict['metadata']['system'] = self.metadata.system.__dict__.copy()
        json_dict['metadata']['background'] = self.metadata.background.__dict__.copy()
        json_dict['metadata']['background']['roi'] = self.metadata.background.roi.to_dict()
        json_dict['metadata']['groundtruth'] = self.metadata.groundtruth.__dict__.copy()

        # Clean up dictionary values
        ut.sanitize_dict(json_dict)

        # Initialize frame state list
        json_dict['frame_state_list'] = {}

        # Default file header is the date and time
        json_dict['metadata']['file_header'] = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Append dataset type if it exists
        if json_dict['metadata']['type'] is not None:
            json_dict['metadata']['file_header'] = json_dict['metadata']['type'] + \
                '_' + json_dict['metadata']['file_header']

        # Append user header if they provide one
        if header is not None:
            json_dict['metadata']['file_header'] = header + '_' + json_dict['metadata']['file_header']

        # Generate save path
        if dataset_path is None:
            out_dir = os.getcwd().replace('\\', '/') + '/' + json_dict['metadata']['file_header'] + '/'
        else:
            out_dir = dataset_path.replace('\\', '/') + json_dict['metadata']['file_header'] + '/'

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        json_dict['frame_state_list'] = applyPrecisionToFrameStateList(self.frame_state_list)

        # Write frame list
        if not metadata_only:
            if self.frame_list is not None:
                data_file_name = os.path.join(out_dir, json_dict['metadata']['file_header'] + '.tif')
                writeMultiPageTiff(data_file_name, self.frame_list, compression=tif_compression, bit_depth=bit_depth)
                print("Wrote tiff file to file: %s" % data_file_name)
            else:
                raise ValueError('ERROR - frame_list does not exist in dataset!')

            # Write background
            if self.background is not None:
                self.metadata.background.filename = 'background.tif'
                filename = os.path.join(out_dir, self.metadata.background.filename)
                writeMultiPageTiff(filename, [self.background], compression=tif_compression)
                print("Wrote background tiff file to file: %s" % filename)

            # Write dark current
            if self.dark_current is not None:
                self.metadata.background.dark_current_filename = 'dark_current.tif'
                filename = os.path.join(out_dir, self.metadata.background.dark_current_filename)
                writeMultiPageTiff(filename, [self.dark_current], compression=tif_compression)
                print("Wrote dark current tiff file to file: %s" % filename)

        # Determinbe file header
        if json_dict['metadata']['file_header'] is None:
            json_dict['metadata']['file_header'] = 'dataset'
        metadata_file_name = out_dir + json_dict['metadata']['file_header'] + '.json'

        # Sanatize json file
        ut.sanitize_dict(json_dict)

        # Assign directory in dataset object
        self.directory = out_dir

        # Write metadata to JSON
        with open(metadata_file_name, 'w+') as out_file:
            json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)
            json.dump(json_dict, out_file, indent=4)

        return json_dict

    def saveReconstruction(self, object, parameters, label='',
                           omit_time_in_filename=False):

        # Generate reconstruction label
        time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        reconstruction_filename = 'reconstruction'

        # Add label if one is provided
        if label is not None:
            reconstruction_filename += ('_' + label)

        # Add time if user indicates
        if not omit_time_in_filename:
            reconstruction_filename += ('_' + time_str)

        # Ensure reconstructions directory exists
        reconstruction_subdirectory = os.path.join(self.directory, 'reconstructions')
        if not os.path.exists(reconstruction_subdirectory):
            os.makedirs(reconstruction_subdirectory)

        # Generate output path
        output_path = os.path.join(reconstruction_subdirectory, reconstruction_filename)

        # Cast reconstruction as an 8-bit tiff_file
        max_value, min_value = ut.max(object), ut.min(object)
        reconstruction = 255.0 * (np.copy(object) - min_value) / (max_value if max_value != 0.0 else 1.0)
        reconstruction = ut.cast(reconstruction, dtype='uint8')

        # Write reconstruction
        writeMultiPageTiff(output_path + '.tif',
                           reconstruction, bit_depth=8, compression=0)

        # Add max_value and min_value to reconstruction parameters
        parameters['file_scale'] = {'min_value': min_value,
                                    'max_value': max_value}

        # Store label
        parameters['label'] = label

        # Sanatize parameters (convert to lists from numpy arrays)
        ut.sanitize_dict(parameters)

        # Write reconstruction parameters
        with open(output_path + '.json', 'w+') as fp:
            json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)
            json.dump(parameters, fp, indent=4)

        # Let the user know we saved the reconstruction
        print('Saved reconstruction to file header: %s' % output_path)

    def loadCalibration(self):
        """Load calibration information from disk."""
        # Check that this directory is a dataset
        assert isDataset(self.directory), "Dataset %s is not a valid Dataset!"

        # See if there are any tif files in directory
        if os.path.isfile(self.directory + '/calibration.json'):
            with open(self.directory + '/calibration.json') as data_file:
                self.metadata.calibration = json.load(data_file)

    def saveCalibration(self):
        """Save calibration information to disk."""
        # Check that this directory is a dataset
        assert isDataset(self.directory), "Dataset %s is not a valid Dataset!"

        # Assign dataset name inside calibration json
        self.metadata.calibration['dataset_header'] = self.metadata.file_header

        # Sanatize dictionary
        ut.sanitize_dict(self.metadata.calibration)

        # Write reconstruction parameters
        with open(self.directory + '/calibration.json', 'w+') as fp:
            json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)
            json.dump(self.metadata.calibration, fp, indent=4)

    def loadBackground(self):
        """Loads background and dark current images if found in the dataset directory."""

        # Check that this directory is a dataset
        assert isDataset(self.directory), "Dataset %s is not a valid Dataset!"

        # Use real background if available
        tif_list = []
        for file in glob.glob(self.directory + "/*.tif*"):
            tif_list.append(file)

        # Load background and dark current
        for tif_filename in tif_list:
            if 'background' in tif_filename or 'bg' in tif_filename:
                self.background = ut.cast(readMultiPageTiff(tif_filename)[0],
                                          dtype=self.dtype)
            elif 'dark_current' in tif_filename:
                self.dark_current = ut.cast(readMultiPageTiff(tif_filename)[0],
                                            dtype=self.dtype)

    def clearFramesFromMemory(self):
        """Erases all frames from memory."""
        for frame_index in range(len(self._frame_list)):
            self._frame_list[frame_index] = None

        gc.collect()

    def getSingleFrame(self, frame_index, do_demosaic=True):
        # Check that this directory is a dataset
        assert isDataset(self.directory), "Dataset %s is not a valid Dataset!"

        # See if there are any tif files in directory
        tif_list = []
        for file in glob.glob(self.directory + "/*.tif*"):
            tif_list.append(file)

        # Load frame from disk
        for tif_filename in tif_list:
            if 'background' not in tif_filename and 'dark_current' not in tif_filename:
                frame = ut.cast(np.squeeze(readMultiPageTiff(tif_filename, frame_subset=[frame_index])[0]),
                                dtype=self.dtype)

        # Remove background (if provided)
        if self.divide_background and self.background is not None:
            frame = removeBackground(frame,
                                     self.background,
                                     max_thresh=self.metadata.background.max_thresh)

        # Remove background (if provided)
        if self.subtract_mean_dark_current and self.dark_current is not None:
            frame = frame - ut.mean(self.dark_current)

        # Demosaic (if color camera)
        if self.metadata.camera.is_color and do_demosaic:
            frame = demosaic(frame, bayer_coupling_matrix=self.metadata.camera.bayer_coupling_matrix)

        # Subsample frame list
        if self.channel_mask is None:
            return frame
        elif len(self.channel_mask) == 1:
            return frame[:, :, self.channel_mask[0]]
        else:
            return frame[:, :, self.channel_mask]

    def loadFrames(self, frame_mask_override=None):
        """
        This function loads a set of frames as indicated by the frame_mask value.
        All other frames will be set to None until this function is told to load
        them.
        """

        # Check that this directory is a dataset
        assert isDataset(self.directory), "Dataset %s is not a valid Dataset!"

        # See if there are any tif files in directory
        tif_list = []
        for file in glob.glob(self.directory + "/*.tif*"):
            tif_list.append(file)

        # Load data from multi-page tiff
        frame_mask = self.frame_mask if frame_mask_override is None else frame_mask_override

        print('Loading %d frames...' % len(frame_mask))

        for tif_filename in tif_list:
            if 'background' not in tif_filename and 'bg' not in tif_filename and 'dark_current' not in tif_filename:
                updated_frame_list = readMultiPageTiff(tif_filename, frame_subset=frame_mask)

        # Update new frames
        for frame_index, new_frame in zip(frame_mask, updated_frame_list):
            self._frame_list[frame_index] = ut.cast(np.squeeze(new_frame), self.dtype)

        # Perform background subtraction and demosaicing if configured
        for frame_index in frame_mask:

            # Subtract dark current
            if self.subtract_mean_dark_current and self.dark_current is not None:
                self._frame_list[frame_index] -= self.dark_current
                self._frame_list[frame_index][self._frame_list[frame_index] < 0] = 0

            # Divide background
            if self.divide_background and self.background is not None:
                self._frame_list[frame_index] = removeBackground(self._frame_list[frame_index],
                                                                 self.background,
                                                                 max_thresh=self.metadata.background.max_thresh)

            # Demosaic
            if self.metadata.camera.is_color:
                self._frame_list[frame_index] = demosaic(self._frame_list[frame_index],
                                                         bayer_coupling_matrix=self.metadata.camera.bayer_coupling_matrix)

            # Median filter
            if ut.mean(self._frame_list[frame_index]) < self.median_filter_threshold and self.use_median_filter:
                self._frame_list[frame_index] = ut.median(self._frame_list[frame_index])

    def loadMetadata(self):
        """Load metadata and frame states from the dataset's JSON file."""

        # Generate empty dataset object to populate
        metadata = Metadata()
        tif_list = []
        json_list = []

        # Get list of files in directory
        tiff_file_list = glob.glob(os.path.join(self.directory, '*'))

        # See if there are any tif files in directory
        for file in tiff_file_list:
            if 'background' not in file and 'backup' not in file and 'dark_current' not in file:
                if '.tif' in file:
                    tif_list.append(file)

        # See if there are any tif files in directory
        for file in tiff_file_list:
            if 'calibration' not in file and 'backup' not in file:
                if '.json' in file:
                    json_list.append(file)

        assert len(tif_list) == 1, "Could not find tif file in directory %s (Found %d files)" % (self.directory, len(tif_list))
        assert len(json_list) == 1, "Could not find json file!"

        # Load Json file
        with open(json_list[0]) as data_file:
            json_file = json.load(data_file)

        def replaceRoiObjects(_dict):
            for key in _dict:
                if 'roi' in key:
                    _dict[key] = {'start': (0,0), 'shape': (0,0), 'units': 'pixels'}
                elif type(_dict[key]) == dict:
                    replaceRoiObjects(_dict[key])

        # Load metadata object
        if json_file['metadata'] is not None:
            replaceRoiObjects(json_file['metadata'])
            loadDictRecursive(metadata, json_file['metadata'])

        # Convert Roi
        convertRoiRecursive(metadata)

        # Get frame state_list
        frame_state_list = json_file['frame_state_list']

        # Set metadata in dataset
        self.metadata = metadata

        # Set frame_state_list in dataset
        self._frame_state_list = frame_state_list

        # Set frame_list to list of None values (for now)
        self._frame_list = [None] * len(frame_state_list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def loadDictRecursive(inner_attr, inner_dict):
    """Populate an object tree from a nested dict, creating unknown attributes."""
    for key_name, value in inner_dict.items():
        if hasattr(inner_attr, key_name) and type(value) is dict and getattr(inner_attr, key_name) is not None:
            loadDictRecursive(getattr(inner_attr, key_name), value)
        else:
            try:
                setattr(inner_attr, key_name, value)
            except AttributeError:
                print('Failed to set attribute %s' % key_name)


def convertRoiRecursive(metadata):
    """Replace serialized ROI dicts in an object tree with Roi objects."""
    for key_name in dir(metadata):
        if '__' in key_name:
            continue
        if key_name == 'roi':
            roi_dict = getattr(metadata, key_name)
            if type(roi_dict) is dict:
                if roi_dict['y_end'] is not None:
                    setattr(metadata, key_name, Roi(start=(roi_dict['y_start'], roi_dict['x_start']),
                                                    end=(roi_dict['y_end'], roi_dict['x_end'])))
                else:
                    setattr(metadata, key_name, None)
        elif hasattr(getattr(metadata, key_name), '__dict__'):
            convertRoiRecursive(getattr(metadata, key_name))


def applyPrecisionToFrameStateList(frame_state_list, precision=5):
    """Round floats in a frame state list (for compact JSON)."""
    filtered = copy.deepcopy(frame_state_list)
    for frame in filtered:
        for device_name in frame:
            if isinstance(frame[device_name], dict) and device_name != 'time_sequence_s':
                for key in frame[device_name]['common']:
                    if isinstance(frame[device_name]['common'][key], float):
                        frame[device_name]['common'][key] = round(frame[device_name]['common'][key], precision)
                for state in frame[device_name]['states']:
                    for state_t in state:
                        for key in state_t:
                            if isinstance(state_t[key], float):
                                state_t[key] = round(state_t[key], precision)
                            elif isinstance(state_t[key], dict):
                                for key2 in state_t[key]:
                                    if isinstance(state_t[key][key2], float):
                                        state_t[key][key2] = round(state_t[key][key2], precision)
    return filtered


def removeBackground(frame, background=None, max_thresh=None):
    """Divide by a background image, or by the frame mean (capped at max_thresh). Clips negatives."""
    if background is not None:
        frame /= background
    else:
        max_thresh = np.inf if max_thresh is None else max_thresh
        mean = ut.mean(frame)
        frame /= mean if mean < max_thresh else max_thresh
    frame[frame < 0.] = 0.
    return frame


def demosaic(frame, order='grbg', bayer_coupling_matrix=None, white_balance=False):
    """Demosaic a Bayer frame to (ny/2, nx/2, 3).

    With `bayer_coupling_matrix` (4x3), solves for RGB by least squares;
    otherwise uses the 2x2 pattern given by `order`.
    """
    dtype = ut.dtype_name(frame)
    out_shape = (int(np.shape(frame)[0] / 2), int(np.shape(frame)[1] / 2), 3)
    frame_out = ut.zeros(out_shape, dtype)

    if bayer_coupling_matrix is not None:
        frame_vec = ut.zeros((4, int(np.shape(frame)[0] * np.shape(frame)[1] / 4)), dtype)
        bayer_coupling_matrix = ut.cast(bayer_coupling_matrix, dtype)

        # Pattern index -> (row, col) offset
        offsets = {0: (1, 1), 1: (0, 1), 2: (1, 0), 3: (0, 0)}
        for index, (r, c) in offsets.items():
            frame_vec[index, :] = np.copy(ut.vec(frame[r::2, c::2]))

        result = np.linalg.lstsq(bayer_coupling_matrix, frame_vec, rcond=None)[0]
        result -= ut.min(result)
        result /= ut.max(result)
        for channel in range(3):
            frame_out[:, :, channel] = np.reshape(result[channel], out_shape[:2])
            if white_balance:
                frame_out[:, :, channel] -= ut.min(frame_out[:, :, channel])
                frame_out[:, :, channel] /= ut.max(frame_out[:, :, channel])
        return frame_out

    # Bayer 2x2 cell indices: 0 1 / 2 3
    b_index, r_index, g1_index = order.find('b'), order.find('r'), order.find('g')
    g2_index = ({0, 1, 2, 3} - {b_index, r_index, g1_index}).pop()

    def start(i):
        return int(i in [2, 3]), int(i in [1, 3])

    r0, g10, g20, b0 = start(r_index), start(g1_index), start(g2_index), start(b_index)
    frame_out[:, :, 0] = frame[r0[0]::2, r0[1]::2]
    frame_out[:, :, 1] = (frame[g10[0]::2, g10[1]::2] + frame[g20[0]::2, g20[1]::2]) / 2.0
    frame_out[:, :, 2] = frame[b0[0]::2, b0[1]::2]
    frame_out /= ut.max(frame_out)

    if white_balance:
        for channel in range(3):
            frame_out[:, :, channel] /= ut.max(frame_out[:, :, channel])

    return frame_out
