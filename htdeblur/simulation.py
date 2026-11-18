"""Simulated test objects. Ported from comptic.simulation (California object only)."""

import os

import numpy as np
import imageio.v3 as iio
from skimage.transform import resize
from . import _util as ut

_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')


def _load_image(filename, shape=None, dtype=None, backend=None, color_channel='average',
                reshape_mode='constant', anti_aliasing=False):
    backend = backend if backend is not None else 'numpy'
    dtype = dtype if dtype is not None else ut.DEFAULT_DTYPE

    image = np.asarray(iio.imread(os.path.join(_RESOURCES, filename)))

    if image.ndim > 2:
        if color_channel == 'average':
            image = np.mean(image, 2)
        elif color_channel is not None:
            image = image[:, :, int(color_channel)]

    if shape is not None:
        if any(image.shape[i] < shape[i] for i in range(len(shape))):
            print('WARNING : Raw image size (%d x %d) is smaller than requested size (%d x %d).'
                  % (image.shape[0], image.shape[1], shape[0], shape[1]))
        image = resize(image, shape, mode=reshape_mode, preserve_range=True,
                       anti_aliasing=anti_aliasing).astype(float)

    return ut.cast(image, dtype)


def california(shape=None, invert=False, min_value=0.9, max_value=1.1, dtype=None, backend=None, **kwargs):
    """California-outline amplitude object scaled to [min_value, max_value]."""
    obj = _load_image('california.png', shape, dtype, backend, **kwargs)
    obj -= ut.min(obj)
    obj /= ut.max(obj)
    if invert:
        obj = 1 - obj
    obj *= (max_value - min_value)
    obj += min_value
    return ut.cast(obj, dtype if dtype is not None else ut.DEFAULT_DTYPE)
