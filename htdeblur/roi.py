"""Region-of-interest class. Ported from llops.roi.

Operators:
    roi + offset / roi - offset : shift start
    roi + roi                   : bounding union
    roi & roi                   : intersection
    outer * inner               : inner ROI expressed in outer's coordinates
"""

import copy

import numpy as np


class Roi:
    def __init__(self, shape=None, start=None, end=None, input_shape=None, units='pixels'):
        assert not (shape and end), 'shape and end cannot both be provided.'
        self._start, self._shape = None, None
        if end is not None:
            self.start = start
            self.end = end
        elif shape is not None:
            self.shape = list(shape)
            self.start = start if start is not None else tuple([0] * len(shape))
            assert len(self.start) == len(self.shape)
        self.input_shape = tuple(input_shape) if input_shape is not None else None
        self.units = units

    # -- geometry ------------------------------------------------------------

    @property
    def ndim(self):
        return len(self.shape) if self._shape else 0

    @property
    def shape(self):
        if self._shape:
            return tuple(int(sh) for sh in self._shape)

    @shape.setter
    def shape(self, new_shape):
        if new_shape is not None:
            self._shape = list(new_shape)

    size = shape

    @property
    def start(self):
        if self._start:
            return tuple(int(st) for st in self._start)

    @start.setter
    def start(self, new_start):
        if new_start is not None:
            self._start = list(new_start)

    @property
    def end(self):
        return tuple(int(st + sh) for (st, sh) in zip(self._start, self._shape))

    @end.setter
    def end(self, new_end):
        self._shape = [int(abs(end - start)) for (start, end) in zip(self.start, new_end)]

    @property
    def center(self):
        return tuple(int(st + sh // 2) for (st, sh) in zip(self._start, self._shape))

    @property
    def slice(self):
        """Slices into an array of input_shape (clipped to it when known)."""
        input_shape = self.input_shape if self.input_shape is not None else [np.inf] * self.ndim
        return tuple(slice(int(max(st, 0)), int(min(st + sh, n)))
                     for (st, sh, n) in zip(self._start, self._shape, input_shape))

    @property
    def mask(self):
        m = np.zeros(self.input_shape)
        m[self.slice] = 1
        return m

    @property
    def valid(self):
        """Part of this ROI inside input_shape."""
        if self.input_shape is None:
            return self
        return Roi(start=[int(max(st, 0)) for st in self._start],
                   end=[int(min(st + sh, n)) for (st, sh, n) in zip(self._start, self._shape, self.input_shape)],
                   input_shape=self.input_shape)

    def overlaps(self, other):
        return (all(s < e for (s, e) in zip(self.start, other.end)) and
                all(s < e for (s, e) in zip(other.start, self.end)))

    def decimate(self, factor=2):
        return Roi(start=[st // factor for st in self.start],
                   shape=[sh // factor for sh in self.shape],
                   input_shape=[n // factor for n in self.input_shape] if self.input_shape is not None else None,
                   units=self.units)

    def copy(self):
        return self.__deepcopy__(None)

    def draw(self, ax=None, c='r', linewidth=2, **kwargs):
        """Outline this ROI on a matplotlib axis."""
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax
        rect = mpatches.Rectangle(list(reversed(self.start)), self.shape[1], self.shape[0],
                                  edgecolor=c, facecolor=None, linewidth=linewidth, fill=False, **kwargs)
        ax.add_artist(rect)

    # -- arithmetic ----------------------------------------------------------

    def _shifted_start(self, val, sign):
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 2:
                return tuple(st + sign * off for (st, off) in zip(self._start, val))
            return None  # llops ignored other lengths
        return tuple(st + sign * val for st in self._start)

    def __add__(self, val):
        new = copy.deepcopy(self)
        if isinstance(val, Roi):
            new.start = tuple(min(a, b) for (a, b) in zip(self.start, val.start))
            new.end = tuple(max(a, b) for (a, b) in zip(self.end, val.end))
        else:
            start = self._shifted_start(val, +1)
            if start is not None:
                new.start = start
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, val):
        if isinstance(val, Roi):
            self.start = tuple(min(a, b) for (a, b) in zip(self.start, val.start))
            self.end = tuple(max(a, b) for (a, b) in zip(self.end, val.end))
        else:
            start = self._shifted_start(val, +1)
            if start is not None:
                self.start = start
        return self

    def __sub__(self, val):
        if isinstance(val, Roi):
            raise ValueError('Subtraction is not supported for Roi objects inputs.')
        new = copy.deepcopy(self)
        start = self._shifted_start(val, -1)
        if start is not None:
            new.start = start
        return new

    def __isub__(self, val):
        if isinstance(val, Roi):
            raise ValueError('Subtraction is not supported for Roi objects inputs.')
        start = self._shifted_start(val, -1)
        if start is not None:
            self.start = start
        return self

    def __mul__(self, inner):
        """`inner` (relative to this ROI) mapped into this ROI's coordinates."""
        if not isinstance(inner, Roi):
            raise ValueError('Multiplication is only supported for two roi objects')
        if inner.input_shape is not None and max(inner.input_shape) != np.inf:
            assert all(n == sh for (n, sh) in zip(inner.input_shape, self.shape))
        new = copy.deepcopy(inner)
        new.start = tuple(a + b for (a, b) in zip(self.start, inner.start))
        new.end = tuple(st + sh for (st, sh) in zip(new.start, inner.shape))
        new.input_shape = self.input_shape
        return new

    def __and__(self, val):
        if not isinstance(val, Roi):
            raise ValueError('Intersection is only supported for two roi objects')
        new = copy.deepcopy(self)
        new.start = tuple(max(a, b) for (a, b) in zip(self.start, val.start))
        new.end = tuple(min(a, b) for (a, b) in zip(self.end, val.end))
        return new

    def __deepcopy__(self, memo):
        return Roi(start=copy.deepcopy(self.start), shape=copy.deepcopy(self.shape),
                   input_shape=copy.deepcopy(self.input_shape), units=self.units)

    def __repr__(self):
        if not self._start:
            return 'Roi(Empty)'
        r = 'Roi: start=%s, end=%s, shape=%s' % (self.start, self.end, self.shape)
        if self.input_shape is not None:
            r += ', input_shape: %s' % (self.input_shape,)
        return r

    __str__ = __repr__

    def to_dict(self):
        return {'start': self.start, 'shape': self.shape, 'units': self.units, 'input_shape': self.input_shape}


def get_overlapping_region(image_pair, roi_pair):
    """Crops of two images to the overlap of their ROIs."""
    image_0, image_1 = image_pair
    roi_0, roi_1 = roi_pair
    overlap = roi_0 & roi_1
    return image_0[(overlap - roi_0.start).slice], image_1[(overlap - roi_1.start).slice]


def crop_roi(x, roi, out_of_bounds_placeholder=None):
    from ._util import crop
    return crop(x, roi.shape, crop_start=roi.start, out_of_bounds_placeholder=out_of_bounds_placeholder)


def pad_roi(x, roi, pad_value=0):
    from ._util import pad
    return pad(x, roi.input_shape, crop_start=roi.start, pad_value=pad_value)
