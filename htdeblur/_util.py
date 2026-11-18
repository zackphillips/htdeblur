"""Array helpers that replace the llops calls htdeblur relied on.

Where numpy has a direct equivalent, htdeblur calls numpy. The functions here
keep llops behavior that numpy does not provide directly (centered FFTs,
center/edge/mean padding, dtype names such as 'complex32', keepdims sums,
float32 scalars), so results match the original code.

llops dtype names: 'complex32' is numpy complex64 and 'complex64' is numpy
complex128 (the number counts bits per component).
"""

import builtins
import math

import numpy as np
import scipy.fftpack
import skimage.filters
import skimage.transform

DEFAULT_DTYPE = 'float32'

# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------

_NAME_TO_NATIVE = {'complex32': np.complex64, 'complex64': np.complex128}


def native_dtype(name=None):
    """numpy dtype for an llops dtype name ('float32', 'complex32', ...)."""
    name = DEFAULT_DTYPE if name is None else name
    if not isinstance(name, str):
        return np.dtype(name)
    return np.dtype(_NAME_TO_NATIVE.get(name, name))


def dtype_name(x):
    """llops dtype name of an array (complex64 -> 'complex32')."""
    d = np.asarray(x).dtype
    if d == np.complex64:
        return 'complex32'
    if d == np.complex128:
        return 'complex64'
    return str(d)


def astype(x, dtype=None):
    """Cast to an llops dtype name; complex -> real takes the real part. No-op if unchanged."""
    dtype = DEFAULT_DTYPE if dtype is None else dtype
    x = np.asarray(x) if isinstance(x, (list, tuple)) else x
    if dtype_name(x) == dtype:
        return x
    if 'complex' in dtype_name(x) and 'complex' not in dtype:
        x = np.real(x)
    return np.asarray(x).astype(native_dtype(dtype))


def cast(x, dtype=None):
    """Convert to a numpy array of an llops dtype name (default float32)."""
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return astype(x, dtype)


def asarray(x, dtype=None):
    """llops.asarray: like cast, but complex input stays complex64."""
    if 'complex' in dtype_name(np.asarray(x)):
        dtype = 'complex32'
    return astype(np.asarray(x), dtype)


def zeros(shape, dtype=None):
    shape = shape if isinstance(shape, (list, tuple, np.ndarray)) else [shape]
    return np.zeros(tuple(int(s) for s in shape), native_dtype(dtype))


def ones(shape, dtype=None):
    shape = shape if isinstance(shape, (list, tuple, np.ndarray)) else [shape]
    return np.ones(tuple(int(s) for s in shape), native_dtype(dtype))


# ---------------------------------------------------------------------------
# Reductions with llops semantics
# ---------------------------------------------------------------------------

def scalar(x):
    """First element as a Python complex, or a float32 for real input (llops.scalar)."""
    if not hasattr(x, '__len__'):
        return x
    a = np.asarray(x)
    if np.iscomplexobj(a):
        return complex(a.item(0))
    return np.float32(a.item(0))


def sum(x, axis=None):
    """Sum; with `axis`, keeps reduced dimensions (llops.sum)."""
    if isinstance(x, list):
        return builtins.sum(x)
    if axis is None:
        return np.sum(x)
    axis = axis if isinstance(axis, (list, tuple, np.ndarray)) else [axis]
    return np.sum(x, axis=tuple(axis), keepdims=True)


def mean(x, axis=None):
    """Mean; without `axis` returns llops.scalar (float32 for real data)."""
    if axis is None:
        return scalar(sum(x) / size(x))
    return sum(x, axis=axis) / np.shape(np.asarray(x))[axis]


def size(x):
    """Element count, as np.prod of the shape (llops.size)."""
    return np.prod(np.shape(x)) if hasattr(x, '__len__') else np.prod((1,))


def max(x, axis=None):
    """Max of the real part."""
    if isinstance(x, (list, tuple)):
        return builtins.max(x)
    return np.max(np.real(x), axis=axis)


def min(x, axis=None):
    """Min of the real part."""
    if isinstance(x, (list, tuple)):
        return builtins.min(x)
    return np.min(np.real(x), axis=axis)


def argmax(x):
    """Coordinates of the max of the real part, as a tuple."""
    x = np.asarray(x)
    return tuple(np.unravel_index(np.argmax(np.real(x)), x.shape))


def argmin(x):
    x = np.asarray(x)
    return tuple(np.unravel_index(np.argmin(np.real(x)), x.shape))


def unique(x):
    """Unique values; order-preserving for lists/tuples, sorted for arrays (llops.unique)."""
    if isinstance(x, (list, tuple)):
        out = []
        for e in x:
            if e not in out:
                out.append(e)
        return out
    return np.unique(x)


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------

def expand_dims(x, ndim):
    """Append trailing axes until x has `ndim` dimensions."""
    while np.ndim(x) < ndim:
        x = x[:, np.newaxis]
    return x


def vec(x):
    """Squeeze, then flatten (llops.vec)."""
    x = np.squeeze(x)
    return x if x.ndim == 1 else x.ravel()


def flip(x, axis=None):
    """Flip along `axis`; with axis=None flips every axis (llops.flip)."""
    axes = list(range(np.ndim(x))) if axis is None else (axis if isinstance(axis, list) else [axis])
    for ax in axes:
        x = np.flip(x, ax)
    return x


def roll(x, shift):
    """Roll by a per-axis shift list (llops.roll with a sequence)."""
    for ax, sh in enumerate(shift):
        x = np.roll(x, int(np.real(sh)), ax)
    return x


def pad(x, M, crop_start=(0, 0), pad_value=0, center=False):
    """Place x into an array of shape M (llops.pad).

    center=True places x at ceil(M/2) - ceil(N/2). Out-of-range placement is
    clipped. pad_value: a number, or 'mean', 'median', 'maximum', 'minimum', 'edge'.
    """
    N = list(np.shape(x))
    M = list(M) if isinstance(M, (list, tuple, np.ndarray)) else [M]
    if len(N) != len(M):
        N = N + [1] * (len(M) - len(N))
        x = np.reshape(x, N)

    if center:
        crop_start = [int(math.ceil(m / 2) - math.ceil(n / 2)) for (m, n) in zip(M, N)]

    input_roi = [slice(0, n) for n in N]
    for i in range(len(M)):
        if crop_start[i] < 0:
            input_roi[i] = slice(int(-crop_start[i]), int(N[i]))
        elif crop_start[i] + N[i] > M[i]:
            input_roi[i] = slice(0, int(M[i] - crop_start[i]))
    input_roi = tuple(input_roi)

    y = np.ones([int(m) for m in M], dtype=np.asarray(x).dtype)

    if isinstance(pad_value, str):
        if pad_value == 'mean':
            y.fill(mean(x))
        elif pad_value == 'median':
            y.fill(np.median(x))
        elif pad_value == 'maximum':
            y.fill(max(x))
        elif pad_value == 'minimum':
            y.fill(min(x))
        elif pad_value == 'edge':
            _pad_edge(x, y, N, M, crop_start)
        else:
            raise ValueError('Invalid pad_value (%s)' % pad_value)
    elif np.isscalar(pad_value):
        y.fill(pad_value)
    else:
        y[:] = pad_value

    output_roi = tuple(slice(int(builtins.max(crop_start[i], 0)), int(builtins.min(crop_start[i] + N[i], M[i])))
                       for i in range(len(N)))
    y[output_roi] = x[input_roi]
    return y


def _pad_edge(x, y, N, M, crop_start):
    """Fill the 8 regions around the placed array with replicated edges (2D)."""
    shifts = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    shifts.remove((0, 0))
    for shift_0 in shifts:
        shift = [-s for s in shift_0]
        shift_amount = [shift[i] * N[i] for i in range(len(N))]
        padded_start = [builtins.max(shift_amount[i] + crop_start[i], 0) for i in range(len(N))]
        padded_end = [builtins.min(shift_amount[i] + crop_start[i] + N[i], M[i]) for i in range(len(N))]
        slc_padded = tuple(slice(padded_start[i], padded_end[i]) for i in range(len(N)))
        slc_input = tuple(slice(0 if shift[a] <= 0 else N[a] - 1, N[a] if shift[a] >= 0 else 1) for a in range(len(N)))
        if all(ps != pe for (ps, pe) in zip(padded_start, padded_end)):
            edge = x[slc_input]
            for a in range(len(N)):
                if edge.shape[a] == 1:
                    reps = [1] * len(N)
                    reps[a] = abs(padded_start[a] - padded_end[a])
                    edge = np.tile(edge, reps)
            if all(s > 0 for s in edge.shape):
                y[slc_padded] = edge


def crop(x, M, crop_start=(0, 0), out_of_bounds_placeholder=None, center=False):
    """Crop x to shape M starting at crop_start; out-of-range pixels get the placeholder (NaN by default)."""
    N = np.shape(x)
    if center:
        crop_start = [int(math.floor(n / 2 - m / 2)) for (m, n) in zip(M, N)]
    output_roi = [slice(0, m) for m in M]
    for i in range(len(N)):
        if crop_start[i] < 0:
            output_roi[i] = slice(int(-crop_start[i]), int(M[i]))
        elif crop_start[i] + M[i] > N[i]:
            output_roi[i] = slice(0, int(N[i] - crop_start[i]))
    input_roi = tuple(slice(int(builtins.max(crop_start[i], 0)), int(builtins.min(crop_start[i] + M[i], N[i])))
                      for i in range(len(N)))
    fill = np.nan if out_of_bounds_placeholder is None else out_of_bounds_placeholder
    y = np.full([int(m) for m in M], fill, dtype=np.asarray(x).dtype)
    y[tuple(output_roi)] = x[input_roi]
    return y


def grid(shape, scale=1, dtype=None):
    """Centered coordinate grids (y, x), zero at index shape // 2 (llops.grid, 2D)."""
    scale = list(scale) if isinstance(scale, (list, tuple, np.ndarray)) else [scale] * len(shape)
    d = native_dtype(dtype)
    lin_y = (np.arange(shape[0], dtype=d) - shape[0] // 2) * scale[0]
    lin_x = (np.arange(shape[1], dtype=d) - shape[1] // 2) * scale[1]
    grid_y = (lin_y[:, np.newaxis] * np.ones_like(lin_x)[np.newaxis, :]).astype(d)
    grid_x = (lin_x[np.newaxis, :] * np.ones_like(lin_y)[:, np.newaxis]).astype(d)
    return grid_y, grid_x


def ramp(shape, axis=0, min_value=0.0, max_value=1.0, reverse=False, dtype=None):
    """Linear ramp along `axis` (llops.ramp)."""
    if max_value == min_value:
        return ones(shape, dtype)
    if reverse:
        delta = builtins.round((min_value - max_value) / shape[axis], 8)
        ramp_1d = slice(max_value, min_value + delta, delta)
    else:
        delta = builtins.round((max_value - min_value) / shape[axis], 8)
        ramp_1d = slice(min_value, max_value - delta, delta)
    coordinates = [slice(0, sz) for sz in shape]
    coordinates[axis] = ramp_1d
    r = resize(np.mgrid[tuple(coordinates)][axis], shape)
    return asarray(r, dtype)


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------

def next_fast_len(n):
    return scipy.fftpack.next_fast_len(int(n))


def next_fast_even_number(val):
    """Smallest value >= val that is both FFT-friendly and even."""
    v = val
    while scipy.fftpack.next_fast_len(v) != int(math.ceil(v / 2) * 2):
        v = scipy.fftpack.next_fast_len(v + 1)
    return v


def Ft(x, axes=None, center=True):
    """Unnormalized centered 2D FFT: fftshift(fft2(fftshift(x))) (llops convention).

    With center=False, `axes` is ignored and the transform runs over the last two axes,
    as in llops.
    """
    if center:
        return scipy.fftpack.fftshift(scipy.fftpack.fft2(scipy.fftpack.fftshift(x, axes=axes), axes=_fft_axes(x, axes)), axes=axes)
    return scipy.fftpack.fft2(x)


def iFt(x, axes=None, center=True):
    """Inverse of Ft: ifftshift(ifft2(ifftshift(x)))."""
    if center:
        return scipy.fftpack.ifftshift(scipy.fftpack.ifft2(scipy.fftpack.ifftshift(x, axes=axes), axes=_fft_axes(x, axes)), axes=axes)
    return scipy.fftpack.ifft2(x)


def _fft_axes(x, axes):
    if axes is None:
        return tuple(range(np.ndim(x)))[-2:]
    return axes if isinstance(axes, (list, tuple)) else (axes,)


def _linear_conv_setup(input_shape, h):
    """Padded-convolution geometry from llops.fft.conv_functions (mode='same', padded, FFT-sized)."""
    support = np.where(np.abs(flip(h)) > 0)
    kernel_size = [int(np.max(ix) - np.min(ix) + 1) for ix in support]
    crop_start = [(k // 2 * 2) // 2 for k in kernel_size]
    conv_shape = [scipy.fftpack.next_fast_len(int(n + k)) for (n, k) in zip(input_shape, kernel_size)]
    kernel_f = Ft(pad(h, conv_shape, center=True))
    return conv_shape, crop_start, kernel_f


def _linear_conv_apply(x, conv_shape, crop_start, kernel_f_mult, dtype, pad_value):
    xp = pad(x, conv_shape, crop_start=crop_start, pad_value=pad_value)
    out = iFt(Ft(xp) * kernel_f_mult)
    if not np.issubdtype(dtype, np.complexfloating):
        out = np.real(out)
    return crop(out.astype(dtype), np.shape(x), crop_start=crop_start)


def convolve(x, h, pad_value='edge'):
    """Linear convolution, output the same size as x, with edge-padded boundaries (llops.convolve).

    h is a centered kernel (any size up to the padded size); its nonzero support sets the padding.
    """
    conv_shape, crop_start, kernel_f = _linear_conv_setup(np.shape(x), h)
    return _linear_conv_apply(x, conv_shape, crop_start, kernel_f, np.asarray(h).dtype, pad_value)


def deconvolve(x, h, reg=0, pad_value='edge'):
    """Tikhonov-regularized inverse of `convolve`: conj(H) / (|H|^2 + reg) (llops.deconvolve)."""
    conv_shape, crop_start, kernel_f = _linear_conv_setup(np.shape(x), h)
    inv = np.conj(kernel_f) / (np.abs(kernel_f) ** 2 + (reg if reg is not None else 0))
    return _linear_conv_apply(x, conv_shape, crop_start, inv, np.asarray(h).dtype, pad_value)


def convolve_circular(x, h):
    """Circular convolution with a centered kernel of the same shape (llops Convolution, mode='circular')."""
    out = iFt(Ft(x) * Ft(h))
    return np.real(out).astype(np.asarray(x).dtype) if not np.iscomplexobj(x) else out.astype(np.asarray(x).dtype)


# ---------------------------------------------------------------------------
# Filters / resampling
# ---------------------------------------------------------------------------

def resize(x, new_size):
    return skimage.transform.resize(x, new_size, anti_aliasing=False, preserve_range=True)


def downsample(x, factor=2):
    if isinstance(factor, (list, tuple)):
        new_size = tuple(sh // f for (sh, f) in zip(np.shape(x), factor))
    else:
        new_size = tuple(sh // factor for sh in np.shape(x))
    return resize(x, new_size)


decimate = downsample


def upsample(x, factor=2):
    return resize(x, tuple(int(sh * factor) for sh in np.shape(x)))


def bin(x, factor):
    """Mean over non-overlapping factor x factor blocks (llops.filter.bin), computed in float."""
    shape = [sh // factor for sh in np.shape(x)[:2]]
    x = np.asarray(x)[:shape[0] * factor, :shape[1] * factor]
    acc = np.float64 if x.dtype == np.float64 else np.float32
    # Reduce over a reshaped view so large mosaics are not copied to float first
    return x.reshape(shape[0], factor, shape[1], factor).mean(axis=(1, 3), dtype=acc)


def gaussian(x, sigma=10):
    """skimage gaussian filter on float32, real and imaginary parts separately."""
    if np.iscomplexobj(x):
        re = skimage.filters.gaussian(np.real(x).astype(np.float32), sigma=sigma)
        im = skimage.filters.gaussian(np.imag(x).astype(np.float32), sigma=sigma)
        return re.astype(x.dtype) + 1j * im.astype(x.dtype)
    return skimage.filters.gaussian(np.real(x).astype(np.float32), sigma=sigma)


def normalize01(x):
    """(|x| - min|x|) / (max|x| - min|x|)."""
    a = np.abs(x)
    return (a - a.min()) / (a.max() - a.min())


def median(x, diameter=3):
    """Median filter with a disk footprint, run on a uint16-scaled copy (llops integer_function)."""
    from skimage.morphology import disk
    original = np.asarray(x).dtype
    lo, hi = min(x), max(x)
    xi = (65535.0 * (x - lo) / hi).astype(np.uint16)
    out = skimage.filters.median(xi, footprint=disk(diameter))
    return (out / 65535.0 * hi + lo).astype(original)


def soft_threshold(x, alpha):
    """exp(i angle(x)) * max(|x| - alpha, 0); real input returns real output."""
    if not np.iscomplexobj(x):
        xc = np.asarray(x).astype(np.complex64)
        return np.real(np.exp(1j * np.angle(xc)) * (np.abs(xc) - alpha) * (np.abs(xc) > alpha))
    return np.exp(1j * np.angle(x)) * (np.abs(x) - alpha) * (np.abs(x) > alpha)


# ---------------------------------------------------------------------------
# Serialization / display
# ---------------------------------------------------------------------------

def sanitize_dict(d):
    """In place: convert numpy arrays and scalars in a nested dict to JSON-friendly Python types."""
    if type(d) is not dict:
        return
    for key in d:
        if '__' in key:
            continue
        v = d[key]
        if type(v) in (list, tuple, np.ndarray):
            d[key] = _sanitize_iterable(v)
        elif type(v) is dict:
            sanitize_dict(v)
        elif np.isscalar(v):
            d[key] = _sanitize_scalar(v)


def _sanitize_iterable(it):
    if type(it) is tuple:
        it = list(it)
    elif type(it) is np.ndarray:
        it = it.tolist()
    for i, item in enumerate(it):
        if type(item) in (list, tuple):
            it[i] = _sanitize_iterable(item)
        elif type(item) is dict:
            sanitize_dict(item)
        elif np.isscalar(item):
            it[i] = _sanitize_scalar(item)
    return it


def _sanitize_scalar(s):
    a = np.asarray(s)
    if np.iscomplexobj(a) and np.imag(a) != 0:
        return {'real': np.real(a).item(), 'imag': np.imag(a).item()}
    return np.real(a).item()


def list_plot_flat(arrays, figsize=None, clim=None, colorbar=False, **kwargs):
    """Show a list of 2D arrays side by side."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(arrays), figsize=figsize, squeeze=False)
    for ax, a in zip(axes[0], arrays):
        im = ax.imshow(np.abs(np.asarray(a)), cmap='gray', clim=clim)
        ax.axis('off')
        if colorbar:
            fig.colorbar(im, ax=ax)
    return fig
