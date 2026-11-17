"""Linear operators, objectives and FISTA for the multi-frame forward model.

A trimmed numpy port of the llops operator/solver stack, limited to what
htdeblur.recon and htdeblur.recon.normalize_measurements build:

    Crop, Convolution (circular), Diagonalize, FourierTransform, DiagStack,
    Segmentation, operator products, an L2 data term with L2 / L1 / TV / wavelet
    regularizers, and FISTA.

Arrays are 2D. A "stacked" measurement is the frames concatenated along axis 0
(see vector_stack). Numerical conventions follow llops so results match the
original code, including two quirks that are kept deliberately:

* L2 terms report 0.5 * ||r||^2 as their value but 2 * A^H r as their gradient.
* Nesterov momentum uses t_k = 1 + sqrt(1 + 4 t_{k-1}^2) / 2, and the gradient
  step updates the iterate in place, so for smooth-only objectives the momentum
  term cancels. Step sizes tuned against the original code carry over.
"""

import math
import random

import numpy as np
import scipy.signal

from . import _util as ut
from .roi import Roi, crop_roi, pad_roi


# ---------------------------------------------------------------------------
# Operator base
# ---------------------------------------------------------------------------

class Operator:
    """Base linear operator: maps arrays of shape N to shape M."""

    M = None
    N = None
    dtype = ut.DEFAULT_DTYPE

    def forward(self, x):
        raise NotImplementedError

    def adjoint(self, y):
        raise NotImplementedError

    def inverse(self, y, reg=None):
        raise NotImplementedError('%s is not invertible' % type(self).__name__)

    def _alloc(self, shape):
        # Fortran order as in llops: float32 sums/norms accumulate in memory order,
        # and FISTA's restart test is sensitive to that rounding.
        return np.zeros(shape, ut.native_dtype(self.dtype), order='F')

    def _out(self, value, shape):
        y = self._alloc(shape)
        if np.iscomplexobj(value) and not np.iscomplexobj(y):
            value = np.real(value)  # llops assigned complex results into real buffers
        y[:] = value
        return y

    def __mul__(self, other):
        if isinstance(other, Operator):
            return Product(self, other)
        return self.forward(other)

    @property
    def H(self):
        return Adjoint(self)

    @property
    def inv(self):
        return _Inverse(self)


class _Inverse:
    def __init__(self, op):
        self.op = op

    def __mul__(self, y):
        return self.op.inverse(y)


class Adjoint(Operator):
    def __init__(self, op):
        self.op, self.M, self.N, self.dtype = op, op.N, op.M, op.dtype

    def forward(self, x):
        return self.op.adjoint(x)

    def adjoint(self, y):
        return self.op.forward(y)


class Product(Operator):
    """A * B."""

    def __init__(self, A, B):
        if np.prod(A.N) != np.prod(B.M):
            raise ValueError('Cannot compose %s with %s: shape mismatch' % (A, B))
        self.A, self.B = A, B
        self.M, self.N, self.dtype = A.M, B.N, A.dtype

    def forward(self, x):
        return self.A.forward(self.B.forward(x))

    def adjoint(self, y):
        return self.B.adjoint(self.A.adjoint(y))

    def inverse(self, y, reg=None):
        return self.B.inverse(self.A.inverse(y, reg), reg)


# ---------------------------------------------------------------------------
# Concrete operators
# ---------------------------------------------------------------------------

class Crop(Operator):
    """Crop from shape N to the ROI; the adjoint pads with `pad_value` (number or 'mean').

    With pad_value='mean' the "adjoint" is affine, as in llops.
    """

    def __init__(self, N=None, M=None, roi=None, crop_start=None, pad_value=0, center=None,
                 out_of_bounds_placeholder=None, dtype=None):
        self.dtype = dtype if dtype is not None else ut.DEFAULT_DTYPE
        assert N is not None or roi is not None, 'Either N or a ROI must be provided!'
        assert not (M is not None and roi is not None), 'Both M and a ROI cannot be provided!'
        if roi is not None:
            self.roi = roi
            if crop_start is not None:
                self.roi.start = crop_start
            if N is not None:
                self.roi.input_shape = N
        else:
            self.roi = Roi(start=(0, 0) if crop_start is None else crop_start, shape=M, input_shape=N)
        self.pad_value = pad_value
        self.out_of_bounds_placeholder = out_of_bounds_placeholder
        self.M, self.N = tuple(self.roi.shape), tuple(self.roi.input_shape)
        if center is not None:
            self.roi.start = [int((n - m) / 2) for (m, n) in zip(M, N)]

    def forward(self, x):
        return self._out(crop_roi(np.reshape(x, self.N), self.roi, self.out_of_bounds_placeholder), self.M)

    def adjoint(self, y):
        return self._out(pad_roi(np.reshape(y, self.M), self.roi, pad_value=self.pad_value), self.N)


class Convolution(Operator):
    """Circular convolution with a centered kernel the same shape as the input."""

    def __init__(self, kernel, mode='circular', pad_value=0, dtype=None):
        if mode != 'circular':
            raise ValueError("Only mode='circular' is supported (got %r)" % mode)
        self.dtype = dtype if dtype is not None else ut.DEFAULT_DTYPE
        self.kernel = ut.cast(kernel, self.dtype).copy()
        self.M = self.N = tuple(np.shape(self.kernel))
        self.kernel_f = ut.Ft(self.kernel)
        self.pad_value = pad_value

    def forward(self, x):
        return self._out(ut.iFt(ut.Ft(x) * self.kernel_f), self.M)

    def adjoint(self, y):
        return self._out(ut.iFt(ut.Ft(y) * np.conj(self.kernel_f)), self.N)

    def inverse(self, y, reg=None):
        intensity = np.abs(self.kernel_f) ** 2
        k_inv = np.conj(self.kernel_f) / (intensity if reg is None else intensity + reg)
        return self._out(ut.iFt(ut.Ft(y) * k_inv), self.N)


class Diagonalize(Operator):
    """Pointwise multiplication by a fixed array."""

    def __init__(self, vector, dtype=None):
        self.dtype = dtype if dtype is not None else ut.dtype_name(vector)
        self.vector = ut.cast(vector, self.dtype)
        self.M = self.N = tuple(np.shape(self.vector))

    def forward(self, x):
        return self._out(x * self.vector, self.M)

    def adjoint(self, y):
        return self._out(np.conj(self.vector) * np.reshape(y, self.N), self.N)

    def inverse(self, y, reg=None):
        reg = 0.0 if reg is None else reg
        return self._out(np.conj(self.vector) / (np.abs(self.vector) ** 2 + reg) * y, self.N)


class FourierTransform(Operator):
    """Centered, unnormalized FFT. As in llops, the adjoint is the inverse FFT (not scaled)."""

    def __init__(self, N, dtype=None):
        self.dtype = dtype if dtype is not None else 'complex32'
        assert 'complex' in self.dtype, 'FourierTransform requires a complex dtype (got %s)' % self.dtype
        self.M = self.N = tuple(N)

    def forward(self, x):
        return self._out(ut.Ft(x), self.M)

    def adjoint(self, y):
        return self._out(ut.iFt(y), self.N)


class DiagStack(Operator):
    """Block-diagonal stack: operator i acts on row-block i of a vertically stacked input."""

    def __init__(self, operators):
        self.stack_operators = list(operators)
        first = self.stack_operators[0]
        self.dtype = first.dtype
        self.M = (sum(op.M[-2] for op in operators), first.M[-1])
        self.N = (sum(op.N[-2] for op in operators), first.N[-1])
        self._in, self._out_slices, a, b = [], [], 0, 0
        for op in operators:
            self._in.append(slice(a, a + op.N[-2]))
            self._out_slices.append(slice(b, b + op.M[-2]))
            a += op.N[-2]
            b += op.M[-2]

    def __getitem__(self, i):
        return self.stack_operators[i]

    def forward(self, x):
        y = self._alloc(self.M)
        for op, si, so in zip(self.stack_operators, self._in, self._out_slices):
            y[so, :] += op.forward(x[si, :])
        return y

    def adjoint(self, y):
        x = self._alloc(self.N)
        for op, si, so in zip(self.stack_operators, self._in, self._out_slices):
            x[si, :] += op.adjoint(y[so, :])
        return x


class Segmentation(Operator):
    """Maps an object to a vertical stack of ROI crops (one per frame).

    adjoint/inverse place the crops back, optionally with Tukey alpha blending, and
    the inverse also divides by how many ROIs cover each pixel (i.e. it stitches).
    """

    def __init__(self, roi_list, N=None, alpha_blend_size=0, dtype=None):
        self.dtype = dtype if dtype is not None else ut.DEFAULT_DTYPE
        self.alpha_blend_size = alpha_blend_size
        for roi in roi_list:
            if not isinstance(roi, Roi):
                raise ValueError('Expected Roi object in roi_list (got %r)' % (roi,))
        self.roi_list = list(roi_list)
        self.N = tuple(sum(self.roi_list).shape if N is None else N)
        M = [max(r.shape[0] for r in self.roi_list), max(r.shape[1] for r in self.roi_list)]
        for roi in self.roi_list[1:]:
            M[0] += roi.shape[0]
        self.M = tuple(M)
        self._update_masks()

    def _update_masks(self):
        self._coverage = self._alloc(self.N)
        for roi in self.roi_list:
            self._coverage[roi.slice] += 1.0
        self._coverage[np.abs(self._coverage) < 1e-2] = 1.0

        if self.alpha_blend_size > 0:
            alpha = self.alpha_blend_size / np.max(self.roi_list[0].shape)
            windows = [scipy.signal.windows.tukey(s, alpha=alpha) for s in self.roi_list[0].shape]
            mask = ut.cast(np.outer(windows[0], windows[1]), self.dtype)
            mask -= ut.min(mask)
            mask /= ut.max(mask)
            mask[np.abs(mask) < 1e-2] = 1e-2
            mask_sum = np.zeros_like(self._coverage)
            for roi in self.roi_list:
                valid = roi.valid - roi.start
                mask_sum[roi.valid.slice] += mask[valid.slice]
            mask_sum[np.abs(mask_sum) < 1e-2] = 1
            mask_sum /= self._coverage
            self._blend, self._blend_sum = mask, mask_sum
        else:
            self._blend, self._blend_sum = None, None

    def forward(self, x):
        y = self._alloc(self.M)
        for index, roi in enumerate(self.roi_list):
            # llops stepped blocks by the object height; equal to the ROI height for single-row strips
            y[index * self.N[0]:(index + 1) * self.N[0], 0:self.M[1]] = crop_roi(x, roi, out_of_bounds_placeholder=0)
        return y

    def _place(self, x, step_from_first):
        y = self._alloc(self.N)
        for index, roi in enumerate(self.roi_list):
            h = self.roi_list[0].shape[0] if step_from_first else roi.shape[0]
            x_single = x[Roi(start=(index * h, 0), shape=roi.shape).slice]
            valid = roi.valid - roi.start
            if self._blend is not None:
                y[roi.slice] += x_single[valid.slice] * self._blend[valid.slice]
            else:
                y[roi.slice] += x_single[valid.slice]
        if self._blend_sum is not None:
            y /= self._blend_sum
        return y

    def adjoint(self, y):
        return self._place(y, step_from_first=True)

    def inverse(self, y, reg=None):
        return self._place(y, step_from_first=False) / self._coverage


def phase_ramp(shape, shift, dtype='complex32'):
    """exp(-i 2 pi k . shift) on a centered frequency grid of `shape` (llops PhaseRamp forward)."""
    coords = np.mgrid[tuple(slice(-m / 2 / m, m / 2 / m, 1.0 / m) for m in shape)]
    coords = [(-2 * np.pi * c).astype(ut.native_dtype(dtype)) for c in coords]
    phase = sum(c * ut.scalar(np.real(s)) for (c, s) in zip(coords, shift))
    phase = phase - np.floor(np.real(phase) / (2 * np.pi)) * (2 * np.pi)  # llops.mod (complex-safe)
    ramp = np.cos(phase) + 1j * np.sin(phase)
    if 'complex' not in dtype:
        ramp = np.real(ramp)  # llops wrote into a buffer of the requested (possibly real) dtype
    return ramp.astype(ut.native_dtype(dtype))


def vector_stack(arrays, dtype=None):
    """Stack 2D arrays vertically, each zero-padded (top-left) to the largest shape."""
    single = [max(np.shape(a)[0] for a in arrays), max(np.shape(a)[1] for a in arrays)]
    dtype = dtype if dtype is not None else ut.dtype_name(arrays[0])
    out = ut.zeros((single[0] * len(arrays), single[1]), dtype)
    for i, a in enumerate(arrays):
        out[i * single[0]:(i + 1) * single[0], :] = ut.pad(a, single)
    return out


def vector_split(vector, count):
    step = np.shape(vector)[0] // count
    return [vector[i * step:(i + 1) * step, :] for i in range(count)]


# ---------------------------------------------------------------------------
# Objective terms
# ---------------------------------------------------------------------------
#
# Values are computed the way llops did: the unscaled value is written to a
# buffer of the problem dtype, then multiplied in place by the weight. Matching
# the rounding matters because FISTA's restart test compares objective values.

def _scaled(value, weight, dtype):
    buf = np.zeros((1, 1), ut.native_dtype(dtype))
    buf[:] = np.real(value) if not np.iscomplexobj(buf) else value
    if weight is not None:
        buf *= weight
    return buf

class L2Data:
    """0.5 * ||A x - y||^2. Gradient reported as 2 A^H (A x - y) (llops convention)."""

    smooth = True

    def __init__(self, A, y):
        self.A, self.y = A, y
        self.N, self.dtype = A.N, A.dtype

    def value(self, x, dtype=None):
        r = self.A._alloc(self.A.M)
        r[:] = self.A.forward(x) - self.y
        return _scaled(0.5 * np.linalg.norm(r) ** 2, None, dtype or self.dtype)

    def gradient(self, x):
        return self.A.adjoint(2 * (self.A.forward(x) - self.y))


class L2Reg:
    """weight * 0.5 * ||x||^2."""

    smooth = True

    def __init__(self, weight):
        self.weight = weight

    def value(self, x, dtype=None):
        return _scaled(0.5 * np.linalg.norm(x) ** 2, self.weight, dtype or ut.dtype_name(x))

    def gradient(self, x):
        return 2 * x * np.conj(self.weight)


class L1Reg:
    """weight * ||x||_1."""

    smooth = False

    def __init__(self, weight):
        self.weight = weight

    def value(self, x, dtype=None):
        return _scaled(np.sum(np.abs(x)), self.weight, dtype or ut.dtype_name(x))

    def prox(self, x, alpha):
        return ut.soft_threshold(x, alpha * self.weight)


class TVReg:
    """Anisotropic TV via shifted Haar shrinkage (llops TotalVariation).

    Value: weight * sum(sqrt(sum_axis |roll(x, 1) - x|)). Odd-length axes are
    edge-padded by one sample for the prox and cropped back (llops required even sizes).
    """

    smooth = False

    def __init__(self, weight, ndim=2):
        self.weight = weight
        self.ndim = ndim
        self.threshold = 1.41 * 2 * ndim
        self.scaling = 1 / math.sqrt(ndim)

    def value(self, x, dtype=None):
        acc = np.zeros_like(x)
        for ax in range(x.ndim):
            acc += np.abs(np.roll(x, 1, axis=ax) - x)
        return _scaled(np.sum(np.sqrt(acc)), self.weight, dtype or ut.dtype_name(x))

    def prox(self, x, alpha):
        alpha = alpha * self.weight
        orig_shape = x.shape
        pad = [(0, s % 2) for s in orig_shape]
        xp = np.pad(x, pad, mode='edge') if any(p[1] for p in pad) else x
        y = np.zeros_like(xp)
        for ax in range(self.ndim):
            for shift in (0, 1):
                w = self._haar(xp, ax, shift)
                m = xp.shape[ax] // 2
                detail = [slice(None)] * self.ndim
                detail[ax] = slice(m, None)
                detail = tuple(detail)
                w[detail] = ut.soft_threshold(w[detail], self.threshold * alpha)
                y += self._ihaar(w, ax, shift)
        y /= (2 * self.ndim)
        return y[tuple(slice(0, s) for s in orig_shape)]

    def _slices(self, shape, ax):
        m = shape[ax] // 2

        def sl(s):
            out = [slice(None)] * self.ndim
            out[ax] = s
            return tuple(out)
        return sl(slice(None, m)), sl(slice(m, None)), sl(slice(None, None, 2)), sl(slice(1, None, 2))

    def _haar(self, x, ax, shift):
        w = np.zeros_like(x)
        x = np.roll(x, -shift, axis=ax)
        ma, mb, a, b = self._slices(x.shape, ax)
        w[ma] = self.scaling * (x[b] + x[a])
        w[mb] = self.scaling * (x[b] - x[a])
        return w

    def _ihaar(self, w, ax, shift):
        y = np.zeros_like(w)
        ma, mb, a, b = self._slices(w.shape, ax)
        y[a] = self.scaling * (w[ma] - w[mb])
        y[b] = self.scaling * (w[ma] + w[mb])
        return np.roll(y, shift, axis=ax)


class WaveletReg:
    """weight * ||W x||_1 with a pywt transform and random cycle spinning (llops WaveletSparsity)."""

    smooth = False

    def __init__(self, weight, shape, wavelet_type='db4', extension_mode='symmetric', level=None,
                 use_cycle_spinning=True):
        import pywt  # noqa: F401  (fail early if missing)
        self.weight = weight
        self.shape = tuple(shape)
        self.wavelet_type, self.mode, self.level = wavelet_type, extension_mode, level
        self.use_cycle_spinning = use_cycle_spinning
        self.pad_shape = [n if not n % 2 else n + 1 for n in self.shape]
        self.randshift = None
        self.coeff_slices = None

    def _forward(self, x):
        import pywt
        x = np.reshape(x, self.shape)
        if list(self.pad_shape) != list(self.shape):
            x = ut.pad(x, self.pad_shape)
        if self.use_cycle_spinning:
            self.randshift = [random.randint(0, x.shape[i]) for i in range(x.ndim)]
            x = ut.roll(x, self.randshift)
        coeffs = pywt.wavedecn(x, self.wavelet_type, mode=self.mode, level=self.level)
        arr, self.coeff_slices = pywt.coeffs_to_array(coeffs)
        return arr

    def _adjoint(self, arr):
        import pywt
        coeffs = pywt.array_to_coeffs(arr, self.coeff_slices)
        y = pywt.waverecn(coeffs, self.wavelet_type, mode=self.mode)
        if self.use_cycle_spinning and self.randshift is not None:
            y = ut.roll(y, [-s for s in self.randshift])
        if list(self.pad_shape) != list(self.shape):
            y = ut.crop(y, self.shape)
        return y[tuple(slice(0, s) for s in self.shape)]

    def value(self, x, dtype=None):
        return _scaled(np.sum(np.abs(self._forward(x))), self.weight, dtype or ut.dtype_name(x))

    def prox(self, x, alpha):
        return self._adjoint(ut.soft_threshold(self._forward(x), alpha * self.weight)).astype(x.dtype)


class Objective:
    """Sum of a data term and regularizers. At most one non-smooth term (as in llops)."""

    def __init__(self, data_term, regularizers=()):
        self.terms = [data_term] + list(regularizers)
        self.N, self.dtype = data_term.N, data_term.dtype
        smooth = [t for t in self.terms if t.smooth]
        rough = [t for t in self.terms if not t.smooth]
        if len(rough) > 1:
            raise ValueError('Only one non-smooth regularizer (l1, tv, wavelet) can be used at a time.')
        self.smooth_terms = smooth
        self.non_smooth_part = rough[0] if rough else None

    def __call__(self, x):
        """Objective value, accumulated in the problem dtype as llops did."""
        total = np.zeros((1, 1), ut.native_dtype(self.dtype))
        for t in self.terms:
            total += t.value(x, self.dtype)
        return total

    def smooth_gradient(self, x):
        g = None
        for t in self.smooth_terms:
            gt = t.gradient(x)
            g = gt if g is None else g + gt
        return g


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class _Nesterov:
    def __init__(self, objective, restart_enabled):
        self.objective, self.restart_enabled = objective, restart_enabled
        self.y_prev = self.x_prev = None
        self.beta, self.t, self.t_prev = 0.5, 0., 1.

    def iterate(self, y):
        _t = self.t
        self.t = 1 + math.sqrt(1 + 4 * self.t_prev ** 2) / 2
        self.t_prev = _t
        if self.y_prev is not None:
            x = (1 - self.beta) * y + self.beta * self.y_prev
            if self.restart_enabled and abs(ut.scalar(self.objective(self.x_prev))) < abs(ut.scalar(self.objective(x))):
                self.beta = 0
                x = y
            else:
                self.beta = (1 - self.t_prev) / self.t
        else:
            x = y
        self.x_prev = x
        self.y_prev = y
        self.t_prev = self.t
        return x


class Fista:
    """Proximal gradient with (llops-style) Nesterov momentum and objective-based restart."""

    def __init__(self, objective, convergence_tol=1e-14):
        self.objective = objective
        self.convergence_tol = convergence_tol
        self.cost = []

    def solve(self, initialization=None, iteration_count=10, step_size=0.1,
              use_nesterov_acceleration=True, nesterov_restart_enabled=True, let_diverge=True,
              verbose=False, **_ignored):
        let_diverge = let_diverge or nesterov_restart_enabled
        if initialization is None:
            x = np.zeros(self.objective.N, ut.native_dtype(self.objective.dtype), order='F')
        else:
            x = np.array(initialization, copy=True, order='F')
        nesterov = _Nesterov(self.objective, nesterov_restart_enabled) if use_nesterov_acceleration else None

        self.cost = []
        for iteration in range(iteration_count):
            # Gradient step on the smooth part, in place (llops semantics; see module docstring)
            if step_size is None:
                g = self.objective.smooth_gradient(x)
                x[:] -= np.linalg.norm(g) ** 2 / (np.linalg.norm(g) ** 2 + 1e-15) * g
            else:
                x -= step_size * self.objective.smooth_gradient(x)
            if self.objective.non_smooth_part is not None:
                x = np.asfortranarray(self.objective.non_smooth_part.prox(x, alpha=step_size), dtype=x.dtype)
            if nesterov is not None:
                x = nesterov.iterate(x)

            value = abs(ut.scalar(self.objective(x)))
            self.cost.append(value)
            if value == np.inf:
                raise ValueError('Objective evaluation overflow at iteration %d' % iteration)
            if verbose:
                print('iteration %d: cost %.6g' % (iteration + 1, value))
            if len(self.cost) > 2:
                if (abs(self.cost[-1] - self.cost[-2]) / max(self.cost[-1], 1e-10) < self.convergence_tol
                        or self.cost[-1] < 1e-20):
                    return x
                if self.cost[-1] > self.cost[-2] and not let_diverge:
                    return x
        return x
