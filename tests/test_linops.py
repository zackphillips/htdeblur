"""Properties of the operators and solver that do not depend on reference data."""
import numpy as np
import pytest

from htdeblur import _util as ut
from htdeblur import linops
from htdeblur.roi import Roi

rng = np.random.RandomState(0)


def _dot(a, b):
    return np.vdot(a.astype(np.float64), b.astype(np.float64))


def _adjoint_gap(op):
    x = rng.randn(*op.N).astype(np.float32)
    y = rng.randn(*op.M).astype(np.float32)
    lhs, rhs = _dot(op.forward(x), y), _dot(x, op.adjoint(y))
    return abs(lhs - rhs) / max(abs(lhs), 1e-12)


def test_crop_adjoint():
    assert _adjoint_gap(linops.Crop(N=(20, 30), M=(12, 16), crop_start=(3, 5))) < 1e-6


def test_convolution_adjoint():
    kernel = rng.rand(24, 32).astype(np.float32)
    assert _adjoint_gap(linops.Convolution(kernel)) < 1e-5


def test_segmentation_adjoint_without_blending():
    rois = [Roi(start=(0, 10 * i), shape=(12, 16)) for i in range(3)]
    G = linops.Segmentation(rois)
    for r in rois:
        r.input_shape = G.N
    assert _adjoint_gap(G) < 1e-6


def test_segmentation_inverse_stitches():
    obj = rng.rand(12, 36).astype(np.float32)
    rois = [Roi(start=(0, 10 * i), shape=(12, 16), input_shape=(12, 36)) for i in range(3)]
    G = linops.Segmentation(rois)
    np.testing.assert_allclose(G.inverse(G.forward(obj)), obj, rtol=1e-6)


def test_diagstack_blocks():
    ops = [linops.Crop(N=(8, 10), M=(4, 10), crop_start=(i, 0)) for i in range(3)]
    D = linops.DiagStack(ops)
    x = rng.rand(*D.N).astype(np.float32)
    y = D.forward(x)
    for i, op in enumerate(ops):
        np.testing.assert_array_equal(y[4 * i:4 * (i + 1)], op.forward(x[8 * i:8 * (i + 1)]))
    assert _adjoint_gap(D) < 1e-6


def test_phase_ramp_is_a_shifted_delta():
    shape = (16, 20)
    ramp = linops.phase_ramp(shape, np.asarray([3.0, -5.0]))
    image = np.abs(ut.iFt(ramp))
    peak = np.unravel_index(np.argmax(image), shape)
    assert peak == (shape[0] // 2 + 3, shape[1] // 2 - 5)


@pytest.mark.parametrize('regs', [[], [linops.L2Reg(1e-3)], [linops.TVReg(1e-3)], [linops.L1Reg(1e-3)]])
def test_fista_decreases_objective(regs):
    kernel = np.zeros((24, 32), np.float32)
    kernel[12, 10:22] = rng.rand(12)
    kernel /= kernel.sum()
    A = linops.Crop(N=(24, 32), M=(20, 26), center=True) * linops.Convolution(kernel)
    truth = rng.rand(24, 32).astype(np.float32)
    objective = linops.Objective(linops.L2Data(A, A.forward(truth)), regs)
    solver = linops.Fista(objective)
    x = solver.solve(iteration_count=30, step_size=0.5)
    assert solver.cost[-1] < 0.1 * float(np.ravel(objective(np.zeros_like(x)))[0])


def test_tv_prox_handles_odd_shapes():
    x = rng.rand(9, 13).astype(np.float32)
    assert linops.TVReg(1.0).prox(x, 0.01).shape == x.shape


def test_only_one_nonsmooth_regularizer():
    A = linops.Crop(N=(8, 8), M=(8, 8))
    with pytest.raises(ValueError):
        linops.Objective(linops.L2Data(A, np.zeros((8, 8), np.float32)), [linops.TVReg(1), linops.L1Reg(1)])
