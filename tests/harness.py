"""Characterization harness shared by the reference generator and the regression tests.

Uses only htdeblur's public API and numpy, so the same cases run against the
original llops-backed code (to produce tests/fixtures/reference.npz) and the
numpy port (to check it).
"""

import os
import tempfile

import numpy as np


def _arr(x):
    """Convert results (arrays, scalars, tuples, Roi) to a numpy array."""
    if hasattr(x, 'start') and hasattr(x, 'shape') and not isinstance(x, np.ndarray):
        return np.asarray(list(x.start) + list(x.shape), dtype=float)
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

FRAME_SHAPE = (96, 128)
PULSE_POSITIONS = 24       # blur length in pixels
FRAME_COUNT = 4
FRAME_SPACING = 96         # pixels between frame start positions


def make_dataset(directory, htdeblur):
    """Write a synthetic single-segment coded-illumination dataset to `directory`."""
    from htdeblur.mddataset import MotionDeblurDataset
    from htdeblur import simulation

    rng = np.random.RandomState(0)
    obj_shape = (FRAME_SHAPE[0], FRAME_SPACING * FRAME_COUNT + FRAME_SHAPE[1] + PULSE_POSITIONS)
    obj = np.asarray(simulation.california(shape=obj_shape, min_value=0.2, max_value=1.0), dtype=float)

    frames, states = [], []
    for j in range(FRAME_COUNT):
        code = (rng.uniform(size=PULSE_POSITIONS) > 0.5).astype(float)
        code[0] = code[-1] = 1.0
        x0 = j * FRAME_SPACING
        frame = np.zeros(FRAME_SHAPE)
        for k, w in enumerate(code):
            frame += w * obj[:, x0 + k: x0 + k + FRAME_SHAPE[1]]
        frame = frame / code.sum() * 20000 + rng.normal(0, 50, FRAME_SHAPE) + 1000
        frames.append(np.clip(frame, 0, 65535).astype(np.uint16))

        states.append({
            'position': {
                'common': {'linear_segment_index': 0, 'velocity': 1.0, 'led_update_rate_us': 1.0},
                'states': [[{'time_index': k, 'value': {'x': float(x0 + k), 'y': 0.0, 'units': 'pixels'}}]
                           for k in range(PULSE_POSITIONS)]},
            'illumination': {
                'common': {},
                'states': [[{'index': 0, 'time_index': k, 'value': {'w': float(w)}}] for k, w in enumerate(code)]},
        })

    ds = MotionDeblurDataset()
    ds.frame_list = frames
    ds.frame_state_list = states
    ds.metadata.type = 'motion_deblur'
    ds.metadata.camera.pixel_size_um = 6.5
    ds.metadata.objective.mag = 10
    ds.metadata.objective.na = 0.25
    ds.metadata.system.mag = 1.0
    ds.metadata.illumination.spectrum.center = {'w': 0.53}
    ds.save(directory + '/', header='synthetic')
    return ds.directory


def load_dataset(path):
    from htdeblur.mddataset import MotionDeblurDataset
    return MotionDeblurDataset(dataset_path=path)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_blurkernel(htdeblur, out):
    bk = htdeblur.blurkernel
    for method in ('random_phase', 'random'):
        for metric in ('dnf', 'cond'):
            np.random.seed(1)
            v, m = bk.vector(10, kernel_length=25, method=method, metric=metric, n_tests=20)
            out['bk_vector_%s_%s' % (method, metric)] = _arr(v)
            out['bk_vector_metric_%s_%s' % (method, metric)] = _arr(m)

    np.random.seed(2)
    out['bk_kernel'] = _arr(bk.kernel((32, 48), 8, kernel_length=20, n_tests=10))

    for method in ('constant', 'random_phase', 'random'):
        np.random.seed(3)
        out['bk_generate_%s' % method] = _arr(bk.generate((30, 40), 15, method=method))

    vec = np.linspace(1, 2, 11)
    for position in ('center', 'center_left', 'center_right', 'origin'):
        for axis in (0, 1):
            out['bk_fromVector_%s_%d' % (position, axis)] = _arr(bk.fromVector(vec.copy(), (24, 30), axis=axis, position=position))
    out['bk_fromVector_reverse'] = _arr(bk.fromVector(vec.copy(), (24, 30), reverse=True))
    out['bk_fromVector_interp'] = _arr(bk.fromVector(vec.copy(), (24, 40), interpolation_factor=2.0))
    out['bk_fromVector_nonorm'] = _arr(bk.fromVector(vec.copy(), (24, 30), normalize=False))

    np.random.seed(4)
    v, r = bk.genRandInitialization(30, 0.5)
    out['bk_genRandInitialization'] = _arr(v)
    out['bk_dnf'] = _arr(bk.dnf(vec))
    out['bk_cond'] = _arr(bk.cond(vec))
    out['bk_bbox'] = _arr(bk.getPositionListBoundingBox([[1.2, 3.0], [4.7, 10.0], [2.0, 6.5]]))
    out['bk_bbox_mean'] = _arr(bk.getPositionListBoundingBox([[1.2, 3.0], [4.7, 10.0], [2.0, 6.5]], use_mean=True))


def case_analysis(htdeblur, out):
    an = htdeblur.analysis
    params = an.getDefaultSystemParams()
    out['an_default_keys'] = np.asarray(sorted(params.keys()))
    out['an_default_values'] = np.asarray([float(params[k]) if isinstance(params[k], (int, float, bool, np.floating)) else np.nan
                                           for k in sorted(params.keys())])
    exposures = np.asarray([1e-4, 1e-3, 1e-2])
    out['an_exposureTimeToSnr'] = np.asarray([an.exposureTimeToSnr(t, **params) for t in exposures])
    rest = {k: v for k, v in params.items() if k != 'illuminance'}
    out['an_illuminanceToPhotonPixelRate'] = _arr(an.illuminanceToPhotonPixelRate(1000.0, **rest))
    for strategy in ('strobe', 'stop_and_stare', 'coded'):
        np.random.seed(5)
        res = an.frameRateToExposure(10, 1e4, strategy, **rest)
        out['an_frameRateToExposure_%s' % strategy] = _arr(res)
    np.random.seed(6)
    out['an_getOptimalDnf'] = _arr(an.getOptimalDnf(40, pulse_count=10, n_tests=10))
    out['an_snr'] = _arr(an.snr(np.arange(1, 101, dtype=float).reshape(10, 10)))


def case_optics(htdeblur, out):
    from htdeblur import optics
    out['op_pupil'] = _arr(optics.pupil((32, 32), camera_pixel_size=6.5, objective_magnification=10,
                                        illumination_wavelength=0.53, objective_numerical_aperture=0.25))
    out['op_pupil_shift'] = _arr(optics.pupil((32, 32), center=False))


def case_registration(htdeblur, out):
    from htdeblur import registration, simulation
    img = np.asarray(simulation.california(shape=(64, 96)), dtype=np.float32)
    shifted = np.roll(img, (3, -5), axis=(0, 1))
    for pre in ((), ('normalize',), ('normalize', 'highpass')):
        s, t = registration.registerImage(img, shifted, method='xc', preprocess_methods=list(pre))
        out['reg_xc_%s' % '_'.join(pre or ('none',))] = _arr(s)
        out['reg_xc_trust_%s' % '_'.join(pre or ('none',))] = _arr(t)
    s, t = registration.registerImage(img, shifted, method='xc', axis=1)
    out['reg_xc_axis1'] = _arr(s)


def case_dataset(htdeblur, out, tmp):
    path = make_dataset(tmp, htdeblur)
    ds = load_dataset(path)
    out['ds_shape'] = _arr(ds.shape)
    out['ds_frames'] = np.stack([np.asarray(f) for f in ds.frame_list])
    vectors, rois = ds.blur_vectors()
    out['ds_blur_vectors'] = np.stack([np.asarray(v) for v in vectors])
    out['ds_blur_rois'] = np.stack([_arr(r) for r in rois])
    out['ds_roi_list'] = np.stack([_arr(r) for r in ds.roi_list])
    out['ds_segment_directions'] = _arr(ds.frame_segment_direction_list)

    ds.metadata.calibration['blur_vector'] = {'scale': {'axis': 1, 'factor': 1}}
    vectors, _ = ds.blur_vectors()
    out['ds_blur_vectors_phase_ramp'] = np.stack([np.asarray(v) for v in vectors])
    return path


def _fresh_dataset(path):
    ds = load_dataset(path)
    ds.metadata.calibration['blur_vector'] = {'scale': {'axis': 1, 'factor': 1}}
    return ds


def new_reconstruction(path):
    from htdeblur.recon import Reconstruction
    return Reconstruction(_fresh_dataset(path), alpha_blend_distance=1000, normalize=False, use_psf=False,
                          estimate_background_poly=True)


def case_recon_init(htdeblur, out, path):
    recon = new_reconstruction(path)
    out['rc_object_shape'] = _arr(recon.object_shape)
    out['rc_blur_kernels'] = np.stack([np.asarray(k) for k in recon.blur_kernel_list])
    out['rc_y_list'] = np.stack([np.asarray(y) for y in recon.y_list])
    out['rc_roi_list'] = np.stack([_arr(r) for r in recon.roi_list])
    out['rc_roi_list_conv'] = np.stack([_arr(r) for r in recon.roi_list_convolution])
    out['rc_roi_list_meas'] = np.stack([_arr(r) for r in recon.roi_list_measurement_support])
    out['rc_object_valid_roi'] = _arr(recon.object_valid_roi)


def case_recon_solve(htdeblur, out, path):
    # Regularization weights are chosen so FISTA's restart comparisons are well
    # above float32 rounding; near-ties there make results depend on the CPU's
    # summation order (e.g. tv=5e-4 ties at 7e-8 relative on iteration 3).
    for mode, reg in (('global', {'tv': 0.5}), ('global', {'l2': 1e-3}), ('global', {}),
                      ('global', {'l1': 10.0}), ('static', {}), ('sequential', {'l2': 1e-3})):
        recon = new_reconstruction(path)
        recon.reconstruct(iteration_count=8, step_size=0.5, mode=mode, reg_types=reg, display_type=None)
        key = 'rc_%s_%s' % (mode, '_'.join(sorted(reg)) or 'none')
        out[key] = np.asarray(recon.object_recovered)


def case_mddataset_processing(htdeblur, out, path):
    ds = load_dataset(path)
    from htdeblur.recon import normalize_measurements
    ny, nx = normalize_measurements(ds, decimation_factor=1, debug=False)
    out['md_norm_y'] = np.stack([np.asarray(v) for v in ny])
    out['md_norm_x'] = np.stack([np.asarray(v) for v in nx])

    ds = load_dataset(path)
    ds.register(force=True, frame_registration_mode='xc', segment_registration_mode=None)
    out['md_frame_offsets'] = np.asarray(ds.metadata.calibration['registration']['frame_offsets'], dtype=float)


REFERENCE_CASES = ('blurkernel', 'analysis', 'optics', 'registration', 'dataset', 'recon_init', 'recon_solve',
                   'mddataset_processing')


def run_reference_cases(htdeblur):
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        case_blurkernel(htdeblur, out)
        case_analysis(htdeblur, out)
        case_optics(htdeblur, out)
        case_registration(htdeblur, out)
        path = case_dataset(htdeblur, out, tmp)
        case_recon_init(htdeblur, out, path)
        case_recon_solve(htdeblur, out, path)
        case_mddataset_processing(htdeblur, out, path)
    return out
