"""Pairwise image registration. Ported from comptic.registration.

Only the methods htdeblur uses are kept: cross-correlation ('xc') and ORB
features ('orb'), with 'normalize' and 'highpass' preprocessing.
`register_roi_list` lives in `htdeblur.recon`.
"""

import numpy as np
import scipy.fft
import llops as yp
import llops.filter as filters
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import EuclideanTransform

_PREPROCESS_METHODS = ('normalize', 'highpass')


def _preprocess(image0, image1, methods):
    unknown = set(methods) - set(_PREPROCESS_METHODS)
    if unknown:
        raise ValueError('Unsupported preprocess methods %s (supported: %s)' % (sorted(unknown), _PREPROCESS_METHODS))

    if 'normalize' in methods:
        image0 = filters._normalize(image0)
        image1 = filters._normalize(image1)

    if 'highpass' in methods:
        image0 = filters.gaussian(image0, sigma=2) - filters.gaussian(image0, sigma=4)
        image1 = filters.gaussian(image1, sigma=2) - filters.gaussian(image1, sigma=4)

    return image0, image1


def _register_xc(image0, image1, axis, trust_threshold):
    # Pad to an FFT-friendly size
    pad_size = tuple(scipy.fft.next_fast_len(s) for s in yp.shape(image0))
    if pad_size != tuple(yp.shape(image0)):
        image0 = yp.pad(image0, pad_size, pad_value='edge', center=True)
        image1 = yp.pad(image1, pad_size, pad_value='edge', center=True)

    src_freq, target_freq = yp.Ft(image0, axes=axis), yp.Ft(image1, axes=axis)
    cross_correlation = yp.iFt(src_freq * yp.conj(target_freq), center=False, axes=axis)

    # 1D registration: collapse the other axes
    if axis is not None:
        axis_to_sum = list(range(yp.ndim(image1)))
        del axis_to_sum[axis]
        cross_correlation = yp.sum(cross_correlation, axis=axis_to_sum)

    shape = yp.shape(src_freq)
    maxima = yp.argmax(yp.abs(cross_correlation))
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    # A singleton dimension carries no shift information
    for dim in range(yp.ndim(src_freq)):
        if shape[dim] == 1:
            shifts[dim] = 0

    # Peak-to-mean energy ratio of the correlation
    trust_metric = yp.scalar(yp.max(yp.abs(cross_correlation) ** 2) / yp.mean(yp.abs(cross_correlation) ** 2))
    return shifts, trust_metric, trust_metric / trust_threshold


def _register_orb(image0, image1, axis, trust_threshold, orb_feature_threshold):
    match_count = 0
    fast_threshold = 0.05
    while match_count < orb_feature_threshold:
        extractor = ORB(n_keypoints=500, fast_n=9, harris_k=0.1, fast_threshold=fast_threshold)

        extractor.detect_and_extract(np.asarray(image0).astype(np.double))
        keypoints0, descriptors0 = extractor.keypoints, extractor.descriptors

        extractor.detect_and_extract(np.asarray(image1).astype(np.double))
        keypoints1, descriptors1 = extractor.keypoints, extractor.descriptors

        match_count = min(len(keypoints0), len(keypoints1))
        fast_threshold -= 0.01
        if fast_threshold == 0:
            raise RuntimeError('Could not find any keypoints (even after shrinking fast threshold).')

    matches = match_descriptors(descriptors0, descriptors1, cross_check=True)

    # Keep only matches aligned with the requested axis
    if axis is not None:
        filtered = []
        for index_0, index_1 in matches:
            unit_vec = keypoints0[index_0, :] - keypoints1[index_1, :]
            unit_vec /= np.linalg.norm(unit_vec)
            if abs(unit_vec[axis]) > 0.99:
                filtered.append((index_0, index_1))
        matches = np.asarray(filtered)

    _, inliers = ransac((keypoints0[matches[:, 0]], keypoints1[matches[:, 1]]),
                        EuclideanTransform, min_samples=3, residual_threshold=2, max_trials=100)
    matches = matches[inliers, :]

    if matches.shape[0] == 0:
        return np.asarray([0, 0]), 1e10, 0.0

    # Mean keypoint displacement (more stable than the RANSAC model translation)
    difference = keypoints0[matches[:, 0]] - keypoints1[matches[:, 1]]
    shifts = difference.sum(axis=0) / difference.shape[0]
    # comptic had np.round(shifts[0]), which returned a scalar and crashed every caller.
    shifts = np.round(shifts)

    if axis is not None:
        _shifts = [0, 0]
        _shifts[axis] = shifts[axis]
        shifts = _shifts

    residuals = np.sqrt(np.sum(np.abs(keypoints0[matches[:, 0]] + np.asarray(shifts) - keypoints1[matches[:, 1]]) ** 2))
    trust_metric = residuals / matches.shape[0]
    return shifts, trust_metric, trust_threshold / trust_metric


def registerImage(image0, image1, method='xc', axis=None, preprocess_methods=(), debug=False,
                  energy_ratio_threshold=1.5, mean_residual_threshold=40.0, orb_feature_threshold=25):
    """Estimate the integer shift between two images.

    Returns:
        (shifts, trust_ratio). Shifts are zeroed when trust_ratio <= 1.
    """
    if preprocess_methods:
        image0, image1 = _preprocess(image0, image1, preprocess_methods)

    if method == 'xc':
        trust_threshold = energy_ratio_threshold
        shifts, trust_metric, trust_ratio = _register_xc(image0, image1, axis, trust_threshold)
    elif method == 'orb':
        trust_threshold = mean_residual_threshold
        shifts, trust_metric, trust_ratio = _register_orb(image0, image1, axis, trust_threshold, orb_feature_threshold)
    else:
        raise ValueError('Invalid registration method %s' % method)

    if not trust_ratio > 1:
        if debug:
            print('Ignoring shift with trust metric %g (threshold is %g)' % (trust_metric, trust_threshold))
        shifts = np.zeros_like(np.asarray(shifts)).tolist()

    return shifts, trust_ratio
