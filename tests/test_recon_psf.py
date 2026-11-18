"""Reconstruction with the optional system PSF (Reconstruction(use_psf=True))."""
import tempfile

import numpy as np

import htdeblur
import harness
from htdeblur.recon import Reconstruction


def test_reconstruct_with_psf():
    with tempfile.TemporaryDirectory() as tmp:
        path = harness.case_dataset(htdeblur, {}, tmp)
        recon = Reconstruction(harness._fresh_dataset(path), alpha_blend_distance=1000, normalize=False,
                               use_psf=True, estimate_background_poly=True)
        assert all(abs(float(np.sum(k)) - 1) < 1e-5 for k in recon.blur_kernel_list)
        recon.reconstruct(iteration_count=4, step_size=0.5, mode='global', reg_types={'l2': 1e-3}, display_type=None)
        assert np.isfinite(recon.object_recovered).all()
