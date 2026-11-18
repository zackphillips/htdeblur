# htdeblur: high-throughput imaging using motion deblur

**Status: archived research code (last active development 2019).** Code for [Phillips et al., *Biomed. Opt. Express* 11(1), 2020](https://doi.org/10.1364/BOE.11.000281). The code is kept for reference and figure reproduction. It is not maintained, and the API will not change.

## Layout

| Path | Contents |
|---|---|
| `htdeblur/blurkernel.py` | Blur kernel generation, illumination optimization, kernel recovery |
| `htdeblur/recon.py` | Multi-frame forward model and reconstruction |
| `htdeblur/linops.py` | Operators (crop, circular convolution, segmentation/stitching), L2/L1/TV/wavelet objective terms, FISTA |
| `htdeblur/roi.py` | `Roi` region-of-interest class |
| `htdeblur/_util.py` | Array helpers that keep llops numerics (centered FFTs, padding modes, dtype names) |
| `htdeblur/dataset.py` | `Dataset` (multi-page TIFF + JSON) and metadata classes |
| `htdeblur/mddataset.py` | `MotionDeblurDataset`, the motion-deblur subclass of `Dataset` |
| `htdeblur/registration.py` | Pairwise image registration (cross-correlation, ORB) |
| `htdeblur/optics.py` | Pupil and OTF models |
| `htdeblur/simulation.py` | Test objects (California, UCB logo/seal) used by the figure notebooks |
| `htdeblur/analysis.py` | SNR / deconvolution noise factor (DNF) analysis, `snr` |
| `htdeblur/solver.py`, `pgd.py`, `project_simplex_box.py` | Kernel optimization: projected gradient descent, simplex/box projection |
| `htdeblur/blurkernelplot.py` | Kernel plotting helpers |
| `htdeblur/acquisition/` | Hardware control (stage, LED array, camera via Micro-Manager). Optional |
| `notebooks/publications/` | Figure notebooks for the journal paper and FOM 2019 abstract |
| `notebooks/archive/` | Sandbox, conference, analysis and acquisition notebooks. Unmaintained; most need `libwallerlab` |

The parts of two deprecated, private lab libraries that this code used were folded in and trimmed: `comptic` (dataset container, registration, pupil/OTF, SNR, test object) and `llops` (array backend, operators, solvers). The package depends only on PyPI packages.

### Differences from the llops-era code

- Numerics match the llops implementation: centered FFT conventions, padding rules, float32 accumulation order and FISTA's momentum and restart behavior are kept, including two quirks documented in `linops.py` (L2 gradients carry a factor of 2; the momentum term cancels for smooth-only objectives). Step sizes tuned against the original code carry over.
- Regularizers `l1F`, `bilateral`, `median` and `tv_wavelet`, the iterative path of `blurKernelRecoveryFromStatic`, `genConvolutionSupportList`, and the spline autocalibration code (`getAutocalibrationFns`, `BlurKernelBasis`) were removed. They depended on llops/ndoperators code that is not available and did not run.
- TV regularization edge-pads odd-length axes by one sample for the prox; llops raised on odd sizes.
- Only circular convolution is supported (the only mode the reconstruction used). Direct inversion (`iteration_count < 0`) works only for invertible forward models.

## Installation

Requires Python 3.12 and numpy 1.x. The code is not tested on numpy 2.

```shell
git clone https://github.com/zackphillips/htdeblur
cd htdeblur
pip install -e .                 # core
pip install -e ".[acquisition]"  # + hardware control
pip install -e ".[notebooks]"    # + jupyter
pip install -e ".[test]" && pytest
```

`uv.lock` pins a resolved environment (`uv sync`).

### Dependencies not on PyPI

Only hardware control needs these: `illuminate` (LED array controller, [sci-microscopy/illuminate](https://github.com/sci-microscopy/illuminate)) and `MMCorePy` (ships with Micro-Manager 2.0). The publication notebooks no longer use `llops`, `libwallerlab` or `ndoperators`; the archived notebooks in `notebooks/archive/` still do.

## Reproducing figures

The figure notebooks are in `notebooks/publications/`; see its README for the per-notebook status and data layout. Seven of the twelve need no external data and run in CI. The other five need the original reconstructions and datasets, which are not in the repository; point `HTDEBLUR_DATA` at them.

## Tests

`pytest` runs in a few seconds:

- `tests/test_reference.py` checks 69 outputs (blur kernels, analysis, optics, registration, dataset I/O, ROI geometry, measurement normalization, frame registration and FISTA reconstructions in global/static/sequential modes with l2/l1/tv) against `tests/fixtures/reference.npz`, recorded from the llops-backed code on a synthetic coded-illumination dataset. Tolerance is rtol 1e-5; most arrays match exactly.
- `tests/test_linops.py` and `tests/test_roi.py` check operator adjoints, stitching, the phase-ramp shift theorem, FISTA convergence and ROI algebra.

`tests/reference/` documents how the fixtures were made. The last llops revision is a renamed refactor of the API htdeblur used and has two bugs in its operator sums (an unsupported `cache` argument, and accumulation into uninitialized `np.empty` buffers); `llops_compat.py` maps the names and patches those bugs.

## Citation

Z. F. Phillips, S. Dean, B. Recht, and L. Waller, "High-throughput fluorescence microscopy using multi-frame motion deblurring," *Biomedical Optics Express* 11(1), 281–300 (2020). https://doi.org/10.1364/BOE.11.000281

```bibtex
@article{phillips2020htdeblur,
  author  = {Phillips, Zachary F. and Dean, Sarah and Recht, Benjamin and Waller, Laura},
  title   = {High-throughput fluorescence microscopy using multi-frame motion deblurring},
  journal = {Biomedical Optics Express},
  volume  = {11},
  number  = {1},
  pages   = {281--300},
  year    = {2020},
  doi     = {10.1364/BOE.11.000281}
}
```

## License

BSD 3-clause. See `LICENSE`.
