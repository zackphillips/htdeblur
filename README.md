# htdeblur: high-throughput imaging using motion deblur

**Status: archived research code (last active development 2019).** Code for [Phillips et al., *Biomed. Opt. Express* 11(1), 2020](https://doi.org/10.1364/BOE.11.000281). The code is kept for reference and figure reproduction. It is not maintained, and the API will not change.

## Layout

| Path | Contents |
|---|---|
| `htdeblur/blurkernel.py` | Blur kernel generation, illumination optimization, kernel recovery |
| `htdeblur/recon.py` | Multi-frame forward model and reconstruction |
| `htdeblur/mddataset.py` | `MotionDeblurDataset` loader (subclasses `comptic.containers.Dataset`) |
| `htdeblur/analysis.py` | SNR / deconvolution noise factor (DNF) analysis |
| `htdeblur/solver.py`, `pgd.py`, `project_simplex_box.py` | Kernel optimization: projected gradient descent, simplex/box projection |
| `htdeblur/blurkernelplot.py` | Kernel plotting helpers |
| `htdeblur/acquisition/` | Hardware control (stage, LED array, camera via Micro-Manager). Optional |
| `notebooks/publications/` | Figure notebooks for the journal paper |
| `notebooks/archive/` | Sandbox, conference, analysis and acquisition notebooks. Unmaintained; most need `libwallerlab` |

## Installation

Requires Python 3.10–3.12 and numpy 1.x. The code is not tested on numpy 2.

```shell
git clone https://github.com/zackphillips/htdeblur
cd htdeblur
pip install -e .                 # core
pip install -e ".[acquisition]"  # + hardware control
pip install -e ".[notebooks]"    # + jupyter
```

`uv.lock` pins a resolved environment for Python 3.12 (`uv sync`).

### Dependencies not on PyPI

`pip install` does not pull these in. Install them manually before `import htdeblur` will work.

| Package | Used by | Notes |
|---|---|---|
| `llops` | almost everything (`import llops as yp`) | Linear operators, solvers, backend abstraction. Use the numpy backend |
| `comptic` | `recon`, `mddataset`, `blurkernel` (registration), `acquisition.hardware` | Dataset container, registration, pupil/OTF |
| `illuminate` | `acquisition.hardware` only | LED array controller |
| `MMCorePy` | `acquisition.hardware` only | Ships with Micro-Manager 2.0 |
| `libwallerlab` | most notebooks (not the package) | Predecessor of this package, llops and comptic |
| `ndoperators` | `fig_multiframe_schematic`, `fig_optimization`, `fig_multiframe_operator` and some archive notebooks | Operator library; not used by the package |

`pgd` and `project_simplex_box` depend only on numpy and import without any of these.

## Reproducing figures

The notebooks in `notebooks/publications/` were run on the authors' machines. Before running them:

- Edit the hardcoded data and output paths (under `/Users/zfphil/...`, `/home/sarah/...`, `/home/deansarah/...`) at the top of each notebook.
- Most expect `deblur.mplstyle`, a matplotlib style file that is not in this repo. Remove the `plt.style.use(...)` line, or provide your own.
- Raw datasets and intermediate reconstructions (`*.npz`, `*.tiff`) are not in this repo.

| Notebook | Imports | Data |
|---|---|---|
| `fig_limiting_analysis.ipynb` | `htdeblur` | `journalpaper/data` |
| `fig_multiframe_schematic.ipynb` | `htdeblur`, `llops`, `comptic`, `ndoperators` | none (simulated test object) |
| `fig_snr_comparison.ipynb` | `htdeblur`, `llops`, `comptic` | beads and resolution target reconstructions |
| `fig_reg_comparison_thinkpad.ipynb` | `llops` | regularized bead reconstructions (`*.npz`) |
| `fig_main.ipynb`, `fig_full_field.ipynb` | `libwallerlab`, `llops` | stitched bead reconstructions |
| `fig_fom_2019.ipynb` | `libwallerlab`, `llops` | `fom2019/` |
| `fig_optimization.ipynb` | `libwallerlab`, `llops`, `ndoperators` | `dnf_*.npz` (`fom2019/dnf_v_method.npz` included) |
| `fig_reg_comparison.ipynb` | `libwallerlab`, `llops` | raw bead dataset, 2019-02-04 |
| `fig_multiframe_operator.ipynb` | `libwallerlab`, `ndoperators` | single-illumination presentation data |
| `fig_limiting_analysis_sarah.ipynb`, `_zack.ipynb` | `libwallerlab` | earlier variants of `fig_limiting_analysis` |

Notebooks that import `libwallerlab` need porting to `htdeblur` imports before they will run against this package.

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
