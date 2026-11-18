# Publication notebooks

Figure notebooks for Phillips et al., *Biomed. Opt. Express* 11(1), 2020, plus the FOM 2019 abstract. All of them import only `htdeblur`, numpy/scipy/matplotlib, and the local helpers in this directory.

```shell
pip install -e ".[notebooks]"        # from the repository root
cd notebooks/publications
jupyter lab
```

Figures are written to `./figures` (override with `HTDEBLUR_FIGURES`). Notebooks that need the original data read it from `HTDEBLUR_DATA` (default `~/htdeblur-data`) and stop at the first load with a message naming the missing file.

## Status

| Notebook | Produces | Inputs | Runs here |
|---|---|---|---|
| `fig_limiting_analysis.ipynb` | System analysis (mechanical/optical limits) | none; sweeps are computed and cached in `figures/limiting_analysis_cache` | yes (~30 s) |
| `fig_limiting_analysis_sarah.ipynb`, `_zack.ipynb` | Earlier variants of the above | none | yes |
| `fig_multiframe_schematic.ipynb` | Multi-frame system schematic panels | none (California test object) | yes |
| `fig_multiframe_operator.ipynb` | Multi-frame smear matrix (`multiframe_smear.pdf`) | `resources/brain.png`; optional `single_illums.npz` | yes |
| `fig_optimization.ipynb` | Illumination optimization (box plot, spectra, DNF power law) | `fom2019/dnf_v_method.npz` (included); pulse-count sweep is recomputed (~90 s) | yes |
| `fig_fom_2019.ipynb` | FOM 2019 SNR figures | none | yes |
| `fig_snr_comparison.ipynb` | SNR comparison (resolution target, beads) | `res_target_snr_comparison/`, `beads_output/` reconstructions | needs data |
| `fig_main.ipynb`, `fig_full_field.ipynb` | Main bead figure, full-field mosaic and ROI panels | `beads_stitched/` strip reconstructions and stitched TIFF | needs data |
| `fig_reg_comparison.ipynb`, `_thinkpad.ipynb` | Regularization comparison | `regularized_output/` reconstructions; raw dataset (scale bar) | needs data |

The seven self-contained notebooks run in CI (`tests/test_notebooks.py`) and are saved with outputs from the current code. The data notebooks keep their original outputs.

## Expected data layout

Paths are relative to `HTDEBLUR_DATA`, which corresponds to the authors' `~/datasets/motiondeblur/`:

```
beads_stitched/
    beads2_raster_45ms_{coded,stopandstare,strobe}_..._strip_<i>_regularize=....npz   # keys: object_recovered, y (coded), roi
    stitched_recon_codedbeadschannel=0-2.tiff
res_target_snr_comparison/res_line_bright_{coded,strobe,constant,stopandstare}_..._reprocess_regularize=[].npz
beads_output/beads2_line_45ms_{coded,strobe,constant,stopandstare}_..._no_median_regularize=[].npz
regularized_output/beads2_line_45ms_coded_raster_100_motion_deblur_2019_02_04_16_45_36_ss=1_regularize=<reg>.npz
02-04-19-MotionDeblur-beads2/beads2_line_45ms_coded_raster_100_motion_deblur_2019_02_04_16_45_36/   # raw dataset
```

Exact file names are in each notebook's load cell. `tools/make_standin_data.py` writes placeholder files with this layout (constant/random content, correct names, keys and minimum sizes); all five data notebooks execute end to end against it, which checks the code paths but not the figures.

## What changed from the 2019 notebooks

- Imports moved from `libwallerlab`, `llops` and `ndoperators` to `htdeblur` (`linops`, `roi`, `_util`, `simulation`, `analysis`); hardcoded Dropbox paths replaced by `nbconfig`.
- `%matplotlib notebook` became `%matplotlib inline`; `nbconfig` keeps figures open across cells so cells that save or restyle a figure created earlier still work. The `deblur` matplotlib style is applied only if installed.
- `fig_fom_2019` used an older SNR model whose API differs from `htdeblur.analysis`. `fom2019_analysis.py` is that model, ported from libwallerlab with two fixes where it raised (missing `motion_velocity`; more pulses than kernel positions).
- `fig_optimization` passes `method='search'` and `pulse_count` explicitly: `htdeblur.analysis.getOptimalDnf` defaults to a power-law estimate with 10 pulses.
- Fixed bugs that stopped the originals from running top to bottom: a `lambdaaa` syntax error and missing definitions in `fig_main` (copied from `fig_full_field`), an undefined `decimation_factor` (set to 8, matching the binning used later), builtin `bin()` used for image binning, an out-of-range index in `fig_fom_2019`, cells referencing variables defined later, removed matplotlib APIs (`cm.get_cmap`, `tick.label`), and `np.load` of archives with pickled ROIs (`allow_pickle=True`).
- `fig_multiframe_operator` falls back to a seeded random illumination code when `single_illums.npz` is absent; this only affects the first section, whose figure is not saved.
- `blurkernel.generateVector` no longer exists; `fig_fom_2019` uses `blurkernel.vector`, which performs the same random pulse search.
