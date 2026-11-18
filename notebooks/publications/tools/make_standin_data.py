"""Write placeholder inputs with the layout the data-dependent notebooks expect.

The original datasets and reconstruction archives are not in the repository.
This script recreates their directory layout, file names, archive keys and
minimum array extents (taken from the ROIs the notebooks slice) with constant or
random content, so the notebooks can be executed end to end to check the code.
The figures it produces are meaningless.

    python tools/make_standin_data.py /path/to/standin
    HTDEBLUR_DATA=/path/to/standin jupyter nbconvert --execute fig_main.ipynb ...

Peak memory when running fig_full_field against this data is about 8 GB.
"""
import os
import sys

import numpy as np
import tifffile

root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'standin_data')
rng = np.random.RandomState(0)


def path(*parts):
    p = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def rand(shape):
    return (1000 + 100 * rng.rand(*shape)).astype(np.float32)


# --- beads_stitched: per-strip reconstructions (fig_main, fig_full_field) -------------
# Strip index must appear both as 'strip_<i>_regularize=' and 'strip=<i>_channel=' (both
# naming schemes are parsed). ROIs reach row 1600 and column 29200.
strip_shape, strip_count, strip_pitch = (1600, 29300), 18, 1500
for kind in ('coded', 'stopandstare', 'strobe'):
    for i in range(strip_count):
        name = 'beads2_raster_45ms_%s_strip_%d_regularize=[]_strip=%d_channel=0.npz' % (kind, i, i)
        arrays = {'object_recovered': np.full(strip_shape, 1000, np.float32),
                  'roi': np.array({'start': (i * strip_pitch, 0), 'shape': strip_shape}, dtype=object)}
        if kind == 'coded':
            arrays['y'] = np.full(strip_shape, 1000, np.float32)
        np.savez_compressed(path('beads_stitched', name), **arrays)

# Stitched mosaic; slices reach (23831, 30900)
tifffile.imwrite(path('beads_stitched', 'stitched_recon_codedbeadschannel=0-2.tiff'),
                 np.full((23900, 31000), 100, np.uint16), photometric='minisblack', compression='zlib')

# --- SNR comparison archives (fig_snr_comparison) --------------------------------------
snr_shape = (1600, 6000)
snr_files = [
    'res_target_snr_comparison/res_line_bright_coded_raster_100_motion_deblur_2019_02_05_10_52_50_reprocess_regularize=[].npz',
    'res_target_snr_comparison/res_line_bright_strobe_raster_1_motion_deblur_2019_02_05_10_53_59_reprocess_regularize=[].npz',
    'res_target_snr_comparison/res_line_bright_constant_raster_100_motion_deblur_2019_02_05_10_54_51_reprocess_regularize=[].npz',
    'res_target_snr_comparison/res_line_bright_stopandstare_stop_and_stare_2019_02_05_10_57_04_reprocess_regularize=[].npz',
    'beads_output/beads2_line_45ms_coded_raster_100_motion_deblur_2019_02_04_16_45_36_no_median_regularize=[].npz',
    'beads_output/beads2_line_45ms_strobe_raster_1_motion_deblur_2019_02_04_16_45_48_no_median_regularize=[].npz',
    'beads_output/beads2_line_45ms_constant_raster_1_motion_deblur_2019_02_04_16_45_23_no_median_regularize=[].npz',
    'beads_output/beads2_line_45ms_stopandstare_stop_and_stare_2019_02_04_16_44_59_no_median_regularize=[].npz',
]
for f in snr_files:
    np.savez_compressed(path(*f.split('/')), object_recovered=rand(snr_shape), y=rand(snr_shape))

# --- Regularization comparison archives (fig_reg_comparison*) --------------------------
prefix = 'beads2_line_45ms_coded_raster_100_motion_deblur_2019_02_04_16_45_36_ss=1_regularize='
for reg in ('[]', '[l21.0e-02]', '[median1.0e-02]'):
    np.savez_compressed(path('regularized_output', prefix + reg + '.npz'), object_recovered=rand(snr_shape), y=rand(snr_shape))
np.savez_compressed(path('regularized_output', 'beads2_line_45ms_stopandstare_stop_and_stare_2019_02_04_16_44_59_no_median_regularize=[].npz'),
                    object_recovered=rand(snr_shape), y=rand(snr_shape))

# Raw dataset; only its metadata (effective pixel size) is used
from htdeblur.dataset import Dataset  # noqa: E402
ds = Dataset()
ds.frame_list = [np.full((64, 64), 100, np.uint16)]
ds.frame_state_list = [{}]
ds.metadata.type = 'motion_deblur'
ds.metadata.camera.pixel_size_um = 6.5
ds.metadata.objective.mag = 10
ds.metadata.system.mag = 1.0
raw_parent = path('02-04-19-MotionDeblur-beads2', 'x')
ds.save(os.path.dirname(raw_parent) + '/')
target = os.path.join(os.path.dirname(raw_parent), 'beads2_line_45ms_coded_raster_100_motion_deblur_2019_02_04_16_45_36')
os.rename(ds.directory.rstrip('/'), target)

print('wrote stand-in data to', root)
