# Platform imports
import os, glob
from os.path import expanduser
import configargparse
import ast

import matplotlib.pyplot as plt
import numpy as np

libwallerlab = True
# Load motiondeblur module and Dataset class
if libwallerlab:
    import libwallerlab.projects.motiondeblur as md
    from libwallerlab.utilities.io import Dataset, isDataset
else:
    import htdeblur as md
    from comptic.containers import Dataset, isDataset
    from htdeblur.mddataset import MotionDeblurDataset
import llops as yp # must match libwallerlab llops

yp.config.setDefaultBackend('numpy') 
yp.config.setDefaultDatatype('float32')

def main(args):
    output_path = os.path.expanduser('~/deblurring/datasets/regularized_output')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if args.beads:
        dataset_full_path = '/home/deansarah/Dropbox/02-04-19-MotionDeblur-beads2/beads2_line_45ms_coded_raster_100_motion_deblur_2019_02_04_16_45_36'
    else:
        dataset_full_path = '/home/deansarah/Dropbox/res_line_bright_coded_raster_100_motion_deblur_2019_02_05_10_52_50'
    print(dataset_full_path)
    # Create dataset object (loads metadata)
    if libwallerlab:
        dataset = Dataset(dataset_full_path, use_median_filter=False, subtract_mean_dark_current=False, force_type='motion_deblur')
    else:
        dataset = MotionDeblurDataset(dataset_path=dataset_full_path, use_median_filter=False, subtract_mean_dark_current=False, force_type='')
    force = False

    if libwallerlab:
        dataset.motiondeblur.register(force=force)
        dataset.motiondeblur.normalize(force=force)
    else:
        dataset.register(force=force)
        dataset.normalize(force=force)
    dataset.metadata.calibration['blur_vector'] = {'scale': {'axis': 1, 'factor': 1}}

    if args.beads:
        dataset.frame_mask = [10,11,12,13,14]
        ss = args.ss # 0.5
        nit = 100 # 30
    else:
        dataset.frame_mask =  [8,9,10,11]
        ss = args.ss
        nit = 100

    # Create recon object
    recon = md.recon.Reconstruction(dataset, alpha_blend_distance=1000, normalize=False, use_psf=False, estimate_background_poly=True)

    # Perform reconstruction
    recon.reconstruct(iteration_count=nit, step_size=ss, mode='global', reg_types=args.reg_types)
    recon.save(output_path, filename=recon.dataset.metadata.file_header + '_ss={}'.format(ss), formats=['png', 'npz'], save_raw=True, downsample=4)

def parse():
    parser = configargparse.ArgParser()
    parser.add_argument('--beads', type=int, default=1)
    parser.add_argument('--reg-types', default="{'tv': 5e-4}")
    parser.add_argument('--ss', type=float, default=1)
    return parser.parse_args()
    
if __name__ == "__main__":
#if True:
    args = parse()
    args.reg_types = ast.literal_eval(args.reg_types)
    print(args)
    main(args)