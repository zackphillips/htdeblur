#!/usr/bin/env python
# coding: utf-8

# # Reconstruction 


# Load motiondeblur module and Dataset class
import libwallerlab.projects.motiondeblur as md
from libwallerlab.utilities.io import Dataset, isDataset

# Platform imports
import os, glob
import configargparse
import ast

# Debugging imports
import llops as yp
import numpy as np
import matplotlib.pyplot as plt

yp.config.setDefaultDatatype('float32')

try:
    yp.config.setDefaultBackend('arrayfire')
    import arrayfire as af
    af.device_gc()
except ValueError as e:
    print('not using arrayfire')

def initialize(args):
    print('Loading Dataset... ', end='')
    # Find files in this directory
    folder_list = glob.glob(os.path.join(args.dataset_path, '*/'))
    dataset_list = [folder for folder in folder_list if isDataset(folder)]

    # Filter datasets in directory
    filtered_dataset_list = [folder_name for folder_name in folder_list if (args.dataset_type in folder_name) and (args.dataset_label in folder_name)]
    assert not len(filtered_dataset_list) > 1, "More than one dataset with criterion found: " + str(filtered_dataset_list)
    assert not len(filtered_dataset_list) == 0, "No dataset with criterion found!" + str(folder_list)
    
    # Create dataset object (loads metadata)
    dataset_full_path = filtered_dataset_list[0]
    print(dataset_full_path)
    dataset = Dataset(dataset_full_path, subtract_mean_dark_current=False, use_median_filter=args.median, force_type='motion deblur') 

    # Select channel
    dataset.channel_mask = [args.channel]

    # Preprocess dataset (registration, normalization -- should read from precalibrated file)
    dataset.motiondeblur.register(force=False, frame_registration_mode='xc', segment_registration_mode='xc')
    dataset.motiondeblur.normalize(force=False)
    # dataset.motiondeblur.skip_first_frame_segment = True
    dataset.metadata.calibration['blur_vector'] = {'scale': {'axis': 1, 'factor': 1}} # 0.98875}}
    
    # set strip_index list as necessary
    if args.strip_index == [-1]:
        args.strip_index = dataset.position_segment_indicies_full
    
    return dataset

def reconstruct_strip(dataset, args, strip_ind):
    
    # Clear frames from memory
    print(af.device.device_mem_info()['alloc']['bytes'])
    dataset.clearFramesFromMemory()
    print(af.device.device_mem_info()['alloc']['bytes'])

    # Set position segment
    dataset.motiondeblur.position_segment_indicies = [strip_ind]
    print('all frames in strip', dataset.frame_mask)
    if args.frame_mask != [-1]:
        dataset.frame_mask = [dataset.frame_mask[i] for i in args.frame_mask]
    
    # Create recon object
    print('Creating reconstruction object')
    recon = md.recon.Reconstruction(dataset, alpha_blend_distance=args.alpha,
                                    use_psf=args.use_psf, verbose=True, estimate_background_poly=True)
    # Perform reconstruction
    print('Starting reconstruction')
    recon.reconstruct(iteration_count=args.iteration_count, 
                      step_size=args.step_size, mode=args.recon_mode, 
                      reg_types=args.reg_types)
    
    recon.reg_types_recon = None
    filename = recon.dataset.metadata.file_header + '_strip=' + str(strip_ind) + '_channel=' + str(args.channel)
    if args.frame_mask != [-1]:
        filename += '_subset=' + str(args.frame_mask)
    recon.save(args.output_dir, filename=filename + args.save_tag, formats=['npz'], save_raw=True)

def main(args):
    dataset = initialize(args)
    print('all strips:', dataset.motiondeblur.position_segment_indicies_full)
    for ind in args.strip_index:
        reconstruct_strip(dataset, args, ind)
        
#     datasets = [None] * len(args.gpu_strips.keys())
#     def init_on_gpu(gpu_ind):
#         print("INITIALIZING" + gpu_ind)
#         af.set_device(gpu_ind)
#         datasets[gpu_ind] = initialize(args)
#     def run_on_gpu(gpu_ind):
#         print("RECONS" + gpu_ind)
#         for ind in args.gpu_strips[gpu_ind]:
#             reconstruct_strip(datasets[gpu_ind], args, ind)
#     #num_cores = len(args.gpu_strips.keys()) # multiprocessing.cpu_count() - 2
#     #assert num_cores <= multiprocessing.cpu_count()
#     #results = Parallel(n_jobs=num_cores)(delayed(run_on_gpu)(i) for i in args.gpu_strips.keys())
#     for gpu_ind in args.gpu_strips.keys():
#         init_on_gpu(gpu_ind)
#     for gpu_ind in args.gpu_strips.keys():
#         run_on_gpu(gpu_ind)

def parse():
    parser = configargparse.ArgParser()
    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add_argument('--dataset-path', default='/home/ubuntu/datasets/')
    parser.add_argument('--dataset-type', default='coded')
    parser.add_argument('--dataset-label', default='res')
    parser.add_argument('--output-dir', default='/home/ubuntu/reconstructions/')
    parser.add_argument('--strip-index', nargs='+', type=int, default=[0])
    parser.add_argument('--frame-mask', nargs='+', type=int, default=[-1])
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--step-size', type=float, default=0.7)
    parser.add_argument('--recon-mode', default='global')
    parser.add_argument('--save-tag', default='')
    parser.add_argument('--use-psf', type=bool, default=False)
    parser.add_argument('--reg-types', default="{'tv': 5e-4}")
    parser.add_argument('--alpha', type=int, default=1000)
    parser.add_argument('--iteration-count', type=int, default=500)
    parser.add_argument('--median', type=bool, default=False)
    return parser.parse_args()
    
if __name__ == "__main__":
#if True:
    args = parse()
    args.reg_types = ast.literal_eval(args.reg_types)
    print(args)
    main(args)



