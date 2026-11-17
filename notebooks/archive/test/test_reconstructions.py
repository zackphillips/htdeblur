#!/usr/bin/env python
# coding: utf-8

# # Reconstruction Sandbox
# This notebook is a test-bed for regularization and reconstruction methods

# In[2]:

# Load motiondeblur module and Dataset class
import libwallerlab.projects.motiondeblur as md
from libwallerlab.utilities.io import Dataset, isDataset

# Platform imports
import os, glob

# Debugging imports
import llops as yp
import matplotlib.pyplot as plt

yp.config.setDefaultBackend('arrayfire')


# ## Load Data

# In[3]:


# Define user for path setting
dataset_path = '/home/ubuntu/datasets/'

# Define which dataset to use
dataset_type = 'coded'
dataset_label =  'res'

# Define group truth for simulation 
simulate_file = dataset_path + 'res_target_color_strobe_raster_motiondeblur_2018_05_22_19_17_18_recovered_[19, 20, 21, 22, 23].npz' # None


# Find files in this directory
folder_list = glob.glob(os.path.join(dataset_path, '*/'))
dataset_list = [folder for folder in folder_list if isDataset(folder)]

# Filter datasets in directory
filtered_dataset_list = [folder_name for folder_name in folder_list if (dataset_type in folder_name) and (dataset_label in folder_name)]
assert not len(filtered_dataset_list) > 1, "More than one dataset with criterion found!"
assert not len(filtered_dataset_list) == 0, "No dataset with criterion found!"

# Create dataset object (loads metadata)
dataset = Dataset(filtered_dataset_list[0])

# Force type to be motiondeblur
dataset.metadata.type = 'motiondeblur'

# Select green channel
dataset.channel_mask = [0]

# Preprocess dataset (md-specific)
md.preprocess(dataset)


# # Create Reconstruction Object

# In[3]:


# Set position segment
# dataset.position_segment_indicies = [3]
dataset.frame_mask = [19, 20, 21, 22] # [20, 21]

# Create recon object
recon = md.recon.Reconstruction(dataset, alpha_blend_distance=0, pad_mode='mean')

# Apply frame-dependent position offset
recon.applyFrameDependentOffset()

# Apply frame lateral offset
recon.applySegmentDependentOffset()

# Update crop with current overlap
recon.updateCropOperatorWithOverlap()

# Register Measurements
# offsets = recon.register_measurements(method='xc', preprocess_methods=['normalize', 'highpass'], axis=1, debug=False, write_results=True)


# ## Ensure Crop Operator Has Correct Overlap


# ## Simulation
if simulate_file is not None:
    recon.simulate_measurements(simulate_file) #,crop_offset=(0,1000))

# ## Stitch Measurements Together (No Deconvolution)

# In[9]:


recon.reconstruct(iteration_count=100, mode='static', step_size=0.1, reg_types={'l2': 1e-3})
recon.show(savepath='./0',show_raw=True)


# ## Sequential Recon

# In[12]:


recon.reconstruct(iteration_count=50, step_size=0.5, mode='sequential', reg_types={'l2': 1e-2, 'tv': 1e-3})
recon.show(savepath='./1',show_raw=True)


# ##  Global Recon

# In[ ]:


recon.reconstruct(iteration_count=500, step_size=1, mode='global', reg_types={'l2': 1e-4, 'tv': 1e-3})
recon.show(savepath='./2',show_raw=True)


# ## Perform Single-Frame Recon



# In[ ]:


recon.reconstruct(iteration_count=100, step_size=1, mode='single', frame_number=1, reg_types={'tv': 1e-1, 'l2': 1e-3})
recon.show(savepath='./3',show_raw=True)



