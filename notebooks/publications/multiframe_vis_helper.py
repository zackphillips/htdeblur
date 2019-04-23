import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage.transform as skt
import scipy

## methods for constructing matrices. eventually should depend only on operator library

def blur_matrix_1d(illum, n):
    p = len(illum)
    blur_matrix = np.zeros([n+p,n])
    for i in range(n):
        blur_matrix[i:(i+p),i] = illum
    return blur_matrix

def crop_operator(n, m, offset=0):
    crop = np.zeros([m,n])
    for i in range(m):
        if i+offset < n:
            crop[i,i+offset] = 1
    return crop

def spatially_varying_blur_matrix(illums, n, window_overlap=0):
    assert not window_overlap, "variable overlap not yet implemented"
    n_segments = len(illums)   
    p = len(illums[0]) # assuming all the same length
    offsets, heights = get_segments(n_segments, n+p)
    
    crop_blur_list = []
    for i in range(n_segments):
        assert len(illums[i]) == p, "variable illumination length not yet supported"
        crop_blur_list.append(crop_operator(n+p, heights[i], offset=offsets[i]).dot(blur_matrix_1d(illums[i], n)))
    return np.vstack(crop_blur_list)

def get_segments(n_segments, total_length):
    segment_length = int(np.ceil((total_length) / n_segments))
    offsets = [i*segment_length for i in range(n_segments)]
    heights = [segment_length for i in range(n_segments-1)]
    heights.append(total_length - segment_length*(n_segments-1))
    return offsets, heights

def generate_circ(colk, k):
    n = len(colk)
    col1 = np.roll(colk, -k)
    return scipy.linalg.circulant(col1)

def gen_illum(blur_length, throughput, illum_type = 'pure_rand'):
    if illum_type == 'pure_rand':
        illumination = np.random.rand(blur_length)
        illumination = throughput * illumination / (np.sum(illumination)) *  blur_length
    elif illum_type == 'box':
        illumination = np.zeros(blur_length)
        on_indices = np.random.choice(blur_length, int(np.ceil(throughput*blur_length)), replace=False)
        partial_on = np.random.choice(on_indices, 1, replace=False)
        illumination[on_indices] = 1
        illumination[partial_on] = 1 - ( np.ceil(throughput*blur_length) - throughput*blur_length )
    return illumination