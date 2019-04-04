import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from libwallerlab.utilities import io

def plotBlurKernel(blur_kernel, ax):
    min_plot = np.amin(np.where(blur_kernel != 0), axis=1) - np.array([20,20])
    max_plot = np.amax(np.where(blur_kernel != 0), axis=1) + np.array([20,20])
    ax.imshow(np.abs(blur_kernel[min_plot[0]:max_plot[0], min_plot[1]:max_plot[1]]), cmap='gray', interpolation='none',
              extent=[min_plot[1],max_plot[1],min_plot[0],max_plot[0]])

def plotBlurKernelSupport(blur_kernel_roi_list, fig_width=10):

    # Calculate kernel support of all frames to determine full object size
    object_support_roi = io.Roi(x_start=blur_kernel_roi_list[0].x_start,
                                     y_start=blur_kernel_roi_list[0].y_start,
                                     size=blur_kernel_roi_list[0].size())

    for blur_kernel_roi in blur_kernel_roi_list:
        object_support_roi.x_start = min(blur_kernel_roi.x_start, object_support_roi.x_start)
        object_support_roi.y_start = min(blur_kernel_roi.y_start, object_support_roi.y_start)
        object_support_roi.x_end = max(blur_kernel_roi.x_end, object_support_roi.x_end)
        object_support_roi.y_end = max(blur_kernel_roi.y_end, object_support_roi.y_end)

    fig1 = plt.figure(figsize=(1 + fig_width, 1 + fig_width * object_support_roi.shape()[0] /  object_support_roi.shape()[1] ))
    ax1 = fig1.add_subplot(111, aspect='equal')
    xmin, xmax, ymin, ymax = 0, 0, 0, 0

    ax1.set_xlim((-100 + object_support_roi.x_start, object_support_roi.x_end + 100))
    ax1.set_ylim((-100 + object_support_roi.y_start, object_support_roi.y_end + 100))

    patterns = ['-', '+', 'x', 'o', 'O', '.', '*']  # more patterns
    colors = ['b', 'r', 'y', 'g', 'm', 'lightblue', 'b', 'r', 'y', 'g', 'm', 'lightblue', 'b', 'r', 'y', 'g', 'm', 'lightblue']  # more patterns
    for i, blur_kernel_roi in enumerate(blur_kernel_roi_list):
        ax1.add_patch(
            patches.Rectangle(
                (blur_kernel_roi.x_start, blur_kernel_roi.y_start),   # (x,y)
                blur_kernel_roi.shape()[1],          # width
                blur_kernel_roi.shape()[0],          # height
                color=colors[i],
                fill=True,
                alpha=0.7
            )
        )
    plt.xlabel('x position (pixels)')
    plt.ylabel('y position (pixels)')
    plt.title('Blur kernel measurement support')
