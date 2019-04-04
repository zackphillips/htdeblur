h_star_list = []
x_list, y_list = [],  []

# for strip_index in yp.display.progressBar(range(len(data_list))):
strip_index = 6

if strip_index % 2 == 0:
    # Extract strips
    y = data_list[strip_index][1]
    x = data_list[strip_index][0]
        
    # Normalize
    y /= yp.scalar(yp.sum(y))
    x /= yp.scalar(yp.sum(x))
    
    # Compute indicies to section array into
    n = [y_sh // m_sh for (m_sh, y_sh) in zip(measurement_shape, yp.shape(y))]
    x_indicies = np.arange(measurement_shape[1], yp.shape(y)[1], measurement_shape[1])
    y_indicies = np.arange(measurement_shape[0], yp.shape(y)[0], measurement_shape[0])

    # Split static (input) array in 2D blocks in x
    _x = np.array_split(x, x_indicies, axis=1)[:-1]
    _y = np.array_split(y, x_indicies, axis=1)[:-1]
    
    # Split static (input) array in 2D blocks in y
    for (__x, __y) in zip(_x, _y):
        _x_list = np.array_split(__x, y_indicies, axis=0)[:-1]
        _y_list = np.array_split(__y, y_indicies, axis=0)[:-1]
        h_star_list += [yp.deconvolve(___y, ___x, reg=1e-12) for (___x, ___y) in zip(_x_list, _y_list)]
        x_list += _x_list
        y_list += _y_list

# Calculate average blur kernel
blur_vector_estimated = np.asarray(sum(h_star_list) / len(h_star_list))
blur_vector_estimated /= yp.sum(blur_vector_estimated)

plt.figure()
plt.subplot(211)
plt.imshow(blur_vector_estimated)
plt.subplot(212)
plt.imshow(yp.gaussian(yp.pad(np.flip(blur_vector)[np.newaxis, :], blur_vector_estimated.shape, center=True), sigma=1))