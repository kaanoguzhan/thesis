import numpy as np


def find_beam_center(img, x_axis_is_space=True):
    """
    Finds the row with the highest total value in the image.
    The first dimension of image must be "time" and the second dimension must be "space".
    """

    # Swap axes if x_axis_is_space is False
    if not x_axis_is_space:
        img = np.transpose(img)

    max_row = 0
    max_val = 0
    for i in range(0, len(img)):
        cur_val = np.sum(img[i])
        if cur_val > max_val:
            max_val = cur_val
            max_row = i
    return max_row


def find_marker_laser_pulse(img, search_first_n_rows=120, x_axis_is_space=True):
    """
    Finds the laser pulse in the image with a very naive approach.
    #TODO: This will be improved in the future.
    """

    # Swap axes if x_axis_is_space is False
    if not x_axis_is_space:
        img = np.transpose(img)

    img = img[:search_first_n_rows]

    # Set max value to 5000
    img[img > 5000] = 5000
    # img[img < 500] = 0

    sum_y_axis = np.sum(img, axis=0)

    # Normalize sum_y_axis
    sum_y_axis_min = np.min(sum_y_axis)
    sum_y_axis_max = np.max(sum_y_axis)
    sum_y_axis = (sum_y_axis - sum_y_axis_min) / (sum_y_axis_max - sum_y_axis_min)

    # Beam start is the first time where the sum_y_axis is above 0.7
    start = np.argmax(sum_y_axis > 0.70)
    # Beam end is the last time where the sum_y_axis was above 0.7
    end = len(sum_y_axis) - np.argmax(np.flip(sum_y_axis) > 0.70)

    # # # Plot the sum_y_axis and start and end for debugging
    # plt.figure(figsize=(10, 5))
    # plt.plot(sum_y_axis, 'o')
    # plt.plot(start, sum_y_axis[start], 'o', color='r')
    # plt.plot(end, sum_y_axis[end], 'o', color='r')

    # # # Plot the image and the laser pulse borders for debugging
    # plt.figure(figsize=(20, 20))
    # plt.imshow(img, vmax=5000)
    # plt.plot(np.ones(img.shape[0]) * start, np.arange(img.shape[0]))
    # plt.plot(np.ones(img.shape[0]) * end, np.arange(img.shape[0]))

    return start, end


def get_window_around_beam_centre(img, window_size, x_axis_is_space=True):
    """
    Returns a window of the image centered on the beam center.
    """

    # Swap axes if x_axis_is_space is False
    if not x_axis_is_space:
        img = np.transpose(img)

    beam_center = find_beam_center(img)
    print(f'beam_center = {beam_center}')
    window_half_size = int(window_size/2)
    return beam_center-window_half_size, beam_center+window_half_size
