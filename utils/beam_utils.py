import numpy as np


def find_beam_center(img):
    """
    Finds the row with the highest total value in the image.
    The first dimension of image must be "time" and the second dimension must be "space".
    """
    max_row = 0
    max_val = 0
    for i in range(0, len(img)):
        cur_val = np.sum(img[i])
        if cur_val > max_val:
            max_val = cur_val
            max_row = i
    return max_row


def get_window_around_beam_center(img, window_size):
    """
    Returns a window of the image centered on the beam center.
    """
    beam_center = find_beam_center(img)
    print(f'beam_center = {beam_center}')
    window_half_size = int(window_size/2)
    return img[beam_center-window_half_size:beam_center+window_half_size, :]
