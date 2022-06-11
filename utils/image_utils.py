import numpy as np


def split_image_with_stride(image, window_size, stride):
    """
    Split image into sub images with window size and stride
    :param image: image to split
    :param window_size: window size of the sub images
    :param stride: stride of the sub images
    :return: list of sub images
    """
    image_shape = image.shape
    n_rows = int(np.floor((image_shape[0] - window_size) / stride) + 1)
    sub_images = np.zeros((n_rows, window_size, image_shape[1]))
    for i in range(n_rows):
        sub_images[i, :, :] = image[i*stride:i*stride+window_size, :]
    return sub_images
