import skimage.io
import skimage.transform
import numpy as np


def read(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.
    Boldly copied from the caffe.io routine load_image: https://github.com/BVLC/caffe/blob/master/python/caffe/io.py

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def resize(im, size):
    return skimage.transform.resize(im, size)
