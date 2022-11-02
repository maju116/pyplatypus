import numpy as np
from numpy import ndarray
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pydicom
import tifffile
from skimage.transform import resize
from skimage.color import rgb2gray, gray2rgb
from typing import Union, Tuple, List


def read_image(path: str, channels: int, target_size: Union[int, Tuple[int, int]]) -> ndarray:
    """
    Loads image as numpy array.

    Parameters
    ----------
    path: str
        Image path.
    channels: int
        Number of color channels.
    target_size: Union[int, Tuple[int, int]]
        Target size for the image to be loaded.

    Returns
    -------
    pixel_array: ndarray
        Image as numpy array.
    """
    if path.lower().endswith(('.tif', 'tiff')):
        pixel_array = tifffile.imread(path)
        if channels == 1:
            if len(pixel_array.shape) == 3 and pixel_array.shape[0] == 3:
                pixel_array = np.expand_dims(rgb2gray(pixel_array), axis=-1)
            elif len(pixel_array.shape) == 2:
                pixel_array = np.expand_dims(pixel_array, axis=-1)
        elif channels == 3:
            if len(pixel_array.shape) == 2:
                pixel_array = gray2rgb(pixel_array)
        elif len(pixel_array.shape) == 3:
            pixel_array = np.moveaxis(pixel_array, 0, -1)
        pixel_array = resize(pixel_array, target_size, preserve_range=True)
        # ToDo: Check if special cases should be added - rgb2gray, ...
    elif path.lower().endswith('.dcm'):
        pixel_array = pydicom.dcmread(path).pixel_array
        if channels == 1:
            if len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
                pixel_array = np.expand_dims(rgb2gray(pixel_array), axis=-1)
            elif len(pixel_array.shape) == 2:
                pixel_array = np.expand_dims(pixel_array, axis=-1)
        elif channels == 3:
            if len(pixel_array.shape) == 2:
                pixel_array = gray2rgb(pixel_array)
        else:
            # ToDo: Check if any other type of DICOM should be implemented https://dicom.innolitics.com/ciods/rt-dose/image-pixel/00280004
            raise ValueError('For DICOM images number of channels can be set to 1 or 3!')
        pixel_array = resize(pixel_array, target_size, preserve_range=True)
    else:
        if channels == 1:
            color_mode = "grayscale"
        elif channels == 3:
            color_mode = "rgb"
        elif channels == 4:
            color_mode = "rgba"
        else:
            raise ValueError('For classical (PNG, JPG, ...) images number of channels can be set to 1, 3 or 4!')
        pixel_array = img_to_array(load_img(path, color_mode=color_mode, target_size=target_size))
    return pixel_array


def split_images(
        images: List[ndarray],
        h_splits: int,
        w_splits: int,
) -> list:
    """
    Splits list of images/masks onto smaller ones.

    Parameters
    ----------
    images: List[ndarray]
        List of images or masks.
    h_splits: int
        Number of vertical splits of the image.
    w_splits: int
        Number of horizontal splits of the image.

    Returns
    -------
    images: list
        List of images or masks.
    """
    if h_splits > 1:
        images = [np.vsplit(se, h_splits) for se in images]
        images = [item for sublist in images for item in sublist]
    if w_splits > 1:
        images = [np.hsplit(se, w_splits) for se in images]
        images = [item for sublist in images for item in sublist]
    return images


def read_and_concatenate_images(
        paths: List[List[str]],
        channels: List[int],
        target_size: Union[int, Tuple[int, int]]
) -> list:
    """
    Read and concatenates images along channel axis.

    Parameters
    ----------
    paths: List[List[str]]
        List of images paths to concatenate.
    channels: List[int]
        Number of color channels.
    target_size: Union[int, Tuple[int, int]]
        Target size for the image to be loaded.

    Returns
    -------
    images: list
        List of images.
    """
    images = [np.concatenate([read_image(path, channels=ch, target_size=target_size)
                              for path, ch in zip(sub_list, channels)], axis=-1) for sub_list in paths]
    return images
