import tensorflow as tf
from typing import Tuple, List, Optional
import numpy as np
import os
import pandas as pd


def read_images_from_directory(
        paths: List,
        indices: Optional[List],
        target_size: Tuple[int, int],
        grayscale: bool,
        scale: float,
        colormap: List[Tuple[int, int, int]]
):
    """
    Reads images from directories.

    Args:
     paths (List): Images or masks paths.
     indices (List) Indices of selected images. If `None` all images in `paths` will be selected.
     target_size (Tuple[int, int]):
     colormap (List[Tuple[int, int, int]]): Class color map.
     grayscale (bool): Defines input layer color channels -  `1` if `True`, `3` if `False`.
     scale (float): Scaling factor for images pixel values.
     colormap (List[Tuple[int, int, int]]): Class color map.
    """

    return 4


def create_images_masks_paths(
        path: str,
        mode: str,
        only_images: bool,
        subdirs: Tuple[str, str],
        column_sep: str
) -> dict:
    """
        Generates images/masks path from selected configuration.

        Args:
         path (str): Images and masks directory.
         mode (str): Character. One of "dir", "nested_dirs", "config_file"
         only_images (bool): Should generator read only images (e.g. on train set for predictions).
         subdirs (Tuple[str, str]): Vector of two characters containing names of subdirectories with images and masks.
         column_sep (str): Character. Configuration file separator.
        """
    if mode in ["dir", 1]:
        images_paths = [os.path.join(path, subdirs[0], s) for s in os.listdir(os.path.join(path, subdirs[0]))]
        if not only_images:
            masks_paths = [os.path.join(path, subdirs[1], s) for s in os.listdir(os.path.join(path, subdirs[1]))]
    elif mode in ["nested_dirs", 2]:
        nested_dirs = os.listdir(path)
        images_paths = [[os.path.join(path, nd, subdirs[0], s) for s in os.listdir(os.path.join(path, nd, subdirs[0]))][0] for nd in nested_dirs]
        if not only_images:
            masks_paths = [[os.path.join(path, nd, subdirs[1], s) for s in os.listdir(os.path.join(path, nd, subdirs[1]))] for nd in nested_dirs]
    elif mode in ["config_file", 3]:
        config = pd.read_csv(path)
        images_paths = config.images.to_list()
        if not only_images:
            masks_paths = [s.split(column_sep) for s in config.masks.to_list()]
    if not only_images:
        return {"images_paths": images_paths, "masks_paths": masks_paths}
    else:
        return {"images_paths": images_paths}


class SegmentationGenerator(tf.keras.utils.Sequence):
    """
    Generates batches of data (images and masks). The data will be looped over (in batches).

    Args:
     path (str): Images and masks directory.
     colormap (List[Tuple[int, int, int]]): Class color map.
     mode (str): Character. One of "dir", "nested_dirs", "config_file"
     only_images (bool): Should generator read only images (e.g. on train set for predictions).
     net_h (int): Input layer height. Must be equal to `2^x, x - natural`.
     net_w (int): Input layer width. Must be equal to `2^x, x - natural`.
     grayscale (bool): Defines input layer color channels -  `1` if `TRUE`, `3` if `FALSE`.
     scale (float): Scaling factor for images pixel values. Default to `1 / 255`.
     batch_size (int): Batch size.
     shuffle (bool): Should data be shuffled.
     subdirs (Tuple[str, str]): Vector of two characters containing names of subdirectories with images and masks.
     column_sep (str): Character. Configuration file separator.
    """

    def __init__(
            self,
            path: str,
            colormap: List[Tuple[int, int, int]],
            mode: str = "dir",
            only_images: bool = False,
            net_h: int = 256,
            net_w: int = 256,
            grayscale: bool = False,
            scale: float = 1 / 255,
            batch_size: int = 32,
            shuffle: bool = True,
            subdirs: Tuple[str, str] = ("images", "masks"),
            column_sep: str = ";"
    ) -> None:
        self.path = path
        self.colormap = colormap
        self.mode = mode
        self.only_images = only_images
        self.net_h = net_h
        self.net_w = net_w
        self.grayscale = grayscale
        self.scale = scale
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subdirs = subdirs
        self.column_sep = column_sep
        self.target_size = (net_h, net_w)
        self.classes = len(colormap)
        # Add checks for generator
        self.config = create_images_masks_paths(path, mode, only_images, subdirs, column_sep)
        print(len(self.config["images_paths"]), "images detected!")
        print("Set 'steps_per_epoch' to:", int(np.ceil(len(self.config["images_paths"]) / self.batch_size)))
        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes on epoch end."""

        self.indexes = list(range(len(self.config)))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        images = read_images_from_directory()

    def __len__(self):
        """Number of batches in 1 epoch."""

        return int(np.ceil(len(self.config["images_paths"]) / self.batch_size))
