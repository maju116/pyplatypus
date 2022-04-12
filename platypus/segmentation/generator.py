import tensorflow as tf
from typing import Tuple, List, Optional, Union, Any
import numpy as np
import os
import pandas as pd
from numpy import ndarray
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import albumentations as A


def split_masks_into_binary(
        mask: np.ndarray,
        colormap: List[Tuple[int, int, int]]
) -> np.ndarray:
    """
    Splits multi-class mask into binary masks.

    Args:
        mask (np.ndarray): Segmentation mask.
        colormap (List[Tuple[int, int, int]]): Class color map.
    """
    return np.stack([np.all(mask == c, axis=-1) * 1 for c in colormap], axis=-1)


class segmentation_generator(tf.keras.utils.Sequence):
    """
    Generates batches of data (images and masks). The data will be looped over (in batches).

    Args:
     path (str): Images and masks directory.
     colormap (List[Tuple[int, int, int]]): Class color map.
     mode (str): Character. One of "dir", "nested_dirs", "config_file"
     only_images (bool): Should generator read only images (e.g. on train set for predictions).
     net_h (int): Input layer height. Must be equal to `2^x, x - natural`.
     net_w (int): Input layer width. Must be equal to `2^x, x - natural`.
     grayscale (bool): Defines input layer color channels -  `1` if `True`, `3` if `False`.
     augmentation_pipeline (Optional[A.core.composition.Compose]): Augmentation pipeline.
     batch_size (int): Batch size.
     shuffle (bool): Should data be shuffled.
     subdirs (Tuple[str, str]): Vector of two characters containing names of subdirectories with images and masks.
     column_sep (str): Character. Configuration file separator.
    """

    def __init__(
            self,
            path: str,
            colormap: Optional[List[Tuple[int, int, int]]],
            mode: str = "dir",
            only_images: bool = False,
            net_h: int = 256,
            net_w: int = 256,
            grayscale: bool = False,
            augmentation_pipeline: Optional[A.core.composition.Compose] = None,
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
        self.augmentation_pipeline = augmentation_pipeline
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subdirs = subdirs
        self.column_sep = column_sep
        self.target_size = (net_h, net_w)
        self.classes = len(colormap)
        # Add checks for generator
        self.config = self.create_images_masks_paths(self.path, self.mode, self.only_images, self.subdirs,
                                                     self.column_sep)
        self.indexes = None
        print(len(self.config["images_paths"]), "images detected!")
        print("Set 'steps_per_epoch' to:", int(np.ceil(len(self.config["images_paths"]) / self.batch_size)))
        self.on_epoch_end()

    @staticmethod
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
             mode (str): Character. One of "nested_dirs", "config_file"
             only_images (bool): Should generator read only images (e.g. on train set for predictions).
             subdirs (Tuple[str, str]): Vector of two characters containing names of subdirectories with images and masks.
             column_sep (str): Character. Configuration file separator.
            """
        if mode in ["nested_dirs", 1]:
            nested_dirs = os.listdir(path)
            nested_dirs.sort()
            images_paths = [
                [os.path.join(path, nd, subdirs[0], s) for s in os.listdir(os.path.join(path, nd, subdirs[0]))] for nd
                in
                nested_dirs]
            if not only_images:
                masks_paths = [
                    [os.path.join(path, nd, subdirs[1], s) for s in
                     sorted(os.listdir(os.path.join(path, nd, subdirs[1])))] for nd
                    in nested_dirs]
        elif mode in ["config_file", 2]:
            config = pd.read_csv(path)
            images_paths = [[s] for s in config.images.to_list()]
            if not only_images:
                masks_paths = [s.split(column_sep) for s in config.masks.to_list()]
        else:
            raise ValueError("Incorrect 'mode' selected!")
        if not only_images:
            return {"images_paths": images_paths, "masks_paths": masks_paths}
        else:
            return {"images_paths": images_paths}

    def read_images_and_masks_from_directory(
            self,
            indices: Optional[List]
    ) -> Union[tuple[list[Any], list[ndarray]], list[Any]]:
        """
        Reads images from directories.

        Args:
         indices (List) Indices of selected images. If `None` all images in `paths` will be selected.
        """
        selected_images_paths = [self.config["images_paths"][idx] for idx in indices] if indices is not None else \
            self.config["images_paths"]
        selected_images = [img_to_array(load_img(img_path[0], grayscale=self.grayscale, target_size=self.target_size))
                           for img_path in selected_images_paths]
        if self.only_images is not None:
            selected_masks_paths = [self.config["masks_paths"][idx] for idx in indices] if indices is not None else \
                self.config["masks_paths"]
            selected_masks = [
                sum([img_to_array(load_img(si, grayscale=False, target_size=self.target_size)) for si in
                     sub_list]) for
                sub_list
                in selected_masks_paths]
            selected_masks = [split_masks_into_binary(mask, self.colormap) for mask in selected_masks]
        return (selected_images, selected_masks) if self.only_images is not None else selected_images

    def on_epoch_end(self):
        """Updates indexes on epoch end."""

        self.indexes = list(range(len(self.config["images_paths"])))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        if not self.only_images:
            images, masks = self.read_images_and_masks_from_directory(indexes)
            if self.augmentation_pipeline is not None:
                transformed = [self.augmentation_pipeline(image=image, mask=mask) for image, mask in zip(images, masks)]
                images = np.stack([tr['image'] for tr in transformed], axis=0)
                masks = np.stack([tr['mask'] for tr in transformed], axis=0)
            else:
                images = np.stack(images, axis=0)
                masks = np.stack(masks, axis=0)
        else:
            images = self.read_images_and_masks_from_directory(indexes)
            if self.augmentation_pipeline is not None:
                transformed = [self.augmentation_pipeline(image=image) for image in images]
                images = np.stack([tr['image'] for tr in transformed], axis=0)
            else:
                images = np.stack(images, axis=0)
        return (images, masks) if not self.only_images else images

    def __len__(self):
        """Number of batches in 1 epoch."""

        return int(np.ceil(len(self.config["images_paths"]) / self.batch_size))
