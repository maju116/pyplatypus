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

    def __init__(
            self,
            path: str,
            colormap: Optional[List[Tuple[int, int, int]]],
            mode: str = "dir",
            only_images: bool = False,
            net_h: int = 256,
            net_w: int = 256,
            h_splits: int = 1,
            w_splits: int = 1,
            grayscale: bool = False,
            augmentation_pipeline: Optional[A.core.composition.Compose] = None,
            batch_size: int = 32,
            shuffle: bool = True,
            subdirs: Tuple[str, str] = ("images", "masks"),
            column_sep: str = ";"
    ) -> None:
        """
        Generates batches of data (images and masks). The data will be looped over (in batches).

        Args:
        path (str): Images and masks directory.
        colormap (List[Tuple[int, int, int]]): Class color map.
        mode (str): Character. One of "dir", "nested_dirs", "config_file"
        only_images (bool): Should generator read only images (e.g. on train set for predictions).
        net_h (int): Input layer height. Must be equal to `2^x, x - natural`.
        net_w (int): Input layer width. Must be equal to `2^x, x - natural`.
        h_splits (int): Number of vertical splits of the image.
        w_splits (int): Number of horizontal splits of the image.
        grayscale (bool): Defines input layer color channels -  `1` if `True`, `3` if `False`.
        augmentation_pipeline (Optional[A.core.composition.Compose]): Augmentation pipeline.
        batch_size (int): Batch size.
        shuffle (bool): Should data be shuffled.
        subdirs (Tuple[str, str]): Vector of two characters containing names of subdirectories with images and masks.
        column_sep (str): Character. Configuration file separator.
        """
        self.path = path
        self.colormap = colormap
        self.mode = mode
        self.only_images = only_images
        self.net_h = net_h
        self.net_w = net_w
        self.h_splits = h_splits
        self.w_splits = w_splits
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
        self.steps_per_epoch = int(np.ceil(len(self.config["images_paths"]) / self.batch_size))
        print(len(self.config["images_paths"]), "images detected!")
        print("Set 'steps_per_epoch' to:", self.steps_per_epoch)
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

            Returns:
                Dictionary with images and masks paths.
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

    @staticmethod
    def __read_image__(
            path: str,
            grayscale: bool,
            target_size: Union[int, Tuple[int, int]]
    ) -> ndarray:
        """
        Loads image as numpy array.

        Args:
            path (str): Image path.
            grayscale (bool): Should image be loaded as grayscale.
            target_size (Union[int, Tuple[int, int]]): Target size for the image to be loaded.

        Returns:
            Image as numpy array.
        """
        return img_to_array(load_img(path, grayscale=grayscale, target_size=target_size))

    @staticmethod
    def __split_images__(
            images: List[ndarray],
            h_splits: int,
            w_splits: int,
    ):
        """
        Splits list of images/masks onto smaller ones.

        Args:
            images (List[ndarray]): List of images or masks.
            h_splits (int): Number of vertical splits of the image.
            w_splits (int): Number of horizontal splits of the image.

        Returns:
            List of images or masks.
        """
        if h_splits > 1:
            images = [np.vsplit(se, h_splits) for se in images]
            images = [item for sublist in images for item in sublist]
        if w_splits > 1:
            images = [np.hsplit(se, w_splits) for se in images]
            images = [item for sublist in images for item in sublist]
        return images

    def read_images_and_masks_from_directory(
            self,
            indices: Optional[List]
    ) -> Union[tuple[list[Any], list[ndarray]], list[Any]]:
        """
        Reads images from directories.

        Args:
         indices (List) Indices of selected images. If `None` all images in `paths` will be selected.

        Returns:
            List of images and masks.
        """
        selected_images_paths = [self.config["images_paths"][idx] for idx in indices] if indices is not None else \
            self.config["images_paths"]
        if not self.only_images:
            selected_masks_paths = [self.config["masks_paths"][idx] for idx in indices] if indices is not None else \
                self.config["masks_paths"]
        if self.h_splits > 1 or self.w_splits > 1:
            selected_images = [
                self.__read_image__(img_path[0], grayscale=self.grayscale,
                                    target_size=(self.h_splits * self.net_h, self.w_splits * self.net_w))
                for img_path in selected_images_paths]
            selected_images = self.__split_images__(selected_images, self.h_splits, self.w_splits)
            if not self.only_images:
                selected_masks = [
                    sum([self.__read_image__(si, grayscale=False,
                                             target_size=(self.h_splits * self.net_h, self.w_splits * self.net_w))
                         for si in sub_list]) for sub_list in selected_masks_paths]
                selected_masks = [split_masks_into_binary(mask, self.colormap) for mask in selected_masks]
                selected_masks = self.__split_images__(selected_masks, self.h_splits, self.w_splits)
        else:
            selected_images = [
                self.__read_image__(img_path[0], grayscale=self.grayscale, target_size=self.target_size)
                for img_path in selected_images_paths]
            if not self.only_images:
                selected_masks = [
                    sum([self.__read_image__(si, grayscale=False, target_size=self.target_size)
                         for si in sub_list]) for sub_list in selected_masks_paths]
                selected_masks = [split_masks_into_binary(mask, self.colormap) for mask in selected_masks]
        return (selected_images, selected_masks) if self.only_images is not None else selected_images

    def on_epoch_end(
            self
    ) -> None:
        """Updates indexes on epoch end."""

        self.indexes = list(range(len(self.config["images_paths"])))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(
            self,
            index: int
    ) -> Union[tuple[ndarray, ndarray], ndarray]:
        """
        Returns one batch of data.

        Args:
            index (int): Batch index.

        Returns:
            Batch of data - images and masks.
        """
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

    def __len__(
            self
    ) -> int:
        """Number of batches in 1 epoch."""

        return int(np.ceil(len(self.config["images_paths"]) / self.batch_size))
