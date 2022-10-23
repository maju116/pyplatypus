import tensorflow as tf
from typing import Tuple, List, Optional, Union, Any
import numpy as np
import os
import pandas as pd
from numpy import ndarray
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import albumentations as A
import pydicom
import tifffile
from skimage.transform import resize
from skimage.color import rgb2gray, gray2rgb
from pyplatypus.utils.toolbox import split_masks_into_binary
from pyplatypus.segmentation.models.u_shaped_models import u_shaped_model
from pyplatypus.data_models.semantic_segmentation_datamodel import SemanticSegmentationData, \
    SemanticSegmentationModelSpec

import logging as log


class SegmentationGenerator(tf.keras.utils.Sequence):
    """The class can be used as train, validation or test generator. It is also utilized by the engine
    while producing the output plots, based on the trained models.

    Methods
    -------
    create_images_masks_paths(path: str, mode: str, only_images: bool, subdirs: Tuple[str, str], column_sep: str)
        Generates the dictionary storing the paths to the images and optionally coresponding masks.

    read_images_and_masks_from_directory(self, indices: Optional[List])
        Composes the batch of data out of the loaded images and optionally masks.
    """

    def __init__(
            self,
            path: str,
            colormap: Optional[List[Tuple[int, int, int]]],
            mode: str = "nested_dirs",
            only_images: bool = False,
            net_h: int = 256,
            net_w: int = 256,
            h_splits: int = 1,
            w_splits: int = 1,
            channels: Union[int, List[int]] = 3,
            augmentation_pipeline: Optional[A.core.composition.Compose] = None,
            batch_size: int = 32,
            shuffle: bool = True,
            subdirs: Tuple[str, str] = ("images", "masks"),
            column_sep: str = ";",
            return_paths: bool = False
    ) -> None:
        """
        Generates batches of data (images and masks). The data will be looped over (in batches).

        Parameters
        ----------
        path: str
            Images and masks directory.
        colormap: List[Tuple[int, int, int]]
            Class color map.
        mode: str
            Character. One of "nested_dirs", "config_file"
        only_images: bool
            Should generator read only images (e.g. on train set for predictions).
        net_h: int
            Input layer height. Must be equal to `2^x, x - natural`.
        net_w: int
            Input layer width. Must be equal to `2^x, x - natural`.
        h_splits: int
            Number of vertical splits of the image.
        w_splits: int
            Number of horizontal splits of the image.
        channels: Union[int, List[int]]
            Defines inputs layer color channels.
        augmentation_pipeline: Optional[A.core.composition.Compose]
            Augmentation pipeline.
        batch_size: int
            Batch size.
        shuffle: bool
            Should data be shuffled.
        subdirs: Tuple[str, str]
            Vector of two characters containing names of subdirectories with images and masks.
        column_sep: str
            Character. Configuration file separator.
        return_paths: bool
            Indicates whether the generator is supposed return images paths.
        """
        self.path = path
        self.colormap = colormap
        self.mode = mode
        self.only_images = only_images
        self.net_h = net_h
        self.net_w = net_w
        self.h_splits = h_splits
        self.w_splits = w_splits
        self.channels = channels if isinstance(channels, list) else [channels]
        self.augmentation_pipeline = augmentation_pipeline
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subdirs = subdirs
        self.column_sep = column_sep
        self.target_size = (net_h, net_w)
        self.classes = len(colormap)
        self.return_paths = return_paths
        self.config = self.create_images_masks_paths(self.path, self.mode, self.only_images, self.subdirs,
                                                     self.column_sep)
        self.indexes = None
        self.steps_per_epoch = self.calculate_steps_per_epoch()
        print(len(self.config["images_paths"]), "images detected!")
        print("Set 'steps_per_epoch' to:", self.steps_per_epoch)
        self.on_epoch_end()

    def calculate_steps_per_epoch(self) -> int:
        """Calculates the number of steps needed to go through all the images given the batch size.

        Returns
        -------
        steps_per_epoch: int
            Steps that the generator is to take in order to complete an epoch.
        """
        steps_per_epoch = int(np.ceil(len(self.config["images_paths"]) / self.batch_size))
        return steps_per_epoch

    @staticmethod
    def create_images_masks_paths(
            path: str, mode: str, only_images: bool, subdirs: Tuple[str, str], column_sep: str
    ) -> dict:
        """
        Generates the dictionary storing the paths to the images and optionally coresponding masks.
        It is the latter foundation upon which the batches are generated.

        Parameters
        ----------
        path: str
            Images and masks directory.
        mode: str
            Character. One of "nested_dirs", "config_file"
        only_images: bool
            Should generator read only images (e.g. on train set for predictions).
        subdirs: Tuple[str, str]
            Vector of two characters containing names of subdirectories with images and masks.
        column_sep: str
            Configuration file separator.

        Returns
        -------
        path_dict: dict
            Dictionary with images and optionally masks paths.
        """
        if mode in ["nested_dirs", 1]:
            nested_dirs = os.listdir(path)
            nested_dirs.sort()

            images_paths = []
            masks_paths = []
            for nd in nested_dirs:
                try:
                    images_paths_batch = [
                        os.path.join(path, nd, subdirs[0], s) for s in sorted(
                            os.listdir(os.path.join(path, nd, subdirs[0]))
                        )
                    ]
                    images_paths.append(images_paths_batch)
                    if not only_images:
                        masks_paths_batch = [
                            os.path.join(path, nd, subdirs[1], s) for s in sorted(
                                os.listdir(os.path.join(path, nd, subdirs[1]))
                            )
                        ]
                        masks_paths.append(masks_paths_batch)
                except FileNotFoundError:
                    log.warning(f"The current image {nd} is incomplete for it contains only masks or images!")
                    pass

        elif mode in ["config_file", 2]:
            config = pd.read_csv(path)
            images_paths = [s.split(column_sep) for s in config.images.to_list()]
            if not only_images:
                masks_paths = [s.split(column_sep) for s in config.masks.to_list()]
        else:
            raise ValueError("Incorrect 'mode' selected!")
        if not only_images:
            path_dict = {"images_paths": images_paths, "masks_paths": masks_paths}
            return path_dict
        else:
            path_dict = {"images_paths": images_paths}
            return path_dict

    @staticmethod
    def __read_image__(path: str, channels: int, target_size: Union[int, Tuple[int, int]]) -> ndarray:
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

    @staticmethod
    def __split_images__(
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

    def read_images_and_masks_from_directory(
            self, indices: Optional[List]
    ) -> Union[tuple[list[Any], list[ndarray]], list[Any]]:
        """
        Composes the batch of data out of the loaded images and optionally masks.

        Parameters
        ----------
        indices: List
            Indices of selected images. If `None` all images in `paths` will be selected.

        Returns
        -------
        loaded_images: list
            List of images and masks.
        """
        selected_images_paths = [self.config["images_paths"][idx] for idx in indices] if indices is not None else \
            self.config["images_paths"]
        if not self.only_images:
            selected_masks_paths = [self.config["masks_paths"][idx] for idx in indices] if indices is not None else \
                self.config["masks_paths"]
        if self.h_splits > 1 or self.w_splits > 1:
            selected_images = [
                np.concatenate([self.__read_image__(si, channels=ch,
                                                    target_size=(
                                                        self.h_splits * self.net_h, self.w_splits * self.net_w))
                                for si, ch in zip(sub_list, self.channels)], axis=-1) for sub_list in
                selected_images_paths]
            selected_images = self.__split_images__(selected_images, self.h_splits, self.w_splits)
            if not self.only_images:
                selected_masks = [
                    sum([self.__read_image__(si, channels=3,
                                             target_size=(self.h_splits * self.net_h, self.w_splits * self.net_w))
                         for si in sub_list]) for sub_list in selected_masks_paths]
                selected_masks = [split_masks_into_binary(mask, self.colormap) for mask in selected_masks]
                selected_masks = self.__split_images__(selected_masks, self.h_splits, self.w_splits)
        else:
            selected_images = [
                np.concatenate([self.__read_image__(si, channels=ch, target_size=self.target_size)
                                for si, ch in zip(sub_list, self.channels)], axis=-1) for sub_list in
                selected_images_paths]
            if not self.only_images:
                selected_masks = [
                    sum([self.__read_image__(si, channels=3, target_size=self.target_size)
                         for si in sub_list]) for sub_list in selected_masks_paths]
                selected_masks = [split_masks_into_binary(mask, self.colormap) for mask in selected_masks]
        if not self.only_images:
            loaded_images = (selected_images, selected_masks, selected_images_paths)
        else:
            loaded_images = (selected_images, selected_images_paths)
        return loaded_images

    def on_epoch_end(self) -> None:
        """Updates indexes on epoch end, optionally shuffles them for the sake of randomization."""
        self.indexes = list(range(len(self.config["images_paths"])))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index: int) -> Union[tuple[ndarray, ndarray], ndarray]:
        """
        Returns one batch of data.

        Parameters
        ----------
        index: int
            Batch index.

        Returns
        -------
        batch: tuple
            Batch of data being images and masks.
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        if not self.only_images:
            images, masks, paths = self.read_images_and_masks_from_directory(indexes)
            if self.augmentation_pipeline is not None:
                transformed = [self.augmentation_pipeline(image=image, mask=mask) for image, mask in zip(images, masks)]
                images = np.stack([tr['image'] for tr in transformed], axis=0)
                masks = np.stack([tr['mask'] for tr in transformed], axis=0)
            else:
                images = np.stack(images, axis=0)
                masks = np.stack(masks, axis=0)
            batch = (images, masks)
        else:
            images, paths = self.read_images_and_masks_from_directory(indexes)
            if self.augmentation_pipeline is not None:
                transformed = [self.augmentation_pipeline(image=image) for image in images]
                images = np.stack([tr['image'] for tr in transformed], axis=0)
            else:
                images = np.stack(images, axis=0)
            batch = images
        if self.return_paths:
            batch = (images, paths)
        return batch

    def __len__(self) -> int:
        """Returns the number of batches that one epoch is comprised of."""
        return int(np.ceil(len(self.config["images_paths"]) / self.batch_size))


def prepare_data_generator(
        data: SemanticSegmentationData, model_cfg: SemanticSegmentationModelSpec,
        augmentation_pipeline: Optional[A.Compose] = None, path: Optional[str] = None,
        only_images: bool = False, return_paths: bool = False
) -> SegmentationGenerator:
    """Prepares the train, validation and test generators, for each model separately.

    Parameters
    ----------
    data : SemanticSegmentationData
        Stores the data crucial for designing the data flow within the generators.
    model_cfg : SemanticSegmentationModelSpec
        From this data model information regarding the model is taken.
    augmentation_pipeline : Optional[Compose]
        Albumentations package native augmentation pipeline, None is allowed.
    path : str
        Path for the generator.
    only_images: bool
        Should generator read only images (e.g. on train set for predictions).
    return_paths: bool
        Indicates whether the generator should output images paths.

    Returns
    -------
    generators: tuple
        Tuple composed of the generators.
    """
    generator = SegmentationGenerator(
        path=path,
        mode=data.mode,
        colormap=data.colormap,
        only_images=only_images,
        net_h=model_cfg.net_h,
        net_w=model_cfg.net_w,
        h_splits=model_cfg.h_splits,
        w_splits=model_cfg.w_splits,
        channels=model_cfg.channels,
        augmentation_pipeline=augmentation_pipeline,
        batch_size=model_cfg.batch_size,
        shuffle=False,
        subdirs=data.subdirs,
        column_sep=data.column_sep,
        return_paths=return_paths
    )
    return generator


def predict_from_generator(model: u_shaped_model, generator: SegmentationGenerator) -> tuple:
    """Serves the batches of images to the supplied model and returns predictions alongside the paths to the
    images that the batch is comprised of.

    Parameters
    ----------
    model : u_shaped_model
        For now it is the U-shaped one but in the future it is expected to be one from the models
        associated with the tasks implemented within Platypus.
    generator : SegmentationGenerator
        Generator created on the course of preparing the modelling pipeline.

    Returns
    -------
    predictions: np.array
        Consists of the predictions for all the data yielded by the generator.
    paths: list
        Paths to the original images.
    """
    predictions = []
    paths = []
    for images_batch, paths_batch in generator:
        prediction = model.predict(images_batch)
        predictions.append(prediction)
        paths += [pt[0] for pt in paths_batch]
    predictions = np.concatenate(predictions, axis=0)
    return predictions, paths
