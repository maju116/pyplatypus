import tensorflow as tf
from typing import Tuple, List, Optional, Union, Any
import numpy as np
from numpy import ndarray
from skimage.transform import resize
import albumentations as A
from pyplatypus.utils.mask import split_masks_into_binary, read_and_sum_masks
from pyplatypus.utils.path import create_images_masks_paths, filter_paths_by_indices
from pyplatypus.utils.image import split_images, read_and_concatenate_images
from pyplatypus.segmentation.models.u_shaped_models import u_shaped_model
from pyplatypus.data_models.semantic_segmentation import SemanticSegmentationData, \
    SemanticSegmentationModelSpec


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
            net_h: Union[int, List[int]] = 256,
            net_w: Union[int, List[int]] = 256,
            ensemble_net_h: Optional[int] = None,
            ensemble_net_w: Optional[int] = None,
            h_splits: int = 1,
            w_splits: int = 1,
            channels: Union[int, List[int], List[Union[int, List[int]]]] = 3,
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
        net_h: Union[int, List[int]]
            Input layer height or list of heights for multiple inputs.
        net_w: Union[int, List[int]]
            Input layer width or list of widths for multiple inputs.
        ensemble_net_h: Optional[int]
            Output layer height for the ensemble model.
        ensemble_net_w: Optional[int]
            Output layer width for the ensemble model.
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
        self.ensemble_net_h = ensemble_net_h
        self.ensemble_net_w = ensemble_net_w
        self.h_splits = h_splits
        self.w_splits = w_splits
        self.channels = channels
        self.is_ensemble = self.check_model_is_ensemble()
        self.augmentation_pipeline = augmentation_pipeline
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subdirs = subdirs
        self.column_sep = column_sep
        self.classes = len(colormap)
        self.return_paths = return_paths
        self.config = create_images_masks_paths(self.path, self.mode, self.only_images, self.subdirs, self.column_sep)
        self.indexes = None
        self.steps_per_epoch = self.calculate_steps_per_epoch()
        print(len(self.config["images_paths"]), "images detected!")
        print("Set 'steps_per_epoch' to:", self.steps_per_epoch)
        self.on_epoch_end()

    def check_model_is_ensemble(self) -> bool:
        """Checks if generator is used for ensemble model."""
        single_input_h = isinstance(self.net_h, int)
        single_input_w = isinstance(self.net_w, int)
        single_input_channels = isinstance(self.channels, int) or all([isinstance(subitem, int) for subitem in self.channels])
        if single_input_h != single_input_w:
            raise ValueError("Width and height are set for different number of models!")
        elif (single_input_h and single_input_w) and not single_input_channels:
            raise ValueError("Height and width is set for single model, but channels for multiple models!")
        elif (not single_input_h and not single_input_w) and single_input_channels:
            raise ValueError("Height and width is set for multiple models, but channels for single model!")
        elif single_input_h and single_input_w and single_input_channels:
            return False
        elif not isinstance(self.ensemble_net_h, int) or not isinstance(self.ensemble_net_w, int):
            raise ValueError("For the ensemble model 'ensemble_net_h' and 'ensemble_net_w' must be set!")
        elif len(self.net_h) == len(self.net_w) and len(self.net_w) == len(self.channels):
            return True
        else:
            raise ValueError("Heights, widths and channels set to different number of inputs!")

    def calculate_steps_per_epoch(self) -> int:
        """Calculates the number of steps needed to go through all the images given the batch size.

        Returns
        -------
        steps_per_epoch: int
            Steps that the generator is to take in order to complete an epoch.
        """
        steps_per_epoch = int(np.ceil(len(self.config["images_paths"]) / self.batch_size))
        return steps_per_epoch

    def calculate_masks_target_size(self) -> Tuple[int, int]:
        """
        Calculates masks target size.

        Returns
        -------
        target_size: Tuple[int, int]
            Masks target size.
        """
        if self.is_ensemble:
            if self.h_splits > 1 or self.w_splits > 1:
                target_size = (self.h_splits * self.ensemble_net_h, self.w_splits * self.ensemble_net_w)
            else:
                target_size = (self.ensemble_net_h, self.ensemble_net_w)
        else:
            if self.h_splits > 1 or self.w_splits > 1:
                target_size = (self.h_splits * self.net_h, self.w_splits * self.net_w)
            else:
                target_size = (self.net_h, self.net_w)
        return target_size

    def calculate_images_target_sizes(self) -> List[Tuple[int, int]]:
        """
        Calculates images target sizes.

        Returns
        -------
        target_size: List[Tuple[int, int]]
            Images target sizes.
        """
        if self.h_splits > 1 or self.w_splits > 1:
            target_sizes = [(self.h_splits * h, self.w_splits * w) for h, w in zip(self.net_h, self.net_w)]
        else:
            target_sizes = [(h, w) for h, w in zip(self.net_h, self.net_w)]
        return target_sizes

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
        loaded_data: list
            List of images and masks.
        """
        selected_images_paths = filter_paths_by_indices(self.config["images_paths"], indices)
        target_size = self.calculate_masks_target_size()
        channels = self.channels[0] if self.is_ensemble else self.channels
        selected_images = read_and_concatenate_images(selected_images_paths, channels, target_size)
        if self.h_splits > 1 or self.w_splits > 1:
            selected_images = split_images(selected_images, self.h_splits, self.w_splits)
        if not self.only_images:
            selected_masks_paths = filter_paths_by_indices(self.config["masks_paths"], indices)
            selected_masks = read_and_sum_masks(selected_masks_paths, target_size)
            selected_masks = [split_masks_into_binary(mask, self.colormap) for mask in selected_masks]
            if self.h_splits > 1 or self.w_splits > 1:
                selected_masks = split_images(selected_masks, self.h_splits, self.w_splits)
        if not self.only_images:
            loaded_data = (selected_images, selected_masks, selected_images_paths)
        else:
            loaded_data = (selected_images, selected_images_paths)
        return loaded_data

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
        if self.is_ensemble:
            target_sizes = self.calculate_images_target_sizes()
        if not self.only_images:
            images, masks, paths = self.read_images_and_masks_from_directory(indexes)
            if self.augmentation_pipeline is not None:
                transformed = [self.augmentation_pipeline(image=image, mask=mask) for image, mask in zip(images, masks)]
                images = [tr['image'] for tr in transformed]
                masks = [tr['mask'] for tr in transformed]
            if self.is_ensemble:
                images = [[resize(im, ts) for im in images] for ts in target_sizes]
                images = [np.stack(im, axis=0) for im in images]
            else:
                images = np.stack(images, axis=0)
            masks = np.stack(masks, axis=0)
            batch = (images, masks)
            if self.return_paths:
                batch = (images, masks, paths)
        else:
            images, paths = self.read_images_and_masks_from_directory(indexes)
            if self.augmentation_pipeline is not None:
                transformed = [self.augmentation_pipeline(image=image) for image in images]
                images = [tr['image'] for tr in transformed]
            if self.is_ensemble:
                images = [[resize(im, ts) for im in images] for ts in target_sizes]
                images = [np.stack(im, axis=0) for im in images]
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
