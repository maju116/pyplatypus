import pandas as pd
import logging as log
import os
from typing import Tuple, List, Optional


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


def filter_paths_by_indices(paths: List[List[str]], indices: Optional[Tuple[int]]) -> List[List[str]]:
    """
    Filters path list by indices.

    Parameters
    ----------
        paths: List[str]
            Images paths.
        indices: Optional[Tuple[int]]
            Indices.

    Returns
    -------
    filtered_paths: List[str]
        Filtered paths list.
    """
    filtered_paths = [paths[idx] for idx in indices] if indices is not None else paths
    return filtered_paths
