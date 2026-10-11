"""Read instances off disk instead of destroying them.

The Data Science Bowl stores **one PNG per nucleus** - 70 files in one validation case, 375
in another, 2 in the smallest. `unite_masks()` is what the training pipeline does with those
files and its own docstring says what it costs: the element-wise maximum merges them, so
instance identity is gone at read time. Nothing new has to be collected to score instances;
it only has to be read differently.

This reads for **scoring only**. Making the dataset able to *train* on instances is the next
piece of work and is the same constraint as multilabel - the mask layer encodes one class per
pixel (`pyplatypus#158`, `pyplatypus#91`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pyplatypus.instances.metrics import InstanceError, label_map

#: Where the challenge puts the per-object files inside a case directory.
MASKS_DIRECTORY = "masks"


def read_instance_masks(case: Path, directory: str = MASKS_DIRECTORY) -> list[np.ndarray]:
    """Every per-object mask in one case directory, as boolean arrays, in sorted order.

    Sorted by filename so two runs read the same objects in the same order. The order is not
    load-bearing for the metric - it compares every pair - but an unordered read makes two
    runs' per-case output impossible to diff, and that is the first thing anybody does when a
    number moves.
    """
    folder = Path(case) / directory
    if not folder.is_dir():
        raise InstanceError(f"no {directory}/ directory in {case}; this reads per-object files")
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise InstanceError(f"{folder} holds no .png files")

    masks = []
    for path in paths:
        with Image.open(path) as handle:
            mask = np.asarray(handle.convert("L")) > 0
        if not mask.any():
            raise InstanceError(
                f"{path.name} is empty. An object that covers no pixel is not an object, and "
                "counting it would make a false negative out of nothing"
            )
        masks.append(mask)
    return masks


def read_instances(case: Path, directory: str = MASKS_DIRECTORY) -> np.ndarray:
    """The same case as a label map, which is what the metric compares."""
    return label_map(read_instance_masks(case, directory))
