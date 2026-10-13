"""The training target of an instance model, built from a label map.

**Built, never corrected, and built last.** A geometric transform moves an object, and a
target that encodes direction or distance has to be rebuilt from the moved objects rather
than transformed alongside them: negating a flow field's x component under a horizontal
flip is correct and insufficient, because a crop moves an object's centre and there is no
component to negate. Measured on `pyplatypus#210`: 49.5% of foreground voxels disagreed by
a median of 8.9 degrees under a corrected-rather-than-rebuilt crop, against 0.0 for a
corrected flip. So the order is read, resize, tile, augment, **then** build - and that
order is what this module exists to be the last step of.

Two targets, and each one's channel order is a contract that the decoding reads back:

    rim     (H, W, n_class + 1)   0 background, 1 interior, 2 rim
    flows   (H, W, 2 + n_class)   0 row flow, 1 column flow, then the classes

The vector channels come **first** in `flows` and the classes keep the order every other
task gives them, which is what `Separation.semantic_from` says in one number. Reading the
foreground off the flow channels is the one thing not to do: the field is exactly zero at
every object's sink, so its non-zero voxels are the foreground minus precisely the seeds -
and the seeds are what limits this whole family of methods (`pyplatypus#211`).
"""

from __future__ import annotations

import numpy as np

from pyplatypus.data.masks import classes_to_onehot
from pyplatypus.instances.flows import flows_from_instances
from pyplatypus.instances.metrics import InstanceError
from pyplatypus.instances.postprocess import parts_from_instances
from pyplatypus.spec.instances import Separation


def resize_labels(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A label map at another size: nearest-neighbour, then relabelled consecutively.

    **An object can vanish here, and one that vanished must not leave a hole in the
    numbering.** The metric counts objects by their values, so a gap would read as an
    object the truth claims and the prediction missed - a false negative invented by the
    resize. Relabelling consecutively is what keeps the count honest, and it is the reason
    this is a function rather than one call to PIL.

    Nearest-neighbour, always: interpolating a label map averages identities and produces
    values belonging to no object.

    Args:
        labels: `(H, W)` integers, 0 for background.
        size: `(height, width)` to resize to, in this package's own axis order.

    Returns:
        `(height, width)` int32, with the objects that survived numbered 1..n.

    Both halves of that, in one example: the single pixel at (6, 6) does not survive the
    halving, and the object that does survive comes back numbered 1 rather than keeping the
    7 it went in with - so a reader counting values never sees a gap where the vanished
    object was.

    >>> labels = np.zeros((8, 8), dtype=np.int32)
    >>> labels[0:4, 0:4] = 1
    >>> labels[6, 6] = 7
    >>> resized = resize_labels(labels, (4, 4))
    >>> resized.shape, sorted(np.unique(resized).tolist())
    ((4, 4), [0, 1])
    >>> int(resize_labels(labels, (8, 8)).max())  # nothing vanishes, and 7 still renumbers
    2
    """
    from PIL import Image

    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise InstanceError(
            f"expected a 2-D label map, got shape {labels.shape}. Instances in volumes are "
            "a separate question and this does not answer it"
        )
    height, width = int(size[0]), int(size[1])
    # mode="I" is 32-bit signed integers, which is what holds an object count of any size a
    # case can have; "L" would wrap at 256 and the 375-object case in this dataset's own
    # validation split would come back with objects renamed rather than resized.
    picture = Image.fromarray(labels.astype(np.int32), mode="I")
    small = np.asarray(picture.resize((width, height), Image.Resampling.NEAREST))

    present = np.unique(small[small > 0])
    lookup = np.zeros(int(small.max()) + 1 if small.size else 1, dtype=np.int32)
    lookup[present] = np.arange(1, present.size + 1, dtype=np.int32)
    return lookup[small]


def instance_target(
    labels: np.ndarray,
    separation: Separation,
    n_class: int = 2,
    *,
    width: int | None = None,
    mode: str | None = None,
) -> np.ndarray:
    """The channels an instance model is trained against, channels-last float32.

    Args:
        labels: the label map this sample's objects are in, **after** every geometric step.
        separation: which target to build. `Separation`'s own description has what each one
            is worth and what it costs.
        n_class: how many classes the data names, which is 2 for every instance dataset
            today - background and object. Passed rather than assumed so the one place that
            would have to change is an argument rather than a literal in the middle of a
            concatenate.
        width: the rim's width in pixels, for `rim`.
        mode: how the field is built, for `flows`.

    Returns:
        `(H, W, Separation.head_channels(n_class))` float32. The channel order is the
        contract in this module's docstring.

    Raises:
        InstanceError: if a setting belongs to the other mechanism. The specification
            refuses that combination before anything is read; this refuses it again,
            because the engine is not the only caller - an example script assembling the
            pieces by hand reaches here directly.

    >>> labels = np.zeros((8, 8), dtype=np.int32)
    >>> labels[2:6, 2:6] = 1
    >>> instance_target(labels, Separation.RIM, width=1).shape
    (8, 8, 3)
    >>> target = instance_target(labels, Separation.FLOWS, mode="centroid")
    >>> target.shape
    (8, 8, 4)
    >>> bool((target[..., 3] == (labels > 0)).all())
    True
    """
    labels = np.asarray(labels)
    if separation is Separation.RIM:
        if mode is not None:
            raise InstanceError(f"mode={mode!r} is for flows; rim takes a width")
        interior, rim = parts_from_instances(labels, width=1 if width is None else width)
        classes = np.zeros(labels.shape, dtype=np.int64)
        classes[interior] = 1
        classes[rim] = 2
        return classes_to_onehot(classes, n_class + 1).astype(np.float32)

    if width is not None:
        raise InstanceError(f"width={width!r} is for rim; flows takes a mode")
    flow = flows_from_instances(labels, mode="diffusion" if mode is None else mode)
    classes = (labels > 0).astype(np.int64)
    semantic = classes_to_onehot(classes, n_class).astype(np.float32)
    return np.concatenate([np.moveaxis(flow, 0, -1), semantic], axis=-1)
