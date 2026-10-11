"""Turn a semantic mask into instances the naive way.

**This is a yardstick, not a contribution.** Connected components is what the published
`dsbowl-unet` weights can be scored with at all, and their card already says why it will do
badly: *semantic, not instance* - touching nuclei come back as one region, and one region is
one instance here. The number it produces is the baseline every model in this milestone is
measured against, and it quantifies a limitation the documentation has so far only named.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from pyplatypus.instances.metrics import InstanceError

#: 4-connectivity, not 8. Two nuclei touching only at a corner are two objects under 4 and
#: one under 8, and a diagonal join is the cheapest way to merge things that are not joined.
#: It makes this baseline slightly kinder, which is the direction to err for a yardstick the
#: later work has to beat.
CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def instances_from_mask(mask: np.ndarray, minimum_area: int = 0) -> np.ndarray:
    """Label every connected foreground region of a binary or class-index mask.

    `minimum_area` drops regions below a pixel count and relabels so the result stays
    consecutive. Zero by default: a floor is a tuning knob, and a baseline that has been
    tuned is no longer the thing later work gets compared against.
    """
    if mask.ndim != 2:
        raise InstanceError(
            f"expected a 2-D mask, got shape {mask.shape}. Instances in volumes are a "
            "separate question from instances in images and this does not answer it"
        )
    labels, found = ndimage.label(mask > 0, structure=CONNECTIVITY)
    if minimum_area <= 0 or found == 0:
        return labels.astype(np.int32)

    areas = np.bincount(labels.ravel())
    keep = np.nonzero(areas[1:] >= minimum_area)[0] + 1
    lookup = np.zeros(found + 1, dtype=np.int32)
    lookup[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    return lookup[labels]
