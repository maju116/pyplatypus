"""Flows: a vector per pixel pointing at its own object, and the decoding that follows them.

`#159` closed measured-and-no, and the reason is structural rather than a matter of tuning: an
interior and a rim are two mutually exclusive classes competing for the same pixels, so a rim
wide enough to learn is wide enough to hollow the object out. Rim Dice climbed 0.48 to 0.76
across widths 1 to 4 while interior Dice fell 0.90 to 0.81, and instance average precision
peaked at 0.5587 and turned over.

**A flow field has no such competition.** Every foreground pixel carries a unit vector towards
its own object's centre; no pixel is spent on being a boundary, and nothing is subtracted from
the object to make room for the signal. That is the premise of `#212`, and this module is its
first half - the target and the decoding - written and measured against the ground truth
**before anything is trained**, for the reason `parts_from_instances` was: the post-process is
what caps the approach, and a cap can be measured without the thing it caps.

Two ways to build the target, and the difference is the whole question this module's
measurement answers:

    centroid    the straight-line direction to the object pixel nearest its centroid.
                Cheap, exact for a convex blob, and wrong wherever the straight path leaves
                the object - the pixels on the far side of the gap then aim across it
                instead of along the object.
    diffusion   CellPose's: heat diffused from that same pixel *within the object*, and the
                field is the gradient of it. Correct for a concave object by construction,
                because heat cannot cross a boundary the object does not have.

**What makes the cheap one wrong here is finer than "a concave object", and the obvious
reading was measured at almost zero.** On the Data Science Bowl's 134 held-out cases, 6000
objects, **19 have their centroid outside themselves - 0.32%** - and the cheap field still
loses **0.0446** of instance AP and hands back 6268 objects where the truth has 6000. The
cause is one level down: a straight line between two pixels of a *discrete* set can step
outside the set even when the set is convex to the eye, because its boundary is ragged.
Measured on the first 6 of those cases, 274 objects: 0.27% of foreground pixels have such a
path, 6 objects come back split, and **every one of the 6 has pixels whose path leaves it**
while 10 objects have such pixels and survive. So a leaving path is necessary and not
sufficient, and the per-case average of AP is what turns a few objects into 0.0446.

**The ceilings, one run over those 134 cases** - `separation_ceiling.py` prints them and the
set they belong to, and the two must travel together:

    flows:diffusion + follow     0.9924      6043 objects back
    flows:centroid  + follow     0.9478      6268
    rim + nearest_seed           0.9807      6074
    rim + watershed              0.9703
    connected_components         0.7583      the bar any model using it is capped by

So the flow route's ceiling is **0.0117 above the rim route's** on the same pictures, and
neither is what caps a model: the bar is 0.7583. **Do not quote these beside the 0.9846 and
0.7655 already on the record** - no set of cases on this disk reproduces that pair, and
`separation_ceiling.py` carries the four that were tried.

Both use the same sink - the foreground pixel of the object closest to its centroid - so the
pair differs in one variable and nothing else. The centroid itself is not the sink: for a
crescent it can fall outside the object entirely, and a field pointing at a place the object
does not occupy drags pixels off it.

**The field is zero at the sink and unit length everywhere else on the object**, and the zero
is written there rather than arrived at - see `_field_in_window`. Unit length
means the decoding's step size is measured in pixels, and it means the regression target is
bounded in [-1, 1], which is what a linear head and `mse` can represent without being asked
to predict a magnitude as well as a direction. CellPose instead scales each object's field by
its own maximum gradient, which is a difference worth measuring and is not what this does.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from pyplatypus.instances.metrics import InstanceError
from pyplatypus.instances.postprocess import _drop_small

#: How the vectors are built. `centroid` is the cheap one, `diffusion` is CellPose's.
FLOW_MODES = ("centroid", "diffusion")

#: 8-connectivity, and only for grouping the places the flows *land* - not for objects, which
#: this package labels under 4 (`postprocess.CONNECTIVITY`). The argument for it is that a
#: single object's endpoints form one tight cluster a pixel or two across and can straddle a
#: diagonal, while two objects' centres are far enough apart that joining diagonals cannot
#: merge them.
#:
#: **Nothing measurable depends on it here**, and that is said rather than left implied: over
#: 25 held-out cases and 1002 objects, 4-connectivity returns *byte-identical* label maps.
#: So this is the conservative choice against over-splitting and not a finding - the mutation
#: that swaps it is the one mutation this module's tests do not catch, and it is uncaught
#: because there is nothing there to catch.
LANDING_CONNECTIVITY = np.ones((3, 3), dtype=bool)


def flows_from_instances(
    labels: np.ndarray, *, mode: str = "centroid", iterations: int | None = None
) -> np.ndarray:
    """An instance label map as a two-channel unit vector field pointing at each object's sink.

    Returns `(2, H, W)` float32: channel 0 is the **row** component and channel 1 the
    **column** component, in numpy's own axis order, so a field can be indexed the same way
    as the image it describes. Both channels are zero off the foreground and at the sink.

    `mode` is one of `FLOW_MODES`; `iterations` only applies to `diffusion` and defaults to
    twice an object's bounding-box perimeter in pixels, which is enough for heat to reach the
    far end of any path inside it.

    The background's zero is **not a vector pointing nowhere**, it is a value the target does
    not define - which is why a regression loss on this needs `where: foreground`
    (`#209`). Reading the foreground back off this array is the one thing not to do: the
    field is exactly zero at every object's sink, so the non-zero voxels are the foreground
    *minus precisely the seeds*, and `#211`'s oracle measured the seeds as the limit of this
    whole family of methods.
    """
    if mode not in FLOW_MODES:
        raise InstanceError(f"mode is {mode!r}; it has to be one of {FLOW_MODES}")
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise InstanceError(
            f"expected a 2-D label map, got shape {labels.shape}. Instances in volumes are a "
            "separate question from instances in images and this does not answer it"
        )

    flow = np.zeros((2,) + labels.shape, dtype=np.float32)
    present = np.unique(labels[labels > 0])
    if present.size == 0:
        return flow

    objects = ndimage.find_objects(labels.astype(np.intp))
    for value in present:
        window = objects[int(value) - 1]
        if window is None:  # a value with no pixels cannot happen here, but say so rather
            continue  # than index into None if the label map is ever handed in sparse
        mask = labels[window] == value
        rows, columns = _field_in_window(mask, mode=mode, iterations=iterations)
        flow[0][window][mask] = rows[mask]
        flow[1][window][mask] = columns[mask]
    return flow


def _sink(mask: np.ndarray) -> tuple[int, int]:
    """The pixel of this object closest to its centroid, which is always inside it.

    The centroid of a crescent lies in the bay rather than in the object, and a field aimed at
    a place the object does not occupy points pixels across the gap instead of along it. This
    is what both modes aim at, so the comparison between them is about the *path* and not
    about where the path ends - which is the only reason the two are one variable apart.

    On this data the correction is never needed: no object of 274 has its centroid outside
    itself. It is here because the cost of being wrong is an object that cannot be recovered
    at all, and the check is one `argmin` over the object's own pixels.
    """
    coordinates = np.argwhere(mask)
    centre = coordinates.mean(axis=0)
    nearest = np.argmin(((coordinates - centre) ** 2).sum(axis=1))
    return tuple(int(value) for value in coordinates[nearest])


def _field_in_window(
    mask: np.ndarray, *, mode: str, iterations: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """The two components for one object, inside its own bounding box."""
    sink = _sink(mask)
    if mode == "centroid":
        rows, columns = _straight_line_to(mask, sink)
    else:
        rows, columns = _diffusion_gradient(mask, sink, iterations)
    # The sink is zero by **construction and not by arithmetic**, which is the one thing the
    # normalisation below cannot be trusted with. The straight line to the sink is exactly
    # zero there and needs nothing; the diffusion's central difference across a maximum is a
    # residue of the iteration order, and normalising a residue produces a full unit vector -
    # measured at (0.707, 0.707) before this line existed, pointing diagonally away from the
    # very pixel the method is about. A model trained on that learns noise at every object's
    # centre, and the decoding survives it, which is how it went unnoticed.
    rows, columns = rows.copy(), columns.copy()
    rows[sink] = 0.0
    columns[sink] = 0.0
    return _to_unit_length(rows, columns, mask)


def _straight_line_to(mask: np.ndarray, sink: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    grid_rows, grid_columns = np.indices(mask.shape, dtype=np.float32)
    return sink[0] - grid_rows, sink[1] - grid_columns


def _diffusion_gradient(
    mask: np.ndarray, sink: tuple[int, int], iterations: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Heat from the sink, confined to the object, and the gradient that climbs back to it.

    One pad of background around the object so the stencil never wraps, an 8-neighbourhood
    average including the pixel itself, and one unit of heat added at the sink every
    iteration. `log1p` before the gradient because the field spans several orders of magnitude
    between the sink and the far end of a long object, and a gradient of the raw field is
    numerically all sink.
    """
    height, width = mask.shape
    if iterations is None:
        iterations = 2 * (height + width)
    inside = np.zeros((height + 2, width + 2), dtype=bool)
    inside[1:-1, 1:-1] = mask
    heat = np.zeros_like(inside, dtype=np.float64)
    row, column = sink[0] + 1, sink[1] + 1

    for _ in range(iterations):
        heat[row, column] += 1.0
        neighbours = (
            heat[:-2, :-2]
            + heat[:-2, 1:-1]
            + heat[:-2, 2:]
            + heat[1:-1, :-2]
            + heat[1:-1, 1:-1]
            + heat[1:-1, 2:]
            + heat[2:, :-2]
            + heat[2:, 1:-1]
            + heat[2:, 2:]
        )
        heat[1:-1, 1:-1] = np.where(mask, neighbours / 9.0, 0.0)

    warm = np.log1p(heat)
    rows = (warm[2:, 1:-1] - warm[:-2, 1:-1]) / 2.0
    columns = (warm[1:-1, 2:] - warm[1:-1, :-2]) / 2.0
    return rows.astype(np.float32), columns.astype(np.float32)


def _to_unit_length(
    rows: np.ndarray, columns: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unit length on the object, exactly zero where the vector was zero to begin with."""
    length = np.hypot(rows, columns)
    moving = mask & (length > 0)
    unit_rows = np.zeros(mask.shape, dtype=np.float32)
    unit_columns = np.zeros(mask.shape, dtype=np.float32)
    unit_rows[moving] = rows[moving] / length[moving]
    unit_columns[moving] = columns[moving] / length[moving]
    return unit_rows, unit_columns


def follow_flows(
    flow: np.ndarray,
    foreground: np.ndarray,
    *,
    steps: int = 200,
    step_size: float = 1.0,
) -> np.ndarray:
    """Where every foreground pixel ends up after following the field, as float coordinates.

    Returns `(2, N)` for the `N` foreground pixels in row-major order: the row and column each
    one arrived at. Euler steps on a bilinearly interpolated field, which is the standard
    integration here and the reason the field is unit length - a step is then `step_size`
    pixels wherever it is taken.

    **Exposed on its own so the convergence can be probed rather than assumed**, by taking two
    runs at different `steps` and comparing - the same trick as `#211`'s oracle row.

    Done on an exact field, it comes back saying the step count does not matter: one step
    scores 0.9914 against 0.9923 converged over fifteen cases. The reason is structural. The
    field is continuous, so one object's pixels land in a connected set however far along they
    get; and it *diverges* at a seam, so the gap between two objects opens on the first step.
    Under-stepping therefore preserves the separation rather than losing it. **The step size is
    the parameter that is not safe** - above one pixel the integration jumps past the sink and
    the endpoints scatter, 0.9923 at 1.0 against 0.4715 at 2.0.

    Interpolating a field that is zero off the foreground is what stops a pixel that wanders
    out: the vectors it samples shrink towards nothing and it halts. That is deliberate. The
    alternative - clamping to the foreground - would push it along a boundary it has no
    business following.
    """
    flow = np.asarray(flow, dtype=np.float64)
    foreground = np.asarray(foreground)
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise InstanceError(
            f"expected a (2, H, W) field, got shape {flow.shape}; channel 0 is the row "
            "component and channel 1 the column component"
        )
    if flow.shape[1:] != foreground.shape:
        raise InstanceError(
            f"the field is {flow.shape[1:]} and the foreground is {foreground.shape}; they "
            "are two readings of one picture and must be the same size"
        )
    if steps < 1:
        raise InstanceError(
            f"steps is {steps}; following a field for no steps leaves every pixel where it "
            "started, which labels the foreground rather than its basins"
        )
    if step_size <= 0:
        raise InstanceError(f"step_size is {step_size}; a step has to move the pixel somewhere")

    height, width = foreground.shape
    rows, columns = np.nonzero(foreground > 0)
    position = np.stack([rows.astype(np.float64), columns.astype(np.float64)])
    for _ in range(steps):
        moved_rows = _sample(flow[0], position)
        moved_columns = _sample(flow[1], position)
        position[0] = np.clip(position[0] + step_size * moved_rows, 0, height - 1)
        position[1] = np.clip(position[1] + step_size * moved_columns, 0, width - 1)
    return position


def _sample(field: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of one channel at float coordinates."""
    height, width = field.shape
    rows, columns = position
    row_low = np.clip(np.floor(rows).astype(np.intp), 0, height - 1)
    column_low = np.clip(np.floor(columns).astype(np.intp), 0, width - 1)
    row_high = np.minimum(row_low + 1, height - 1)
    column_high = np.minimum(column_low + 1, width - 1)
    row_fraction = rows - row_low
    column_fraction = columns - column_low
    top = field[row_low, column_low] * (1 - column_fraction) + (
        field[row_low, column_high] * column_fraction
    )
    bottom = field[row_high, column_low] * (1 - column_fraction) + (
        field[row_high, column_high] * column_fraction
    )
    return top * (1 - row_fraction) + bottom * row_fraction


def instances_by_flows(
    flow: np.ndarray,
    foreground: np.ndarray,
    *,
    steps: int = 200,
    step_size: float = 1.0,
    minimum_area: int = 0,
) -> np.ndarray:
    """Instances from a flow field: follow every foreground pixel, then label the basins.

    Pixels that arrive at the same place are one object. The places are found rather than
    assumed: the endpoints are rounded to the pixel grid, the grid cells anybody landed on are
    grouped under `LANDING_CONNECTIVITY`, and every pixel takes the group its own endpoint
    fell into.

    **No seed, no interior, nothing eroded** - which is the point of `#212`. The two decodings
    this package already has both need an interior to be predicted as its own class, and that
    class is what `#159` measured as costing the object more than the separation was worth.

    A foreground pixel whose endpoint landed outside every group cannot happen - its own
    endpoint is a landing site by construction - so there is no equivalent here of
    `instances_from_parts`'s foreground with no interior anywhere. The failure that replaces
    it is a field whose basins do not correspond to objects, and no arithmetic in this function
    can tell that from a field whose basins do.
    """
    positions = follow_flows(flow, foreground, steps=steps, step_size=step_size)
    foreground = np.asarray(foreground)
    labels = np.zeros(foreground.shape, dtype=np.int32)
    if positions.shape[1] == 0:
        return labels

    landed_rows = np.rint(positions[0]).astype(np.intp)
    landed_columns = np.rint(positions[1]).astype(np.intp)
    sites = np.zeros(foreground.shape, dtype=bool)
    sites[landed_rows, landed_columns] = True
    basins, found = ndimage.label(sites, structure=LANDING_CONNECTIVITY)
    if found == 0:
        return labels

    rows, columns = np.nonzero(foreground > 0)
    labels[rows, columns] = basins[landed_rows, landed_columns]
    return _drop_small(labels, found, minimum_area)
