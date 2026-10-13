"""Instances rather than classes: the third task's data, model and decodings.

A semantic mask says *what* every pixel is. An instance map says *which object* it belongs
to, and the difference is not a post-process bolted onto the first: the masks arrive one
file per object and keep their identity, the head is wider than the class count, and the
score is an average precision over matched objects rather than an overlap.

**Two fields are new and everything else is inherited** - `input_shape`, `channels`,
`blocks`, `filters`, `batch_size`, `epochs`, `splits`, the pretrained encoders, the
callbacks and the augmentation all mean here exactly what they mean for a semantic model,
which is why `InstanceModel` is a subclass rather than a sibling:

    separation:  one block         what the network predicts, which sets the head's
                                   width and which losses are admissible, carrying
                                   that mechanism's own settings
    decode:      a list of blocks  how those channels become objects, with
                                   `connected_components` always available

**`separation` is a noun and `decode` is a verb, and no value appears in both.** That is
the rule that settled the shape, after a first version put `rim` and `watershed` in one
field and was refuted by a single question: *can `rim` go in both?* It cannot, and if any
value could, the field names would be wrong.

`decode` can be a list and `separation` cannot, because several decodings share one target
while several targets do not share a training run. `rim` + watershed and `rim` +
nearest-seed are the same channels and the same loss, so one trained model serves both.
Measured on the Data Science Bowl's 134 held-out cases: a second decoding of all of them
costs **1.41 s** against **16 minutes** for a second training run, a hundredth of one
epoch - so the right analogy is `metrics:` and not `callbacks:`.

**What the numbers say about the axis itself, because they are the reason to read its
documentation twice.** Trained on the 134 held-out cases, against the same model's own mask
through `connected_components`: `rim` is worth **+0.0640** of instance average precision
and `flows` **+0.0680**, and the 0.0040 between them is below the 0.0074 two runs of one
configuration differ by. A perfect separation surface is worth +0.003 to the rim route; a
**perfect mask is worth +0.2398**. So on this data the separation signal is not what limits
instance segmentation - the mask is, by a factor of 3.6 - and this axis names a choice that
moves the score by less than the noise between two runs. It is here because a configuration
format exists so somebody can ask the question of *their* data, where the answer may differ,
and because every number above was measured on microscopy of touching nuclei. It is not
here because it was expected to pay. `JOURNAL.md` §4bn, §4bo, §4bq; `pyplatypus#212`.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, ClassVar, Literal

from pydantic import Field, model_validator

from pyplatypus.spec.common import SpecModel
from pyplatypus.spec.components import LossSpec, MseLoss
from pyplatypus.spec.data import DataSpec
from pyplatypus.spec.models import SegmentationModel


class Separation(str, Enum):
    """The signal that makes one mask into many objects.

    One per model, because this is what the network is trained to predict: it sets the
    width of the head and which losses mean anything against it.

    **This enum is the vocabulary rather than the field.** A specification names a signal
    inside a block - `separation: {name: rim, width: 3}` - so that the mechanism's own
    settings sit in the mechanism they belong to, and `_Separation.signal` is what ties a
    block to its member here. The enum is what the *numeric* layer takes, because
    `instance_target` and `decode_instances` have to know the signal and must know nothing
    about the spec; it is also what a weights fingerprint records, since the settings
    change what the channels were trained towards and not what they are.

    **`hv` is deliberately not a member**, and that absence is a measurement rather than a
    gap in the implementation. HoVer-Net's two distance maps were swept both ways they can
    be turned into markers, on the ground truth, which is the test `pyplatypus#213` set
    itself: the maps' own zeros reach an instance average precision of 0.9382 and the
    published gradient route 0.7995, where a one-pixel erosion of the same truth reaches
    **0.9807** and plain connected components 0.7583. The better of the two recipes loses
    to the cheapest marker there is by 0.0425 on perfect inputs, so a trained one cannot
    win; `#213` is closed measured-and-no. Naming the value would promise a mechanism this
    package measured as absent.
    """

    #: Three mutually exclusive classes - background, object interior, object rim - so the
    #: head is one channel wider than the data's class count and the ordinary softmax
    #: losses apply unchanged. Measured at its best width (3 px) it reaches 0.5587 where a
    #: perfect semantic mask through connected components reaches 0.7655, and the curve has
    #: a maximum because **a rim wide enough to learn is wide enough to hollow the object
    #: out**: rim Dice climbs 0.48 to 0.76 across widths 1 to 4 while interior Dice falls
    #: 0.90 to 0.81. `pyplatypus#159`.
    RIM = "rim"

    #: A two-channel unit vector field pointing at each object's sink, plus the semantic
    #: channels. **No pixel is spent on being a boundary**, which is the structural argument
    #: for it and the reason it was tried after `rim` closed - and on this data it is worth
    #: 0.0040 more than `rim`, below the noise between two runs. Its decoding's ceiling on
    #: perfect parts is 0.9873 against the rim route's 0.9629. `pyplatypus#212`.
    FLOWS = "flows"

    def head_channels(self, n_class: int) -> int:
        """How wide the output has to be for this signal, given what the data names.

        **Derived, never stated**, and that is the whole of this method's reason to exist.
        `n_class` already comes from the data - a model restating it could only disagree
        with its own masks, which is a mistake this package made once and fixed - and the
        two new targets add channels of their own: `rim` adds the third class and `flows`
        adds two continuous ones in front of the semantic block. A field saying `channels:
        4` would be a setting that can lie about the target it was built for, and the
        shape error it causes surfaces deep inside torch rather than here.

        It is also the one line that forced `examples/train_instance_flows.py` to assemble
        its model outside `Engine` and call `build_model(spec, n_class=4)` by hand.

        >>> Separation.RIM.head_channels(2), Separation.FLOWS.head_channels(2)
        (3, 4)
        """
        if self is Separation.RIM:
            return n_class + 1
        return 2 + n_class

    @property
    def semantic_from(self) -> int:
        """The first channel of the class block, which is where the mask is read from.

        0 for `rim`, where every channel is already a class, and 2 for `flows`, where the
        vector field comes first. Everything that reads a mask out of an instance model's
        output - the metrics, the foreground a decoding is allowed to label, `predict` -
        needs this one number, and deriving it beside the width keeps the two from
        disagreeing.

        **The class block runs to the end of the head, and it is not the same width at the
        two separations.** `flows` leaves the classes as the data names them, so its block
        is `n_class` wide; `rim` splits the foreground into interior and rim, so its block
        is `n_class + 1` wide and the extra class is where its own channel went. That
        asymmetry is why both numbers are derived here rather than written down once as
        "the last `n_class` channels", which would be right about one target and quietly
        wrong about the other.

        >>> Separation.RIM.semantic_from, Separation.FLOWS.semantic_from
        (0, 2)
        """
        return 2 if self is Separation.FLOWS else 0


_SEPARATION_NAME = (
    "Which separation signal, and it decides which other keys this block accepts: `rim` "
    "takes `width`, `flows` takes `mode`, `loss` and `weight`. A setting that belongs to "
    "the other mechanism is not refused here, it is **unwritable** - `width` under `name: "
    "flows` is an unknown key, which is the answer every typo in a specification gets.\n\n"
    "That is this block's reason to be a block, and it was built flat first: `separation: "
    "rim` beside `rim_width`, `flow_mode`, `flow_loss` and `flow_weight`. The flat form "
    "parses, so it needs a validator and a table of which field belongs to which "
    "mechanism to refuse half of what it accepts, and it shows every reader four fields of "
    "which two can apply to them. This form shows a reader of `name: rim` one field, and "
    'the refusal is gone rather than written - `extra="forbid"` was already doing that '
    "job for every other key in the file."
)


class _Separation(SpecModel):
    """What the network predicts, so that one mask can become many objects.

    One block per model, because this is what the network is trained against: it sets the
    width of the head, which losses mean anything against it, and which decodings can read
    what comes out. Its other keys are the target builder's own settings, and each lives in
    the block it applies to - so there is no such thing here as a rim width on a flow field.

    **This is a noun and `decode` is a verb, and no value appears in both.** The module
    docstring has the question that settled it.
    """

    #: Which signal this block configures, as the enum the numeric layer speaks:
    #: `instance_target` and `decode_instances` take it and know nothing about the spec at
    #: all. A `ClassVar` rather than a field because it is what the block *is*, where
    #: `name` is how that is spelled in a file. What holds the two together is
    #: `test_each_separation_block_names_its_signal`, rather than a `Literal[Separation.RIM]`
    #: that would put the enum into the schema for a reader who only writes a string.
    signal: ClassVar[Separation]

    def head_channels(self, n_class: int) -> int:
        """How wide the head has to be for this signal. See `Separation.head_channels`."""
        return self.signal.head_channels(n_class)

    @property
    def semantic_from(self) -> int:
        """Which channel the class block starts at. See `Separation.semantic_from`."""
        return self.signal.semantic_from

    @property
    def target_settings(self) -> dict:
        """This block's own keys, as the target builder's keyword arguments.

        One place anything asks, so that neither the engine nor an example script has to
        know which settings belong to which signal. With the block form that question has
        no second place left to be answered wrongly: a setting cannot be written outside
        the block it belongs to, so this is a translation rather than a selection.
        """
        raise NotImplementedError  # pragma: no cover - unreachable while the union is exhaustive

    @property
    def objective(self) -> LossSpec | None:
        """The loss on this signal's own channels, where it has any that are not classes.

        `None` for every signal whose head is all classes, which `rim` is: there the
        model's `loss` covers the whole head and there is no second term to weigh it
        against. `Flows` is what overrides it, and `InstanceObjective` is what reads it.
        """
        return None


class Rim(_Separation):
    """Background, object interior, object rim - three classes, one extra channel.

    The cheapest signal there is: the head is one channel wider than the data's class
    count and the ordinary softmax losses apply to all of it unchanged, so nothing about
    the objective or the activation is special. What it costs is that the rim is taken out
    of the objects, which is the trade `Separation.RIM` has the numbers for.

    ```yaml
    separation:
      name: rim
      width: 2          # in pixels of the model's input, and not the published sweep's 3
    decode:
      - name: nearest_seed
      - name: connected_components
    ```
    """

    signal: ClassVar[Separation] = Separation.RIM

    name: Literal["rim"] = Field("rim", description=_SEPARATION_NAME)

    width: int = Field(
        2,
        ge=1,
        description=(
            "How many pixels wide the rim class is, **in pixels of the model's own "
            "input** - so it means the same thing whether the image was resized or tiled, "
            "because the target is built after the resize and the augmentation rather than "
            "before (`pyplatypus#210`).\n\n"
            "**There is a measured sweep of this quantity and it is not a sweep of this "
            "field, which is worth knowing before quoting either.** Four runs differing in "
            "nothing else scored 0.5169, 0.5486, **0.5587** and 0.5306 of instance average "
            "precision at widths 1 to 4, against a 0.0074 spread between two runs of one "
            "configuration - but that pipeline derived the rim at the *source* resolution "
            "and resized the three-class map afterwards. A Data Science Bowl case is around "
            "520 pixels and the model reads 256, so the peak of that sweep is roughly 1.5 "
            "pixels of model input, not 3. The default is the nearest width above it. A "
            "sweep in these units has not been run.\n\n"
            "What that sweep does establish is the shape of the trade, which is the part "
            "that transfers: **a rim wide enough to learn is wide enough to hollow the "
            "object out**, so rim Dice climbed 0.48 to 0.76 across those widths while "
            "interior Dice fell 0.90 to 0.81, and past the peak the model found *more* "
            "regions and matched fewer of them. Both halves are reported per run, because "
            "a single instance number cannot say which of them moved."
        ),
    )

    @property
    def target_settings(self) -> dict:
        """The rim's width, under the name `instance_target` takes it by."""
        return {"width": self.width}


class Flows(_Separation):
    """A unit vector field pointing at each object's sink, plus the semantic channels.

    Two continuous channels in front of the classes, and **no pixel spent on being a
    boundary** - which is the structural argument for it, and the reason it was built after
    `rim` closed measured-and-no. It makes the objective composite and the activation
    partial: the vectors stay as the network produced them while the classes are softmaxed,
    and `InstanceObjective` is where those two halves meet. `Separation.FLOWS` has what it
    turned out to be worth.

    ```yaml
    separation:
      name: flows
      mode: diffusion
      loss:
        name: smooth_l1
        where: foreground
      weight: 1.0
    ```
    """

    signal: ClassVar[Separation] = Separation.FLOWS

    name: Literal["flows"] = Field("flows", description=_SEPARATION_NAME)

    mode: Literal["centroid", "diffusion"] = Field(
        "diffusion",
        description=(
            "How the vector field is built: towards each object's centroid in a straight "
            "line, or down the gradient of heat diffused inside it.\n\n"
            "**`diffusion` is the default because the two were measured on the ground truth "
            "before anything was trained**: decoded back into objects they reach 0.9924 and "
            "0.9478. The obvious explanation for the gap is not the cause - only 19 of 6000 "
            "objects have their centroid outside themselves, 0.32% - and the real one is "
            "one level down: a straight line between two pixels of a discrete set can leave "
            "the set even when the object looks convex, which splits an object whose own "
            "centroid is inside it.\n\n"
            "`centroid` is 23 times cheaper to build (4.9 ms a case against 113.6 ms), "
            "which is the reason to reach for it and is small beside an epoch."
        ),
    )

    loss: LossSpec | None = Field(
        None,
        description=(
            "The objective on the two vector channels, which is a different question from "
            "the model's own `loss` - that one scores the mask. A continuous target has "
            "nothing in the overlap-and-cross-entropy losses to be trained against, so "
            "this is one of `mse`, `l1` or `smooth_l1`, and anything else is refused by "
            "name: `loss: dice` against a vector field would return a number and a "
            "gradient, which is the worst available outcome.\n\n"
            "Left unset it is `mse` with `where: foreground`, which is what the measured "
            "run used. `where` is the field to read before changing it, because the "
            "foreground is 13.97% of a Data Science Bowl image and `everywhere` spends 86% "
            "of the objective on a background zero."
        ),
    )

    weight: float = Field(
        1.0,
        gt=0,
        description=(
            "How much the vector term counts against the mask term, which are summed. The "
            "default is 1.0 and it is measured rather than conventional: on the 60-epoch "
            "run the two terms are the same order of magnitude throughout - 0.5937 against "
            "0.6345 after the first epoch and 0.0734 against 0.0659 at the last - so "
            "neither half of the objective is drowning the other at this weight."
        ),
    )

    @property
    def target_settings(self) -> dict:
        """How to build the field, under the name `instance_target` takes it by."""
        return {"mode": self.mode}

    @property
    def objective(self) -> LossSpec:
        """The vector term, derived when the block stayed silent.

        The same silent-means-derived rule as `InstanceModel.decodings`, and for the same
        reason: `loss` stays `None` in `save_spec`'s output, so an older engine never
        receives a key it does not know and `model_fields_set` can still tell a stated
        objective from the default that was there anyway.
        """
        return self.loss if self.loss is not None else MseLoss()

    @model_validator(mode="after")
    def a_vector_field_needs_a_continuous_loss(self):
        """`loss: dice` over two directions is refused by name, before anything is built.

        The `continuous` flag on the loss itself is what answers it, rather than a list of
        names kept here that a new regression loss would have to be added to. The other
        half of the question - an overlap loss is what the *mask* needs - is on
        `InstanceModel`, because `loss` there applies at either separation.
        """
        if self.loss is not None and not self.loss.continuous:
            raise ValueError(
                f"separation.loss '{self.loss.name}' is defined over classes, and the two "
                f"flow channels are a vector field - there is no overlap between a "
                f"direction and a direction. Use mse, l1 or smooth_l1; unset it is mse "
                f"with where: foreground."
            )
        return self


#: Keyed on `name`, exactly as `loss` and `decode` are. The two mechanisms' settings live
#: in their own blocks, so `width` with `flows` is an unknown key rather than a refusal
#: somebody had to write - which is the whole difference from the flat form this replaced.
SeparationSpec = Annotated[Rim | Flows, Field(discriminator="name")]


_DECODE_NAME = (
    "Which decoding, and it decides which other keys this block accepts - `follow_flows` "
    "takes `steps` and `step_size`, the others do not. A decoding the chosen `separation` "
    "cannot feed is refused by name with both halves stated, rather than returning a label "
    "map of something else."
)
_MINIMUM_AREA = (
    "Drop an object smaller than this many pixels. 0 keeps everything, which is the "
    "default because the right floor is a property of the data rather than of the method - "
    "a nucleus at 256x256 in the Data Science Bowl runs from 4 pixels up, so a floor "
    "chosen here would silently delete the smallest real objects in one dataset while "
    "letting speckle through in another.\n\n"
    "It is applied after the objects are formed, so it removes answers rather than "
    "changing how they were reached."
)


class _Decode(SpecModel):
    #: Which separation signals this decoding can read, empty meaning every one of them.
    #: A class variable rather than a field: it is a fact about the algorithm and not a
    #: setting, and `a_decoding_needs_the_signal_it_reads` is what turns it into a refusal
    #: naming both halves. Kept here rather than as a table in the validator so that a new
    #: decoding declares its own requirement beside its own description.
    needs: ClassVar[frozenset[Separation]] = frozenset()


class ConnectedComponents(_Decode):
    """Label the thresholded foreground and ignore the separation signal entirely.

    **The yardstick, and the reason it is compatible with every target.** It is what any
    semantic model already gives for free, so a row of it beside the signal's own row is
    what makes the signal's number readable: the same model, the same mask, the same
    pictures, with and without the thing being measured. Everything else is 1 ms an image
    against the 11 ms the others cost, so there is no reason to leave it out.

    It is also a ceiling nothing can pass: touching objects are one region, so on the Data
    Science Bowl's held-out cases a **perfect** semantic mask scores 0.7655 through this at
    native size and 0.7505 at 256x256. A run that lands between the published weights'
    0.4888 and that number has improved its mask and said nothing about separation.

    ```yaml
    decode:
      - name: connected_components
    ```
    """

    name: Literal["connected_components"] = Field("connected_components", description=_DECODE_NAME)
    minimum_area: int = Field(0, ge=0, description=_MINIMUM_AREA)


class NearestSeed(_Decode):
    """Grow the predicted interiors outward: every other foreground pixel joins the nearest.

    A Voronoi partition of the foreground around the interiors, which one distance
    transform answers. Not a watershed: it splits on distance alone, so the join lands on
    the geometric midline wherever the picture says the objects part. For nuclei - small,
    roughly convex, meeting along a short seam - the two agree closely.

    **The better of the two rim decodings, measured**: 0.5561 against the watershed's
    0.5458 on one trained model, and 0.9846 against 0.9703 on the ground truth. A
    foreground pixel no seed can reach stays background rather than becoming an object of
    its own, which is the honest answer when the model said "object here" and "no centre
    anywhere".
    """

    needs: ClassVar[frozenset[Separation]] = frozenset({Separation.RIM})

    name: Literal["nearest_seed"] = Field("nearest_seed", description=_DECODE_NAME)
    minimum_area: int = Field(0, ge=0, description=_MINIMUM_AREA)


class Watershed(_Decode):
    """Flood the foreground from the predicted interiors, over the rim evidence as terrain.

    The interiors are the markers and the model's own rim probability is the surface, so a
    join lands on a ridge rather than on the midline between two seeds - which is the
    information `nearest_seed` throws away when it thresholds the rim and never looks at it
    again. The engine passes that probability; there is nothing to configure, because the
    one defensible choice of surface is the model's own.

    **Measured, and it does not pay here.** On one trained model it reaches 0.5458 against
    `nearest_seed`'s 0.5561 - 0.0103 below, which is 7.8 times the 0.0013 standard deviation
    of three runs of the identical configuration. Surface quality matters a great deal to
    it, and to nothing: a *perfect* rim as the surface is worth +0.0132, ten times that
    standard deviation, and only draws level with the decoding that ignores the surface
    altogether. So the seeds decide which objects exist and the seam between them is
    second-order. `pyplatypus#211`, closed measured-and-no.
    """

    needs: ClassVar[frozenset[Separation]] = frozenset({Separation.RIM})

    name: Literal["watershed"] = Field("watershed", description=_DECODE_NAME)
    minimum_area: int = Field(0, ge=0, description=_MINIMUM_AREA)


class FollowFlows(_Decode):
    """Follow the field from every foreground pixel and label the basins they arrive in.

    **No seed, no interior, nothing eroded**, which is the point of the flow target: pixels
    that arrive at the same place are one object, and the places are found rather than
    assumed. The highest ceiling of the four on perfect parts, 0.9873.
    """

    needs: ClassVar[frozenset[Separation]] = frozenset({Separation.FLOWS})

    name: Literal["follow_flows"] = Field("follow_flows", description=_DECODE_NAME)
    minimum_area: int = Field(0, ge=0, description=_MINIMUM_AREA)
    steps: int = Field(
        200,
        ge=1,
        description=(
            "How many Euler steps each pixel takes. **Measured as not mattering**: one step "
            "scores 0.9914 against 0.9923 converged, a span below the 0.0013 two runs "
            "differ by. The field is continuous, so one object's pixels stay together, and "
            "it diverges at a seam, so the gap between two objects opens on the first step "
            "rather than after many. Left adjustable because that argument is about this "
            "data's object sizes, and lowering it is the cheap thing to try on larger ones."
        ),
    )
    step_size: float = Field(
        1.0,
        gt=0,
        description=(
            "How far one step moves, in pixels, on a unit-length field. **This one is a "
            "cliff**: 0.9923 at 1.0, 0.4715 at 2.0, 0.0303 at 10.0. A step longer than a "
            "pixel jumps over the sink it was converging on and lands in a neighbour's "
            "basin, so this is not a speed knob - raise it and the decoding stops working."
        ),
    )


#: Keyed on `name`, exactly as `metrics:` and `callbacks:` are, so a misspelled decoding
#: fails by name at load time rather than silently doing nothing. No new machinery.
DecodeSpec = Annotated[
    ConnectedComponents | NearestSeed | Watershed | FollowFlows,
    Field(discriminator="name"),
]


class InstanceData(DataSpec):
    """Where the images and their per-object masks are.

    **The one thing this says that a semantic data block cannot**: the masks are read one
    file per object and keep their identity. `unite_masks()` is what the semantic pipeline
    does with the same files, and its own docstring says the cost - the element-wise maximum
    merges them, so instance identity is gone at read time. Nothing new has to be collected
    to train or score instances on the Data Science Bowl, which ships 670 cases of exactly
    this shape; it only has to be read differently.

    There is no `colormap`, no `labels` and no `regions` here, because none of them is how
    this data names anything: a directory of per-object files says where the objects are and
    says nothing about what kind they are. `n_class` is therefore 2 and derived rather than
    declared - see the property.
    """

    subdirs: tuple[str, str] = Field(
        ("images", "masks"),
        description=(
            "For `nested_dirs`, the two subdirectories of each sample directory: the images, "
            "and **one mask file per object**. A case with 375 nuclei has 375 files here, "
            "which is the Data Science Bowl's own layout; they are read as a label map with "
            "one value per object rather than united into one mask, which is the whole "
            "difference between this task and `semantic_segmentation`."
        ),
    )

    @property
    def n_class(self) -> int:
        """Two: background, and object.

        Derived rather than declared because nothing in this data could say otherwise. The
        per-object files carry identity and no class - a nucleus mask does not say what kind
        of nucleus - so a `colormap` or a `labels` list would be a naming with nothing to
        name. Several *kinds* of object is a real question and it is `classes_from`'s, on
        `pyplatypus#91`, which is deferred and has nothing built on purpose.

        It is 2 rather than 1 because the semantic block of the head is the ordinary
        background-and-foreground pair the rest of the package already trains, scores and
        thresholds; `Separation.head_channels` adds the target's own channels to it.
        """
        return 2

    @property
    def label_map(self) -> bool:
        """False: these are per-object pictures, not a label map on disk.

        The label map is built by reading them, which is a different thing from the files
        already being one - and the readers branch on this.
        """
        return False

    @property
    def multilabel(self) -> bool:
        """False, and not because it was not thought about.

        A voxel belongs to one object, and the measurement is on the issue that paired these
        two: every per-object mask in all 670 Data Science Bowl cases, 29,461 objects, **0 of
        670 overlapping anywhere**. So instances need no mask layer holding several things
        per pixel, which is what dissolved the pairing with `pyplatypus#158` rather than
        answering it.
        """
        return False


class InstanceModel(SegmentationModel):
    """One instance model: a U-shaped network, a separation target, and how to decode it.

    Everything a semantic model has, plus the two axes. `loss` and `metrics` keep their
    meaning and apply to the **semantic channels** - the mask - which is why the first
    number any of these runs reports is foreground Dice: if the mask moved, the instance
    number moved for a reason that is not separation, and nothing else in the table can
    tell those apart.
    """

    separation: SeparationSpec = Field(
        description=(
            "What the network predicts so that one mask can become many objects: a block "
            "named `rim` or `flows`, carrying that mechanism's own settings. **Required, "
            "with no default**, because it decides the width of the head and which losses "
            "are admissible - the two things a model cannot be built without - and because "
            "neither value is a safe guess at what somebody meant.\n\n"
            "A block rather than a bare name because the settings differ between the two "
            "and belong inside the one they describe: `width` under `name: flows` is an "
            "unknown key rather than something a validator has to refuse.\n\n"
            "It is also the field whose documentation carries a result: on the Data Science "
            "Bowl the two are worth +0.0640 and +0.0680 of instance average precision over "
            "the same mask through `connected_components`, a difference below the noise "
            "between two runs of one configuration, while a perfect mask is worth +0.2398. "
            "Read the blocks' own descriptions for what that means before planning a "
            "comparison around this axis."
        ),
    )

    decode: list[DecodeSpec] | None = Field(
        None,
        min_length=1,
        description=(
            "How the predicted channels become objects. **A list**, because several "
            "decodings share one forward pass: a second decoding of 134 images costs 1.41 s "
            "against 16 minutes for a second training run, so this is `metrics:` rather "
            "than `callbacks:` - `evaluate()` returns one row per (model, decoding).\n\n"
            "Each entry is a block with a `name` and that decoding's own parameters. A "
            "decoding the `separation` cannot feed - `follow_flows` with `rim`, say - is "
            "refused at load time with both halves named, rather than returning a label map "
            "of something else.\n\n"
            "**Left unset it is the signal's own decoding plus `connected_components`**, "
            "which is the pair that makes a row readable: the same model and the same mask "
            "with and without the separation signal. Write the list to choose; leave it out "
            "and the comparison every one of these runs needs is already there.\n\n"
            "With `splits`, decoding happens after the tiles are stitched, never per tile - "
            "an object crossing a seam would otherwise become two objects, and that is a "
            "constraint of the task rather than a setting."
        ),
    )

    @property
    def signal(self) -> Separation:
        """Which separation signal, without the block's settings around it.

        What the numeric layer takes: `instance_target` and `decode_instances` are handed
        this enum, and everything that branches on the mechanism rather than reading one of
        its settings asks here. One named hop rather than `separation.signal` spelled out
        at every call site, and the same shape as the three delegations below it.
        """
        return self.separation.signal

    @property
    def decodings(self) -> tuple[DecodeSpec, ...]:
        """The decodings this model asks for - **derived when it stayed silent**.

        The derived pair is the signal's own decoding and `connected_components`, in that
        order, because the second is what the first is read against. Deriving rather than
        defaulting in the field keeps the specification a description of what somebody
        wrote: `decode` stays `None` in `save_spec`'s output and in what R sends, so an
        older engine never receives a key it does not know, and `model_fields_set` can still
        tell a stated single decoding from the pair that was there anyway.
        """
        if self.decode is not None:
            return tuple(self.decode)
        own = NearestSeed() if self.signal is Separation.RIM else FollowFlows()
        return (own, ConnectedComponents())

    @property
    def objective(self) -> LossSpec | None:
        """The vector term's loss: derived for `flows`, absent for `rim`.

        The block answers it, because the objective on a signal's own channels is a
        property of that signal - `rim` has none and `Flows.objective` is where the
        silent-means-`mse` rule is written.
        """
        return self.separation.objective

    def head_channels(self, n_class: int) -> int:
        """How wide this model's output is, given what the data names. See `Separation`."""
        return self.separation.head_channels(n_class)

    @property
    def target_settings(self) -> dict:
        """The mechanism's own parameters, as the target builder's keyword arguments.

        Asked of the block, which is the only thing that has them - so nothing outside it
        has to know which settings belong to which signal, and with the block form nothing
        *can* ask the question wrongly.
        """
        return self.separation.target_settings

    def weights_fingerprint(self) -> dict:
        """What must match for a file of weights to belong to this model.

        The separation **signal** is in it, which is the point of overriding: a `rim` head
        and a `flows` head of the same architecture, blocks and filters have the same shape
        at `n_class` 2 and 1 respectively, and they mean entirely different things. Without
        this, weights trained to predict a vector field would load cleanly into a model
        whose channels are classes and every prediction would be confidently wrong - the
        same failure the colormap went into the comparison for.

        The block's **settings** stay out, and that is a decision rather than an omission:
        `width` and `mode` change what the channels were trained *towards* and not what
        they are, so weights from a 2-pixel rim are both the right shape and the right
        meaning for a model asking for 3, which is the ordinary way to start one run from
        another.
        """
        return {**super().weights_fingerprint(), "separation": self.signal.value}

    @model_validator(mode="after")
    def instances_are_two_dimensional(self):
        """Both targets and all four decodings are written for images.

        `flows_from_instances` and `instances_from_parts` refuse a label map that is not 2-D
        by name, and this refuses the specification that would reach them - before anything
        is read, rather than on the first sample. Instances in volumes are a separate
        question and this does not answer it: a flow field in 3-D is three channels, a rim
        is a shell whose thickness is a fraction of an object rather than of its outline,
        and neither follows from the 2-D version by changing a loop bound.
        """
        if self.rank != 2:
            raise ValueError(
                f"instance_segmentation is 2-D; input_shape={tuple(self.input_shape)} is "
                f"{self.rank}D. Instances in volumes are a separate question - a flow field "
                f"gains a component and a rim becomes a shell - and nothing here answers it."
            )
        return self

    @model_validator(mode="after")
    def a_decoding_needs_the_signal_it_reads(self):
        """`follow_flows` with `rim` means nothing, so it is refused by name.

        The way `lovasz` with `regions` is: at load time, with the reason stated, rather
        than returning a number and a gradient. Both halves are named because either could
        be the one that is wrong - somebody may have meant a different decoding or a
        different target, and a message naming only the decoding guesses which.
        """
        if self.decode is None:
            return self
        wrong = [
            f"{block.name} needs separation: " + " or ".join(sorted(s.value for s in block.needs))
            for block in self.decode
            if block.needs and self.signal not in block.needs
        ]
        if wrong:
            listed = "\n  - ".join(wrong)
            raise ValueError(
                f"separation is '{self.signal.value}', and these decodings read a "
                f"different signal:\n  - {listed}\n"
                "connected_components reads none of them and works with every target, "
                "which is why it is the yardstick every other row is compared against."
            )
        return self

    @model_validator(mode="after")
    def the_mask_needs_a_loss_over_classes(self):
        """A continuous loss over a one-hot target, refused at either separation.

        `loss` applies to the semantic channels - the mask - whatever the separation is, so
        it is always an overlap or a cross-entropy loss. The other half of the question is
        `Flows.a_vector_field_needs_a_continuous_loss`, which is where the vector channels'
        objective now lives; both read the `continuous` flag on the loss itself rather than
        a list of names kept here that a new loss would have to be added to.

        **The validator that used to stand beside this one is gone rather than moved.**
        `a_mechanism_setting_needs_its_mechanism` refused `rim_width` under `flows` and the
        five other crossings of a flat field with the wrong separation, from a `_BELONGS_TO`
        table that a fifth setting would have had to be added to. In the block form those
        combinations cannot be written at all, so `extra="forbid"` answers them with the
        message every other unknown key gets, and there is no table left to drift. That is
        the whole argument for the block, and it is worth having written down where the
        refusal used to be: a refusal is code to keep in step, and an unwritable thing is
        not.
        """
        if self.loss.continuous:
            raise ValueError(
                f"loss '{self.loss.name}' scores a continuous target, and `loss` is the "
                f"objective on the semantic channels - the mask - at either separation. Run "
                f"as it stands it would score raw output against 0 and 1 with no activation "
                f"between them. Use an overlap or cross-entropy loss here; for "
                f"separation: flows the vector channels have their own `loss` inside the "
                f"block."
            )
        return self
