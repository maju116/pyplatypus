"""YOLOv3: Darknet-53 and three heads.

Written to be **layer-for-layer what the published COCO weights were trained as**, because
those weights are the reason this architecture rather than a newer one, and a structure
that merely resembles Darknet-53 loads them cleanly and predicts nonsense. The order of
the convolutions, the 0.1 slope on the leaky activation, the stride-2 convolutions in
place of pooling, the two upsample-and-concatenate joins: all of it is fixed by that.
Verified by parameter count, which for 80 classes and three anchors is a published number.

What is **not** fixed is the part the old package got right and most implementations
hard-code: `anchors_per_grid` and `n_class` come from the caller and set the width of the
three 1x1 output convolutions, and the input size is free as long as 32 divides it. The
backbone is unaffected by both, which is exactly why COCO's backbone weights can be reused
on three classes of blood cell while its heads cannot.

The heads emit `(batch, grid_h, grid_w, anchors, 5 + n_class)` - channels last at the end,
matching what `encode` produces, so the loss compares like with like and nothing has to
remember a transpose.
"""

from __future__ import annotations

import torch
from torch import nn

from pyplatypus.detection.encode import STRIDES, grid_shapes
from pyplatypus.detection.metrics import DetectionError

#: Darknet's leaky slope. Not torch's default of 0.01, and not a free choice: the weights
#: were fitted against this function.
LEAKY_SLOPE = 0.1


class DarknetConv(nn.Module):
    """Convolution, batch norm, leaky ReLU - Darknet's unit, in Darknet's order.

    The bias is dropped wherever batch norm follows, because the norm's shift subsumes it.
    That is not a saving, it is what the published weights contain: a bias tensor this
    package expected and the file did not have would stop the load, and one it did not
    expect would be silently ignored.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        *,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=not batch_norm,
        )
        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = nn.LeakyReLU(LEAKY_SLOPE, inplace=True) if batch_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.activation(self.norm(x))
        return x


class DarknetResidual(nn.Module):
    """Squeeze to half the channels with a 1x1, restore with a 3x3, add the input."""

    def __init__(self, channels: int):
        super().__init__()
        self.narrow = DarknetConv(channels, channels // 2, 1)
        self.widen = DarknetConv(channels // 2, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.widen(self.narrow(x))


class Darknet53(nn.Module):
    """The backbone, returning the three feature maps the heads join.

    Independent of `n_class` and of `anchors_per_grid`, which is the whole reason COCO's
    backbone transfers to a three-class problem while its heads do not.
    """

    #: (channels after the downsample, how many residual blocks) per stage.
    STAGES = ((64, 1), (128, 2), (256, 8), (512, 8), (1024, 4))

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.stem = DarknetConv(in_channels, 32, 3)
        channels = 32
        stages = []
        for width, repeats in self.STAGES:
            block = [DarknetConv(channels, width, 3, stride=2)]
            block += [DarknetResidual(width) for _ in range(repeats)]
            stages.append(nn.Sequential(*block))
            channels = width
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        routes = []
        for stage in self.stages:
            x = stage(x)
            routes.append(x)
        # Strides 8, 16 and 32 - the last three stages. Returned coarsest first, matching
        # the order the anchors and the heads are in.
        return routes[4], routes[3], routes[2]


class YoloNeck(nn.Module):
    """The five alternating convolutions before a head, and the branch that feeds the next."""

    def __init__(self, in_channels: int, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            DarknetConv(in_channels, channels, 1),
            DarknetConv(channels, channels * 2, 3),
            DarknetConv(channels * 2, channels, 1),
            DarknetConv(channels, channels * 2, 3),
            DarknetConv(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class YoloHead(nn.Module):
    """A 3x3 widening convolution and the 1x1 that emits the predictions.

    The 1x1's width is `anchors * (5 + n_class)`, which is why the published COCO heads -
    3 * 85 = 255 channels - cannot be reused for any other number of classes, and why the
    backbone can.
    """

    def __init__(self, channels: int, anchors: int, n_class: int):
        super().__init__()
        self.widen = DarknetConv(channels, channels * 2, 3)
        self.predict = DarknetConv(channels * 2, anchors * (5 + n_class), 1, batch_norm=False)
        self.anchors = anchors
        self.n_class = n_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.predict(self.widen(x))
        batch, _, height, width = out.shape
        # To channels-last, split into anchors and attributes. This is the shape `encode`
        # writes, so the loss never transposes and a mismatch cannot hide in a permutation.
        out = out.view(batch, self.anchors, 5 + self.n_class, height, width)
        return out.permute(0, 3, 4, 1, 2).contiguous()


class Yolo3(nn.Module):
    """Darknet-53 and three heads, coarsest first.

    `forward` returns a tuple of three tensors shaped
    `(batch, grid_h, grid_w, anchors, 5 + n_class)`, raw - offsets and objectness in logit
    space, as `decode(..., raw=True)` expects.
    """

    def __init__(self, *, n_class: int = 80, anchors_per_grid: int = 3, in_channels: int = 3):
        super().__init__()
        if n_class < 1:
            raise DetectionError(f"n_class must be at least 1; got {n_class}")
        if anchors_per_grid < 1:
            raise DetectionError(f"anchors_per_grid must be at least 1; got {anchors_per_grid}")
        self.n_class = n_class
        self.anchors_per_grid = anchors_per_grid
        self.in_channels = in_channels

        self.backbone = Darknet53(in_channels)
        self.neck_large = YoloNeck(1024, 512)
        self.head_large = YoloHead(512, anchors_per_grid, n_class)

        self.reduce_medium = DarknetConv(512, 256, 1)
        self.neck_medium = YoloNeck(256 + 512, 256)
        self.head_medium = YoloHead(256, anchors_per_grid, n_class)

        self.reduce_small = DarknetConv(256, 128, 1)
        self.neck_small = YoloNeck(128 + 256, 128)
        self.head_small = YoloHead(128, anchors_per_grid, n_class)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.ndim != 4:
            raise DetectionError(
                f"expected a batch of images shaped (n, channels, height, width); got "
                f"{tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise DetectionError(
                f"this model takes {self.in_channels} channels and the batch has {x.shape[1]}"
            )
        height, width = x.shape[2], x.shape[3]
        if height % 32 or width % 32:
            raise DetectionError(
                f"an input of {(height, width)} is not divisible by 32, so the three grids "
                f"would not tile it. See `grid_shapes`."
            )

        deep, middle, shallow = self.backbone(x)

        branch = self.neck_large(deep)
        large = self.head_large(branch)

        joined = torch.cat([self.upsample(self.reduce_medium(branch)), middle], dim=1)
        branch = self.neck_medium(joined)
        medium = self.head_medium(branch)

        joined = torch.cat([self.upsample(self.reduce_small(branch)), shallow], dim=1)
        branch = self.neck_small(joined)
        small = self.head_small(branch)

        return (large, medium, small)

    def grid_shapes(self, input_shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        return grid_shapes(input_shape, scales=3, strides=STRIDES)


def build_yolo3(*, n_class: int = 80, anchors_per_grid: int = 3, in_channels: int = 3) -> Yolo3:
    """The one entry point, matching `build_model` on the segmentation side."""
    return Yolo3(n_class=n_class, anchors_per_grid=anchors_per_grid, in_channels=in_channels)
