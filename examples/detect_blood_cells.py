"""Train YOLOv3 on BCCD, from nothing, and say what it scores.

    python examples/detect_blood_cells.py --data path/to/BCCD --epochs 120

BCCD is 364 blood-smear photographs with 4888 boxes in three classes - red cells, white
cells and platelets - from `github.com/Shenggan/BCCD_Dataset`, MIT licence, with its own
train/val/test split. **Taken from that repository rather than the Kaggle mirror**: the
images are the same, but the Kaggle copy is governed by competition rules accepted at
download, and a trained model is a derivative of the data behind it.

This script exists because it is the first point at which anything about detection in this
package is demonstrable rather than merely tested. The metrics, the encoder and the anchors
were each verified against something external; none of that says a model trained with them
finds a blood cell.

Three things about this dataset decide how it has to be run, and all three were measured
rather than assumed:

- **The classes are wildly unbalanced**: 4155 red cells against 372 white and 361
  platelets. Average precision is reported per class, so a mean over the three is not
  dominated by the red cells - but a single number would be, which is why there is no
  single number here without the three beside it.
- **Two of the 4888 annotations are points rather than boxes** - `xmin == xmax` - which
  Pascal VOC's 1-based reading turns into 1x1 pixels. They are not cells; they are a click
  without a drag. `drop_degenerate` removes them once the letterbox has shrunk them below
  a pixel, and the count is printed rather than hidden.
- **The encoder can lose objects, and on this dataset it barely does.** Two cells of one
  shape whose centres fall in one grid cell share a slot, so the second is dropped. A
  synthetic test that packed 45 equal boxes into a 416 frame lost 9% of them, and that
  number does not transfer: BCCD averages about fourteen objects an image, and the real
  loss at 416 is **3 boxes out of 2804**. The run prints it before training rather than
  after, so a dataset where it does matter can be noticed in time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pyplatypus.detection import (
    COCO_THRESHOLDS,
    Letterbox,
    detection_report,
    drop_degenerate,
    encode,
    generate_anchors,
    non_max_suppression,
    read_voc,
)
from pyplatypus.detection.encode import decode
from pyplatypus.detection.loss import Yolo3Loss
from pyplatypus.detection.yolo3 import build_yolo3

LABELS = ["RBC", "WBC", "Platelets"]


def split_names(root: Path, which: str) -> list[str]:
    """BCCD's own split, so the numbers are comparable to anybody else's on it."""
    listing = root / "ImageSets" / "Main" / f"{which}.txt"
    if not listing.exists():
        raise SystemExit(
            f"{listing} is missing. Point --data at the BCCD directory of "
            f"github.com/Shenggan/BCCD_Dataset, which carries the canonical splits."
        )
    return [line.strip() for line in listing.read_text().splitlines() if line.strip()]


class BloodCells(Dataset):
    def __init__(self, root: Path, names: list[str], *, size: int, anchors, train: bool):
        self.root = root
        self.names = names
        self.size = size
        self.anchors = anchors
        self.train = train

    def __len__(self) -> int:
        return len(self.names)

    def read(self, index: int):
        name = self.names[index]
        annotation = read_voc(self.root / "Annotations" / f"{name}.xml", LABELS)
        image = np.asarray(
            Image.open(self.root / "JPEGImages" / f"{name}.jpg").convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        fit = Letterbox.fit((annotation.height, annotation.width), (self.size, self.size))
        boxes = fit.forward(annotation.boxes)
        boxes, labels, dropped = drop_degenerate(boxes, annotation.labels)
        return fit.apply_to_image(image), boxes, labels, dropped, fit, annotation

    def __getitem__(self, index: int):
        image, boxes, labels, _, _, _ = self.read(index)
        # Horizontal flip only. A smear has no up or down, and a left-right flip is free:
        # the boxes follow it exactly, with no interpolation and no rounding. The decision
        # is separate from applying it, so an image that happens to have no boxes is still
        # flipped - otherwise its background would only ever be seen one way round.
        if self.train and np.random.rand() < 0.5:
            image = image[:, ::-1].copy()
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
        encoded = encode(boxes, labels, anchors=self.anchors,
                         input_shape=(self.size, self.size), n_class=len(LABELS))
        return (torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                *[torch.from_numpy(t) for t in encoded.targets])


def survey_targets(root: Path, names: list[str], *, anchors, size: int) -> dict:
    """What the encoder can and cannot represent, before any training.

    Measured here rather than accumulated during the run: the first version of this kept
    counters on the Dataset, which `DataLoader` increments **inside worker processes**, so
    the main process saw zeros and the number printed was silence. Up front is also where
    it is useful - knowing the target drops a tenth of the objects is a reason to change
    `--size` before spending half an hour, not after.
    """
    placed = unplaced = dropped = 0
    for name in names:
        annotation = read_voc(root / "Annotations" / f"{name}.xml", LABELS)
        fit = Letterbox.fit((annotation.height, annotation.width), (size, size))
        boxes, labels, lost = drop_degenerate(fit.forward(annotation.boxes),
                                              annotation.labels)
        dropped += lost
        encoded = encode(boxes, labels, anchors=anchors, input_shape=(size, size),
                         n_class=len(LABELS))
        placed += encoded.placed
        unplaced += encoded.unplaced
    return {"placed": placed, "unplaced": unplaced, "dropped": dropped}


def collate(batch):
    images = torch.stack([row[0] for row in batch])
    targets = [torch.stack([row[1 + g] for row in batch]) for g in range(3)]
    return images, targets


def evaluate(model, root, names, *, anchors, size, device, objectness, nms_iou,
             operating_point=0.5):
    """Predictions through decode, NMS and the inverse letterbox, then scored.

    Two thresholds, and they are not the same thing. `objectness` is kept low because
    average precision is a property of the **whole ranking**: cutting the tail off throws
    away the part of the precision-recall curve that AP integrates over, and inflates the
    score. `operating_point` is where precision and recall are read, because those are a
    single choice of confidence and have no meaning without one.
    """
    model.eval()
    data = BloodCells(root, names, size=size, anchors=anchors, train=False)
    predictions, truths = [], []
    with torch.no_grad():
        for index in range(len(data)):
            image, _, _, _, fit, annotation = data.read(index)
            tensor = torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )[None].to(device)
            outputs = [out[0].cpu().numpy() for out in model(tensor)]
            boxes, scores, labels = decode(
                outputs, anchors=anchors, input_shape=(size, size),
                n_class=len(LABELS), objectness=objectness, raw=True,
            )
            keep = non_max_suppression(boxes, scores, labels, iou_threshold=nms_iou)
            predictions.append({
                # Back onto the photograph's own pixels, which is where the truth lives.
                "boxes": fit.inverse(boxes[keep]) if len(keep) else np.zeros((0, 4)),
                "scores": scores[keep],
                "labels": labels[keep],
            })
            truths.append(annotation.as_truth())
    return (
        detection_report(predictions, truths, labels=LABELS,
                         iou_thresholds=(0.5,), interpolation="101"),
        detection_report(predictions, truths, labels=LABELS,
                         iou_thresholds=COCO_THRESHOLDS, interpolation="101"),
        # The same predictions at one confidence, which is where precision and recall live.
        detection_report(predictions, truths, labels=LABELS, iou_thresholds=(0.5,),
                         interpolation="101", score_threshold=operating_point),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="the BCCD directory")
    parser.add_argument("--size", type=int, default=416, help="divisible by 32")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rate", type=float, default=1e-4)
    parser.add_argument("--anchors-per-grid", type=int, default=3)
    parser.add_argument("--objectness", type=float, default=0.01,
                        help="kept low: average precision integrates the whole ranking")
    parser.add_argument("--operating-point", type=float, default=0.5,
                        help="the confidence at which precision and recall are reported")
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="measurements")
    parser.add_argument("--save", default=None,
                        help="where to write the trained weights; without it the run "
                             "cannot be asked anything afterwards")
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    root = Path(arguments.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_names = split_names(root, "train")
    val_names = split_names(root, "val")
    test_names = split_names(root, "test")
    print(f"BCCD: {len(train_names)} train, {len(val_names)} val, {len(test_names)} test")
    print(f"device: {device}, input {arguments.size}\n")

    # Anchors from this data, not COCO's. `anchor_coverage` is the way to see the difference.
    annotations = [read_voc(root / "Annotations" / f"{name}.xml", LABELS)
                   for name in train_names]
    fit = generate_anchors(annotations, anchors_per_grid=arguments.anchors_per_grid,
                           scales=3, input_shape=(arguments.size, arguments.size),
                           seed=arguments.seed)
    print(f"anchors: mean IoU {fit.mean_iou:.4f} over {fit.boxes_used} boxes "
          f"({fit.boxes_dropped} dropped as degenerate)")
    for row in fit.as_rows():
        print(f"  grid {row['grid']} slot {row['slot']}: "
              f"{row['width_pixels']:6.1f} x {row['height_pixels']:6.1f} px, "
              f"{row['boxes']:4} boxes, IoU {row['mean_iou']:.3f}")
    print()

    model = build_yolo3(n_class=len(LABELS),
                        anchors_per_grid=arguments.anchors_per_grid).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    loss_fn = Yolo3Loss(anchors=fit.anchors, n_class=len(LABELS),
                        input_shape=(arguments.size, arguments.size))
    optimiser = torch.optim.Adam(model.parameters(), lr=arguments.rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, arguments.epochs)

    survey = survey_targets(root, train_names, anchors=fit.anchors, size=arguments.size)
    total = survey["placed"] + survey["unplaced"]
    print(f"targets: {survey['placed']} boxes placed, {survey['unplaced']} could not be "
          f"({survey['unplaced'] / max(total, 1):.1%}), {survey['dropped']} dropped as "
          f"degenerate\n")

    train_data = BloodCells(root, train_names, size=arguments.size,
                            anchors=fit.anchors, train=True)
    loader = DataLoader(train_data, batch_size=arguments.batch, shuffle=True,
                        num_workers=arguments.workers, collate_fn=collate, drop_last=True)

    history = []
    started = time.time()
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        totals = {}
        steps = 0
        for images, targets in loader:
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            parts = loss_fn(model(images), targets)
            optimiser.zero_grad()
            parts.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimiser.step()
            for key, value in parts.as_dict().items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        schedule.step()
        record = {"epoch": epoch, **{k: v / steps for k, v in totals.items()}}
        history.append(record)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:>4}  " + "  ".join(
                f"{k} {v:7.3f}" for k, v in record.items() if k != "epoch"))

    print(f"\ntrained in {(time.time() - started) / 60:.1f} min")
    if arguments.save:
        destination = Path(arguments.save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "anchors": fit.anchors,
            "n_class": len(LABELS),
            "labels": LABELS,
            "input": arguments.size,
            "anchors_per_grid": arguments.anchors_per_grid,
        }, destination)
        print(f"weights written to {destination}")
    print()

    for split, names in (("validation", val_names), ("test", test_names)):
        half, coco, point = evaluate(
            model, root, names, anchors=fit.anchors, size=arguments.size, device=device,
            objectness=arguments.objectness, nms_iou=arguments.nms_iou,
            operating_point=arguments.operating_point,
        )
        print(f"=== {split} ({len(names)} images) ===")
        print(f"  mAP@0.5       {half.mean_average_precision:.4f}")
        print(f"  mAP@[.50:.95] {coco.mean_average_precision:.4f}")
        # The gap between those two is localisation, and this is it as one number: how well
        # the boxes that matched actually fit, rather than merely that they cleared 0.5.
        if half.mean_matched_iou is not None:
            print(f"  mean IoU of matched boxes {half.mean_matched_iou:.4f}")
        print(f"  precision and recall at confidence {arguments.operating_point}")
        print(f"  {'class':12} {'AP@0.5':>8} {'IoU':>6} {'truth':>6} {'pred':>7} "
              f"{'prec':>7} {'rec':>7}")
        for row, at_point in zip(half.per_class, point.per_class, strict=True):
            ap = f"{row['average_precision']:.4f}" if row['average_precision'] is not None else "   -  "
            iou = f"{row['mean_matched_iou']:.3f}" if row['mean_matched_iou'] is not None else "  -  "
            pr = f"{at_point['precision']:.3f}" if at_point['precision'] is not None else "  -  "
            rc = f"{at_point['recall']:.3f}" if at_point['recall'] is not None else "  -  "
            print(f"  {row['label']:12} {ap:>8} {iou:>6} {row['n_truth']:>6} "
                  f"{at_point['n_predicted']:>7} {pr:>7} {rc:>7}")
        print()
        if split == "test":
            out = Path(arguments.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "bccd-yolo3.json").write_text(json.dumps({
                "anchors": fit.anchors,
                "anchor_mean_iou": fit.mean_iou,
                "input": arguments.size,
                "epochs": arguments.epochs,
                "history": history,
                "targets": survey,
                "test_map_50": half.mean_average_precision,
                "test_map_coco": coco.mean_average_precision,
                "test_mean_matched_iou": half.mean_matched_iou,
                "test_per_class": half.per_class,
            }, indent=2, default=float) + "\n")
            print(f"wrote {out / 'bccd-yolo3.json'}")


if __name__ == "__main__":
    main()
