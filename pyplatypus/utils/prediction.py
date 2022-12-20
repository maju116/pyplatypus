from pathlib import Path
from PIL import Image
import numpy as np


def save_masks(image_masks: list, paths: list, model_name: str, mode: str = "nested_dirs"):
    for im, path in zip(image_masks, paths):
        image_path = Path(path)
        img_name, img_type = image_path.name.split(".")
        if mode == "nested_dirs":
            img_directory = image_path.parents[1]
        else:
            img_directory = image_path.parents[0]
        masks_path = img_directory/"predicted_masks"
        if not masks_path.exists():
            Path.mkdir(masks_path, exist_ok=False)
        mask_as_image = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        mask_path = masks_path/f"{img_name}_{model_name}_predicted_mask.png"
        mask_as_image.save(mask_path)
