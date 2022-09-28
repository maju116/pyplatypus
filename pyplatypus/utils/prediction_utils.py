from pathlib import Path
from PIL import Image
import numpy as np


def save_masks(image_masks: list, paths: list, model_name: str, mode: str = "nested_dirs"):
    for im, path in zip(image_masks, paths):
        image_path = Path(path)
        img_name, img_type = image_path.name.split(".")
        if img_type != "png":
            raise NotImplementedError("Types other than PNG to be handled soon.")
        if mode == "nested_dirs":
            img_directory = image_path.parents[1]
            img_name = img_directory.name
        else:
            img_directory = image_path.parents[0]
            img_name = img_directory.name
        masks_path = img_directory/"predicted_masks"
        Path.mkdir(masks_path, exist_ok=True)
        mask_as_image = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        mask_path = masks_path/f"{img_name}_{model_name}_predicted_mask.png"
        mask_as_image.save(mask_path)
