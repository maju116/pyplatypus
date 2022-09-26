from pathlib import Path
from PIL import Image
import numpy as np


def save_masks(image_masks: list, paths: list, model_name: str):
    for im, path in zip(image_masks, paths):
        img_directory = Path(path).parents[1]
        masks_path = img_directory/"predicted_masks"
        Path.mkdir(masks_path, exist_ok=True)
        img_name, img_type = Path(path).name.split(".")
        if img_type != "png":
            raise NotImplementedError("Types other than PNG to be handled soon.")
        mask_as_image = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        mask_path = masks_path/f"{model_name}_predicted_mask.png"
        mask_as_image.save(mask_path)
