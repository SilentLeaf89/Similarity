"""
Модуль очистки от фона
"""

import os

import cv2
from ultralytics import YOLO
import numpy as np


def clear_image_background(
    filename: str, save_dir: str = "masked_photo", model_name="yolov8x-seg.pt"
) -> None:
    """Clear background from filename to save_dir using YOLO segmentation model

    Args:
        filename (str): file name or Path
        save_dir (str): dir path or Path
        model_name (str, optional): name of YOLO model.
        Defaults to "yolov8x-seg.pt".
    """

    model = YOLO(model_name)
    os.makedirs(save_dir, exist_ok=True)
    results = model(filename)
    for r in results:
        img = np.copy(r.orig_img)
        for ci, c in enumerate(r):
            b_mask = np.zeros(img.shape[:2], np.uint8)
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            isolated = np.dstack([img, b_mask])
            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            iso_crop = isolated[y1:y2, x1:x2]
            original_name = filename.split(".")[-2].split("/")[-1]
            mask_name = f"{original_name}_mask_{ci}.png"
            mask_path = os.path.join(save_dir, mask_name)
            _ = cv2.imwrite(mask_path, iso_crop)
