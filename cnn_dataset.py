"""
Generates patches to train CNN model on
"""

import os
import cv2
from pathlib import Path
from mold_detector import MoldDetector

INPUT_DIR = "resources/Photos-Mold"
OUTPUT_DIR = "resources/Mold_Patches"
MIN_AREA = 2500
PATCH_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)
detector = MoldDetector(INPUT_DIR, OUTPUT_DIR)
image_files = list(Path(INPUT_DIR).glob("*.jpg")) + list(Path(INPUT_DIR).glob("*.png"))

for image_path in image_files:
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        continue

    # Detect mold regions
    mold_mask, contours, _ = detector.detect_mold_regions(image)
    
    patch_count = 1
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue  # Skip small patches

        x, y, w, h = cv2.boundingRect(contour)
        square_size = max(w, h)

        # Create square patch
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + square_size, image.shape[1])
        y2 = min(y + square_size, image.shape[0])

        patch = image[y1:y2, x1:x2]

        # Resize patch to fixed size
        patch_resized = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))

        # Save resized patch
        patch_filename = f"{Path(image_path).stem}_patch{patch_count}.png"
        patch_path = os.path.join(OUTPUT_DIR, patch_filename)
        cv2.imwrite(patch_path, patch_resized)

        patch_count += 1
