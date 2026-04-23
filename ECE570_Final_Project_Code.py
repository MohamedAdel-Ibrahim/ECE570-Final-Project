# -*- coding: utf-8 -*-
"""
ECE570 Final Project: Robust In-Cabin Distracted Driver Detection
Author: Mohamed Adel Elsayed Abdelmoeti Ibrahim
"""

import os
import shutil
import zipfile
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from ultralytics import YOLO

# ==============================================================================
# GRADER NOTE: DATASET REPRODUCIBILITY
# Due to the proprietary nature of the human-subject data (NDA restricted), 
# the full dataset cannot be downloaded automatically. 
#
# TO RUN THIS PIPELINE: 
# Please upload 'ai_distracted2.zip' to the same folder as this script.
# ==============================================================================

ZIP_NAME = "ai_distracted2.zip"
BASE_PATH = "ai_distracted2"
RAW_ROOT = "ai_distracted2"
ENH_ROOT = "ai_distracted_enhanced2"

# 1. Extraction Logic
if not os.path.exists(BASE_PATH):
    if os.path.exists(ZIP_NAME):
        print(f"[INFO] Found {ZIP_NAME}. Extracting...")
        with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall("./")
        print("[INFO] Extraction complete.")
    else:
        print(f"[ERROR] {ZIP_NAME} not found. Please ensure the zip is in this folder.")

# 2. Data Purging
CLASSES_TO_REMOVE = ["c6","c7","c8"]
SUBSETS = ["train" , "test"]

for subset in SUBSETS:
    subset_path = os.path.join(BASE_PATH , subset)
    if os.path.exists(subset_path):
        print(f"[INFO] Cleaning subset directory: {subset_path}")
        for cls in CLASSES_TO_REMOVE:
            cls_path = os.path.join(subset_path, cls)
            if os.path.exists(cls_path):
                shutil.rmtree(cls_path)
                print(f"[INFO] Deleted class: {cls_path}")

print("[INFO] Target classes successfully purged from dataset.")

# 3. Illumination Recovery Pipeline
os.makedirs(ENH_ROOT, exist_ok=True)

def adjust_gamma(image , gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image, table)

"""
Applies a 3-stage illumination recovery pipeline (Median Blur, Gamma Correction, CLAHE)
to resolve severe cabin shadowing and sensor noise.
"""
def apply_night_enhancement(img):
    denoised = cv2.medianBlur(img , 3)
    # gamma 1.5 selected as the optimal hyperparameter during phase 1 testing.
    gamma_corrected = adjust_gamma(denoised, gamma=1.5)
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab , cv2.COLOR_LAB2BGR)

def enhance_split(split):
    src_split = os.path.join(RAW_ROOT , split)
    dst_split = os.path.join(ENH_ROOT , split)
    os.makedirs(dst_split, exist_ok=True)
    if not os.path.exists(src_split): return
    classes = sorted([d for d in os.listdir(src_split) if os.path.isdir(os.path.join(src_split , d))])
    print(f"[INFO] Initiating enhancement pipeline for split: {split}")
    for cls in classes:
        src_cls = os.path.join(src_split, cls)
        dst_cls = os.path.join(dst_split, cls)
        os.makedirs(dst_cls, exist_ok=True)
        images = glob.glob(os.path.join(src_cls, "*.jpg"))
        for img_path in tqdm(images, desc=f"Processing {cls}", leave=False):
            img = cv2.imread(img_path)
            if img is not None:
                enhanced = apply_night_enhancement(img)
                cv2.imwrite(os.path.join(dst_cls , os.path.basename(img_path)) , enhanced)

enhance_split("train")
enhance_split("test")
print("[INFO] Night enhancement pipeline completed successfully.")

# 4. SIDE-BY-SIDE VISUAL VALIDATION
""" Generates a side-by-side visual validation grid to verify the
 structural integrity of the preprocessing pipeline across all active classes."""
print("[INFO] Generating visual validation grid...")
active_classes = sorted(['c0','c1','c2','c3','c4','c5','c9'])
plt.figure(figsize=(10,20))
for i, cls in enumerate(active_classes):
    src_path = os.path.join(RAW_ROOT, "train", cls)
    images = glob.glob(os.path.join(src_path, "*.jpg"))
    if images:
        img_name = os.path.basename(random.choice(images))
        raw_img = cv2.imread(os.path.join(src_path, img_name))
        enh_img = cv2.imread(os.path.join(ENH_ROOT, "train", cls, img_name))
        if raw_img is not None and enh_img is not None:
            plt.subplot(len(active_classes), 2, i*2 + 1)
            plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
            plt.title(f"{cls}: Original")
            plt.axis('off')
            plt.subplot(len(active_classes), 2, i*2 + 2)
            plt.imshow(cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB))
            plt.title(f"{cls}: Enhanced")
            plt.axis('off')
plt.tight_layout()
plt.savefig("pipeline_validation.png")
print("[INFO] Validation grid saved as 'pipeline_validation.png'")

# 5. Data Segregation (80/20 Split)
"""
Partitions the augmented training data into a train/validation split
to monitor YOLOv8-nano generalization and prevent overfitting.
"""
random.seed(42)
TRAIN_DIR = os.path.join(ENH_ROOT, "train")
VAL_DIR   = os.path.join(ENH_ROOT, "val")
if os.path.exists(TRAIN_DIR):
    os.makedirs(VAL_DIR, exist_ok=True)
    VAL_RATIO = 0.2
    classes = sorted(os.listdir(TRAIN_DIR))
    for cls in classes:
        cls_train = os.path.join(TRAIN_DIR, cls)
        cls_val   = os.path.join(VAL_DIR, cls)
        os.makedirs(cls_val, exist_ok=True)
        images = glob.glob(os.path.join(cls_train, "*"))
        random.shuffle(images)
        n_val = int(len(images)*VAL_RATIO)
        for img in images[:n_val]:
            shutil.move(img , os.path.join(cls_val , os.path.basename(img)))
print(f"[INFO] Dataset partitioned successfully.")

# 6. Model Training
""" Executes the YOLOv8-nano training pipeline.
Implements heavy HSV color shifting and spatial augmentations to
prevent overfitting during the restricted 15-epoch run. """
model = YOLO("yolov8n-cls.pt")
model.train(
    data=os.path.abspath(ENH_ROOT),
    epochs=15,
    imgsz=224,
    batch=32,
    degrees=15,
    scale=0.4,
    hsv_v=0.4,
    project="driver_radar",
    name="v2_enhanced_final"
)

# 7. Final Inference Test
"""
Loads the fine-tuned YOLOv8-nano weights and executes a single-image
inference test, mapping the raw class output to a human-readable telemetry log.
"""
CLASS_MAP = {
    "c0": "Normal Driving", "c1": "Texting (Hand)", "c2": "Talking on Phone",
    "c3": "Texting" , "c4": "Talking on Phone", "c5": "Carscreen", "c9": "Distracted"
}

BEST_WEIGHTS_PATH = os.path.join("driver_radar", "v2_enhanced_final", "weights", "best.pt")
img_path = "image_test.png"

if os.path.exists(BEST_WEIGHTS_PATH) and os.path.exists(img_path):
    model = YOLO(BEST_WEIGHTS_PATH)
    res = model.predict(source=img_path, imgsz=224, verbose=False)[0]
    pred_idx = int(res.probs.top1)
    pred_conf = float(res.probs.top1conf)
    readable_name = CLASS_MAP.get(res.names[pred_idx], res.names[pred_idx])
    print(f"[INFERENCE] File: {img_path} | Prediction: {readable_name} ({pred_conf*100:.1f}%)")
else:
    print("[WARNING] Inference skipped: Weights or test image not found.")
