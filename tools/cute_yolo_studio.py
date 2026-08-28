#!/usr/bin/env python3
"""
Cute-YOLO Studio

Repository location:
    training/cute_yolo_adapt_gui.py

Purpose:
    Fine-tune an already-trained fixed Cute-YOLO 8+24 model using
    verified real deployment-domain data.

GUI terminology:
    Base model              = export ZIP produced by Cute_YOLO_Trainer.ipynb
    Original training data  = the ZIP used for the base training
    Real labeled data       = ZIP exported by cute_label_gui.py

Default adaptation:
    75% original/source samples
    25% real/target samples
    target-only validation/model selection
    1e-4 fine-tuning learning rate
    target-heavy INT8 calibration

Outputs:
    <class>_adapted_export.zip
    <class>_adapted.cute        (optional, if zip_to_cute.py is available)

Dependencies:
    pip install tensorflow numpy opencv-python matplotlib

Tkinter normally ships with Python on Windows/macOS.
On Ubuntu/Debian, if necessary:
    sudo apt install python3-tk
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
import webbrowser

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import cv2
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# ============================================================
# Fixed Cute-YOLO contract
# ============================================================

IMG_SIZE = 128
GRID_SIZE = 16
OUTPUT_CHANNELS = 5

ARCHITECTURE_ID = "cute_yolo_fixed_dualcore_hybrid_8_24"

HYBRID_WIDTH = 32
HYBRID_NORMAL_OUT = 8
HYBRID_EFFICIENT_OUT = 24
HYBRID_BLOCKS = 5

EXPECTED_TRAINABLE_PARAMETERS = 23_925
EXPECTED_CONV_MACS = 6_995_968


# ============================================================
# Cute-YOLO BLE deployment
# ============================================================

CUTE_BLE_DEVICE_NAME = "Cute-YOLO"

CUTE_BLE_SERVICE_UUID = (
    "7f1d0001-9f3b-4c2b-8f3e-5b51c0de0001"
)

CUTE_BLE_CTRL_UUID = (
    "7f1d0002-9f3b-4c2b-8f3e-5b51c0de0001"
)

CUTE_BLE_DATA_UUID = (
    "7f1d0003-9f3b-4c2b-8f3e-5b51c0de0001"
)

CUTE_BLE_STATUS_UUID = (
    "7f1d0004-9f3b-4c2b-8f3e-5b51c0de0001"
)


# ============================================================
# Default training / loss / deployment settings
# ============================================================

SEED = 123

ALLOW_MISSING_LABEL_FILES = False
DEFAULT_DATASET_CLASS = "X"
CROP_MIN_RETAINED_AREA = 0.50

BACKGROUND_LOSS_WEIGHT = 1.5
HARD_NEGATIVES = 32

# User-selectable objectness/loss profiles.
#
# COMPLEX / DISTINCTIVE:
#   Faces, vehicles, animals, tools, etc. The target has richer visual
#   structure, so ordinary positive BCE + hard-negative mining is the
#   conservative default.
#
# PRIMITIVE / REPETITIVE:
#   Circles, blobs, simple geometric primitives, etc. Many similar nearby
#   objects can create pseudo-object patterns in the gaps, so enable the
#   additional footprint/halo and crowd-aware negative emphasis.
LOSS_PRESET_COMPLEX = "complex_distinctive"
LOSS_PRESET_PRIMITIVE = "primitive_repetitive"
LOSS_PRESET_CUSTOM = "custom"
DEFAULT_LOSS_PRESET = LOSS_PRESET_COMPLEX

# Data-preparation defaults paired with the object-characteristic presets.
# Cute-YOLO's 16x16 head has an 8-pixel stride at 128x128 input.
# For complex/distinctive targets (especially faces), extremely tiny labels
# can contain too little visual structure to be learnable and can dominate
# crowded datasets such as WIDER FACE.  The complex preset therefore drops
# boxes smaller than one detector cell in either dimension.
# Primitive/repetitive datasets keep all valid boxes by default.
COMPLEX_MIN_OBJECT_SIZE_PX = 8.0
PRIMITIVE_MIN_OBJECT_SIZE_PX = 0.0
DEFAULT_MIN_OBJECT_SIZE_PX = COMPLEX_MIN_OBJECT_SIZE_PX

# Mild training augmentation inherited from the proven WIDER FACE recipe.
# Horizontal flips require a geometrically transformed target; brightness
# and contrast are photometric only and therefore leave boxes unchanged.
AUGMENT_HORIZONTAL_FLIP_DEFAULT = True
AUGMENT_BRIGHTNESS_DEFAULT = True
AUGMENT_CONTRAST_DEFAULT = True
AUGMENT_SHUFFLE_DEFAULT = True
AUGMENT_FLIP_PROBABILITY = 0.50
AUGMENT_BRIGHTNESS_RANGE = (0.85, 1.15)
AUGMENT_CONTRAST_RANGE = (0.85, 1.15)

CURRENT_LOSS_PRESET = DEFAULT_LOSS_PRESET
HARD_NEGATIVE_MINING_ENABLED = True

# Box-aware spatial negative emphasis (training only).
#
# FOOTPRINT:
#   Detector-grid cells whose AREA overlaps a ground-truth box receive
#   extra background emphasis unless they are true object-center cells.
#
# HALO:
#   A one-cell dilation around the box footprint receives a smaller
#   extra penalty. This targets structured gaps just outside tight boxes.
#
# All real positive center cells are excluded globally, so overlapping
# objects in distinct grid cells remain valid positives.
BOX_AWARE_NEGATIVE_ENABLED = False
BOX_FOOTPRINT_NEGATIVE_WEIGHT = 0.50
BOX_HALO_NEGATIVE_WEIGHT = 0.25
BOX_HALO_KERNEL = 3

# Crowd-aware weighting inside the footprint/halo losses.
# C_ij counts how many GT objects influence a background cell through
# either their footprint or their one-cell halo. Cells surrounded by
# several nearby objects therefore receive a stronger negative gradient.
# True positive center cells are still excluded globally.
CROWD_AWARE_NEGATIVE_ENABLED = False
CROWD_COUNT_GAMMA = 0.50
CROWD_COUNT_CAP = 4.0

# Dual-branch specialized domain adaptation (training only).
# Branch A (Conv 32->8, mapped to Core 0 at inference) is
# localization-dominant. Branch B (DW/PW 32->24, mapped to Core 1)
# is objectness-dominant. A small cross-task weight prevents complete
# branch isolation.
SPECIALIZED_FINETUNE_DEFAULT = True
SPECIALIZED_CROSS_LOSS_WEIGHT = 0.25

BOX_WEIGHT = 5.0
IOU_WEIGHT = 2.0

# User-selectable box-overlap loss (training only).
#
# IoU reproduces the earlier / simpler Cute-YOLO face recipe:
#     L_overlap = 1 - IoU
#
# CIoU adds center-distance and aspect-ratio penalties:
#     L_overlap = 1 - CIoU
#
# This switch changes only the training objective; model topology,
# exported tensor shapes, .cute packaging, and ESP32 inference are unchanged.
BOX_OVERLAP_LOSS_IOU = "iou"
BOX_OVERLAP_LOSS_CIOU = "ciou"
DEFAULT_BOX_OVERLAP_LOSS = BOX_OVERLAP_LOSS_CIOU
BOX_OVERLAP_LOSS = DEFAULT_BOX_OVERLAP_LOSS

MIN_BOX_W = 0.05
MIN_BOX_H = 0.05

# Deployment cap written into runtime_config/.cute for the ESP32.
RUNTIME_MAX_DETECTIONS = 32

# Training/validation/debug cap. Keep this higher so crowded
# validation images are not artificially truncated at 16 boxes.
EVAL_MAX_DETECTIONS = 128

CONFIDENCE_GRID = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

NMS_IOU_GRID = [
    0.25,
    0.35,
    0.45,
    0.55,
]


# These globals are assigned by run_adaptation().
MODEL_LABEL = ""
TASK_SLUG = ""
EXPERIMENT_TAG = ""

SOURCE_RATIO = 0.75
TARGET_RATIO = 0.25

SELECTED_CONFIDENCE_THRESHOLD = 0.65
SELECTED_NMS_IOU_THRESHOLD = 0.35

INT8_EXPORT_DIR = Path(".")
TFLITE_PATH = Path("model.tflite")
MANIFEST_PATH = Path("model_deployment_manifest.json")

trainable_parameter_count = EXPECTED_TRAINABLE_PARAMETERS
cute_yolo_macs = EXPECTED_CONV_MACS

EXPORT_DOMAIN_ADAPTATION = True


def slugify_label(text: str) -> str:
    text = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text).strip(),
    )
    return text.strip("_.") or "object"


def log_line(logger: Callable[[str], None], text=""):
    logger(str(text))


def configure_objectness_loss(
    preset: str,
    hard_negative_mining_enabled: bool,
    box_aware_negative_enabled: bool,
    crowd_aware_negative_enabled: bool,
    logger: Optional[Callable[[str], None]] = None,
    context: str = "training",
):
    """Apply the selected training-only objectness configuration."""
    global CURRENT_LOSS_PRESET
    global HARD_NEGATIVE_MINING_ENABLED
    global BOX_AWARE_NEGATIVE_ENABLED
    global CROWD_AWARE_NEGATIVE_ENABLED

    preset = str(preset or LOSS_PRESET_CUSTOM)
    hard = bool(hard_negative_mining_enabled)
    spatial = bool(box_aware_negative_enabled)
    crowd = bool(crowd_aware_negative_enabled) and spatial

    CURRENT_LOSS_PRESET = preset
    HARD_NEGATIVE_MINING_ENABLED = hard
    BOX_AWARE_NEGATIVE_ENABLED = spatial
    CROWD_AWARE_NEGATIVE_ENABLED = crowd

    if logger is not None:
        preset_name = {
            LOSS_PRESET_COMPLEX: "Complex / distinctive object",
            LOSS_PRESET_PRIMITIVE: "Primitive / repetitive object",
            LOSS_PRESET_CUSTOM: "Custom",
        }.get(preset, preset)

        log_line(logger, f"Loss preset ({context}): {preset_name}")
        log_line(
            logger,
            "Objectness switches: "
            f"HNM={'ON' if hard else 'OFF'}, "
            f"footprint/halo={'ON' if spatial else 'OFF'}, "
            f"crowd-aware={'ON' if crowd else 'OFF'}",
        )

        terms = ["L_pos"]
        if hard:
            terms.append(f"{BACKGROUND_LOSS_WEIGHT:g} L_hard")
        if spatial:
            terms.extend([
                f"{BOX_FOOTPRINT_NEGATIVE_WEIGHT:g} L_foot",
                f"{BOX_HALO_NEGATIVE_WEIGHT:g} L_halo",
            ])
        suffix = " (spatial terms crowd-weighted)" if crowd else ""
        log_line(
            logger,
            "Effective objectness: L_obj = " + " + ".join(terms) + suffix,
        )


def objectness_training_metadata():
    return {
        "loss_preset": CURRENT_LOSS_PRESET,
        "background_loss_weight": float(BACKGROUND_LOSS_WEIGHT),
        "hard_negative_mining_enabled": bool(HARD_NEGATIVE_MINING_ENABLED),
        "hard_negatives_per_image": int(HARD_NEGATIVES),
        "box_aware_negative_enabled": bool(BOX_AWARE_NEGATIVE_ENABLED),
        "box_footprint_negative_weight": float(BOX_FOOTPRINT_NEGATIVE_WEIGHT),
        "box_halo_negative_weight": float(BOX_HALO_NEGATIVE_WEIGHT),
        "box_halo_kernel": int(BOX_HALO_KERNEL),
        "crowd_aware_negative_enabled": bool(CROWD_AWARE_NEGATIVE_ENABLED),
        "crowd_count_gamma": float(CROWD_COUNT_GAMMA),
        "crowd_count_cap": float(CROWD_COUNT_CAP),
        "box_aware_negative_definition": (
            "GT-box/grid-cell footprint plus per-object one-cell halo; "
            "optional spatial BCE is weighted by the number of GT objects "
            "influencing each negative cell; all true positive center cells "
            "are excluded"
        ),
    }




def training_augmentation_metadata(cfg):
    return {
        "horizontal_flip_enabled": bool(cfg.augment_horizontal_flip),
        "horizontal_flip_probability": float(AUGMENT_FLIP_PROBABILITY),
        "horizontal_flip_label_handling": (
            "mirror normalized boxes with x_center -> 1-x_center, then re-encode 16x16 target"
        ),
        "brightness_enabled": bool(cfg.augment_brightness),
        "brightness_range": [
            float(AUGMENT_BRIGHTNESS_RANGE[0]),
            float(AUGMENT_BRIGHTNESS_RANGE[1]),
        ],
        "contrast_enabled": bool(cfg.augment_contrast),
        "contrast_range": [
            float(AUGMENT_CONTRAST_RANGE[0]),
            float(AUGMENT_CONTRAST_RANGE[1]),
        ],
        "shuffle_each_epoch": bool(cfg.shuffle_each_epoch),
        "validation_augmentation": False,
    }


def configure_box_overlap_loss(
    mode: str,
    logger: Optional[Callable[[str], None]] = None,
    context: str = "training",
):
    """Select IoU or CIoU for the training-only overlap term."""
    global BOX_OVERLAP_LOSS

    mode = str(mode or DEFAULT_BOX_OVERLAP_LOSS).strip().lower()
    if mode not in {BOX_OVERLAP_LOSS_IOU, BOX_OVERLAP_LOSS_CIOU}:
        raise ValueError(
            "Box overlap loss must be either 'iou' or 'ciou'."
        )

    BOX_OVERLAP_LOSS = mode

    if logger is not None:
        label = "IoU" if mode == BOX_OVERLAP_LOSS_IOU else "CIoU"
        detail = (
            "plain 1-IoU"
            if mode == BOX_OVERLAP_LOSS_IOU
            else "Complete-IoU (overlap + center distance + aspect ratio)"
        )
        log_line(
            logger,
            f"Box overlap loss ({context}): {label} — {detail}",
        )


def box_overlap_loss_label():
    return "IoU" if BOX_OVERLAP_LOSS == BOX_OVERLAP_LOSS_IOU else "CIoU"


def encode_target(boxes, return_collisions=False):
    """
    Encode normalized boxes into Cute-YOLO's 16x16x5 target.

    Each box is:
        {
            "x": center_x,
            "y": center_y,
            "w": width,
            "h": height,
        }

    Coordinates are normalized relative to the detector input
    AFTER any center-square crop.

    The current Cute-YOLO head supports one object center per grid cell.
    If multiple boxes land in the same cell, keep the larger box and
    count the collision.
    """
    target = np.zeros(
        (GRID_SIZE, GRID_SIZE, OUTPUT_CHANNELS),
        dtype=np.float32,
    )

    cell_area = np.full(
        (GRID_SIZE, GRID_SIZE),
        -1.0,
        dtype=np.float32,
    )

    collisions = 0

    for box in boxes:
        x = float(np.clip(box["x"], 0.0, 0.999999))
        y = float(np.clip(box["y"], 0.0, 0.999999))
        width = float(np.clip(box["w"], 0.0, 1.0))
        height = float(np.clip(box["h"], 0.0, 1.0))

        if width <= 0.0 or height <= 0.0:
            continue

        grid_x = int(x * GRID_SIZE)
        grid_y = int(y * GRID_SIZE)

        area = width * height

        if target[grid_y, grid_x, 0] > 0.5:
            collisions += 1

            # Current head can encode only one object in this cell.
            # Keep the larger box so the choice is deterministic.
            if area <= cell_area[grid_y, grid_x]:
                continue

        target[grid_y, grid_x, 0] = 1.0
        target[grid_y, grid_x, 1] = x * GRID_SIZE - grid_x
        target[grid_y, grid_x, 2] = y * GRID_SIZE - grid_y
        target[grid_y, grid_x, 3] = width
        target[grid_y, grid_x, 4] = height

        cell_area[grid_y, grid_x] = area

    if return_collisions:
        return target, collisions

    return target


def flip_boxes_horizontally(boxes):
    """Mirror normalized YOLO boxes left/right without changing size.

    This is performed on box geometry BEFORE target encoding, matching the
    original WIDER FACE pipeline. Re-encoding after x -> 1-x updates both
    the responsible 16x16 cell and the within-cell dx offset correctly.
    """
    return [
        {
            "x": float(np.clip(1.0 - float(box["x"]), 0.0, 0.999999)),
            "y": float(np.clip(float(box["y"]), 0.0, 0.999999)),
            "w": float(np.clip(float(box["w"]), 0.0, 1.0)),
            "h": float(np.clip(float(box["h"]), 0.0, 1.0)),
        }
        for box in boxes
    ]


def parse_yolo_text(text):
    """
    Parse standard single-class YOLO labels.

    The returned coordinates are normalized relative to the
    ORIGINAL source image. They are transformed later if a
    center-square crop is required.
    """
    boxes = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO line {line_number}: {line!r}"
            )

        class_id = int(parts[0])

        if class_id != 0:
            raise ValueError(
                "Cute-YOLO is single-class; expected class id 0, "
                f"got {class_id} on line {line_number}."
            )

        cx, cy, width, height = map(float, parts[1:])

        boxes.append({
            "x": float(np.clip(cx, 0.0, 1.0)),
            "y": float(np.clip(cy, 0.0, 1.0)),
            "w": float(np.clip(width, 0.0, 1.0)),
            "h": float(np.clip(height, 0.0, 1.0)),
        })

    return boxes


def center_square_crop_and_transform_boxes(
    image,
    boxes,
    min_retained_area=CROP_MIN_RETAINED_AREA,
):
    """
    Match the ESP32 detector geometry:

        W x H
          -> centered S x S crop, S=min(W,H)
          -> resize later to IMG_SIZE x IMG_SIZE

    `boxes` are standard YOLO normalized boxes relative to the
    ORIGINAL W x H image.

    Returns:
        cropped_image
        transformed_boxes  # normalized relative to S x S crop
        stats

    A box crossing the crop edge is clipped. It is kept only if
    `retained_area / original_area >= min_retained_area`.
    """
    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale HxW image, got shape {image.shape}"
        )

    height, width = image.shape

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions: {width}x{height}"
        )

    side = min(width, height)

    crop_x = (width - side) // 2
    crop_y = (height - side) // 2

    crop_x2 = crop_x + side
    crop_y2 = crop_y + side

    cropped = image[
        crop_y:crop_y2,
        crop_x:crop_x2,
    ]

    transformed = []

    clipped_boxes = 0
    dropped_boxes = 0
    fully_inside_boxes = 0

    for box in boxes:
        # ------------------------------------------------------
        # YOLO normalized original-image box -> original pixels
        # ------------------------------------------------------
        cx = float(box["x"]) * width
        cy = float(box["y"]) * height
        bw = float(box["w"]) * width
        bh = float(box["h"]) * height

        x1 = cx - 0.5 * bw
        y1 = cy - 0.5 * bh
        x2 = cx + 0.5 * bw
        y2 = cy + 0.5 * bh

        # Clip first to the actual source image in case labels
        # contain tiny numerical excursions beyond [0,1].
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))

        original_w = max(0.0, x2 - x1)
        original_h = max(0.0, y2 - y1)
        original_area = original_w * original_h

        if original_area <= 0.0:
            dropped_boxes += 1
            continue

        # ------------------------------------------------------
        # Intersect original box with the square crop
        # ------------------------------------------------------
        ix1 = max(x1, float(crop_x))
        iy1 = max(y1, float(crop_y))
        ix2 = min(x2, float(crop_x2))
        iy2 = min(y2, float(crop_y2))

        kept_w = max(0.0, ix2 - ix1)
        kept_h = max(0.0, iy2 - iy1)
        kept_area = kept_w * kept_h

        retained_fraction = kept_area / original_area

        if (
            kept_area <= 0.0
            or retained_fraction < float(min_retained_area)
        ):
            dropped_boxes += 1
            continue

        was_clipped = (
            ix1 > x1 + 1e-6
            or iy1 > y1 + 1e-6
            or ix2 < x2 - 1e-6
            or iy2 < y2 - 1e-6
        )

        if was_clipped:
            clipped_boxes += 1
        else:
            fully_inside_boxes += 1

        # ------------------------------------------------------
        # Shift into crop coordinates
        # ------------------------------------------------------
        nx1 = ix1 - crop_x
        ny1 = iy1 - crop_y
        nx2 = ix2 - crop_x
        ny2 = iy2 - crop_y

        # ------------------------------------------------------
        # Crop-pixel box -> normalized crop-relative YOLO box
        # ------------------------------------------------------
        new_cx = 0.5 * (nx1 + nx2) / side
        new_cy = 0.5 * (ny1 + ny2) / side
        new_w = (nx2 - nx1) / side
        new_h = (ny2 - ny1) / side

        transformed.append({
            "x": float(np.clip(new_cx, 0.0, 0.999999)),
            "y": float(np.clip(new_cy, 0.0, 0.999999)),
            "w": float(np.clip(new_w, 0.0, 1.0)),
            "h": float(np.clip(new_h, 0.0, 1.0)),
        })

    stats = {
        "original_width": int(width),
        "original_height": int(height),
        "crop_side": int(side),
        "crop_x": int(crop_x),
        "crop_y": int(crop_y),
        "input_boxes": int(len(boxes)),
        "output_boxes": int(len(transformed)),
        "fully_inside_boxes": int(fully_inside_boxes),
        "clipped_boxes_kept": int(clipped_boxes),
        "dropped_boxes": int(dropped_boxes),
    }

    return cropped, transformed, stats


def resize_detector_input(square_gray):
    """
    Resize a square grayscale crop to the fixed 128x128 detector input.
    """
    height, width = square_gray.shape

    if height != width:
        raise ValueError(
            "resize_detector_input expects a square image, "
            f"got {width}x{height}"
        )

    if width == IMG_SIZE:
        return square_gray.copy()

    interpolation = (
        cv2.INTER_AREA
        if width > IMG_SIZE
        else cv2.INTER_LINEAR
    )

    return cv2.resize(
        square_gray,
        (IMG_SIZE, IMG_SIZE),
        interpolation=interpolation,
    )


def _member_parts(member):
    return PurePosixPath(member).parts


def _find_dataset_image_members(zf):
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    members = []

    for member in zf.namelist():
        if member.endswith("/"):
            continue

        parts = _member_parts(member)

        if "images" not in parts:
            continue

        suffix = PurePosixPath(member).suffix.lower()

        if suffix in supported:
            members.append(member)

    return sorted(members)


def _label_member_for_image(image_member):
    """
    Map:
        optional/root/images/path/file.png
    to:
        optional/root/labels/path/file.txt
    """
    parts = list(_member_parts(image_member))

    image_index = parts.index("images")

    label_parts = (
        parts[:image_index]
        + ["labels"]
        + parts[image_index + 1:]
    )

    label_path = PurePosixPath(*label_parts).with_suffix(".txt")

    return str(label_path)


def _read_classes_txt(zf):
    """
    Read the optional Cute-YOLO class name.

    Preferred:
        classes.txt -> exactly one class name

    Fallback:
        no classes.txt -> class name "X"

    This lets already-YOLO-formatted public datasets be used directly
    without repacking them just to add classes.txt.
    """
    candidates = [
        member
        for member in zf.namelist()
        if PurePosixPath(member).name == "classes.txt"
    ]

    if not candidates:
        return DEFAULT_DATASET_CLASS

    # Prefer the shortest path, e.g. dataset/classes.txt.
    member = min(
        candidates,
        key=lambda name: len(_member_parts(name)),
    )

    text = zf.read(member).decode(
        "utf-8",
        errors="replace",
    )

    classes = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if len(classes) != 1:
        raise ValueError(
            "Cute-YOLO requires exactly one class in classes.txt; "
            f"found {len(classes)}: {classes}"
        )

    return classes[0]


def load_yolo_zip_arrays(
    zip_path,
    val_fraction=0.20,
    seed=SEED,
    min_object_size_px=0.0,
):
    """
    Load a single-class YOLO ZIP into memory.

    Real-image preprocessing intentionally matches deployment geometry:

        arbitrary W x H
            -> centered S x S crop
            -> transform/clip labels
            -> resize to 128 x 128 grayscale

    The target coordinates therefore describe exactly the geometry seen
    by the fixed Cute-YOLO detector.

    min_object_size_px is measured AFTER the center-square crop, at the
    final 128x128 detector input.  Boxes smaller than this threshold in
    either width or height are excluded from BOTH training and validation
    targets.  Set to 0 to disable size filtering.
    """
    zip_path = Path(zip_path)
    min_object_size_px = max(0.0, float(min_object_size_px))

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    images = []
    targets = []
    flipped_targets = []

    total_boxes_before_crop = 0
    total_boxes_after_crop = 0
    total_boxes_after_size_filter = 0
    total_boxes_dropped_by_min_size = 0
    total_boxes_dropped = 0
    total_boxes_clipped_kept = 0
    total_boxes_fully_inside = 0

    negative_images_before_crop = 0
    negative_images_after_crop = 0
    negative_images_after_size_filter = 0
    grid_collisions = 0

    original_shapes = set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        dataset_class = _read_classes_txt(zf)

        image_members = _find_dataset_image_members(zf)

        if not image_members:
            raise RuntimeError(
                f"No images found inside {zip_path}"
            )

        zip_names = set(zf.namelist())

        for image_member in image_members:
            encoded = np.frombuffer(
                zf.read(image_member),
                dtype=np.uint8,
            )

            original_image = cv2.imdecode(
                encoded,
                cv2.IMREAD_GRAYSCALE,
            )

            if original_image is None:
                raise RuntimeError(
                    f"Could not decode image: {image_member}"
                )

            original_h, original_w = original_image.shape

            original_shapes.add(
                (int(original_h), int(original_w))
            )

            label_member = _label_member_for_image(
                image_member
            )

            if label_member in zip_names:
                label_text = zf.read(
                    label_member
                ).decode(
                    "utf-8",
                    errors="replace",
                )
            elif ALLOW_MISSING_LABEL_FILES:
                label_text = ""
            else:
                raise FileNotFoundError(
                    "Missing matching YOLO label file for:\n"
                    f"  {image_member}\n"
                    f"Expected:\n"
                    f"  {label_member}"
                )

            original_boxes = parse_yolo_text(
                label_text
            )

            if not original_boxes:
                negative_images_before_crop += 1

            total_boxes_before_crop += len(
                original_boxes
            )

            (
                square_crop,
                transformed_boxes,
                crop_stats,
            ) = center_square_crop_and_transform_boxes(
                original_image,
                original_boxes,
                min_retained_area=CROP_MIN_RETAINED_AREA,
            )

            detector_image = resize_detector_input(
                square_crop
            )

            if not transformed_boxes:
                negative_images_after_crop += 1

            total_boxes_after_crop += len(
                transformed_boxes
            )

            total_boxes_dropped += (
                crop_stats["dropped_boxes"]
            )

            total_boxes_clipped_kept += (
                crop_stats["clipped_boxes_kept"]
            )

            total_boxes_fully_inside += (
                crop_stats["fully_inside_boxes"]
            )

            # Optional detector-scale size filtering.  This happens after
            # the square crop, so width/height map directly to the final
            # 128x128 detector input.  The same rule is applied before the
            # train/validation split, keeping evaluation consistent with
            # what the model was asked to learn.
            filtered_boxes = []
            for box in transformed_boxes:
                box_w_px = float(box["w"]) * IMG_SIZE
                box_h_px = float(box["h"]) * IMG_SIZE

                if (
                    box_w_px + 1e-9 >= min_object_size_px
                    and box_h_px + 1e-9 >= min_object_size_px
                ):
                    filtered_boxes.append(box)
                else:
                    total_boxes_dropped_by_min_size += 1

            total_boxes_after_size_filter += len(filtered_boxes)

            if not filtered_boxes:
                negative_images_after_size_filter += 1

            target, collisions = encode_target(
                filtered_boxes,
                return_collisions=True,
            )

            # Precompute the geometrically correct target for a horizontal
            # image flip. Transform boxes first (x -> 1-x), then re-encode.
            flipped_target = encode_target(
                flip_boxes_horizontally(filtered_boxes)
            )

            grid_collisions += collisions

            images.append(
                detector_image[..., None]
            )

            targets.append(
                target
            )
            flipped_targets.append(
                flipped_target
            )

    images = np.asarray(
        images,
        dtype=np.uint8,
    )

    targets = np.asarray(
        targets,
        dtype=np.float32,
    )

    flipped_targets = np.asarray(
        flipped_targets,
        dtype=np.float32,
    )

    indices = np.arange(len(images))

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    if len(indices) < 2:
        raise RuntimeError(
            "Need at least two images to create train/validation splits."
        )

    val_count = int(
        round(len(indices) * float(val_fraction))
    )

    val_count = max(
        1,
        min(len(indices) - 1, val_count),
    )

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    stats = {
        "dataset_zip": str(zip_path),
        "class_name": dataset_class,
        "class_name_source": (
            "classes.txt"
            if dataset_class != DEFAULT_DATASET_CLASS
            else "fallback_X_if_classes_txt_missing"
        ),
        "geometry_policy": "center_square_crop_then_resize_128",
        "crop_min_retained_area": float(
            CROP_MIN_RETAINED_AREA
        ),
        "detector_input_size_px": int(IMG_SIZE),
        "min_object_size_px_at_detector_input": float(
            min_object_size_px
        ),
        "images": int(len(images)),
        "train_images": int(len(train_indices)),
        "validation_images": int(len(val_indices)),
        "ground_truth_boxes_before_crop": int(
            total_boxes_before_crop
        ),
        "ground_truth_boxes_after_crop": int(
            total_boxes_after_crop
        ),
        "ground_truth_boxes_after_size_filter": int(
            total_boxes_after_size_filter
        ),
        "boxes_dropped_by_min_object_size": int(
            total_boxes_dropped_by_min_size
        ),
        "boxes_fully_inside_crop": int(
            total_boxes_fully_inside
        ),
        "boxes_clipped_but_kept": int(
            total_boxes_clipped_kept
        ),
        "boxes_dropped_by_crop": int(
            total_boxes_dropped
        ),
        "negative_images_before_crop": int(
            negative_images_before_crop
        ),
        "negative_images_after_crop": int(
            negative_images_after_crop
        ),
        "negative_images_after_size_filter": int(
            negative_images_after_size_filter
        ),
        "grid_cell_collisions": int(
            grid_collisions
        ),
        "original_image_shapes": [
            list(shape)
            for shape in sorted(
                original_shapes
            )
        ],
    }

    return (
        dataset_class,
        images[train_indices],
        targets[train_indices],
        flipped_targets[train_indices],
        images[val_indices],
        targets[val_indices],
        flipped_targets[val_indices],
        stats,
    )


def target_boxes(target):
    target = np.asarray(target)
    boxes = []

    ys, xs = np.nonzero(
        target[..., 0] > 0.5
    )

    for gy, gx in zip(ys, xs):
        dx, dy, width, height = (
            target[gy, gx, 1:5]
        )

        cx = (
            gx + dx
        ) / GRID_SIZE

        cy = (
            gy + dy
        ) / GRID_SIZE

        boxes.append({
            "x1": float(
                np.clip(
                    cx - width / 2,
                    0,
                    1,
                )
            ),
            "y1": float(
                np.clip(
                    cy - height / 2,
                    0,
                    1,
                )
            ),
            "x2": float(
                np.clip(
                    cx + width / 2,
                    0,
                    1,
                )
            ),
            "y2": float(
                np.clip(
                    cy + height / 2,
                    0,
                    1,
                )
            ),
        })

    return boxes


def draw_box(
    ax,
    box,
    linestyle="-",
    linewidth=1.5,
):
    rect = patches.Rectangle(
        (
            box["x1"] * IMG_SIZE,
            box["y1"] * IMG_SIZE,
        ),
        (
            box["x2"]
            - box["x1"]
        ) * IMG_SIZE,
        (
            box["y2"]
            - box["y1"]
        ) * IMG_SIZE,
        fill=False,
        linewidth=linewidth,
        linestyle=linestyle,
    )

    ax.add_patch(rect)


# ============================================================
# Debug images
# ============================================================

DEBUG_SAMPLE_COUNT = 12

# Before/after adaptation prediction figures must show the exact
# same validation samples for a meaningful visual comparison.
ADAPT_COMPARE_DEBUG_SEED = SEED + 3


def _choose_debug_indices(
    count,
    sample_count=DEBUG_SAMPLE_COUNT,
    seed=SEED,
):
    if count <= 0:
        return np.asarray(
            [],
            dtype=np.int64,
        )

    sample_count = min(
        int(sample_count),
        int(count),
    )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.choice(
        int(count),
        size=sample_count,
        replace=False,
    )

    return np.sort(
        indices
    )


def save_dataset_debug_preview(
    images,
    targets,
    output_path,
    title,
    sample_count=DEBUG_SAMPLE_COUNT,
    seed=SEED,
):
    output_path = Path(
        output_path
    )

    indices = _choose_debug_indices(
        len(images),
        sample_count=
            sample_count,
        seed=seed,
    )

    if len(indices) == 0:
        return None

    columns = 4
    rows = int(
        math.ceil(
            len(indices)
            / columns
        )
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            3.4 * columns,
            3.4 * rows,
        ),
        squeeze=False,
    )

    axes = axes.reshape(
        -1
    )

    for panel_index, ax in enumerate(
        axes
    ):
        ax.axis(
            "off"
        )

        if panel_index >= len(indices):
            continue

        sample_index = int(
            indices[
                panel_index
            ]
        )

        image = images[
            sample_index,
            ...,
            0,
        ]

        gt = target_boxes(
            targets[
                sample_index
            ]
        )

        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        for box in gt:
            draw_box(
                ax,
                box,
                linestyle="-",
                linewidth=1.3,
            )

        ax.set_title(
            (
                f"index={sample_index}  "
                f"GT={len(gt)}"
            ),
            fontsize=9,
        )

    fig.suptitle(
        title
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def save_training_history_debug(
    history,
    debug_dir,
    prefix="training",
):
    debug_dir = Path(
        debug_dir
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = np.asarray(
        history.get(
            "train",
            [],
        ),
        dtype=np.float64,
    )

    # Base training stores validation metrics under "validation".
    # Domain adaptation stores target-domain validation metrics under
    # "target_validation". Support both so the same plotting helper
    # works for Train and Adapt.
    validation_values = history.get(
        "validation",
        None,
    )

    if validation_values is None:
        validation_values = history.get(
            "target_validation",
            [],
        )

    validation = np.asarray(
        validation_values,
        dtype=np.float64,
    )

    learning_rate = np.asarray(
        history.get(
            "learning_rate",
            [],
        ),
        dtype=np.float64,
    )

    created = []

    if (
        train.ndim == 2
        and validation.ndim == 2
        and len(train) > 0
        and len(validation) > 0
    ):
        epochs = np.arange(
            1,
            min(
                len(train),
                len(validation),
            ) + 1,
        )

        count = len(
            epochs
        )

        # ----------------------------------------------------
        # Total training / validation loss
        # ----------------------------------------------------
        fig, ax = plt.subplots(
            figsize=(
                8.5,
                5.0,
            )
        )

        ax.plot(
            epochs,
            train[
                :count,
                0,
            ],
            label="train total",
        )

        ax.plot(
            epochs,
            validation[
                :count,
                0,
            ],
            label="validation total",
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Loss"
        )

        ax.set_title(
            "Cute-YOLO training loss"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend()

        fig.tight_layout()

        path = (
            debug_dir
            / f"{prefix}_loss.png"
        )

        fig.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        created.append(
            path
        )

        # ----------------------------------------------------
        # Loss components
        # [total, objectness, box, selected overlap loss, footprint-negative, halo-negative]
        # ----------------------------------------------------
        if (
            train.shape[1] >= 4
            and validation.shape[1] >= 4
        ):
            fig, ax = plt.subplots(
                figsize=(
                    9.5,
                    5.5,
                )
            )

            component_names = [
                "objectness",
                "box",
                box_overlap_loss_label(),
            ]

            line_styles = [
                "-",
                "--",
                ":",
            ]

            # Box-aware runs log footprint and halo BCE separately.
            if (
                train.shape[1] >= 5
                and validation.shape[1] >= 5
            ):
                component_names.append(
                    "footprint-negative"
                )
                line_styles.append(
                    "-."
                )

            if (
                train.shape[1] >= 6
                and validation.shape[1] >= 6
            ):
                component_names.append(
                    "halo-negative"
                )
                line_styles.append(
                    (0, (3, 1, 1, 1))
                )

            for component_index, (
                component_name,
                line_style,
            ) in enumerate(
                zip(
                    component_names,
                    line_styles,
                ),
                start=1,
            ):
                ax.plot(
                    epochs,
                    train[
                        :count,
                        component_index,
                    ],
                    linestyle=
                        line_style,
                    label=(
                        f"train "
                        f"{component_name}"
                    ),
                )

                ax.plot(
                    epochs,
                    validation[
                        :count,
                        component_index,
                    ],
                    linestyle=
                        line_style,
                    label=(
                        f"validation "
                        f"{component_name}"
                    ),
                    alpha=0.65,
                )

            ax.set_xlabel(
                "Epoch"
            )

            ax.set_ylabel(
                "Loss component"
            )

            ax.set_title(
                "Cute-YOLO loss components"
            )

            ax.grid(
                True,
                alpha=0.25,
            )

            ax.legend(
                ncol=2
            )

            fig.tight_layout()

            path = (
                debug_dir
                / (
                    f"{prefix}"
                    "_loss_components.png"
                )
            )

            fig.savefig(
                path,
                dpi=180,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

            created.append(
                path
            )

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------
    if len(
        learning_rate
    ) > 0:
        epochs = np.arange(
            1,
            len(
                learning_rate
            ) + 1,
        )

        fig, ax = plt.subplots(
            figsize=(
                8.0,
                4.5,
            )
        )

        ax.plot(
            epochs,
            learning_rate,
            marker="o",
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Learning rate"
        )

        ax.set_title(
            "Learning-rate schedule"
        )

        if np.all(
            learning_rate
            > 0
        ):
            ax.set_yscale(
                "log"
            )

        ax.grid(
            True,
            alpha=0.25,
        )

        fig.tight_layout()

        path = (
            debug_dir
            / f"{prefix}_learning_rate.png"
        )

        fig.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        created.append(
            path
        )

    return created


def save_operating_point_debug(
    operating_points,
    output_path,
):
    output_path = Path(
        output_path
    )

    if not operating_points:
        return None

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.0,
        )
    )

    for nms_iou in NMS_IOU_GRID:
        subset = [
            item
            for item
            in operating_points
            if np.isclose(
                item[
                    "nms_iou_threshold"
                ],
                nms_iou,
            )
        ]

        subset = sorted(
            subset,
            key=lambda item:
                item[
                    "confidence_threshold"
                ],
        )

        if not subset:
            continue

        ax.plot(
            [
                item[
                    "confidence_threshold"
                ]
                for item
                in subset
            ],
            [
                item[
                    "f1"
                ]
                for item
                in subset
            ],
            marker="o",
            label=(
                f"NMS={nms_iou:.2f}"
            ),
        )

    ax.set_xlabel(
        "Confidence threshold"
    )

    ax.set_ylabel(
        "F1"
    )

    ax.set_title(
        "Validation operating-point sweep"
    )

    ax.set_ylim(
        bottom=0.0
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def save_validation_prediction_debug(
    images,
    targets,
    predictions,
    confidence_threshold,
    nms_iou_threshold,
    output_path,
    sample_count=DEBUG_SAMPLE_COUNT,
    seed=SEED + 1,
    title=None,
):
    output_path = Path(
        output_path
    )

    indices = _choose_debug_indices(
        len(images),
        sample_count=
            sample_count,
        seed=seed,
    )

    if len(indices) == 0:
        return None

    columns = 4
    rows = int(
        math.ceil(
            len(indices)
            / columns
        )
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            3.6 * columns,
            3.7 * rows,
        ),
        squeeze=False,
    )

    axes = axes.reshape(
        -1
    )

    for panel_index, ax in enumerate(
        axes
    ):
        ax.axis(
            "off"
        )

        if panel_index >= len(indices):
            continue

        sample_index = int(
            indices[
                panel_index
            ]
        )

        image = images[
            sample_index,
            ...,
            0,
        ]

        ground_truth = target_boxes(
            targets[
                sample_index
            ]
        )

        detections = (
            non_maximum_suppression(
                decode_predictions(
                    predictions[
                        sample_index
                    ],
                    confidence_threshold=
                        confidence_threshold,
                ),
                iou_threshold=
                    nms_iou_threshold,
                max_detections=
                    EVAL_MAX_DETECTIONS,
            )
        )

        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        # Solid rectangles = GT.
        for box in ground_truth:
            draw_box(
                ax,
                box,
                linestyle="-",
                linewidth=1.4,
            )

        # Dashed rectangles = model prediction.
        for detection in detections:
            draw_box(
                ax,
                detection,
                linestyle="--",
                linewidth=1.2,
            )

            ax.text(
                detection["x1"]
                * IMG_SIZE,
                max(
                    4.0,
                    detection["y1"]
                    * IMG_SIZE
                    - 2.0,
                ),
                (
                    f"{detection['confidence']:.2f}"
                ),
                fontsize=7,
            )

        ax.set_title(
            (
                f"index={sample_index}  "
                f"GT={len(ground_truth)}  "
                f"P={len(detections)}"
            ),
            fontsize=9,
        )

    if title is None:
        title = (
            "Validation predictions: "
            "solid = ground truth, dashed = prediction"
        )

    fig.suptitle(
        title
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def save_dataset_distribution_debug(
    train_targets,
    val_targets,
    output_path,
):
    output_path = Path(
        output_path
    )

    all_targets = np.concatenate(
        [
            train_targets,
            val_targets,
        ],
        axis=0,
    )

    object_counts = np.sum(
        all_targets[
            ...,
            0,
        ] > 0.5,
        axis=(
            1,
            2,
        ),
    ).astype(
        np.int32
    )

    widths_px = []
    heights_px = []

    for target in all_targets:
        ys, xs = np.nonzero(
            target[
                ...,
                0,
            ] > 0.5
        )

        for gy, gx in zip(
            ys,
            xs,
        ):
            widths_px.append(
                float(
                    target[
                        gy,
                        gx,
                        3,
                    ]
                    * IMG_SIZE
                )
            )

            heights_px.append(
                float(
                    target[
                        gy,
                        gx,
                        4,
                    ]
                    * IMG_SIZE
                )
            )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            11.0,
            4.5,
        ),
    )

    axes[0].hist(
        object_counts,
        bins=min(
            30,
            max(
                5,
                int(
                    object_counts.max()
                    + 1
                )
                if len(
                    object_counts
                )
                else 5,
            ),
        ),
    )

    axes[0].set_xlabel(
        "Encoded objects per image"
    )

    axes[0].set_ylabel(
        "Image count"
    )

    axes[0].set_title(
        "Objects per detector input"
    )

    if widths_px:
        axes[1].hist(
            widths_px,
            bins=30,
            alpha=0.65,
            label="width",
        )

    if heights_px:
        axes[1].hist(
            heights_px,
            bins=30,
            alpha=0.65,
            label="height",
        )

    axes[1].set_xlabel(
        "Box size at 128×128 (pixels)"
    )

    axes[1].set_ylabel(
        "Box count"
    )

    axes[1].set_title(
        "Encoded box-size distribution"
    )

    if (
        widths_px
        or heights_px
    ):
        axes[1].legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def cute_yolo_loss(prediction, target):
    prediction = tf.cast(prediction, tf.float32)
    target = tf.cast(target, tf.float32)

    true_objectness = target[..., 0:1]
    raw_objectness = prediction[..., 0:1]

    elementwise_bce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=true_objectness,
        logits=raw_objectness,
    )

    positive_mask = true_objectness > 0.5
    negative_mask = tf.logical_not(positive_mask)

    zero = tf.reduce_sum(elementwise_bce) * 0.0

    positive_values = tf.boolean_mask(elementwise_bce, positive_mask)
    positive_loss = tf.cond(
        tf.size(positive_values) > 0,
        lambda: tf.reduce_mean(positive_values),
        lambda: zero,
    )

    negative_values = tf.boolean_mask(elementwise_bce, negative_mask)

    def hard_negative_loss():
        batch_size = tf.shape(prediction)[0]
        hard_k = tf.minimum(
            tf.size(negative_values),
            HARD_NEGATIVES * batch_size,
        )
        hardest = tf.math.top_k(
            negative_values,
            k=hard_k,
            sorted=False,
        ).values
        return tf.reduce_mean(hardest)

    if HARD_NEGATIVE_MINING_ENABLED:
        negative_loss = tf.cond(
            tf.size(negative_values) > 0,
            hard_negative_loss,
            lambda: zero,
        )
    else:
        negative_loss = zero

    # --------------------------------------------------------
    # Box-aware spatial negative emphasis
    #
    # Rasterize every encoded GT box onto the 16x16 detector grid.
    # A grid cell enters the footprint when its AREA overlaps any GT
    # box. Then add a one-cell halo around the complete footprint.
    #
    # True object-center cells are excluded from both masks globally.
    # --------------------------------------------------------
    if BOX_AWARE_NEGATIVE_ENABLED:
        true_box_for_mask = target[..., 1:5]

        src_grid_y, src_grid_x = tf.meshgrid(
            tf.range(GRID_SIZE, dtype=tf.float32),
            tf.range(GRID_SIZE, dtype=tf.float32),
            indexing="ij",
        )

        src_grid_x = src_grid_x[None, ..., None]
        src_grid_y = src_grid_y[None, ..., None]

        gt_center_x = (
            src_grid_x + true_box_for_mask[..., 0:1]
        ) / GRID_SIZE
        gt_center_y = (
            src_grid_y + true_box_for_mask[..., 1:2]
        ) / GRID_SIZE

        gt_width = true_box_for_mask[..., 2:3]
        gt_height = true_box_for_mask[..., 3:4]

        gt_x1 = gt_center_x - gt_width / 2.0
        gt_y1 = gt_center_y - gt_height / 2.0
        gt_x2 = gt_center_x + gt_width / 2.0
        gt_y2 = gt_center_y + gt_height / 2.0

        # Candidate detector-cell boundaries.
        cell_left = (
            tf.range(GRID_SIZE, dtype=tf.float32) / GRID_SIZE
        )[None, None, None, None, :]
        cell_right = (
            tf.range(1, GRID_SIZE + 1, dtype=tf.float32) / GRID_SIZE
        )[None, None, None, None, :]
        cell_top = (
            tf.range(GRID_SIZE, dtype=tf.float32) / GRID_SIZE
        )[None, None, None, :, None]
        cell_bottom = (
            tf.range(1, GRID_SIZE + 1, dtype=tf.float32) / GRID_SIZE
        )[None, None, None, :, None]

        # Expand every source GT box over all candidate detector cells.
        gt_x1_e = gt_x1[..., None]
        gt_y1_e = gt_y1[..., None]
        gt_x2_e = gt_x2[..., None]
        gt_y2_e = gt_y2[..., None]

        source_positive = (
            true_objectness[..., 0] > 0.5
        )[:, :, :, None, None]

        overlaps_x = tf.logical_and(
            cell_left < gt_x2_e,
            cell_right > gt_x1_e,
        )
        overlaps_y = tf.logical_and(
            cell_top < gt_y2_e,
            cell_bottom > gt_y1_e,
        )

        box_cell_overlap = tf.logical_and(
            source_positive,
            tf.logical_and(overlaps_x, overlaps_y),
        )

        # Per-cell footprint count. This is not merely a Boolean union:
        # it records how many GT boxes overlap each detector cell.
        footprint_count = tf.reduce_sum(
            tf.cast(box_cell_overlap, tf.float32),
            axis=[1, 2],
        )[..., None]

        footprint_mask = footprint_count > 0.0

        # Build an object-specific halo by expanding every GT box by the
        # requested number of detector cells BEFORE reducing across objects.
        # This preserves crowd information: if a gap is influenced by four
        # nearby objects, its influence count becomes four rather than one.
        halo_radius_cells = max(0, (BOX_HALO_KERNEL - 1) // 2)
        halo_pad = tf.cast(halo_radius_cells, tf.float32) / GRID_SIZE

        influence_overlaps_x = tf.logical_and(
            cell_left < (gt_x2_e + halo_pad),
            cell_right > (gt_x1_e - halo_pad),
        )
        influence_overlaps_y = tf.logical_and(
            cell_top < (gt_y2_e + halo_pad),
            cell_bottom > (gt_y1_e - halo_pad),
        )

        influence_overlap = tf.logical_and(
            source_positive,
            tf.logical_and(
                influence_overlaps_x,
                influence_overlaps_y,
            ),
        )

        influence_count = tf.reduce_sum(
            tf.cast(influence_overlap, tf.float32),
            axis=[1, 2],
        )[..., None]

        influence_mask = influence_count > 0.0

        # True object centers are never negative, including centers of
        # other overlapping objects.
        footprint_negative_mask = tf.logical_and(
            footprint_mask,
            negative_mask,
        )

        # Halo is the expanded influence region outside every actual
        # footprint. It is intentionally disjoint from the footprint.
        halo_negative_mask = tf.logical_and(
            influence_mask,
            negative_mask,
        )
        halo_negative_mask = tf.logical_and(
            halo_negative_mask,
            tf.logical_not(footprint_mask),
        )

        # ----------------------------------------------------
        # Crowd-aware weighting
        #
        # C_ij = number of GT object footprint/halo regions that influence
        # this cell. Ordinary single-object negatives get a modest weight,
        # while inter-object gaps surrounded by several objects are
        # emphasized more strongly. The cap prevents extreme gradients.
        #
        #   w_ij = 1 + gamma * min(C_ij, C_max)
        #
        # The weights are normalized inside each spatial term, so this
        # changes the RELATIVE emphasis among spatial negatives without
        # silently multiplying the whole footprint/halo loss magnitude.
        # ----------------------------------------------------
        if CROWD_AWARE_NEGATIVE_ENABLED:
            clipped_count = tf.minimum(
                influence_count,
                tf.cast(CROWD_COUNT_CAP, tf.float32),
            )
            crowd_weight = (
                1.0
                + tf.cast(CROWD_COUNT_GAMMA, tf.float32)
                * clipped_count
            )
        else:
            crowd_weight = tf.ones_like(influence_count)

        footprint_values = tf.boolean_mask(
            elementwise_bce,
            footprint_negative_mask,
        )
        footprint_weights = tf.boolean_mask(
            crowd_weight,
            footprint_negative_mask,
        )

        footprint_negative_loss = tf.cond(
            tf.size(footprint_values) > 0,
            lambda: tf.reduce_sum(
                footprint_values * footprint_weights
            ) / tf.maximum(
                tf.reduce_sum(footprint_weights),
                1e-6,
            ),
            lambda: zero,
        )

        halo_values = tf.boolean_mask(
            elementwise_bce,
            halo_negative_mask,
        )
        halo_weights = tf.boolean_mask(
            crowd_weight,
            halo_negative_mask,
        )

        halo_negative_loss = tf.cond(
            tf.size(halo_values) > 0,
            lambda: tf.reduce_sum(
                halo_values * halo_weights
            ) / tf.maximum(
                tf.reduce_sum(halo_weights),
                1e-6,
            ),
            lambda: zero,
        )

    else:
        footprint_negative_loss = zero
        halo_negative_loss = zero

    objectness_loss = (
        positive_loss
        + BACKGROUND_LOSS_WEIGHT * negative_loss
        + BOX_FOOTPRINT_NEGATIVE_WEIGHT * footprint_negative_loss
        + BOX_HALO_NEGATIVE_WEIGHT * halo_negative_loss
    )

    predicted_box = tf.sigmoid(prediction[..., 1:5])
    true_box = target[..., 1:5]

    positive_count = tf.maximum(
        tf.reduce_sum(tf.cast(positive_mask, tf.float32)),
        1.0,
    )

    box_loss = (
        tf.reduce_sum(
            tf.square(predicted_box - true_box) * true_objectness
        )
        / (positive_count * 4.0)
    )

    grid_y, grid_x = tf.meshgrid(
        tf.range(GRID_SIZE, dtype=tf.float32),
        tf.range(GRID_SIZE, dtype=tf.float32),
        indexing="ij",
    )

    grid_x = grid_x[None, ..., None]
    grid_y = grid_y[None, ..., None]

    predicted_center_x = (grid_x + predicted_box[..., 0:1]) / GRID_SIZE
    predicted_center_y = (grid_y + predicted_box[..., 1:2]) / GRID_SIZE
    true_center_x = (grid_x + true_box[..., 0:1]) / GRID_SIZE
    true_center_y = (grid_y + true_box[..., 1:2]) / GRID_SIZE

    predicted_x1 = predicted_center_x - predicted_box[..., 2:3] / 2
    predicted_y1 = predicted_center_y - predicted_box[..., 3:4] / 2
    predicted_x2 = predicted_center_x + predicted_box[..., 2:3] / 2
    predicted_y2 = predicted_center_y + predicted_box[..., 3:4] / 2

    true_x1 = true_center_x - true_box[..., 2:3] / 2
    true_y1 = true_center_y - true_box[..., 3:4] / 2
    true_x2 = true_center_x + true_box[..., 2:3] / 2
    true_y2 = true_center_y + true_box[..., 3:4] / 2

    intersection = (
        tf.maximum(
            tf.minimum(predicted_x2, true_x2)
            - tf.maximum(predicted_x1, true_x1),
            0.0,
        )
        *
        tf.maximum(
            tf.minimum(predicted_y2, true_y2)
            - tf.maximum(predicted_y1, true_y1),
            0.0,
        )
    )

    predicted_area = (
        tf.maximum(predicted_x2 - predicted_x1, 0.0)
        * tf.maximum(predicted_y2 - predicted_y1, 0.0)
    )
    true_area = (
        tf.maximum(true_x2 - true_x1, 0.0)
        * tf.maximum(true_y2 - true_y1, 0.0)
    )

    union = (
        predicted_area
        + true_area
        - intersection
    )

    iou = intersection / (
        union + 1e-7
    )

    # ========================================================
    # Selectable overlap loss: IoU or CIoU
    # ========================================================

    iou_loss = (
        tf.reduce_sum(
            (1.0 - iou)
            * true_objectness
        )
        / positive_count
    )

    if BOX_OVERLAP_LOSS == BOX_OVERLAP_LOSS_IOU:
        overlap_loss = iou_loss
    else:
        # Complete IoU (CIoU) adds center-distance and aspect-ratio
        # penalties to the ordinary overlap term.
        enclosing_x1 = tf.minimum(
            predicted_x1,
            true_x1,
        )
        enclosing_y1 = tf.minimum(
            predicted_y1,
            true_y1,
        )
        enclosing_x2 = tf.maximum(
            predicted_x2,
            true_x2,
        )
        enclosing_y2 = tf.maximum(
            predicted_y2,
            true_y2,
        )

        enclosing_w = tf.maximum(
            enclosing_x2 - enclosing_x1,
            0.0,
        )
        enclosing_h = tf.maximum(
            enclosing_y2 - enclosing_y1,
            0.0,
        )

        enclosing_diagonal_sq = (
            tf.square(enclosing_w)
            + tf.square(enclosing_h)
            + 1e-7
        )

        center_distance_sq = (
            tf.square(
                predicted_center_x
                - true_center_x
            )
            + tf.square(
                predicted_center_y
                - true_center_y
            )
        )

        predicted_w = tf.maximum(
            predicted_box[..., 2:3],
            1e-7,
        )
        predicted_h = tf.maximum(
            predicted_box[..., 3:4],
            1e-7,
        )
        true_w = tf.maximum(
            true_box[..., 2:3],
            1e-7,
        )
        true_h = tf.maximum(
            true_box[..., 3:4],
            1e-7,
        )

        aspect_v = (
            4.0
            / (math.pi ** 2)
            * tf.square(
                tf.atan(
                    true_w
                    / true_h
                )
                - tf.atan(
                    predicted_w
                    / predicted_h
                )
            )
        )

        alpha = tf.stop_gradient(
            aspect_v
            / (
                1.0
                - iou
                + aspect_v
                + 1e-7
            )
        )

        ciou = (
            iou
            - center_distance_sq
            / enclosing_diagonal_sq
            - alpha
            * aspect_v
        )

        overlap_loss = (
            tf.reduce_sum(
                (1.0 - ciou)
                * true_objectness
            )
            / positive_count
        )

    total_loss = (
        objectness_loss
        + BOX_WEIGHT * box_loss
        + IOU_WEIGHT * overlap_loss
    )

    # Keep the "iou" dictionary key for compatibility with the
    # existing training/adaptation loops and history format. Its value is
    # whichever overlap loss (IoU or CIoU) is selected for this run.
    return {
        "total": total_loss,
        "objectness": objectness_loss,
        "box": box_loss,
        "iou": overlap_loss,
        "footprint_negative": footprint_negative_loss,
        "halo_negative": halo_negative_loss,
    }


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def box_iou(box_a, box_b):
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(
        0.0, box_a["y2"] - box_a["y1"]
    )
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(
        0.0, box_b["y2"] - box_b["y1"]
    )

    return intersection / (area_a + area_b - intersection + 1e-9)


def decode_predictions(
    prediction,
    confidence_threshold,
    min_box_w=MIN_BOX_W,
    min_box_h=MIN_BOX_H,
):
    prediction = np.asarray(prediction, dtype=np.float32)

    objectness = sigmoid_np(prediction[..., 0])
    box_values = sigmoid_np(prediction[..., 1:5])

    detections = []

    ys, xs = np.nonzero(objectness >= confidence_threshold)

    for grid_y, grid_x in zip(ys, xs):
        confidence = float(objectness[grid_y, grid_x])
        x_offset, y_offset, width, height = [
            float(v) for v in box_values[grid_y, grid_x]
        ]

        if width < min_box_w or height < min_box_h:
            continue

        center_x = (grid_x + x_offset) / GRID_SIZE
        center_y = (grid_y + y_offset) / GRID_SIZE

        detections.append({
            "confidence": confidence,
            "x1": float(np.clip(center_x - width / 2, 0, 1)),
            "y1": float(np.clip(center_y - height / 2, 0, 1)),
            "x2": float(np.clip(center_x + width / 2, 0, 1)),
            "y2": float(np.clip(center_y + height / 2, 0, 1)),
        })

    return detections


def non_maximum_suppression(
    detections,
    iou_threshold,
    max_detections=RUNTIME_MAX_DETECTIONS,
):
    remaining = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True,
    )
    selected = []

    while remaining and len(selected) < max_detections:
        best = remaining.pop(0)
        selected.append(best)

        remaining = [
            candidate
            for candidate in remaining
            if box_iou(best, candidate) < iou_threshold
        ]

    return selected


def evaluate_predictions(
    predictions,
    targets,
    confidence_threshold,
    nms_iou_threshold,
):
    true_positive = false_positive = false_negative = 0
    matched_ious = []

    for prediction, target in zip(predictions, targets):
        detections = non_maximum_suppression(
            decode_predictions(
                prediction,
                confidence_threshold=confidence_threshold,
            ),
            iou_threshold=nms_iou_threshold,
            max_detections=
                EVAL_MAX_DETECTIONS,
        )

        ground_truth = target_boxes(target)
        used_ground_truth = set()

        for detection in sorted(
            detections,
            key=lambda item: item["confidence"],
            reverse=True,
        ):
            choices = [
                (box_iou(detection, box), index)
                for index, box in enumerate(ground_truth)
                if index not in used_ground_truth
            ]

            best_iou, best_index = max(
                choices,
                default=(-1.0, -1),
            )

            if best_iou >= 0.50:
                true_positive += 1
                used_ground_truth.add(best_index)
                matched_ious.append(best_iou)
            else:
                false_positive += 1

        false_negative += len(ground_truth) - len(used_ground_truth)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "confidence_threshold": float(confidence_threshold),
        "nms_iou_threshold": float(nms_iou_threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": (
            float(np.mean(matched_ious))
            if matched_ious
            else 0.0
        ),
    }


def sweep_operating_points(
    model,
    images,
    targets,
    verbose=True,
):
    predictions = model.predict(
        images.astype(np.float32)
        / 255.0,
        batch_size=BATCH_SIZE,
        verbose=(
            1
            if verbose
            else 0
        ),
    )

    operating_points = []

    for nms_iou in NMS_IOU_GRID:
        for confidence in CONFIDENCE_GRID:
            result = evaluate_predictions(
                predictions,
                targets,
                confidence_threshold=
                    confidence,
                nms_iou_threshold=
                    nms_iou,
            )

            operating_points.append(
                result
            )

            if verbose:
                print(
                    f"conf={confidence:.2f} "
                    f"nms={nms_iou:.2f} | "
                    f"P={result['precision']:.3f} "
                    f"R={result['recall']:.3f} "
                    f"F1={result['f1']:.3f} "
                    f"IoU={result['mean_iou']:.3f}"
                )

    best = max(
        operating_points,
        key=lambda item: (
            item["f1"],
            item["mean_iou"],
        ),
    )

    return (
        best,
        operating_points,
    )




@dataclass
class AdaptConfig:
    base_model_zip: Path
    original_data_zip: Path
    real_data_zip: Path
    output_dir: Path

    source_ratio: float = 0.75
    target_ratio: float = 0.25
    target_val_fraction: float = 0.20

    epochs: int = 15
    learning_rate: float = 1e-4
    batch_size: int = 64

    early_stop_patience: int = 4
    plateau_patience: int = 2
    plateau_factor: float = 0.5
    min_learning_rate: float = 1e-6

    weight_decay: float = 1e-5
    target_passes_per_epoch: float = 1.0

    calibration_samples: int = 256
    calibration_target_fraction: float = 0.75

    make_cute: bool = True
    converter_path: Optional[Path] = None

    # Training-only loss profile.
    box_overlap_loss: str = DEFAULT_BOX_OVERLAP_LOSS
    loss_preset: str = DEFAULT_LOSS_PRESET
    hard_negative_mining_enabled: bool = True
    box_aware_negative_enabled: bool = False
    crowd_aware_negative_enabled: bool = False

    # Dataset preparation at the final 128x128 detector scale.
    min_object_size_px: float = DEFAULT_MIN_OBJECT_SIZE_PX

    # Mild training augmentation. Validation is never augmented.
    augment_horizontal_flip: bool = AUGMENT_HORIZONTAL_FLIP_DEFAULT
    augment_brightness: bool = AUGMENT_BRIGHTNESS_DEFAULT
    augment_contrast: bool = AUGMENT_CONTRAST_DEFAULT
    shuffle_each_epoch: bool = AUGMENT_SHUFFLE_DEFAULT

    # Training-only branch-specialized adaptation.
    specialized_finetune: bool = SPECIALIZED_FINETUNE_DEFAULT
    cross_loss_weight: float = SPECIALIZED_CROSS_LOSS_WEIGHT


def load_base_model(
    base_zip: Path,
    expected_label: str,
    work_dir: Path,
    logger: Callable[[str], None],
):
    extract_dir = work_dir / "base_model"

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        base_zip,
        "r",
    ) as archive:
        members = archive.namelist()

        keras_members = [
            member
            for member in members
            if member.endswith(".keras")
        ]

        runtime_members = [
            member
            for member in members
            if PurePosixPath(member).name
            == "runtime_config.json"
        ]

        if len(keras_members) != 1:
            raise RuntimeError(
                "The base export ZIP must contain exactly one .keras model. "
                f"Found: {keras_members}"
            )

        if len(runtime_members) != 1:
            raise RuntimeError(
                "The base export ZIP must contain exactly one runtime_config.json. "
                f"Found: {runtime_members}"
            )

        keras_member = keras_members[0]
        runtime_member = runtime_members[0]

        archive.extract(
            keras_member,
            extract_dir,
        )

        archive.extract(
            runtime_member,
            extract_dir,
        )

    model_path = (
        extract_dir
        / keras_member
    )

    runtime_path = (
        extract_dir
        / runtime_member
    )

    runtime_config = json.loads(
        runtime_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        runtime_config.get(
            "architecture_id"
        )
        != ARCHITECTURE_ID
    ):
        raise ValueError(
            "Base model architecture mismatch.\n"
            f"Expected: {ARCHITECTURE_ID}\n"
            f"Found: {runtime_config.get('architecture_id')}"
        )

    if (
        runtime_config.get("label")
        != expected_label
    ):
        raise ValueError(
            "Base model class does not match the datasets.\n"
            f"Base model: {runtime_config.get('label')!r}\n"
            f"Dataset:    {expected_label!r}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    if tuple(model.input_shape) != (
        None,
        IMG_SIZE,
        IMG_SIZE,
        1,
    ):
        raise ValueError(
            f"Unexpected model input shape: {model.input_shape}"
        )

    if tuple(model.output_shape) != (
        None,
        GRID_SIZE,
        GRID_SIZE,
        OUTPUT_CHANNELS,
    ):
        raise ValueError(
            f"Unexpected model output shape: {model.output_shape}"
        )

    parameter_count = int(
        sum(
            np.prod(variable.shape)
            for variable
            in model.trainable_variables
        )
    )

    if (
        parameter_count
        != EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise ValueError(
            "Unexpected Cute-YOLO parameter count: "
            f"{parameter_count:,}"
        )

    log_line(
        logger,
        f"Loaded base model: {model_path.name}",
    )

    log_line(
        logger,
        f"Architecture: {ARCHITECTURE_ID}",
    )

    log_line(
        logger,
        f"Trainable parameters: {parameter_count:,}",
    )

    return (
        model,
        runtime_config,
        parameter_count,
    )


def make_tf_dataset(
    images,
    targets,
    batch_size,
    *,
    flipped_targets=None,
    training=False,
    horizontal_flip=False,
    brightness=False,
    contrast=False,
    shuffle=False,
    seed=SEED,
):
    """Create a tf.data pipeline with label-safe training augmentation."""
    images = np.asarray(images)
    targets = np.asarray(targets)
    flipped_targets = targets if flipped_targets is None else np.asarray(flipped_targets)

    if len(images) != len(targets) or len(images) != len(flipped_targets):
        raise ValueError("Image/target/flipped-target counts do not match.")

    ds = tf.data.Dataset.from_tensor_slices((images, targets, flipped_targets))

    if training and shuffle and len(images) > 1:
        ds = ds.shuffle(
            buffer_size=len(images),
            seed=int(seed),
            reshuffle_each_iteration=True,
        )

    def prepare(image, target, flipped_target):
        image = tf.cast(image, tf.float32) / 255.0
        target = tf.cast(target, tf.float32)
        flipped_target = tf.cast(flipped_target, tf.float32)

        if training and horizontal_flip:
            do_flip = tf.random.uniform([], 0.0, 1.0) < AUGMENT_FLIP_PROBABILITY
            image, target = tf.cond(
                do_flip,
                lambda: (tf.image.flip_left_right(image), flipped_target),
                lambda: (image, target),
            )

        if training and brightness:
            factor = tf.random.uniform(
                [], AUGMENT_BRIGHTNESS_RANGE[0], AUGMENT_BRIGHTNESS_RANGE[1],
                dtype=tf.float32,
            )
            image = image * factor

        if training and contrast:
            factor = tf.random.uniform(
                [], AUGMENT_CONTRAST_RANGE[0], AUGMENT_CONTRAST_RANGE[1],
                dtype=tf.float32,
            )
            mean = tf.reduce_mean(image, axis=(0, 1), keepdims=True)
            image = (image - mean) * factor + mean

        return tf.clip_by_value(image, 0.0, 1.0), target

    return (
        ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def make_mixed_epoch_dataset(
    source_images,
    source_targets,
    source_flipped_targets,
    real_train_images,
    real_train_targets,
    real_train_flipped_targets,
    source_ratio,
    target_ratio,
    target_passes_per_epoch,
    batch_size,
    epoch_index,
    *,
    horizontal_flip=False,
    brightness=False,
    contrast=False,
    shuffle=False,
):
    rng = np.random.default_rng(
        SEED
        + 10_000
        + int(epoch_index)
    )

    target_count = max(
        1,
        int(
            round(
                len(real_train_images)
                * target_passes_per_epoch
            )
        ),
    )

    source_count = max(
        1,
        int(
            round(
                target_count
                * source_ratio
                / target_ratio
            )
        ),
    )

    source_indices = rng.choice(
        len(source_images),
        size=source_count,
        replace=(
            source_count
            > len(source_images)
        ),
    )

    target_indices = rng.choice(
        len(real_train_images),
        size=target_count,
        replace=(
            target_count
            > len(real_train_images)
        ),
    )

    mixed_images = np.concatenate(
        [
            source_images[
                source_indices
            ],
            real_train_images[
                target_indices
            ],
        ],
        axis=0,
    )

    mixed_targets = np.concatenate(
        [
            source_targets[source_indices],
            real_train_targets[target_indices],
        ],
        axis=0,
    )

    mixed_flipped_targets = np.concatenate(
        [
            source_flipped_targets[source_indices],
            real_train_flipped_targets[target_indices],
        ],
        axis=0,
    )

    order = rng.permutation(
        len(mixed_images)
    )

    mixed_images = mixed_images[
        order
    ]

    mixed_targets = mixed_targets[order]
    mixed_flipped_targets = mixed_flipped_targets[order]

    return (
        make_tf_dataset(
            mixed_images,
            mixed_targets,
            batch_size,
            flipped_targets=mixed_flipped_targets,
            training=True,
            horizontal_flip=horizontal_flip,
            brightness=brightness,
            contrast=contrast,
            shuffle=shuffle,
            seed=SEED + 20_000 + int(epoch_index),
        ),
        source_count,
        target_count,
    )


def mean_validation_loss(
    model,
    images,
    targets,
    batch_size,
):
    values = []

    for start in range(
        0,
        len(images),
        batch_size,
    ):
        batch_images = (
            images[
                start:start+batch_size
            ].astype(np.float32)
            / 255.0
        )

        batch_targets = targets[
            start:start+batch_size
        ]

        predictions = model(
            batch_images,
            training=False,
        )

        losses = cute_yolo_loss(
            predictions,
            batch_targets,
        )

        values.append(
            float(
                losses["total"].numpy()
            )
        )

    return float(
        np.mean(values)
    )


def sweep_predictions_operating_points(
    predictions,
    targets,
):
    """Sweep confidence/NMS using already-computed detector outputs."""
    results = []

    for nms_iou in NMS_IOU_GRID:
        for confidence in CONFIDENCE_GRID:
            results.append(
                evaluate_predictions(
                    predictions,
                    targets,
                    confidence_threshold=confidence,
                    nms_iou_threshold=nms_iou,
                )
            )

    best = max(
        results,
        key=lambda item: (
            item["f1"],
            item["mean_iou"],
        ),
    )

    return best, results


def sweep_operating_points_gui(
    model,
    images,
    targets,
    batch_size,
    return_debug=False,
):
    predictions = model.predict(
        images.astype(np.float32)
        / 255.0,
        batch_size=batch_size,
        verbose=0,
    )

    best, results = sweep_predictions_operating_points(
        predictions,
        targets,
    )

    if return_debug:
        return (
            best,
            results,
            predictions,
        )

    return best


def fine_tune_model(
    model,
    source_images,
    source_targets,
    source_flipped_targets,
    real_train_images,
    real_train_targets,
    real_train_flipped_targets,
    real_val_images,
    real_val_targets,
    cfg: AdaptConfig,
    best_weights_path: Path,
    logger: Callable[[str], None],
    progress: Callable[[float], None],
):
    """Fine-tune with optional branch-specialized objectives.

    Specialized mode freezes the stem and detector head, then updates:
      Branch A / runtime Core 0 with localization-dominant loss
      Branch B / runtime Core 1 with objectness-dominant loss

    L_loc = BOX_WEIGHT * L_box + IOU_WEIGHT * L_overlap
    L_A   = L_loc + alpha * L_obj
    L_B   = L_obj + alpha * L_loc

    This changes training only; the exported graph is unchanged.
    """

    def set_trainability():
        if not cfg.specialized_finetune:
            for layer in model.layers:
                layer.trainable = True
            return

        for layer in model.layers:
            name = layer.name
            is_stem = (
                name.startswith("stem1_")
                or name.startswith("stem2_")
                or name.startswith("stem3_")
            )
            is_head = name == "head" or name.startswith("head_")
            layer.trainable = not (is_stem or is_head)

    def collect(names):
        result = []
        seen = set()
        for name in names:
            for variable in model.get_layer(name).trainable_variables:
                key = id(variable)
                if key not in seen:
                    seen.add(key)
                    result.append(variable)
        return result

    set_trainability()

    if cfg.specialized_finetune:
        names_a = []
        names_b = []
        for block in range(1, HYBRID_BLOCKS + 1):
            names_a += [f"h{block}_a_conv", f"h{block}_a_bn"]
            names_b += [
                f"h{block}_dw", f"h{block}_dw_bn",
                f"h{block}_pw", f"h{block}_pw_bn",
            ]

        vars_a = collect(names_a)
        vars_b = collect(names_b)

        if not vars_a or not vars_b:
            raise RuntimeError("Could not identify specialized branch variables.")

        ids_a = {id(v) for v in vars_a}
        ids_b = {id(v) for v in vars_b}
        ids_model = {id(v) for v in model.trainable_variables}
        if ids_a & ids_b:
            raise RuntimeError("Branch variable groups overlap.")
        if ids_a | ids_b != ids_model:
            raise RuntimeError(
                "Trainable-variable grouping mismatch after freezing stem/head."
            )

        optimizer_a = tf.keras.optimizers.Adam(
            learning_rate=cfg.learning_rate,
        )
        optimizer_b = tf.keras.optimizers.Adam(
            learning_rate=cfg.learning_rate,
        )

        log_line(logger, "Specialized fine-tuning ENABLED")
        log_line(
            logger,
            (
                "Frozen: Stem 1/2/3 + head | "
                f"cross-loss={cfg.cross_loss_weight:.3f}"
            ),
        )
        log_line(
            logger,
            (
                "Branch A/Core 0 localization-dominant: "
                f"{sum(int(np.prod(v.shape)) for v in vars_a):,} params"
            ),
        )
        log_line(
            logger,
            (
                "Branch B/Core 1 objectness-dominant: "
                f"{sum(int(np.prod(v.shape)) for v in vars_b):,} params"
            ),
        )

        @tf.function
        def train_step(images, targets):
            with tf.GradientTape(persistent=True) as tape:
                predictions = model(images, training=True)
                losses = cute_yolo_loss(predictions, targets)

                localization = (
                    BOX_WEIGHT * losses["box"]
                    + IOU_WEIGHT * losses["iou"]
                )
                objective_a = (
                    localization
                    + cfg.cross_loss_weight * losses["objectness"]
                )
                objective_b = (
                    losses["objectness"]
                    + cfg.cross_loss_weight * localization
                )

            grads_a = tape.gradient(objective_a, vars_a)
            grads_b = tape.gradient(objective_b, vars_b)
            del tape

            pairs_a = []
            for grad, var in zip(grads_a, vars_a):
                if grad is not None:
                    pairs_a.append((
                        grad + cfg.weight_decay * tf.cast(var, grad.dtype),
                        var,
                    ))
            pairs_b = []
            for grad, var in zip(grads_b, vars_b):
                if grad is not None:
                    pairs_b.append((
                        grad + cfg.weight_decay * tf.cast(var, grad.dtype),
                        var,
                    ))

            ga, va = zip(*pairs_a)
            gb, vb = zip(*pairs_b)
            ga, _ = tf.clip_by_global_norm(ga, 5.0)
            gb, _ = tf.clip_by_global_norm(gb, 5.0)
            optimizer_a.apply_gradients(zip(ga, va))
            optimizer_b.apply_gradients(zip(gb, vb))

            return (
                losses["total"], losses["objectness"], losses["box"],
                losses["iou"], losses["footprint_negative"],
                losses["halo_negative"], objective_a, objective_b,
            )

        def current_lr():
            return float(tf.keras.backend.get_value(optimizer_a.learning_rate))

        def set_lr(value):
            optimizer_a.learning_rate.assign(value)
            optimizer_b.learning_rate.assign(value)

    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.learning_rate,
        )
        log_line(logger, "Specialized fine-tuning disabled: standard adaptation.")

        @tf.function
        def train_step(images, targets):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                losses = cute_yolo_loss(predictions, targets)

            gradients = tape.gradient(losses["total"], model.trainable_variables)
            pairs = []
            for grad, var in zip(gradients, model.trainable_variables):
                if grad is not None:
                    pairs.append((
                        grad + cfg.weight_decay * tf.cast(var, grad.dtype),
                        var,
                    ))
            grads, variables = zip(*pairs)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, variables))
            zero = losses["total"] * 0.0
            return (
                losses["total"], losses["objectness"], losses["box"],
                losses["iou"], losses["footprint_negative"],
                losses["halo_negative"], zero, zero,
            )

        def current_lr():
            return float(tf.keras.backend.get_value(optimizer.learning_rate))

        def set_lr(value):
            optimizer.learning_rate.assign(value)

    @tf.function
    def validation_step(images, targets):
        predictions = model(images, training=False)
        losses = cute_yolo_loss(predictions, targets)
        localization = BOX_WEIGHT * losses["box"] + IOU_WEIGHT * losses["iou"]

        if cfg.specialized_finetune:
            objective_a = localization + cfg.cross_loss_weight * losses["objectness"]
            objective_b = losses["objectness"] + cfg.cross_loss_weight * localization
        else:
            objective_a = losses["total"] * 0.0
            objective_b = losses["total"] * 0.0

        return (
            losses["total"], losses["objectness"], losses["box"],
            losses["iou"], losses["footprint_negative"],
            losses["halo_negative"], objective_a, objective_b,
        )

    def run_epoch(dataset, training):
        totals = np.zeros(8, dtype=np.float64)
        batches = 0
        for batch_images, batch_targets in dataset:
            values = (
                train_step(batch_images, batch_targets)
                if training
                else validation_step(batch_images, batch_targets)
            )
            totals += np.asarray([float(x.numpy()) for x in values])
            batches += 1
        return totals / max(1, batches)

    real_val_ds = make_tf_dataset(
        real_val_images,
        real_val_targets,
        cfg.batch_size,
        training=False,
    )

    history = {
        "train": [],
        "target_validation": [],
        "learning_rate": [],
        "source_examples_per_epoch": [],
        "target_examples_per_epoch": [],
        "specialized_finetune": bool(cfg.specialized_finetune),
        "cross_loss_weight": float(cfg.cross_loss_weight),
    }

    best_validation = float("inf")
    bad_epochs = 0
    plateau_bad_epochs = 0
    completed_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        started = time.time()
        mixed_ds, source_count, target_count = make_mixed_epoch_dataset(
            source_images, source_targets, source_flipped_targets,
            real_train_images, real_train_targets, real_train_flipped_targets,
            cfg.source_ratio, cfg.target_ratio,
            cfg.target_passes_per_epoch, cfg.batch_size, epoch,
            horizontal_flip=cfg.augment_horizontal_flip,
            brightness=cfg.augment_brightness,
            contrast=cfg.augment_contrast,
            shuffle=cfg.shuffle_each_epoch,
        )

        train_metrics = run_epoch(mixed_ds, training=True)
        val_metrics = run_epoch(real_val_ds, training=False)
        learning_rate = current_lr()

        history["train"].append(train_metrics.tolist())
        history["target_validation"].append(val_metrics.tolist())
        history["learning_rate"].append(learning_rate)
        history["source_examples_per_epoch"].append(int(source_count))
        history["target_examples_per_epoch"].append(int(target_count))
        completed_epochs = epoch

        if val_metrics[0] < best_validation:
            best_validation = float(val_metrics[0])
            bad_epochs = 0
            plateau_bad_epochs = 0
            model.save_weights(best_weights_path)
        else:
            bad_epochs += 1
            plateau_bad_epochs += 1

        branch_text = ""
        if cfg.specialized_finetune:
            branch_text = f"  A={val_metrics[6]:.4f}  B={val_metrics[7]:.4f}"

        log_line(
            logger,
            (
                f"Epoch {epoch:02d}/{cfg.epochs}  "
                f"mixed={train_metrics[0]:.4f}  "
                f"real-val={val_metrics[0]:.4f}  "
                f"src/real={source_count}/{target_count}  "
                f"fp={val_metrics[4]:.4f}  halo={val_metrics[5]:.4f}"
                f"{branch_text}  lr={learning_rate:.2e}  "
                f"{time.time() - started:.1f}s"
            ),
        )

        progress(20.0 + 45.0 * epoch / max(1, cfg.epochs))

        if plateau_bad_epochs >= cfg.plateau_patience:
            old_lr = current_lr()
            new_lr = max(cfg.min_learning_rate, old_lr * cfg.plateau_factor)
            if new_lr < old_lr:
                set_lr(new_lr)
                log_line(logger, f"Learning rate -> {new_lr:.3e}")
            plateau_bad_epochs = 0

        if bad_epochs >= cfg.early_stop_patience:
            log_line(logger, "Early stopping: real validation loss stopped improving.")
            break

    model.load_weights(best_weights_path)
    return history, best_validation, completed_epochs

def make_calibration_pool(
    source_images,
    real_train_images,
    cfg: AdaptConfig,
):
    rng = np.random.default_rng(
        SEED + 50_000
    )

    target_count = int(
        round(
            cfg.calibration_samples
            * cfg.calibration_target_fraction
        )
    )

    source_count = (
        cfg.calibration_samples
        - target_count
    )

    target_indices = rng.choice(
        len(real_train_images),
        size=target_count,
        replace=(
            target_count
            > len(real_train_images)
        ),
    )

    source_indices = rng.choice(
        len(source_images),
        size=source_count,
        replace=(
            source_count
            > len(source_images)
        ),
    )

    images = np.concatenate(
        [
            real_train_images[
                target_indices
            ],
            source_images[
                source_indices
            ],
        ],
        axis=0,
    )

    return images[
        rng.permutation(
            len(images)
        )
    ]


def convert_to_int8(
    model,
    calibration_images,
    tflite_path: Path,
    logger: Callable[[str], None],
):
    def representative_dataset():
        for index in range(
            len(calibration_images)
        ):
            image = (
                calibration_images[
                    index:index+1
                ].astype(np.float32)
                / 255.0
            )

            yield [image]

    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = (
        representative_dataset
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = (
        tf.int8
    )

    converter.inference_output_type = (
        tf.int8
    )

    tflite_model = (
        converter.convert()
    )

    tflite_path.write_bytes(
        tflite_model
    )

    interpreter = tf.lite.Interpreter(
        model_path=str(
            tflite_path
        )
    )

    interpreter.allocate_tensors()

    input_detail = (
        interpreter
        .get_input_details()[0]
    )

    output_detail = (
        interpreter
        .get_output_details()[0]
    )

    if (
        input_detail["dtype"]
        != np.int8
        or output_detail["dtype"]
        != np.int8
    ):
        raise RuntimeError(
            "Strict INT8 conversion failed."
        )

    log_line(
        logger,
        (
            "INT8 model: "
            f"{tflite_path.name} "
            f"({tflite_path.stat().st_size:,} bytes)"
        ),
    )

    return (
        interpreter,
        input_detail,
        output_detail,
    )


def validate_int8(
    model,
    interpreter,
    input_detail,
    output_detail,
    images,
):
    def quantize_input(
        image_float,
    ):
        scale, zero_point = (
            input_detail[
                "quantization"
            ]
        )

        q = np.rint(
            image_float
            / scale
            + zero_point
        )

        return np.clip(
            q,
            -128,
            127,
        ).astype(
            np.int8
        )

    def dequantize_output(
        q_output,
    ):
        scale, zero_point = (
            output_detail[
                "quantization"
            ]
        )

        return scale * (
            q_output.astype(
                np.float32
            )
            - zero_point
        )

    count = min(
        16,
        len(images),
    )

    maes = []

    saturation_low = 0
    saturation_high = 0

    for index in range(count):
        image = (
            images[
                index:index+1
            ].astype(np.float32)
            / 255.0
        )

        float_output = model(
            image,
            training=False,
        ).numpy()

        q_input = quantize_input(
            image
        )

        interpreter.set_tensor(
            input_detail["index"],
            q_input,
        )

        interpreter.invoke()

        q_output = interpreter.get_tensor(
            output_detail["index"]
        )

        int8_output = dequantize_output(
            q_output
        )

        maes.append(
            float(
                np.mean(
                    np.abs(
                        float_output
                        - int8_output
                    )
                )
            )
        )

        saturation_low += int(
            np.sum(
                q_output == -128
            )
        )

        saturation_high += int(
            np.sum(
                q_output == 127
            )
        )

    return {
        "validation_images":
            count,
        "mean_absolute_output_logit_difference":
            float(
                np.mean(maes)
            )
            if maes
            else None,
        "saturation_minus_128":
            saturation_low,
        "saturation_plus_127":
            saturation_high,
    }




def predict_int8_dequantized(
    interpreter,
    input_detail,
    output_detail,
    images,
):
    """Run strict-INT8 TFLite on all images and return dequantized logits."""
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]

    if input_scale == 0 or output_scale == 0:
        raise RuntimeError("Invalid TFLite quantization scale.")

    predictions = []

    for index in range(len(images)):
        image = (
            images[index:index+1].astype(np.float32)
            / 255.0
        )

        q_input = np.rint(
            image / input_scale + input_zero_point
        )
        q_input = np.clip(
            q_input,
            -128,
            127,
        ).astype(np.int8)

        interpreter.set_tensor(
            input_detail["index"],
            q_input,
        )
        interpreter.invoke()

        q_output = interpreter.get_tensor(
            output_detail["index"]
        )

        output = output_scale * (
            q_output.astype(np.float32)
            - output_zero_point
        )
        predictions.append(output[0])

    if not predictions:
        return np.empty(
            (0, GRID_SIZE, GRID_SIZE, OUTPUT_CHANNELS),
            dtype=np.float32,
        )

    return np.stack(predictions, axis=0)


def sweep_int8_operating_points(
    interpreter,
    input_detail,
    output_detail,
    images,
    targets,
):
    """Select deployment confidence/NMS from the actual INT8 TFLite model."""
    predictions = predict_int8_dequantized(
        interpreter,
        input_detail,
        output_detail,
        images,
    )

    best, results = sweep_predictions_operating_points(
        predictions,
        targets,
    )

    return best, results, predictions



def export_noodleq_artifacts(
    interpreter,
    input_detail,
    output_detail,
):
    def get_quantization(
        detail,
    ):
        q = detail[
            "quantization_parameters"
        ]

        scales = np.asarray(
            q["scales"],
            dtype=np.float64,
        )

        zero_points = np.asarray(
            q["zero_points"],
            dtype=np.int64,
        )

        return (
            scales,
            zero_points,
            int(
                q[
                    "quantized_dimension"
                ]
            ),
        )


    def scalar_quant(
        detail,
    ):
        scales, zps, _ = (
            get_quantization(
                detail
            )
        )

        if (
            scales.size != 1
            or zps.size != 1
        ):
            raise RuntimeError(
                "Expected scalar activation quantization for "
                f"{detail['name']}"
            )

        return (
            float(
                scales[0]
            ),
            int(
                zps[0]
            ),
        )


    tensor_details = {
        int(
            detail["index"]
        ): detail
        for detail
        in interpreter.get_tensor_details()
    }

    ops = (
        interpreter
        ._get_ops_details()
    )

    op_counts = {}

    for op in ops:
        op_counts[
            op["op_name"]
        ] = (
            op_counts.get(
                op["op_name"],
                0,
            )
            + 1
        )

    print(
        "TFLite operator counts:"
    )

    for name in sorted(
        op_counts
    ):
        print(
            f"  {name:24s}",
            op_counts[name],
        )


    conv_ops = [
        op
        for op in ops
        if op["op_name"]
        == "CONV_2D"
    ]

    dw_ops = [
        op
        for op in ops
        if op["op_name"]
        == "DEPTHWISE_CONV_2D"
    ]

    concat_ops = [
        op
        for op in ops
        if op["op_name"]
        == "CONCATENATION"
    ]


    print(
        "\nExpected structural counts:"
    )

    print(
        "  CONV_2D:",
        len(conv_ops),
        "(expected 14)",
    )

    print(
        "  DEPTHWISE_CONV_2D:",
        len(dw_ops),
        "(expected 5)",
    )

    print(
        "  CONCATENATION:",
        len(concat_ops),
        "(expected 5)",
    )


    if len(conv_ops) != 14:
        raise RuntimeError(
            "Unexpected CONV_2D count. "
            "Inspect TFLite operator list before deployment export."
        )

    if len(dw_ops) != 5:
        raise RuntimeError(
            "Unexpected DEPTHWISE_CONV_2D count."
        )

    if len(concat_ops) != 5:
        raise RuntimeError(
            "Unexpected CONCATENATION count."
        )


    concat_report = []
    all_concat_quantization_compatible = True


    for concat_number, op in enumerate(
        concat_ops,
        start=1,
    ):
        input_indices = [
            int(i)
            for i in op["inputs"]
            if int(i) >= 0
        ]

        output_index = int(
            op["outputs"][0]
        )

        output_detail_q = (
            tensor_details[
                output_index
            ]
        )

        (
            out_scale,
            out_zp,
        ) = scalar_quant(
            output_detail_q
        )

        input_entries = []
        compatible = True

        for input_index in input_indices:
            detail = (
                tensor_details[
                    input_index
                ]
            )

            (
                scale,
                zp,
            ) = scalar_quant(
                detail
            )

            same = (
                np.isclose(
                    scale,
                    out_scale,
                    rtol=1e-7,
                    atol=1e-12,
                )
                and zp == out_zp
            )

            compatible &= bool(
                same
            )

            input_entries.append({
                "tensor_index":
                    input_index,
                "tensor_name":
                    detail["name"],
                "shape":
                    list(
                        map(
                            int,
                            detail["shape"],
                        )
                    ),
                "scale":
                    scale,
                "zero_point":
                    zp,
                "matches_concat_output":
                    bool(
                        same
                    ),
            })

        all_concat_quantization_compatible &= (
            compatible
        )

        entry = {
            "concat_number":
                concat_number,
            "op_index":
                int(
                    op["index"]
                ),
            "inputs":
                input_entries,
            "output": {
                "tensor_index":
                    output_index,
                "tensor_name":
                    output_detail_q[
                        "name"
                    ],
                "shape":
                    list(
                        map(
                            int,
                            output_detail_q[
                                "shape"
                            ],
                        )
                    ),
                "scale":
                    out_scale,
                "zero_point":
                    out_zp,
            },
            "noodle_concat_directly_compatible":
                bool(
                    compatible
                ),
        }

        concat_report.append(
            entry
        )

        print(
            f"\nConcat {concat_number}: "
            f"direct Noodle compatibility = "
            f"{compatible}"
        )

        for branch_number, inp in enumerate(
            input_entries,
            start=1,
        ):
            print(
                f"  input {branch_number}: "
                f"scale={inp['scale']:.9g} "
                f"zp={inp['zero_point']} "
                f"shape={inp['shape']} "
                f"match={inp['matches_concat_output']}"
            )

        print(
            "  output : "
            f"scale={out_scale:.9g} "
            f"zp={out_zp} "
            f"shape={entry['output']['shape']}"
        )


    print(
        "\nAll five concat boundaries directly "
        "compatible with current Noodle concat:",
        all_concat_quantization_compatible,
    )


    GRAPH_REPORT_PATH = (
        INT8_EXPORT_DIR
        / "hybrid_quantization_report.json"
    )

    GRAPH_REPORT_PATH.write_text(
        json.dumps(
            {
                "architecture":
                    f"Cute-YOLO fixed dual-core Hybrid 8+24 | {MODEL_LABEL}",
                "architecture_id":
                    ARCHITECTURE_ID,
                "label":
                    MODEL_LABEL,
                "experiment_tag":
                    EXPERIMENT_TAG,
                "hybrid_width":
                    HYBRID_WIDTH,
                "hybrid_normal_out":
                    HYBRID_NORMAL_OUT,
                "hybrid_efficient_out":
                    HYBRID_EFFICIENT_OUT,
                "operator_counts":
                    op_counts,
                "concat_report":
                    concat_report,
                "all_concat_directly_noodle_compatible":
                    bool(
                        all_concat_quantization_compatible
                    ),
            },
            indent=2,
        )
    )

    print(
        "Saved:",
        GRAPH_REPORT_PATH,
    )

    # ============================================================
    # Export a model_weights_int8.h compatible with the SAME
    # dual-core NoodleQ firmware used by the fixed Cute-YOLO firmware.
    # ============================================================

    def quantize_multiplier_noodle(real_multiplier):
        real_multiplier = float(real_multiplier)

        if real_multiplier < 0.0 or not math.isfinite(real_multiplier):
            raise ValueError(f'Invalid multiplier: {real_multiplier}')

        if real_multiplier == 0.0:
            return np.int32(0), np.int32(0)

        q, shift = math.frexp(real_multiplier)
        q_fixed = int(math.floor(q * (1 << 31) + 0.5))

        if q_fixed == (1 << 31):
            q_fixed >>= 1
            shift += 1

        if shift < -31:
            return np.int32(0), np.int32(0)

        if shift > 30:
            raise OverflowError(
                f'Requantization shift {shift} is outside Noodle range.'
            )

        return np.int32(q_fixed), np.int32(shift)


    def op_tensor_indices(op):
        return [int(i) for i in op['inputs'] if int(i) >= 0]


    def output_shape(op):
        detail = tensor_details[int(op['outputs'][0])]
        return tuple(int(x) for x in detail['shape'])


    def activation_quant(detail):
        return scalar_quant(detail)


    def extract_quantized_op(op, depthwise=False):
        inputs = op_tensor_indices(op)
        if len(inputs) < 3:
            raise RuntimeError(f"{op['op_name']} does not expose input/weight/bias.")

        activation_detail = tensor_details[inputs[0]]
        weight_detail = tensor_details[inputs[1]]
        bias_detail = tensor_details[inputs[2]]
        out_detail = tensor_details[int(op['outputs'][0])]

        input_scale, input_zp = activation_quant(activation_detail)
        output_scale, output_zp = activation_quant(out_detail)

        weight_scales, weight_zps, qdim = get_quantization(weight_detail)
        q_weight = interpreter.get_tensor(inputs[1])
        q_bias = interpreter.get_tensor(inputs[2])

        if q_weight.dtype != np.int8:
            raise RuntimeError(f"Expected int8 weight for {op['op_name']}.")
        if q_bias.dtype != np.int32:
            raise RuntimeError(f"Expected int32 bias for {op['op_name']}.")
        if np.any(weight_zps != 0):
            raise RuntimeError('Noodle export requires zero-centered INT8 weights.')

        output_channels = int(q_bias.size)

        if weight_scales.size == 1:
            weight_scales = np.repeat(weight_scales, output_channels)

        if weight_scales.size != output_channels:
            raise RuntimeError(
                f'Weight scale count {weight_scales.size} != Cout {output_channels}'
            )

        if depthwise:
            # TFLite depthwise: [1, Ky, Kx, Cout] for depth_multiplier=1.
            if q_weight.ndim != 4 or q_weight.shape[0] != 1:
                raise RuntimeError(
                    f'Unexpected TFLite depthwise weight shape: {q_weight.shape}'
                )
            noodle_weight = q_weight[0].transpose(2, 0, 1).copy()
        else:
            # TFLite Conv2D: [Cout, Ky, Kx, Cin]
            if q_weight.ndim != 4:
                raise RuntimeError(
                    f'Unexpected TFLite Conv2D weight shape: {q_weight.shape}'
                )
            noodle_weight = np.transpose(q_weight, (0, 3, 1, 2)).copy()

        real_multipliers = input_scale * weight_scales / output_scale

        multipliers = np.empty(output_channels, dtype=np.int32)
        shifts = np.empty(output_channels, dtype=np.int32)

        for channel, rm in enumerate(real_multipliers):
            multipliers[channel], shifts[channel] = quantize_multiplier_noodle(rm)

        bias_scales, _, _ = get_quantization(bias_detail)
        expected_bias_scales = input_scale * weight_scales

        if bias_scales.size == 1 and output_channels > 1:
            bias_scales = np.repeat(bias_scales, output_channels)

        if bias_scales.size == output_channels:
            bias_scale_error = float(
                np.max(
                    np.abs(bias_scales - expected_bias_scales)
                    / np.maximum(np.abs(expected_bias_scales), 1e-30)
                )
            )
        else:
            bias_scale_error = None

        return {
            'w': noodle_weight.astype(np.int8),
            'b': q_bias.astype(np.int32),
            'm': multipliers,
            's': shifts,
            'input_scale': input_scale,
            'input_zp': input_zp,
            'output_scale': output_scale,
            'output_zp': output_zp,
            'output_shape': output_shape(op),
            'bias_scale_relative_error': bias_scale_error,
            'op_index': int(op['index']),
            'op_name': op['op_name'],
            'weight_quantized_dimension': qdim,
        }


    # ------------------------------------------------------------
    # Identify the fixed graph by operator type + output shape.
    # ------------------------------------------------------------
    ops = interpreter._get_ops_details()

    conv_ops_all = [op for op in ops if op['op_name'] == 'CONV_2D']
    dw_ops_all = [op for op in ops if op['op_name'] == 'DEPTHWISE_CONV_2D']
    concat_ops_all = [op for op in ops if op['op_name'] == 'CONCATENATION']

    if len(conv_ops_all) != 14:
        raise RuntimeError(f'Expected 14 CONV_2D ops, found {len(conv_ops_all)}.')
    if len(dw_ops_all) != 5:
        raise RuntimeError(f'Expected 5 DEPTHWISE_CONV_2D ops, found {len(dw_ops_all)}.')
    if len(concat_ops_all) != 5:
        raise RuntimeError(f'Expected 5 CONCATENATION ops, found {len(concat_ops_all)}.')


    def convs_with_output(h, w, c):
        result = []
        for op in conv_ops_all:
            shape = output_shape(op)
            if len(shape) == 4 and shape[1:] == (h, w, c):
                result.append(op)
        return sorted(result, key=lambda op: int(op['index']))


    stem1_op = convs_with_output(64, 64, 8)
    stem2_op = convs_with_output(32, 32, 16)
    stem3_candidates = convs_with_output(16, 16, 32)
    hybrid_a_ops = convs_with_output(16, 16, 8)
    hybrid_pw_ops = convs_with_output(16, 16, 24)
    head_ops = convs_with_output(16, 16, 5)

    if len(stem1_op) != 1 or len(stem2_op) != 1:
        raise RuntimeError('Could not uniquely identify Stem 1/2.')
    if len(stem3_candidates) != 1:
        raise RuntimeError(
            'Could not uniquely identify Stem 3 CONV_2D. '
            f'Candidates={len(stem3_candidates)}'
        )
    if len(hybrid_a_ops) != 5 or len(hybrid_pw_ops) != 5:
        raise RuntimeError('Could not identify all five hybrid Conv/PW operators.')
    if len(head_ops) != 1:
        raise RuntimeError('Could not uniquely identify the detector head.')

    stem_ops = [stem1_op[0], stem2_op[0], stem3_candidates[0]]
    hybrid_a_ops = sorted(hybrid_a_ops, key=lambda op: int(op['index']))
    hybrid_pw_ops = sorted(hybrid_pw_ops, key=lambda op: int(op['index']))
    dw_ops_all = sorted(dw_ops_all, key=lambda op: int(op['index']))
    concat_ops_all = sorted(concat_ops_all, key=lambda op: int(op['index']))
    head_op = head_ops[0]


    # ------------------------------------------------------------
    # Extract quantized arrays.
    # ------------------------------------------------------------
    stem_q = [extract_quantized_op(op) for op in stem_ops]
    hybrid_a_q = [extract_quantized_op(op) for op in hybrid_a_ops]
    hybrid_dw_q = [extract_quantized_op(op, depthwise=True) for op in dw_ops_all]
    hybrid_pw_q = [extract_quantized_op(op) for op in hybrid_pw_ops]
    head_q = extract_quantized_op(head_op)

    # Check concat compatibility and collect block output quantization.
    concat_quant = []
    for block_index, op in enumerate(concat_ops_all, start=1):
        out_detail = tensor_details[int(op['outputs'][0])]
        out_scale, out_zp = scalar_quant(out_detail)

        inputs = [int(i) for i in op['inputs'] if int(i) >= 0]
        for tensor_index in inputs:
            scale, zp = scalar_quant(tensor_details[tensor_index])
            if not (
                np.isclose(scale, out_scale, rtol=1e-7, atol=1e-12)
                and zp == out_zp
            ):
                raise RuntimeError(
                    f'Concat {block_index} is not directly compatible with '
                    'the current Noodle concat quantization.'
                )

        concat_quant.append((out_scale, out_zp))


    # ------------------------------------------------------------
    # Verify boundary quantization expected by the firmware.
    # ------------------------------------------------------------
    model_input_scale, model_input_zero_point = scalar_quant(input_detail)
    model_output_scale, model_output_zero_point = scalar_quant(output_detail)

    if not (
        np.isclose(stem_q[0]['input_scale'], model_input_scale, rtol=1e-6)
        and stem_q[0]['input_zp'] == model_input_zero_point
    ):
        raise RuntimeError('Stem1 input quantization != model input quantization.')

    for i in range(2):
        if not (
            np.isclose(stem_q[i]['output_scale'], stem_q[i+1]['input_scale'], rtol=1e-6)
            and stem_q[i]['output_zp'] == stem_q[i+1]['input_zp']
        ):
            raise RuntimeError(f'Stem quantization mismatch at boundary {i+1}.')

    if not (
        np.isclose(head_q['output_scale'], model_output_scale, rtol=1e-6)
        and head_q['output_zp'] == model_output_zero_point
    ):
        raise RuntimeError('Head output quantization != model output quantization.')


    def split_output_channels(param, count_a):
        return {
            'w_a': param['w'][:count_a].copy(),
            'b_a': param['b'][:count_a].copy(),
            'm_a': param['m'][:count_a].copy(),
            's_a': param['s'][:count_a].copy(),
            'w_b': param['w'][count_a:].copy(),
            'b_b': param['b'][count_a:].copy(),
            'm_b': param['m'][count_a:].copy(),
            's_b': param['s'][count_a:].copy(),
        }


    stem_split = [
        split_output_channels(stem_q[0], 4),
        split_output_channels(stem_q[1], 8),
        split_output_channels(stem_q[2], 16),
    ]


    def format_cpp_float(value):
        return f'{float(value):.9e}f'


    def write_cpp_array(handle, ctype, name, array, values_per_line=16):
        flat = np.asarray(array).reshape(-1)
        handle.write(
            f'static const {ctype} {name}[{flat.size}] PROGMEM = {{\n'
        )
        for start in range(0, flat.size, values_per_line):
            chunk = flat[start:start + values_per_line]
            line = ', '.join(str(int(v)) for v in chunk)
            if start + values_per_line < flat.size:
                line += ','
            handle.write('  ' + line + '\n')
        handle.write('};\n\n')


    INT8_HEADER_PATH = INT8_EXPORT_DIR / 'model_weights_int8.h'

    with INT8_HEADER_PATH.open('w', encoding='utf-8') as header:
        header.write(
            '#ifndef CUTE_YOLO_FIXED_HYBRID_MODEL_WEIGHTS_INT8_H\n'
            '#define CUTE_YOLO_FIXED_HYBRID_MODEL_WEIGHTS_INT8_H\n\n'
            '#include <Arduino.h>\n'
            '#include <stdint.h>\n\n'
            '// Cute-YOLO fixed 8+24 Hybrid full-INT8 parameters.\n'
            f"// Task: {MODEL_LABEL} single-class detection.\\n"
            '// Topology matches the fixed dual-core ESP32-S3 firmware.\n'
            '// Noodle Conv layout: [Cout][Cin][Ky][Kx].\n'
            '// Noodle depthwise layout: [Cout][Ky][Kx], depth_multiplier=1.\n\n'
            '#define CUTE_YOLO_FIXED_8_24 1\n'
            '#define CUTE_YOLO_V8_REBALANCED_CNN_DW_PW 1 // legacy firmware alias\n'
            '#define CUTE_YOLO_FIXED_HYBRID_8_24 1\n'
            '#define CUTE_YOLO_INPUT_W 128\n'
            '#define CUTE_YOLO_INPUT_H 128\n'
            '#define CUTE_YOLO_INPUT_C 1\n'
            '#define CUTE_YOLO_GRID_W 16\n'
            '#define CUTE_YOLO_OUTPUT_C 5\n'
            f'#define CUTE_YOLO_CONFIDENCE_THRESHOLD {SELECTED_CONFIDENCE_THRESHOLD:.3f}f\n\n'
            f'#define CUTE_YOLO_INPUT_SCALE {format_cpp_float(model_input_scale)}\n'
            f'#define CUTE_YOLO_INPUT_ZERO_POINT {model_input_zero_point}\n'
            f'#define CUTE_YOLO_STEM1_OUTPUT_SCALE {format_cpp_float(stem_q[0]["output_scale"])}\n'
            f'#define CUTE_YOLO_STEM1_OUTPUT_ZERO_POINT {stem_q[0]["output_zp"]}\n'
            f'#define CUTE_YOLO_STEM2_OUTPUT_SCALE {format_cpp_float(stem_q[1]["output_scale"])}\n'
            f'#define CUTE_YOLO_STEM2_OUTPUT_ZERO_POINT {stem_q[1]["output_zp"]}\n'
            f'#define CUTE_YOLO_STEM3_OUTPUT_SCALE {format_cpp_float(stem_q[2]["output_scale"])}\n'
            f'#define CUTE_YOLO_STEM3_OUTPUT_ZERO_POINT {stem_q[2]["output_zp"]}\n'
        )

        for i in range(5):
            block = i + 1
            header.write(
                f'#define CUTE_YOLO_H{block}_DW_OUTPUT_SCALE '
                f'{format_cpp_float(hybrid_dw_q[i]["output_scale"])}\n'
                f'#define CUTE_YOLO_H{block}_DW_OUTPUT_ZERO_POINT '
                f'{hybrid_dw_q[i]["output_zp"]}\n'
                f'#define CUTE_YOLO_H{block}_OUTPUT_SCALE '
                f'{format_cpp_float(concat_quant[i][0])}\n'
                f'#define CUTE_YOLO_H{block}_OUTPUT_ZERO_POINT '
                f'{concat_quant[i][1]}\n'
            )

        header.write(
            f'#define CUTE_YOLO_OUTPUT_SCALE {format_cpp_float(model_output_scale)}\n'
            f'#define CUTE_YOLO_OUTPUT_ZERO_POINT {model_output_zero_point}\n\n'
            '#define CUTE_YOLO_STEM1_BRANCH_A_OUT 4\n'
            '#define CUTE_YOLO_STEM1_BRANCH_B_OUT 4\n'
            '#define CUTE_YOLO_STEM2_BRANCH_A_OUT 8\n'
            '#define CUTE_YOLO_STEM2_BRANCH_B_OUT 8\n'
            '#define CUTE_YOLO_STEM3_BRANCH_A_OUT 16\n'
            '#define CUTE_YOLO_STEM3_BRANCH_B_OUT 16\n'
            '#define CUTE_YOLO_HYBRID_NORMAL_OUT 8\n'
            '#define CUTE_YOLO_HYBRID_EFFICIENT_OUT 24\n'
            '#define CUTE_YOLO_HYBRID_CHANNELS 32\n'
            '#define CUTE_YOLO_HYBRID_BLOCKS 5\n\n'
        )

        # Split stem arrays.
        for stage, sp in enumerate(stem_split, start=1):
            prefix = f'{stage:02d}'
            for suffix in ('a', 'b'):
                write_cpp_array(header, 'int8_t', f'w{prefix}{suffix}', sp[f'w_{suffix}'])
                write_cpp_array(header, 'int32_t', f'b{prefix}{suffix}', sp[f'b_{suffix}'], 8)
                write_cpp_array(header, 'int32_t', f'm{prefix}{suffix}', sp[f'm_{suffix}'], 8)
                write_cpp_array(header, 'int32_t', f's{prefix}{suffix}', sp[f's_{suffix}'], 8)

        # Five hybrid blocks.
        for i in range(5):
            block = i + 1

            for label, param in (
                ('a', hybrid_a_q[i]),
                ('dw', hybrid_dw_q[i]),
                ('pw', hybrid_pw_q[i]),
            ):
                write_cpp_array(header, 'int8_t', f'w_h{block}{label}', param['w'])
                write_cpp_array(header, 'int32_t', f'b_h{block}{label}', param['b'], 8)
                write_cpp_array(header, 'int32_t', f'm_h{block}{label}', param['m'], 8)
                write_cpp_array(header, 'int32_t', f's_h{block}{label}', param['s'], 8)

        write_cpp_array(header, 'int8_t', 'w_head', head_q['w'])
        write_cpp_array(header, 'int32_t', 'b_head', head_q['b'], 8)
        write_cpp_array(header, 'int32_t', 'm_head', head_q['m'], 8)
        write_cpp_array(header, 'int32_t', 's_head', head_q['s'], 8)

        header.write('#endif\n')


    # Save a machine-readable deployment manifest.
    deployment_manifest = {
        'architecture_id': ARCHITECTURE_ID,
        'task': 'single_class_detection',
        'domain_adaptation': bool(EXPORT_DOMAIN_ADAPTATION),
        'source_ratio': float(SOURCE_RATIO),
        'target_ratio': float(TARGET_RATIO),
        'input': [1, 128, 128],
        'output_nhwc': [16, 16, 5],
        'output_firmware_chw': [5, 16, 16],
        'hybrid_normal_out': HYBRID_NORMAL_OUT,
        'hybrid_efficient_out': HYBRID_EFFICIENT_OUT,
        'hybrid_blocks': HYBRID_BLOCKS,
        'trainable_parameter_count': int(trainable_parameter_count),
        'conv_macs': int(cute_yolo_macs),
        'label': MODEL_LABEL,
        'confidence_threshold': float(SELECTED_CONFIDENCE_THRESHOLD),
        'nms_iou_threshold': float(SELECTED_NMS_IOU_THRESHOLD),
        'max_detections': int(RUNTIME_MAX_DETECTIONS),
        'min_box_w': float(MIN_BOX_W),
        'min_box_h': float(MIN_BOX_H),
        'input_scale': float(model_input_scale),
        'input_zero_point': int(model_input_zero_point),
        'output_scale': float(model_output_scale),
        'output_zero_point': int(model_output_zero_point),
        'concat_directly_noodle_compatible': bool(
            all_concat_quantization_compatible
        ),
    }

    MANIFEST_PATH.write_text(json.dumps(deployment_manifest, indent=2))

    print('Created:', INT8_HEADER_PATH)
    print('Created:', MANIFEST_PATH)
    print('This header is compatible with the current dual-core 8+24 firmware.')
    return INT8_EXPORT_DIR / 'model_weights_int8.h', MANIFEST_PATH





BN_MOMENTUM = 0.9
BN_EPSILON = 1e-5


def conv_bn_relu(x, filters, kernel_size, strides=1, name=None):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=True,
        name=None if name is None else f"{name}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        momentum=BN_MOMENTUM,
        epsilon=BN_EPSILON,
        name=None if name is None else f"{name}_bn",
    )(x)
    return tf.keras.layers.ReLU(
        name=None if name is None else f"{name}_relu"
    )(x)


def hybrid_block(x, block_index):
    # Branch A: conventional spatial + cross-channel convolution.
    a = conv_bn_relu(
        x,
        HYBRID_NORMAL_OUT,
        3,
        strides=1,
        name=f"h{block_index}_a",
    )

    # Branch B: depthwise spatial filtering followed by pointwise mixing.
    b = tf.keras.layers.DepthwiseConv2D(
        3,
        strides=1,
        padding="same",
        depth_multiplier=1,
        use_bias=True,
        name=f"h{block_index}_dw",
    )(x)
    b = tf.keras.layers.BatchNormalization(
        momentum=BN_MOMENTUM,
        epsilon=BN_EPSILON,
        name=f"h{block_index}_dw_bn",
    )(b)
    b = tf.keras.layers.ReLU(name=f"h{block_index}_dw_relu")(b)

    b = tf.keras.layers.Conv2D(
        HYBRID_EFFICIENT_OUT,
        1,
        strides=1,
        padding="same",
        use_bias=True,
        name=f"h{block_index}_pw",
    )(b)
    b = tf.keras.layers.BatchNormalization(
        momentum=BN_MOMENTUM,
        epsilon=BN_EPSILON,
        name=f"h{block_index}_pw_bn",
    )(b)
    b = tf.keras.layers.ReLU(name=f"h{block_index}_pw_relu")(b)

    return tf.keras.layers.Concatenate(
        axis=-1,
        name=f"h{block_index}_concat",
    )([a, b])


def build_cute_yolo():
    inp = tf.keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, 1),
        name="image",
    )

    x = conv_bn_relu(inp, 8, 3, strides=2, name="stem1")
    x = conv_bn_relu(x, 16, 3, strides=2, name="stem2")
    x = conv_bn_relu(x, 32, 3, strides=2, name="stem3")

    for block_index in range(1, HYBRID_BLOCKS + 1):
        x = hybrid_block(x, block_index)

    # Same head bias initialization as the established Cute-YOLO training.
    size_logit = math.log(0.18 / 0.82)
    head_bias = np.array(
        [-4.0, 0.0, 0.0, size_logit, size_logit],
        dtype=np.float32,
    )

    out = tf.keras.layers.Conv2D(
        OUTPUT_CHANNELS,
        1,
        strides=1,
        padding="same",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Constant(head_bias),
        name="head",
    )(x)

    return tf.keras.Model(inp, out, name="CuteYOLO_Fixed_8_24")



# ============================================================
# Base-training backend
# ============================================================

@dataclass
class TrainConfig:
    dataset_zip: Path
    output_dir: Path

    val_fraction: float = 0.20

    epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 64

    weight_decay: float = 1e-5

    plateau_patience: int = 2
    plateau_factor: float = 0.5
    min_learning_rate: float = 1e-5

    calibration_samples: int = 256

    make_cute: bool = True
    converter_path: Optional[Path] = None

    # Training-only loss profile.
    box_overlap_loss: str = DEFAULT_BOX_OVERLAP_LOSS
    loss_preset: str = DEFAULT_LOSS_PRESET
    hard_negative_mining_enabled: bool = True
    box_aware_negative_enabled: bool = False
    crowd_aware_negative_enabled: bool = False

    # Dataset preparation at the final 128x128 detector scale.
    min_object_size_px: float = DEFAULT_MIN_OBJECT_SIZE_PX

    # Mild training augmentation. Validation is never augmented.
    augment_horizontal_flip: bool = AUGMENT_HORIZONTAL_FLIP_DEFAULT
    augment_brightness: bool = AUGMENT_BRIGHTNESS_DEFAULT
    augment_contrast: bool = AUGMENT_CONTRAST_DEFAULT
    shuffle_each_epoch: bool = AUGMENT_SHUFFLE_DEFAULT


def train_base_model(
    model,
    train_images,
    train_targets,
    train_flipped_targets,
    val_images,
    val_targets,
    cfg: TrainConfig,
    best_weights_path: Path,
    logger: Callable[[str], None],
    progress: Callable[[float], None],
):
    train_ds = make_tf_dataset(
        train_images,
        train_targets,
        cfg.batch_size,
        flipped_targets=train_flipped_targets,
        training=True,
        horizontal_flip=cfg.augment_horizontal_flip,
        brightness=cfg.augment_brightness,
        contrast=cfg.augment_contrast,
        shuffle=cfg.shuffle_each_epoch,
        seed=SEED,
    )

    val_ds = make_tf_dataset(
        val_images,
        val_targets,
        cfg.batch_size,
        training=False,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.learning_rate,
    )

    @tf.function
    def train_step(images, targets):
        with tf.GradientTape() as tape:
            predictions = model(
                images,
                training=True,
            )

            losses = cute_yolo_loss(
                predictions,
                targets,
            )

        gradients = tape.gradient(
            losses["total"],
            model.trainable_variables,
        )

        decayed_gradients = []

        for gradient, variable in zip(
            gradients,
            model.trainable_variables,
        ):
            if gradient is None:
                decayed_gradients.append(None)
            else:
                decayed_gradients.append(
                    gradient
                    + cfg.weight_decay
                    * tf.cast(
                        variable,
                        gradient.dtype,
                    )
                )

        decayed_gradients, _ = tf.clip_by_global_norm(
            decayed_gradients,
            5.0,
        )

        optimizer.apply_gradients(
            zip(
                decayed_gradients,
                model.trainable_variables,
            )
        )

        return (
            losses["total"],
            losses["objectness"],
            losses["box"],
            losses["iou"],
            losses["footprint_negative"],
            losses["halo_negative"],
        )

    @tf.function
    def validation_step(images, targets):
        predictions = model(
            images,
            training=False,
        )

        losses = cute_yolo_loss(
            predictions,
            targets,
        )

        return (
            losses["total"],
            losses["objectness"],
            losses["box"],
            losses["iou"],
            losses["footprint_negative"],
            losses["halo_negative"],
        )

    def run_epoch(dataset, training):
        totals = np.zeros(
            6,
            dtype=np.float64,
        )

        batches = 0

        for batch_images, batch_targets in dataset:
            metrics = (
                train_step(
                    batch_images,
                    batch_targets,
                )
                if training
                else validation_step(
                    batch_images,
                    batch_targets,
                )
            )

            totals += np.asarray(
                [
                    float(x.numpy())
                    for x in metrics
                ],
                dtype=np.float64,
            )

            batches += 1

        return totals / max(
            1,
            batches,
        )

    history = {
        "train": [],
        "validation": [],
        "learning_rate": [],
    }

    best_validation = float("inf")
    plateau_bad_epochs = 0

    for epoch in range(
        1,
        cfg.epochs + 1,
    ):
        started = time.time()

        train_metrics = run_epoch(
            train_ds,
            training=True,
        )

        validation_metrics = run_epoch(
            val_ds,
            training=False,
        )

        current_lr = float(
            tf.keras.backend.get_value(
                optimizer.learning_rate
            )
        )

        history["train"].append(
            train_metrics.tolist()
        )

        history["validation"].append(
            validation_metrics.tolist()
        )

        history["learning_rate"].append(
            current_lr
        )

        if (
            validation_metrics[0]
            < best_validation
        ):
            best_validation = float(
                validation_metrics[0]
            )

            plateau_bad_epochs = 0

            model.save_weights(
                best_weights_path
            )

        else:
            plateau_bad_epochs += 1

            if (
                plateau_bad_epochs
                > cfg.plateau_patience
            ):
                new_lr = max(
                    cfg.min_learning_rate,
                    current_lr
                    * cfg.plateau_factor,
                )

                if new_lr < current_lr:
                    optimizer.learning_rate.assign(
                        new_lr
                    )

                    log_line(
                        logger,
                        f"Learning rate -> {new_lr:.3e}",
                    )

                plateau_bad_epochs = 0

        log_line(
            logger,
            (
                f"Epoch {epoch:02d}/{cfg.epochs}  "
                f"train={train_metrics[0]:.4f}  "
                f"val={validation_metrics[0]:.4f}  "
                f"obj={validation_metrics[1]:.4f}  "
                f"box={validation_metrics[2]:.4f}  "
                f"{BOX_OVERLAP_LOSS}={validation_metrics[3]:.4f}  "
                f"fp={validation_metrics[4]:.4f}  "
                f"halo={validation_metrics[5]:.4f}  "
                f"lr={current_lr:.2e}  "
                f"{time.time() - started:.1f}s"
            ),
        )

        progress(
            10.0
            + 55.0
            * epoch
            / max(1, cfg.epochs)
        )

    model.load_weights(
        best_weights_path
    )

    return (
        history,
        best_validation,
    )


def run_training(
    cfg: TrainConfig,
    logger: Callable[[str], None],
    progress: Callable[[float], None],
):
    global MODEL_LABEL
    global TASK_SLUG
    global EXPERIMENT_TAG
    global SOURCE_RATIO
    global TARGET_RATIO
    global SELECTED_CONFIDENCE_THRESHOLD
    global SELECTED_NMS_IOU_THRESHOLD
    global INT8_EXPORT_DIR
    global TFLITE_PATH
    global MANIFEST_PATH
    global trainable_parameter_count
    global cute_yolo_macs
    global EXPORT_DOMAIN_ADAPTATION

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    configure_objectness_loss(
        cfg.loss_preset,
        cfg.hard_negative_mining_enabled,
        cfg.box_aware_negative_enabled,
        cfg.crowd_aware_negative_enabled,
        logger=logger,
        context="base training",
    )

    configure_box_overlap_loss(
        cfg.box_overlap_loss,
        logger=logger,
        context="base training",
    )

    if not cfg.dataset_zip.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found:\n{cfg.dataset_zip}"
        )

    cfg.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPORT_DOMAIN_ADAPTATION = False
    SOURCE_RATIO = 1.0
    TARGET_RATIO = 0.0

    progress(2)

    log_line(
        logger,
        "Loading training dataset...",
    )

    (
        MODEL_LABEL,
        train_images,
        train_targets,
        train_flipped_targets,
        val_images,
        val_targets,
        val_flipped_targets,
        dataset_stats,
    ) = load_yolo_zip_arrays(
        cfg.dataset_zip,
        val_fraction=cfg.val_fraction,
        seed=SEED,
        min_object_size_px=cfg.min_object_size_px,
    )

    TASK_SLUG = slugify_label(
        MODEL_LABEL
    )

    EXPERIMENT_TAG = (
        f"{TASK_SLUG}_base_fixed_8_24"
    )

    log_line(
        logger,
        f"Detected class: {MODEL_LABEL}",
    )

    log_line(
        logger,
        (
            f"Dataset: "
            f"{len(train_images)} train + "
            f"{len(val_images)} validation"
        ),
    )

    log_line(
        logger,
        (
            "Minimum object size at 128x128 input: "
            f"{cfg.min_object_size_px:g} px"
        ),
    )
    log_line(
        logger,
        (
            "GT boxes: "
            f"{dataset_stats['ground_truth_boxes_after_crop']} after crop -> "
            f"{dataset_stats['ground_truth_boxes_after_size_filter']} encoded; "
            f"{dataset_stats['boxes_dropped_by_min_object_size']} dropped by size; "
            f"{dataset_stats['grid_cell_collisions']} grid-cell collisions"
        ),
    )

    progress(6)

    work_dir = (
        cfg.output_dir
        / f".{TASK_SLUG}_base_work"
    )

    if work_dir.exists():
        shutil.rmtree(
            work_dir
        )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir = (
        cfg.output_dir
        / f"{TASK_SLUG}_base_debug"
    )

    if debug_dir.exists():
        shutil.rmtree(
            debug_dir
        )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_line(
        logger,
        f"Debug images: {debug_dir}",
    )

    save_dataset_debug_preview(
        train_images,
        train_targets,
        debug_dir
        / "training_dataset_preview.png",
        title=(
            "Training samples after "
            "center crop + 128×128 resize"
        ),
        seed=SEED,
    )

    save_dataset_debug_preview(
        val_images,
        val_targets,
        debug_dir
        / "validation_dataset_preview.png",
        title=(
            "Validation samples after "
            "center crop + 128×128 resize"
        ),
        seed=SEED + 1,
    )

    save_dataset_distribution_debug(
        train_targets,
        val_targets,
        debug_dir
        / "dataset_distribution.png",
    )

    model_path = (
        work_dir
        / f"{TASK_SLUG}_base.keras"
    )

    report_path = (
        work_dir
        / f"{TASK_SLUG}_base_training_report.json"
    )

    best_weights_path = (
        work_dir
        / f"{TASK_SLUG}_base_best.weights.h5"
    )

    INT8_EXPORT_DIR = (
        work_dir
        / "int8_export"
    )

    INT8_EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TFLITE_PATH = (
        INT8_EXPORT_DIR
        / f"{TASK_SLUG}_base_int8.tflite"
    )

    MANIFEST_PATH = (
        INT8_EXPORT_DIR
        / f"{TASK_SLUG}_deployment_manifest.json"
    )

    final_export_zip = (
        cfg.output_dir
        / f"{TASK_SLUG}_base_export.zip"
    )

    final_cute_path = (
        cfg.output_dir
        / f"{TASK_SLUG}_base.cute"
    )

    log_line(
        logger,
        "Building fixed Cute-YOLO 8+24 model...",
    )

    log_line(
        logger,
        (
            "Box loss: coordinate MSE + "
            f"{box_overlap_loss_label()}"
        ),
    )

    log_line(
        logger,
        "Objectness configuration selected from the Studio loss profile.",
    )

    log_line(
        logger,
        (
            "Training augmentation: "
            f"flip={'ON' if cfg.augment_horizontal_flip else 'OFF'} "
            f"(p={AUGMENT_FLIP_PROBABILITY:.2f}), "
            f"brightness={'ON' if cfg.augment_brightness else 'OFF'} "
            f"({AUGMENT_BRIGHTNESS_RANGE[0]:.2f}-{AUGMENT_BRIGHTNESS_RANGE[1]:.2f}), "
            f"contrast={'ON' if cfg.augment_contrast else 'OFF'} "
            f"({AUGMENT_CONTRAST_RANGE[0]:.2f}-{AUGMENT_CONTRAST_RANGE[1]:.2f}), "
            f"shuffle={'ON' if cfg.shuffle_each_epoch else 'OFF'}"
        ),
    )
    if cfg.augment_horizontal_flip:
        log_line(
            logger,
            "Flip labels: x_center -> 1-x_center, then target is re-encoded before training.",
        )

    model = build_cute_yolo()

    _ = model(
        tf.zeros(
            [
                1,
                IMG_SIZE,
                IMG_SIZE,
                1,
            ],
            dtype=tf.float32,
        )
    )

    trainable_parameter_count = int(
        sum(
            np.prod(v.shape)
            for v in model.trainable_variables
        )
    )

    if (
        trainable_parameter_count
        != EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError(
            "Unexpected trainable parameter count: "
            f"{trainable_parameter_count:,}"
        )

    cute_yolo_macs = (
        EXPECTED_CONV_MACS
    )

    log_line(
        logger,
        (
            "Trainable parameters: "
            f"{trainable_parameter_count:,}"
        ),
    )

    log_line(
        logger,
        (
            "Convolution MACs: "
            f"{cute_yolo_macs:,}"
        ),
    )

    progress(10)

    (
        history,
        best_validation,
    ) = train_base_model(
        model,
        train_images,
        train_targets,
        train_flipped_targets,
        val_images,
        val_targets,
        cfg,
        best_weights_path,
        logger,
        progress,
    )

    log_line(
        logger,
        "Saving training debug plots...",
    )

    save_training_history_debug(
        history,
        debug_dir,
        prefix="training",
    )

    log_line(
        logger,
        "Selecting FP32 confidence/NMS operating point...",
    )

    (
        best_operating_point,
        operating_points,
        validation_predictions,
    ) = sweep_operating_points_gui(
        model,
        val_images,
        val_targets,
        cfg.batch_size,
        return_debug=True,
    )

    SELECTED_CONFIDENCE_THRESHOLD = float(
        best_operating_point[
            "confidence_threshold"
        ]
    )

    SELECTED_NMS_IOU_THRESHOLD = float(
        best_operating_point[
            "nms_iou_threshold"
        ]
    )

    log_line(
        logger,
        (
            f"FP32 validation F1="
            f"{best_operating_point['f1']:.4f}, "
            f"IoU="
            f"{best_operating_point['mean_iou']:.4f}, "
            f"conf="
            f"{SELECTED_CONFIDENCE_THRESHOLD:.2f}, "
            f"NMS="
            f"{SELECTED_NMS_IOU_THRESHOLD:.2f}"
        ),
    )

    save_operating_point_debug(
        operating_points,
        debug_dir
        / "operating_point_f1.png",
    )

    save_validation_prediction_debug(
        val_images,
        val_targets,
        validation_predictions,
        confidence_threshold=
            SELECTED_CONFIDENCE_THRESHOLD,
        nms_iou_threshold=
            SELECTED_NMS_IOU_THRESHOLD,
        output_path=(
            debug_dir
            / "validation_predictions.png"
        ),
    )

    debug_summary = {
        "architecture_id":
            ARCHITECTURE_ID,
        "box_overlap_loss":
            BOX_OVERLAP_LOSS,
        "box_coordinate_loss":
            "mse_dx_dy_w_h",
        "objectness_training": objectness_training_metadata(),
        "training_augmentation": training_augmentation_metadata(cfg),
        "min_object_size_px_at_detector_input":
            float(cfg.min_object_size_px),
        "evaluation_max_detections":
            EVAL_MAX_DETECTIONS,
        "runtime_max_detections":
            RUNTIME_MAX_DETECTIONS,
        "label":
            MODEL_LABEL,
        "dataset_stats":
            dataset_stats,
        "best_validation_loss":
            float(
                best_validation
            ),
        "best_operating_point":
            best_operating_point,
        "debug_files":
            sorted(
                path.name
                for path
                in debug_dir.iterdir()
                if path.is_file()
            ),
        "visual_conventions": {
            "dataset_preview":
                "solid rectangles are encoded ground-truth boxes",
            "validation_predictions":
                (
                    "solid rectangles are ground truth; "
                    "dashed rectangles are model detections"
                ),
        },
    }

    (
        debug_dir
        / "debug_summary.json"
    ).write_text(
        json.dumps(
            debug_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    model.save(
        model_path
    )

    report = {
        "architecture_id":
            ARCHITECTURE_ID,
        "label":
            MODEL_LABEL,
        "domain_adaptation":
            False,
        "box_overlap_loss":
            BOX_OVERLAP_LOSS,
        "box_coordinate_loss":
            "mse_dx_dy_w_h",
        "objectness_training": objectness_training_metadata(),
        "training_augmentation": training_augmentation_metadata(cfg),
        "min_object_size_px_at_detector_input":
            float(cfg.min_object_size_px),
        "evaluation_max_detections":
            EVAL_MAX_DETECTIONS,
        "runtime_max_detections":
            RUNTIME_MAX_DETECTIONS,
        "dataset":
            str(cfg.dataset_zip),
        "dataset_stats":
            dataset_stats,
        "epochs":
            cfg.epochs,
        "learning_rate":
            cfg.learning_rate,
        "batch_size":
            cfg.batch_size,
        "history":
            history,
        "best_validation_loss":
            best_validation,
        "best_operating_point":
            best_operating_point,
        "debug_directory":
            str(debug_dir),
        "debug_files":
            sorted(
                path.name
                for path
                in debug_dir.iterdir()
                if path.is_file()
            ),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    progress(70)

    calibration_count = min(
        cfg.calibration_samples,
        len(train_images),
    )

    calibration_images = (
        train_images[
            :calibration_count
        ]
    )

    log_line(
        logger,
        (
            "Converting strict INT8 model "
            f"with {calibration_count} "
            "representative images..."
        ),
    )

    (
        interpreter,
        input_detail,
        output_detail,
    ) = convert_to_int8(
        model,
        calibration_images,
        TFLITE_PATH,
        logger,
    )

    progress(80)

    int8_validation = validate_int8(
        model,
        interpreter,
        input_detail,
        output_detail,
        val_images,
    )

    log_line(
        logger,
        (
            "INT8 output MAE: "
            f"{int8_validation['mean_absolute_output_logit_difference']:.6f}"
        ),
    )

    log_line(
        logger,
        "Selecting deployment operating point on INT8 TFLite validation outputs...",
    )

    (
        int8_best_operating_point,
        int8_operating_points,
        int8_validation_predictions,
    ) = sweep_int8_operating_points(
        interpreter,
        input_detail,
        output_detail,
        val_images,
        val_targets,
    )

    SELECTED_CONFIDENCE_THRESHOLD = float(
        int8_best_operating_point["confidence_threshold"]
    )
    SELECTED_NMS_IOU_THRESHOLD = float(
        int8_best_operating_point["nms_iou_threshold"]
    )
    int8_validation["operating_point"] = int8_best_operating_point
    int8_validation["confidence_grid"] = CONFIDENCE_GRID
    int8_validation["nms_iou_grid"] = NMS_IOU_GRID

    log_line(
        logger,
        (
            "INT8 deployment F1="
            f"{int8_best_operating_point['f1']:.4f}, "
            "IoU="
            f"{int8_best_operating_point['mean_iou']:.4f}, "
            "conf="
            f"{SELECTED_CONFIDENCE_THRESHOLD:.2f}, "
            "NMS="
            f"{SELECTED_NMS_IOU_THRESHOLD:.2f}"
        ),
    )

    save_operating_point_debug(
        int8_operating_points,
        debug_dir / "operating_point_int8_f1.png",
    )

    save_validation_prediction_debug(
        val_images,
        val_targets,
        int8_validation_predictions,
        confidence_threshold=SELECTED_CONFIDENCE_THRESHOLD,
        nms_iou_threshold=SELECTED_NMS_IOU_THRESHOLD,
        output_path=debug_dir / "validation_predictions_int8.png",
    )

    # Preserve both operating points in reports, but make the INT8 point
    # the final deployment point because .cute contains quantized weights.
    debug_summary["fp32_operating_point"] = best_operating_point
    debug_summary["int8_operating_point"] = int8_best_operating_point
    debug_summary["best_operating_point"] = int8_best_operating_point
    debug_summary["int8_validation"] = int8_validation
    debug_summary["confidence_grid"] = CONFIDENCE_GRID
    debug_summary["nms_iou_grid"] = NMS_IOU_GRID
    debug_summary["debug_files"] = sorted(
        path.name for path in debug_dir.iterdir() if path.is_file()
    )
    (debug_dir / "debug_summary.json").write_text(
        json.dumps(debug_summary, indent=2),
        encoding="utf-8",
    )

    report["fp32_operating_point"] = best_operating_point
    report["int8_operating_point"] = int8_best_operating_point
    report["best_operating_point"] = int8_best_operating_point
    report["int8_validation"] = int8_validation
    report["confidence_grid"] = CONFIDENCE_GRID
    report["nms_iou_grid"] = NMS_IOU_GRID
    report["debug_files"] = sorted(
        path.name for path in debug_dir.iterdir() if path.is_file()
    )
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    progress(84)

    log_line(
        logger,
        "Extracting NoodleQ parameters...",
    )

    export_noodleq_artifacts(
        interpreter,
        input_detail,
        output_detail,
    )

    runtime_config = {
        "architecture_id":
            ARCHITECTURE_ID,
        "label":
            MODEL_LABEL,
        "confidence_threshold":
            SELECTED_CONFIDENCE_THRESHOLD,
        "nms_iou_threshold":
            SELECTED_NMS_IOU_THRESHOLD,
        "min_box_w":
            MIN_BOX_W,
        "min_box_h":
            MIN_BOX_H,
        "max_detections":
            RUNTIME_MAX_DETECTIONS,
    }

    (
        INT8_EXPORT_DIR
        / "runtime_config.json"
    ).write_text(
        json.dumps(
            runtime_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        INT8_EXPORT_DIR
        / "int8_validation.json"
    ).write_text(
        json.dumps(
            int8_validation,
            indent=2,
        ),
        encoding="utf-8",
    )

    progress(92)

    package_export(
        final_export_zip,
        model_path,
        report_path,
        best_weights_path,
        INT8_EXPORT_DIR,
        debug_dir=debug_dir,
    )

    log_line(
        logger,
        f"Created: {final_export_zip}",
    )

    cute_created = False

    if cfg.make_cute:
        converter = (
            cfg.converter_path
            if cfg.converter_path
            else None
        )

        if (
            converter
            and converter.exists()
        ):
            run_zip_to_cute(
                converter,
                final_export_zip,
                final_cute_path,
                logger,
            )

            cute_created = True

            log_line(
                logger,
                f"Created: {final_cute_path}",
            )

        else:
            log_line(
                logger,
                (
                    "zip_to_cute.py was not found; "
                    "the base export ZIP is complete."
                ),
            )

    progress(100)

    log_line(
        logger,
        "",
    )

    log_line(
        logger,
        "Base training complete.",
    )

    return {
        "label":
            MODEL_LABEL,
        "export_zip":
            final_export_zip,
        "cute":
            final_cute_path
            if cute_created
            else None,
        "operating_point":
            best_operating_point,
        "runtime_config":
            runtime_config,
        "debug_dir":
            debug_dir,
    }


def package_export(
    final_export_zip: Path,
    model_path: Path,
    report_path: Path,
    best_weights_path: Path,
    int8_export_dir: Path,
    debug_dir: Optional[Path] = None,
):
    with zipfile.ZipFile(
        final_export_zip,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(
            model_path,
            arcname=
                model_path.name,
        )

        archive.write(
            report_path,
            arcname=
                report_path.name,
        )

        archive.write(
            best_weights_path,
            arcname=
                best_weights_path.name,
        )

        for file_path in sorted(
            int8_export_dir.iterdir()
        ):
            if file_path.is_file():
                archive.write(
                    file_path,
                    arcname=(
                        "export/"
                        + file_path.name
                    ),
                )

        if (
            debug_dir is not None
            and Path(debug_dir).exists()
        ):
            for file_path in sorted(
                Path(debug_dir).iterdir()
            ):
                if file_path.is_file():
                    archive.write(
                        file_path,
                        arcname=(
                            "debug/"
                            + file_path.name
                        ),
                    )


def run_zip_to_cute(
    converter_path: Path,
    export_zip: Path,
    cute_path: Path,
    logger: Callable[[str], None],
):
    command = [
        sys.executable,
        str(converter_path),
        str(export_zip),
        "--out",
        str(cute_path),
    ]

    log_line(
        logger,
        "Creating .cute package...",
    )

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if process.stdout.strip():
        for line in process.stdout.splitlines():
            log_line(
                logger,
                line,
            )

    if process.stderr.strip():
        for line in process.stderr.splitlines():
            log_line(
                logger,
                line,
            )

    if process.returncode != 0:
        raise RuntimeError(
            "zip_to_cute.py failed with "
            f"exit code {process.returncode}"
        )


def run_adaptation(
    cfg: AdaptConfig,
    logger: Callable[[str], None],
    progress: Callable[[float], None],
):
    global MODEL_LABEL
    global TASK_SLUG
    global EXPERIMENT_TAG
    global SOURCE_RATIO
    global TARGET_RATIO
    global SELECTED_CONFIDENCE_THRESHOLD
    global SELECTED_NMS_IOU_THRESHOLD
    global INT8_EXPORT_DIR
    global TFLITE_PATH
    global MANIFEST_PATH
    global trainable_parameter_count
    global cute_yolo_macs
    global EXPORT_DOMAIN_ADAPTATION

    EXPORT_DOMAIN_ADAPTATION = True
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    configure_objectness_loss(
        cfg.loss_preset,
        cfg.hard_negative_mining_enabled,
        cfg.box_aware_negative_enabled,
        cfg.crowd_aware_negative_enabled,
        logger=logger,
        context="domain adaptation",
    )

    configure_box_overlap_loss(
        cfg.box_overlap_loss,
        logger=logger,
        context="domain adaptation",
    )

    if not np.isclose(
        cfg.source_ratio
        + cfg.target_ratio,
        1.0,
    ):
        raise ValueError(
            "Original-data ratio + real-data ratio must equal 1.0."
        )

    if cfg.source_ratio <= 0:
        raise ValueError(
            "Original-data ratio must be > 0."
        )

    if cfg.target_ratio <= 0:
        raise ValueError(
            "Real-data ratio must be > 0."
        )

    for path, description in [
        (
            cfg.base_model_zip,
            "Base model export ZIP",
        ),
        (
            cfg.original_data_zip,
            "Original training dataset ZIP",
        ),
        (
            cfg.real_data_zip,
            "Real labeled dataset ZIP",
        ),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{description} not found:\n{path}"
            )

    cfg.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress(2)

    log_line(
        logger,
        "Loading original training dataset...",
    )

    (
        source_class,
        source_train_images_tmp,
        source_train_targets_tmp,
        source_train_flipped_targets_tmp,
        source_val_images_tmp,
        source_val_targets_tmp,
        source_val_flipped_targets_tmp,
        source_stats,
    ) = load_yolo_zip_arrays(
        cfg.original_data_zip,
        val_fraction=0.05,
        seed=SEED,
        min_object_size_px=cfg.min_object_size_px,
    )

    source_images = np.concatenate(
        [
            source_train_images_tmp,
            source_val_images_tmp,
        ],
        axis=0,
    )

    source_targets = np.concatenate(
        [
            source_train_targets_tmp,
            source_val_targets_tmp,
        ],
        axis=0,
    )

    source_flipped_targets = np.concatenate(
        [
            source_train_flipped_targets_tmp,
            source_val_flipped_targets_tmp,
        ],
        axis=0,
    )

    log_line(
        logger,
        (
            f"Original training data: "
            f"{len(source_images)} images"
        ),
    )
    log_line(
        logger,
        (
            "Source GT boxes: "
            f"{source_stats['ground_truth_boxes_after_crop']} after crop -> "
            f"{source_stats['ground_truth_boxes_after_size_filter']} encoded; "
            f"{source_stats['boxes_dropped_by_min_object_size']} dropped below "
            f"{cfg.min_object_size_px:g}px"
        ),
    )

    progress(6)

    log_line(
        logger,
        "Loading real labeled dataset...",
    )

    (
        real_class,
        real_train_images,
        real_train_targets,
        real_train_flipped_targets,
        real_val_images,
        real_val_targets,
        real_val_flipped_targets,
        real_stats,
    ) = load_yolo_zip_arrays(
        cfg.real_data_zip,
        val_fraction=
            cfg.target_val_fraction,
        seed=SEED + 1000,
        min_object_size_px=cfg.min_object_size_px,
    )

    if source_class != real_class:
        raise ValueError(
            "The two datasets use different class names.\n"
            f"Original training data: {source_class!r}\n"
            f"Real labeled data:      {real_class!r}"
        )

    log_line(
        logger,
        (
            "Real GT boxes: "
            f"{real_stats['ground_truth_boxes_after_crop']} after crop -> "
            f"{real_stats['ground_truth_boxes_after_size_filter']} encoded; "
            f"{real_stats['boxes_dropped_by_min_object_size']} dropped below "
            f"{cfg.min_object_size_px:g}px"
        ),
    )

    MODEL_LABEL = real_class
    TASK_SLUG = slugify_label(
        MODEL_LABEL
    )

    EXPERIMENT_TAG = (
        f"{TASK_SLUG}_domain_adapted_fixed_8_24"
    )

    SOURCE_RATIO = (
        cfg.source_ratio
    )

    TARGET_RATIO = (
        cfg.target_ratio
    )

    log_line(
        logger,
        f"Detected class: {MODEL_LABEL}",
    )

    log_line(
        logger,
        (
            f"Real data: "
            f"{len(real_train_images)} train + "
            f"{len(real_val_images)} validation"
        ),
    )

    progress(10)

    work_dir = (
        cfg.output_dir
        / f".{TASK_SLUG}_adapt_work"
    )

    if work_dir.exists():
        shutil.rmtree(
            work_dir
        )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir = (
        cfg.output_dir
        / f"{TASK_SLUG}_adapted_debug"
    )

    if debug_dir.exists():
        shutil.rmtree(
            debug_dir
        )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_line(
        logger,
        f"Adaptation debug images: {debug_dir}",
    )

    save_dataset_debug_preview(
        source_images,
        source_targets,
        debug_dir
        / "source_dataset_preview.png",
        title=(
            "Source/original samples used "
            "for domain adaptation"
        ),
        seed=SEED,
    )

    save_dataset_debug_preview(
        real_train_images,
        real_train_targets,
        debug_dir
        / "real_training_dataset_preview.png",
        title=(
            "Real training samples used "
            "for domain adaptation"
        ),
        seed=SEED + 1,
    )

    save_dataset_debug_preview(
        real_val_images,
        real_val_targets,
        debug_dir
        / "real_validation_dataset_preview.png",
        title=(
            "Real validation samples used "
            "for domain adaptation"
        ),
        seed=SEED + 2,
    )

    save_dataset_distribution_debug(
        real_train_targets,
        real_val_targets,
        debug_dir
        / "real_dataset_distribution.png",
    )

    model_path = (
        work_dir
        / f"{TASK_SLUG}_adapted.keras"
    )

    report_path = (
        work_dir
        / f"{TASK_SLUG}_adaptation_report.json"
    )

    best_weights_path = (
        work_dir
        / f"{TASK_SLUG}_adapted_best.weights.h5"
    )

    INT8_EXPORT_DIR = (
        work_dir
        / "int8_export"
    )

    INT8_EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TFLITE_PATH = (
        INT8_EXPORT_DIR
        / f"{TASK_SLUG}_adapted_int8.tflite"
    )

    MANIFEST_PATH = (
        INT8_EXPORT_DIR
        / f"{TASK_SLUG}_deployment_manifest.json"
    )

    final_export_zip = (
        cfg.output_dir
        / f"{TASK_SLUG}_adapted_export.zip"
    )

    final_cute_path = (
        cfg.output_dir
        / f"{TASK_SLUG}_adapted.cute"
    )

    (
        model,
        base_runtime_config,
        trainable_parameter_count,
    ) = load_base_model(
        cfg.base_model_zip,
        MODEL_LABEL,
        work_dir,
        logger,
    )

    cute_yolo_macs = (
        EXPECTED_CONV_MACS
    )

    progress(14)

    log_line(
        logger,
        (
            "Box loss for adaptation: coordinate MSE + "
            f"{box_overlap_loss_label()}"
        ),
    )

    log_line(
        logger,
        "Objectness configuration for adaptation uses the selected loss profile.",
    )
    log_line(
        logger,
        (
            "Adaptation augmentation: "
            f"flip={'ON' if cfg.augment_horizontal_flip else 'OFF'} "
            f"(p={AUGMENT_FLIP_PROBABILITY:.2f}), "
            f"brightness={'ON' if cfg.augment_brightness else 'OFF'} "
            f"({AUGMENT_BRIGHTNESS_RANGE[0]:.2f}-{AUGMENT_BRIGHTNESS_RANGE[1]:.2f}), "
            f"contrast={'ON' if cfg.augment_contrast else 'OFF'} "
            f"({AUGMENT_CONTRAST_RANGE[0]:.2f}-{AUGMENT_CONTRAST_RANGE[1]:.2f}), "
            f"shuffle={'ON' if cfg.shuffle_each_epoch else 'OFF'}"
        ),
    )
    if cfg.augment_horizontal_flip:
        log_line(
            logger,
            "Flip labels: x_center -> 1-x_center, then target is re-encoded before adaptation.",
        )

    log_line(
        logger,
        "Evaluating base model on real validation data...",
    )

    baseline_loss = mean_validation_loss(
        model,
        real_val_images,
        real_val_targets,
        cfg.batch_size,
    )

    (
        baseline_op,
        baseline_operating_points,
        baseline_predictions,
    ) = sweep_operating_points_gui(
        model,
        real_val_images,
        real_val_targets,
        cfg.batch_size,
        return_debug=True,
    )

    log_line(
        logger,
        (
            "Before adaptation: "
            f"F1={baseline_op['f1']:.4f}, "
            f"IoU={baseline_op['mean_iou']:.4f}, "
            f"loss={baseline_loss:.4f}"
        ),
    )

    save_operating_point_debug(
        baseline_operating_points,
        debug_dir
        / "operating_point_before.png",
    )

    save_validation_prediction_debug(
        real_val_images,
        real_val_targets,
        baseline_predictions,
        confidence_threshold=
            float(
                baseline_op[
                    "confidence_threshold"
                ]
            ),
        nms_iou_threshold=
            float(
                baseline_op[
                    "nms_iou_threshold"
                ]
            ),
        output_path=(
            debug_dir
            / "before_adaptation_predictions.png"
        ),
        seed=ADAPT_COMPARE_DEBUG_SEED,
        title=(
            "Before domain adaptation: "
            "solid = ground truth, dashed = prediction"
        ),
    )

    progress(20)

    log_line(
        logger,
        (
            "Fine-tuning with "
            f"{cfg.source_ratio:.0%} original + "
            f"{cfg.target_ratio:.0%} real data..."
        ),
    )

    log_line(
        logger,
        (
            "Adaptation mode: branch-specialized"
            if cfg.specialized_finetune
            else "Adaptation mode: standard full-loss"
        ),
    )

    (
        history,
        best_real_val_loss,
        completed_epochs,
    ) = fine_tune_model(
        model,
        source_images,
        source_targets,
        source_flipped_targets,
        real_train_images,
        real_train_targets,
        real_train_flipped_targets,
        real_val_images,
        real_val_targets,
        cfg,
        best_weights_path,
        logger,
        progress,
    )

    log_line(
        logger,
        "Saving adaptation debug plots...",
    )

    save_training_history_debug(
        history,
        debug_dir,
        prefix="adaptation",
    )

    log_line(
        logger,
        "Selecting real-domain FP32 confidence/NMS operating point...",
    )

    (
        adapted_op,
        adapted_operating_points,
        adapted_predictions,
    ) = sweep_operating_points_gui(
        model,
        real_val_images,
        real_val_targets,
        cfg.batch_size,
        return_debug=True,
    )

    SELECTED_CONFIDENCE_THRESHOLD = float(
        adapted_op[
            "confidence_threshold"
        ]
    )

    SELECTED_NMS_IOU_THRESHOLD = float(
        adapted_op[
            "nms_iou_threshold"
        ]
    )

    log_line(
        logger,
        (
            "After adaptation (FP32): "
            f"F1={adapted_op['f1']:.4f}, "
            f"IoU={adapted_op['mean_iou']:.4f}, "
            f"conf={SELECTED_CONFIDENCE_THRESHOLD:.2f}, "
            f"NMS={SELECTED_NMS_IOU_THRESHOLD:.2f}"
        ),
    )

    save_operating_point_debug(
        adapted_operating_points,
        debug_dir
        / "operating_point_after.png",
    )

    save_validation_prediction_debug(
        real_val_images,
        real_val_targets,
        adapted_predictions,
        confidence_threshold=
            SELECTED_CONFIDENCE_THRESHOLD,
        nms_iou_threshold=
            SELECTED_NMS_IOU_THRESHOLD,
        output_path=(
            debug_dir
            / "after_adaptation_predictions.png"
        ),
        seed=ADAPT_COMPARE_DEBUG_SEED,
        title=(
            "After domain adaptation: "
            "solid = ground truth, dashed = prediction"
        ),
    )

    comparison = {
        "baseline": {
            "validation_loss":
                baseline_loss,
            "operating_point":
                baseline_op,
        },
        "adapted": {
            "validation_loss":
                best_real_val_loss,
            "operating_point":
                adapted_op,
        },
        "f1_change":
            float(
                adapted_op["f1"]
                - baseline_op["f1"]
            ),
        "mean_iou_change":
            float(
                adapted_op["mean_iou"]
                - baseline_op["mean_iou"]
            ),
    }

    debug_summary = {
        "architecture_id":
            ARCHITECTURE_ID,
        "box_overlap_loss":
            BOX_OVERLAP_LOSS,
        "box_coordinate_loss":
            "mse_dx_dy_w_h",
        "objectness_training": objectness_training_metadata(),
        "training_augmentation": training_augmentation_metadata(cfg),
        "min_object_size_px_at_detector_input":
            float(cfg.min_object_size_px),
        "evaluation_max_detections":
            EVAL_MAX_DETECTIONS,
        "runtime_max_detections":
            RUNTIME_MAX_DETECTIONS,
        "label":
            MODEL_LABEL,
        "domain_adaptation":
            True,
        "specialized_finetune":
            bool(cfg.specialized_finetune),
        "specialized_cross_loss_weight":
            float(cfg.cross_loss_weight),
        "specialized_branch_a_objective":
            "localization + alpha*objectness",
        "specialized_branch_b_objective":
            "objectness + alpha*localization",
        "frozen_during_specialized_adaptation":
            ["stem1", "stem2", "stem3", "head"],
        "original_dataset_stats":
            source_stats,
        "real_dataset_stats":
            real_stats,
        "baseline_validation_loss":
            float(
                baseline_loss
            ),
        "adapted_validation_loss":
            float(
                best_real_val_loss
            ),
        "comparison":
            comparison,
        "debug_files":
            sorted(
                path.name
                for path
                in debug_dir.iterdir()
                if path.is_file()
            ),
        "visual_conventions": {
            "dataset_preview":
                "solid rectangles are encoded ground-truth boxes",
            "before_after_predictions":
                (
                    "solid rectangles are ground truth; "
                    "dashed rectangles are model detections; "
                    "before and after figures use identical "
                    "validation sample indices"
                ),
        },
    }

    (
        debug_dir
        / "debug_summary.json"
    ).write_text(
        json.dumps(
            debug_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    model.save(
        model_path
    )

    report = {
        "architecture_id":
            ARCHITECTURE_ID,
        "label":
            MODEL_LABEL,
        "domain_adaptation":
            True,
        "specialized_finetune":
            bool(cfg.specialized_finetune),
        "specialized_cross_loss_weight":
            float(cfg.cross_loss_weight),
        "specialized_branch_a_objective":
            "localization + alpha*objectness",
        "specialized_branch_b_objective":
            "objectness + alpha*localization",
        "frozen_during_specialized_adaptation":
            ["stem1", "stem2", "stem3", "head"],
        "box_overlap_loss":
            BOX_OVERLAP_LOSS,
        "box_coordinate_loss":
            "mse_dx_dy_w_h",
        "objectness_training": objectness_training_metadata(),
        "training_augmentation": training_augmentation_metadata(cfg),
        "min_object_size_px_at_detector_input":
            float(cfg.min_object_size_px),
        "evaluation_max_detections":
            EVAL_MAX_DETECTIONS,
        "runtime_max_detections":
            RUNTIME_MAX_DETECTIONS,
        "base_model_export":
            str(cfg.base_model_zip),
        "original_training_data":
            str(cfg.original_data_zip),
        "real_labeled_data":
            str(cfg.real_data_zip),
        "source_ratio":
            cfg.source_ratio,
        "target_ratio":
            cfg.target_ratio,
        "target_val_fraction":
            cfg.target_val_fraction,
        "fine_tune_learning_rate":
            cfg.learning_rate,
        "fine_tune_epochs_requested":
            cfg.epochs,
        "fine_tune_epochs_completed":
            completed_epochs,
        "original_dataset_stats":
            source_stats,
        "real_dataset_stats":
            real_stats,
        "history":
            history,
        "comparison":
            comparison,
        "debug_directory":
            str(debug_dir),
        "debug_files":
            sorted(
                path.name
                for path
                in debug_dir.iterdir()
                if path.is_file()
            ),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    progress(70)

    log_line(
        logger,
        "Preparing target-heavy INT8 calibration data...",
    )

    calibration_images = (
        make_calibration_pool(
            source_images,
            real_train_images,
            cfg,
        )
    )

    (
        interpreter,
        input_detail,
        output_detail,
    ) = convert_to_int8(
        model,
        calibration_images,
        TFLITE_PATH,
        logger,
    )

    progress(78)

    int8_validation = (
        validate_int8(
            model,
            interpreter,
            input_detail,
            output_detail,
            real_val_images,
        )
    )

    log_line(
        logger,
        (
            "INT8 output MAE: "
            f"{int8_validation['mean_absolute_output_logit_difference']:.6f}"
        ),
    )

    log_line(
        logger,
        "Selecting deployment operating point on adapted INT8 TFLite real-validation outputs...",
    )

    (
        int8_adapted_op,
        int8_adapted_operating_points,
        int8_adapted_predictions,
    ) = sweep_int8_operating_points(
        interpreter,
        input_detail,
        output_detail,
        real_val_images,
        real_val_targets,
    )

    SELECTED_CONFIDENCE_THRESHOLD = float(
        int8_adapted_op["confidence_threshold"]
    )
    SELECTED_NMS_IOU_THRESHOLD = float(
        int8_adapted_op["nms_iou_threshold"]
    )
    int8_validation["operating_point"] = int8_adapted_op
    int8_validation["confidence_grid"] = CONFIDENCE_GRID
    int8_validation["nms_iou_grid"] = NMS_IOU_GRID

    log_line(
        logger,
        (
            "Adapted INT8 deployment F1="
            f"{int8_adapted_op['f1']:.4f}, "
            "IoU="
            f"{int8_adapted_op['mean_iou']:.4f}, "
            "conf="
            f"{SELECTED_CONFIDENCE_THRESHOLD:.2f}, "
            "NMS="
            f"{SELECTED_NMS_IOU_THRESHOLD:.2f}"
        ),
    )

    save_operating_point_debug(
        int8_adapted_operating_points,
        debug_dir / "operating_point_after_int8.png",
    )

    save_validation_prediction_debug(
        real_val_images,
        real_val_targets,
        int8_adapted_predictions,
        confidence_threshold=SELECTED_CONFIDENCE_THRESHOLD,
        nms_iou_threshold=SELECTED_NMS_IOU_THRESHOLD,
        output_path=debug_dir / "after_adaptation_predictions_int8.png",
        seed=ADAPT_COMPARE_DEBUG_SEED,
        title=(
            "After domain adaptation (INT8): "
            "solid = ground truth, dashed = prediction"
        ),
    )

    debug_summary["adapted_fp32_operating_point"] = adapted_op
    debug_summary["adapted_int8_operating_point"] = int8_adapted_op
    debug_summary["deployment_operating_point"] = int8_adapted_op
    debug_summary["int8_validation"] = int8_validation
    debug_summary["confidence_grid"] = CONFIDENCE_GRID
    debug_summary["nms_iou_grid"] = NMS_IOU_GRID
    debug_summary["debug_files"] = sorted(
        path.name for path in debug_dir.iterdir() if path.is_file()
    )
    (debug_dir / "debug_summary.json").write_text(
        json.dumps(debug_summary, indent=2),
        encoding="utf-8",
    )

    report["adapted_fp32_operating_point"] = adapted_op
    report["adapted_int8_operating_point"] = int8_adapted_op
    report["deployment_operating_point"] = int8_adapted_op
    report["int8_validation"] = int8_validation
    report["confidence_grid"] = CONFIDENCE_GRID
    report["nms_iou_grid"] = NMS_IOU_GRID
    report["debug_files"] = sorted(
        path.name for path in debug_dir.iterdir() if path.is_file()
    )
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    progress(82)

    log_line(
        logger,
        "Extracting NoodleQ INT8 parameters...",
    )

    export_noodleq_artifacts(
        interpreter,
        input_detail,
        output_detail,
    )

    runtime_config = {
        "architecture_id":
            ARCHITECTURE_ID,
        "label":
            MODEL_LABEL,
        "confidence_threshold":
            SELECTED_CONFIDENCE_THRESHOLD,
        "nms_iou_threshold":
            SELECTED_NMS_IOU_THRESHOLD,
        "min_box_w":
            MIN_BOX_W,
        "min_box_h":
            MIN_BOX_H,
        "max_detections":
            RUNTIME_MAX_DETECTIONS,
    }

    (
        INT8_EXPORT_DIR
        / "runtime_config.json"
    ).write_text(
        json.dumps(
            runtime_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        INT8_EXPORT_DIR
        / "int8_validation.json"
    ).write_text(
        json.dumps(
            int8_validation,
            indent=2,
        ),
        encoding="utf-8",
    )

    progress(90)

    log_line(
        logger,
        "Packaging adapted training export...",
    )

    package_export(
        final_export_zip,
        model_path,
        report_path,
        best_weights_path,
        INT8_EXPORT_DIR,
        debug_dir=debug_dir,
    )

    log_line(
        logger,
        f"Created: {final_export_zip}",
    )

    cute_created = False

    if cfg.make_cute:
        converter = (
            cfg.converter_path
            if cfg.converter_path
            else None
        )

        if (
            converter
            and converter.exists()
        ):
            run_zip_to_cute(
                converter,
                final_export_zip,
                final_cute_path,
                logger,
            )

            cute_created = True

            log_line(
                logger,
                f"Created: {final_cute_path}",
            )

        else:
            log_line(
                logger,
                (
                    "zip_to_cute.py was not found; "
                    "the adapted export ZIP is complete."
                ),
            )

    progress(100)

    log_line(
        logger,
        "",
    )

    log_line(
        logger,
        "Domain adaptation complete.",
    )

    log_line(
        logger,
        (
            "F1 change: "
            f"{comparison['f1_change']:+.4f}"
        ),
    )

    log_line(
        logger,
        (
            "Mean IoU change: "
            f"{comparison['mean_iou_change']:+.4f}"
        ),
    )

    return {
        "label":
            MODEL_LABEL,
        "export_zip":
            final_export_zip,
        "cute":
            final_cute_path
            if cute_created
            else None,
        "comparison":
            comparison,
        "runtime_config":
            runtime_config,
        "debug_dir":
            debug_dir,
    }


# ============================================================
# Tkinter GUI
# ============================================================





async def _find_cute_ble_target(
    address: Optional[str],
    logger: Callable[[str], None],
):
    try:
        from bleak import BleakScanner
    except ImportError as exc:
        raise RuntimeError(
            "BLE deployment requires bleak.\n"
            "Install it with:\n"
            "    pip install bleak"
        ) from exc

    if address:
        log_line(
            logger,
            f"Using BLE address: {address}",
        )
        return address

    log_line(
        logger,
        f"Scanning for {CUTE_BLE_DEVICE_NAME}...",
    )

    device = await BleakScanner.find_device_by_filter(
        lambda dev, adv: (
            dev.name == CUTE_BLE_DEVICE_NAME
            or adv.local_name == CUTE_BLE_DEVICE_NAME
            or CUTE_BLE_SERVICE_UUID.lower()
            in [
                str(value).lower()
                for value in adv.service_uuids
            ]
        ),
        timeout=12.0,
    )

    if device is None:
        raise RuntimeError(
            "Cute-YOLO BLE device not found"
        )

    log_line(
        logger,
        f"Found {device.name} @ {device.address}",
    )

    return device


async def _upload_cute_blob_async(
    cute_path: Path,
    address: Optional[str],
    logger: Callable[[str], None],
    progress: Callable[[float], None],
):
    try:
        from bleak import BleakClient
    except ImportError as exc:
        raise RuntimeError(
            "BLE deployment requires bleak.\n"
            "Install it with:\n"
            "    pip install bleak"
        ) from exc

    blob = cute_path.read_bytes()

    if (
        len(blob) < 1024
        or blob[:4] != b"CUTE"
    ):
        raise ValueError(
            "Not a valid .cute model"
        )

    target = await _find_cute_ble_target(
        address,
        logger,
    )

    status_queue: asyncio.Queue[str] = (
        asyncio.Queue()
    )

    def on_status(
        _sender,
        data: bytearray,
    ):
        text = bytes(data).decode(
            "utf-8",
            errors="replace",
        )

        log_line(
            logger,
            f"[ESP32] {text}",
        )

        status_queue.put_nowait(
            text
        )

    async with BleakClient(
        target
    ) as client:
        log_line(
            logger,
            "BLE connected",
        )

        await client.start_notify(
            CUTE_BLE_STATUS_UUID,
            on_status,
        )

        await client.write_gatt_char(
            CUTE_BLE_CTRL_UUID,
            b"INFO",
            response=True,
        )

        mtu = (
            getattr(
                client,
                "mtu_size",
                23,
            )
            or 23
        )

        chunk = max(
            20,
            min(
                240,
                mtu - 3,
            ),
        )

        log_line(
            logger,
            (
                f"Uploading {len(blob):,} bytes "
                f"with chunk={chunk}"
            ),
        )

        await client.write_gatt_char(
            CUTE_BLE_CTRL_UUID,
            f"BEGIN {len(blob)}".encode(),
            response=True,
        )

        ready = False

        for _ in range(8):
            try:
                text = await asyncio.wait_for(
                    status_queue.get(),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                break

            if text.startswith("ERR"):
                raise RuntimeError(
                    text
                )

            if text.startswith("READY "):
                ready = True
                break

        if not ready:
            log_line(
                logger,
                (
                    "READY notification not observed; "
                    "continuing"
                ),
            )

        sent = 0

        while sent < len(blob):
            part = blob[
                sent:sent + chunk
            ]

            await client.write_gatt_char(
                CUTE_BLE_DATA_UUID,
                part,
                response=True,
            )

            sent += len(part)

            upload_fraction = (
                sent / len(blob)
            )

            progress(
                35.0
                + 55.0
                * upload_fraction
            )

            if (
                sent == len(blob)
                or sent % 4096 < chunk
            ):
                log_line(
                    logger,
                    (
                        f"  {sent:,}/{len(blob):,} "
                        f"({100.0 * upload_fraction:.1f}%)"
                    ),
                )

        await client.write_gatt_char(
            CUTE_BLE_CTRL_UUID,
            b"END",
            response=True,
        )

        log_line(
            logger,
            "Transfer complete; waiting for validation/activation...",
        )

        end_time = (
            asyncio.get_running_loop()
            .time()
            + 12.0
        )

        activated = False

        while (
            asyncio.get_running_loop()
            .time()
            < end_time
        ):
            try:
                text = await asyncio.wait_for(
                    status_queue.get(),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                continue

            if text.startswith("ERR"):
                raise RuntimeError(
                    text
                )

            if text.startswith("ACTIVE "):
                activated = True

                log_line(
                    logger,
                    "Model activated",
                )

                break

        if not activated:
            log_line(
                logger,
                (
                    "Activation notification not seen; "
                    "requesting INFO"
                ),
            )

            await client.write_gatt_char(
                CUTE_BLE_CTRL_UUID,
                b"INFO",
                response=True,
            )

            await asyncio.sleep(
                0.5
            )

        await client.stop_notify(
            CUTE_BLE_STATUS_UUID
        )

    progress(
        100.0
    )

    log_line(
        logger,
        "BLE deployment complete.",
    )


def upload_cute_via_ble(
    cute_path: Path,
    address: Optional[str],
    logger: Callable[[str], None],
    progress: Callable[[float], None],
):
    asyncio.run(
        _upload_cute_blob_async(
            cute_path,
            address,
            logger,
            progress,
        )
    )


def run_deploy_cute(
    cute_path: Path,
    address: Optional[str],
    logger: Callable[[str], None],
    progress: Callable[[float], None],
):
    """Upload an already-built .cute package directly over BLE.

    Train/Adapt are responsible for producing the .cute artifact.
    The Deploy tab intentionally performs no ZIP conversion so the
    exact file selected by the user is the exact file transmitted.
    """
    cute_path = Path(cute_path)

    if not cute_path.exists():
        raise FileNotFoundError(
            f".cute model not found:\n{cute_path}"
        )

    if not cute_path.is_file():
        raise ValueError(
            f"Selected .cute path is not a file:\n{cute_path}"
        )

    if cute_path.suffix.lower() != ".cute":
        raise ValueError(
            "Choose a .cute model package, not a training export ZIP."
        )

    blob = cute_path.read_bytes()

    if len(blob) < 1024 or blob[:4] != b"CUTE":
        raise ValueError(
            f"Not a valid .cute model package:\n{cute_path}"
        )

    progress(5.0)

    log_line(
        logger,
        f"Selected .cute: {cute_path}",
    )

    log_line(
        logger,
        f"Package size: {len(blob):,} bytes",
    )

    log_line(
        logger,
        "Starting BLE upload...",
    )

    upload_cute_via_ble(
        cute_path,
        address,
        logger,
        progress,
    )

    return {
        "label": cute_path.stem,
        "cute": cute_path,
    }

# ============================================================
# Unified four-tab desktop GUI
# ============================================================

class CuteYOLOStudio:
    def __init__(self, root: tk.Tk):
        self.root = root

        self.root.title(
            "Cute-YOLO Studio"
        )

        self.root.geometry(
            "980x900"
        )

        self.root.minsize(
            860,
            760,
        )

        self.messages = queue.Queue()
        self.worker = None
        self.last_output_dir = None

        self.converter_var = tk.StringVar(
            value=str(
                self.default_converter_path()
            )
        )

        # -------------------------
        # Train tab variables
        # -------------------------
        self.train_dataset_var = tk.StringVar()

        self.train_output_var = tk.StringVar(
            value=str(
                Path.cwd()
                / "trained_models"
            )
        )

        self.train_val_pct_var = tk.StringVar(
            value="20"
        )

        self.train_epochs_var = tk.StringVar(
            value="50"
        )

        self.train_lr_var = tk.StringVar(
            value="0.001"
        )

        self.train_batch_var = tk.StringVar(
            value="64"
        )

        self.train_make_cute_var = (
            tk.BooleanVar(
                value=True
            )
        )

        # Loss profile / advanced switches for base training.
        self.train_loss_preset_var = tk.StringVar(
            value=DEFAULT_LOSS_PRESET
        )
        self.train_box_overlap_var = tk.StringVar(
            value=DEFAULT_BOX_OVERLAP_LOSS
        )
        self.train_hnm_var = tk.BooleanVar(value=True)
        self.train_spatial_neg_var = tk.BooleanVar(value=False)
        self.train_crowd_var = tk.BooleanVar(value=False)
        self.train_min_object_px_var = tk.StringVar(
            value=f"{DEFAULT_MIN_OBJECT_SIZE_PX:g}"
        )
        self.train_loss_summary_var = tk.StringVar()
        self.train_aug_flip_var = tk.BooleanVar(value=AUGMENT_HORIZONTAL_FLIP_DEFAULT)
        self.train_aug_brightness_var = tk.BooleanVar(value=AUGMENT_BRIGHTNESS_DEFAULT)
        self.train_aug_contrast_var = tk.BooleanVar(value=AUGMENT_CONTRAST_DEFAULT)
        self.train_shuffle_var = tk.BooleanVar(value=AUGMENT_SHUFFLE_DEFAULT)

        # -------------------------
        # Adapt tab variables
        # -------------------------
        self.adapt_base_var = tk.StringVar()
        self.adapt_original_var = tk.StringVar()
        self.adapt_real_var = tk.StringVar()

        self.adapt_output_var = tk.StringVar(
            value=str(
                Path.cwd()
                / "adapted_models"
            )
        )

        self.adapt_source_pct_var = tk.StringVar(
            value="75"
        )

        self.adapt_target_pct_var = tk.StringVar(
            value="25"
        )

        self.adapt_val_pct_var = tk.StringVar(
            value="20"
        )

        self.adapt_epochs_var = tk.StringVar(
            value="15"
        )

        self.adapt_lr_var = tk.StringVar(
            value="0.0001"
        )

        self.adapt_batch_var = tk.StringVar(
            value="64"
        )

        self.adapt_make_cute_var = (
            tk.BooleanVar(
                value=True
            )
        )

        self.adapt_specialized_var = tk.BooleanVar(
            value=SPECIALIZED_FINETUNE_DEFAULT
        )
        self.adapt_cross_loss_var = tk.StringVar(
            value=str(SPECIALIZED_CROSS_LOSS_WEIGHT)
        )

        # Loss profile / advanced switches for adaptation.
        self.adapt_loss_preset_var = tk.StringVar(
            value=DEFAULT_LOSS_PRESET
        )
        self.adapt_box_overlap_var = tk.StringVar(
            value=DEFAULT_BOX_OVERLAP_LOSS
        )
        self.adapt_hnm_var = tk.BooleanVar(value=True)
        self.adapt_spatial_neg_var = tk.BooleanVar(value=False)
        self.adapt_crowd_var = tk.BooleanVar(value=False)
        self.adapt_min_object_px_var = tk.StringVar(
            value=f"{DEFAULT_MIN_OBJECT_SIZE_PX:g}"
        )
        self.adapt_loss_summary_var = tk.StringVar()
        self.adapt_aug_flip_var = tk.BooleanVar(value=AUGMENT_HORIZONTAL_FLIP_DEFAULT)
        self.adapt_aug_brightness_var = tk.BooleanVar(value=AUGMENT_BRIGHTNESS_DEFAULT)
        self.adapt_aug_contrast_var = tk.BooleanVar(value=AUGMENT_CONTRAST_DEFAULT)
        self.adapt_shuffle_var = tk.BooleanVar(value=AUGMENT_SHUFFLE_DEFAULT)

        # -------------------------
        # Deploy tab variables
        # -------------------------
        # Deploy now consumes an already-built .cute package directly.
        # ZIP -> .cute conversion remains part of Train/Adapt only.
        self.deploy_cute_var = tk.StringVar()
        self.deploy_address_var = tk.StringVar()

        # -------------------------
        # Shared status
        # -------------------------
        self.status_var = tk.StringVar(
            value="Ready"
        )

        self.progress_var = tk.DoubleVar(
            value=0.0
        )

        self.build_ui()
        self.poll_messages()

    def default_converter_path(self):
        here = Path(__file__).resolve()

        candidates = [
            here.parent.parent
            / "tools"
            / "zip_to_cute.py",
            here.parent
            / "tools"
            / "zip_to_cute.py",
            here.parent
            / "zip_to_cute.py",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def build_ui(self):
        outer = ttk.Frame(
            self.root,
            padding=14,
        )

        outer.pack(
            fill="both",
            expand=True,
        )

        ttk.Label(
            outer,
            text="Cute-YOLO Studio",
            font=(
                "TkDefaultFont",
                17,
                "bold",
            ),
        ).pack(
            anchor="w",
        )

        ttk.Label(
            outer,
            text=(
                "Train, adapt, and deploy the fixed Cute-YOLO model over BLE."
            ),
        ).pack(
            anchor="w",
            pady=(2, 12),
        )

        self.tabs = ttk.Notebook(
            outer
        )

        self.tabs.pack(
            fill="both",
            expand=True,
        )

        self.train_tab = ttk.Frame(
            self.tabs,
            padding=12,
        )

        self.adapt_tab = ttk.Frame(
            self.tabs,
            padding=12,
        )

        self.deploy_tab = ttk.Frame(
            self.tabs,
            padding=12,
        )

        self.log_tab = ttk.Frame(
            self.tabs,
            padding=12,
        )

        self.tabs.add(
            self.train_tab,
            text="1  Train",
        )

        self.tabs.add(
            self.adapt_tab,
            text="2  Adapt to Real Data",
        )

        self.tabs.add(
            self.deploy_tab,
            text="3  Deploy via BLE",
        )

        self.tabs.add(
            self.log_tab,
            text="4  Log",
        )

        self.build_train_tab()
        self.build_adapt_tab()
        self.build_deploy_tab()
        self.build_log_tab()

        shared = ttk.LabelFrame(
            outer,
            text="Train/Adapt packaging tool",
            padding=8,
        )

        shared.pack(
            fill="x",
            pady=(12, 0),
        )

        ttk.Label(
            shared,
            text="zip_to_cute.py (used by Train/Adapt)",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Entry(
            shared,
            textvariable=self.converter_var,
        ).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=8,
        )

        ttk.Button(
            shared,
            text="Browse",
            command=self.browse_converter,
        ).grid(
            row=0,
            column=2,
        )

        shared.columnconfigure(
            1,
            weight=1,
        )

        status_frame = ttk.Frame(
            outer
        )

        status_frame.pack(
            fill="x",
            pady=(12, 0),
        )

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
        ).pack(
            side="left",
        )

        ttk.Button(
            status_frame,
            text="Open output folder",
            command=self.open_current_output_folder,
        ).pack(
            side="right",
        )

        ttk.Button(
            status_frame,
            text="Clear log",
            command=self.clear_log,
        ).pack(
            side="right",
            padx=8,
        )

        ttk.Progressbar(
            outer,
            variable=self.progress_var,
            maximum=100,
        ).pack(
            fill="x",
            pady=(8, 8),
        )

    # ========================================================
    # Log tab
    # ========================================================

    def build_log_tab(self):
        frame = self.log_tab

        toolbar = ttk.Frame(frame)
        toolbar.pack(
            fill="x",
            pady=(0, 8),
        )

        ttk.Label(
            toolbar,
            text=(
                "Training, adaptation, conversion, and BLE deployment log."
            ),
            font=(
                "TkDefaultFont",
                10,
                "bold",
            ),
        ).pack(
            side="left",
        )

        ttk.Button(
            toolbar,
            text="Open output folder",
            command=self.open_current_output_folder,
        ).pack(
            side="right",
        )

        ttk.Button(
            toolbar,
            text="Clear log",
            command=self.clear_log,
        ).pack(
            side="right",
            padx=(0, 8),
        )

        log_frame = ttk.LabelFrame(
            frame,
            text="Log",
            padding=6,
        )

        log_frame.pack(
            fill="both",
            expand=True,
        )

        self.log_widget = tk.Text(
            log_frame,
            height=30,
            wrap="word",
            state="disabled",
        )

        scrollbar = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_widget.yview,
        )

        self.log_widget.configure(
            yscrollcommand=scrollbar.set
        )

        self.log_widget.pack(
            side="left",
            fill="both",
            expand=True,
        )

        scrollbar.pack(
            side="right",
            fill="y",
        )

    # ========================================================
    # Train tab
    # ========================================================

    def build_train_tab(self):
        frame = self.train_tab

        ttk.Label(
            frame,
            text=(
                "Train from a single-class YOLO ZIP. "
                'If classes.txt is absent, the class is named "X".'
            ),
            font=(
                "TkDefaultFont",
                11,
                "bold",
            ),
        ).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, 8),
        )

        self.add_file_row(
            frame,
            1,
            "Dataset ZIP",
            self.train_dataset_var,
            self.browse_train_dataset,
        )

        self.add_folder_row(
            frame,
            2,
            "Output folder",
            self.train_output_var,
            self.browse_train_output,
        )

        settings = ttk.LabelFrame(
            frame,
            text="Training settings",
            padding=8,
        )

        settings.grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(12, 0),
        )

        self.add_number(
            settings,
            0,
            0,
            "Validation %",
            self.train_val_pct_var,
        )

        self.add_number(
            settings,
            0,
            2,
            "Epochs",
            self.train_epochs_var,
        )

        self.add_number(
            settings,
            1,
            0,
            "Learning rate",
            self.train_lr_var,
        )

        self.add_number(
            settings,
            1,
            2,
            "Batch size",
            self.train_batch_var,
        )

        self.add_augmentation_controls(
            settings,
            2,
            self.train_aug_flip_var,
            self.train_aug_brightness_var,
            self.train_aug_contrast_var,
            self.train_shuffle_var,
        )

        self.add_loss_profile_controls(
            frame,
            4,
            self.train_loss_preset_var,
            self.train_box_overlap_var,
            self.train_hnm_var,
            self.train_spatial_neg_var,
            self.train_crowd_var,
            self.train_min_object_px_var,
            self.train_loss_summary_var,
        )

        ttk.Checkbutton(
            frame,
            text=(
                "Also create "
                "<class>_base.cute"
            ),
            variable=
                self.train_make_cute_var,
        ).grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(12, 0),
        )

        self.train_run_button = ttk.Button(
            frame,
            text="Train base model",
            command=self.start_training,
        )

        self.train_run_button.grid(
            row=6,
            column=0,
            sticky="w",
            pady=(12, 0),
        )

        ttk.Label(
            frame,
            text=(
                "Outputs: "
                "<class>_base_export.zip, "
                "<class>_base_debug/, "
                "and optionally <class>_base.cute"
            ),
        ).grid(
            row=7,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(10, 0),
        )

        frame.columnconfigure(
            1,
            weight=1,
        )

    # ========================================================
    # Adapt tab
    # ========================================================

    def build_adapt_tab(self):
        frame = self.adapt_tab

        ttk.Label(
            frame,
            text=(
                "Fine-tune a base model using verified "
                "real deployment-domain data."
            ),
            font=(
                "TkDefaultFont",
                11,
                "bold",
            ),
        ).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, 8),
        )

        self.add_file_row(
            frame,
            1,
            "Base model export",
            self.adapt_base_var,
            self.browse_adapt_base,
        )

        self.add_file_row(
            frame,
            2,
            "Original training data",
            self.adapt_original_var,
            self.browse_adapt_original,
        )

        self.add_file_row(
            frame,
            3,
            "Real labeled data",
            self.adapt_real_var,
            self.browse_adapt_real,
        )

        self.add_folder_row(
            frame,
            4,
            "Output folder",
            self.adapt_output_var,
            self.browse_adapt_output,
        )

        ttk.Label(
            frame,
            text=(
                "Real labeled data = ZIP exported by cute_label_gui.py"
            ),
        ).grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

        settings = ttk.LabelFrame(
            frame,
            text="Adaptation settings",
            padding=8,
        )

        settings.grid(
            row=6,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(12, 0),
        )

        self.add_number(
            settings,
            0,
            0,
            "Original data %",
            self.adapt_source_pct_var,
        )

        self.add_number(
            settings,
            0,
            2,
            "Real data %",
            self.adapt_target_pct_var,
        )

        self.add_number(
            settings,
            1,
            0,
            "Real validation %",
            self.adapt_val_pct_var,
        )

        self.add_number(
            settings,
            1,
            2,
            "Epochs",
            self.adapt_epochs_var,
        )

        self.add_number(
            settings,
            2,
            0,
            "Learning rate",
            self.adapt_lr_var,
        )

        self.add_number(
            settings,
            2,
            2,
            "Batch size",
            self.adapt_batch_var,
        )

        ttk.Checkbutton(
            settings,
            text="Dual-branch specialized fine-tuning (freeze stem + head)",
            variable=self.adapt_specialized_var,
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(6, 4),
        )

        self.add_number(
            settings,
            3,
            2,
            "Cross-loss weight",
            self.adapt_cross_loss_var,
        )

        ttk.Label(
            settings,
            text=(
                "A/Core 0: localization-dominant | "
                "B/Core 1: objectness-dominant"
            ),
        ).grid(
            row=4,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(4, 0),
        )

        self.add_augmentation_controls(
            settings,
            5,
            self.adapt_aug_flip_var,
            self.adapt_aug_brightness_var,
            self.adapt_aug_contrast_var,
            self.adapt_shuffle_var,
        )

        self.add_loss_profile_controls(
            frame,
            7,
            self.adapt_loss_preset_var,
            self.adapt_box_overlap_var,
            self.adapt_hnm_var,
            self.adapt_spatial_neg_var,
            self.adapt_crowd_var,
            self.adapt_min_object_px_var,
            self.adapt_loss_summary_var,
        )

        ttk.Checkbutton(
            frame,
            text=(
                "Also create "
                "<class>_adapted.cute"
            ),
            variable=
                self.adapt_make_cute_var,
        ).grid(
            row=8,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(12, 0),
        )

        self.adapt_run_button = ttk.Button(
            frame,
            text="Adapt model",
            command=self.start_adaptation,
        )

        self.adapt_run_button.grid(
            row=9,
            column=0,
            sticky="w",
            pady=(12, 0),
        )

        ttk.Label(
            frame,
            text=(
                "Outputs: "
                "<class>_adapted_export.zip, "
                "<class>_adapted_debug/, "
                "and optionally <class>_adapted.cute"
            ),
        ).grid(
            row=10,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(10, 0),
        )

        frame.columnconfigure(
            1,
            weight=1,
        )

    # ========================================================
    # Deploy tab
    # ========================================================

    def build_deploy_tab(self):
        frame = self.deploy_tab

        ttk.Label(
            frame,
            text=(
                "Upload an existing .cute model package directly "
                "to the ESP32 over BLE."
            ),
            font=(
                "TkDefaultFont",
                11,
                "bold",
            ),
        ).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(0, 8),
        )

        self.add_file_row(
            frame,
            1,
            "Cute model (.cute)",
            self.deploy_cute_var,
            self.browse_deploy_cute,
        )

        ttk.Label(
            frame,
            text="BLE address (optional)",
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=3,
        )

        ttk.Entry(
            frame,
            textvariable=
                self.deploy_address_var,
        ).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=8,
            pady=3,
        )

        ttk.Label(
            frame,
            text=(
                "Leave the BLE address empty to scan automatically "
                'for device name "Cute-YOLO".'
            ),
        ).grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(5, 0),
        )

        pipeline = ttk.LabelFrame(
            frame,
            text="Direct deployment pipeline",
            padding=10,
        )

        pipeline.grid(
            row=4,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(14, 0),
        )

        ttk.Label(
            pipeline,
            text=(
                ".cute  →  BLE scan/connect  →  upload  "
                "→  verify  →  activate"
            ),
        ).pack(
            anchor="w",
        )

        ttk.Label(
            pipeline,
            text=(
                "The selected .cute file is uploaded byte-for-byte; "
                "no ZIP conversion is performed in this tab."
            ),
        ).pack(
            anchor="w",
            pady=(6, 0),
        )

        self.deploy_run_button = ttk.Button(
            frame,
            text="Upload .cute",
            command=self.start_deploy,
        )

        self.deploy_run_button.grid(
            row=5,
            column=0,
            sticky="w",
            pady=(14, 0),
        )

        ttk.Label(
            frame,
            text=(
                "Requires the ESP32 Cute-YOLO firmware to be running "
                "and Python package 'bleak' to be installed."
            ),
        ).grid(
            row=6,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(10, 0),
        )

        frame.columnconfigure(
            1,
            weight=1,
        )

    # ========================================================
    # Generic widget helpers
    # ========================================================

    def add_file_row(
        self,
        parent,
        row,
        label,
        variable,
        command,
    ):
        ttk.Label(
            parent,
            text=label,
        ).grid(
            row=row,
            column=0,
            sticky="w",
            pady=3,
        )

        ttk.Entry(
            parent,
            textvariable=variable,
        ).grid(
            row=row,
            column=1,
            sticky="ew",
            padx=8,
            pady=3,
        )

        ttk.Button(
            parent,
            text="Browse",
            command=command,
        ).grid(
            row=row,
            column=2,
            pady=3,
        )

        parent.columnconfigure(
            1,
            weight=1,
        )

    def add_augmentation_controls(
        self,
        parent,
        row,
        flip_var,
        brightness_var,
        contrast_var,
        shuffle_var,
    ):
        box = ttk.LabelFrame(
            parent,
            text="Training augmentation (training only)",
            padding=6,
        )
        box.grid(
            row=row,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(8, 2),
        )

        ttk.Checkbutton(
            box,
            text=f"Horizontal flip (p={AUGMENT_FLIP_PROBABILITY:.2f}; labels re-encoded)",
            variable=flip_var,
        ).grid(row=0, column=0, sticky="w", padx=(0, 14))

        ttk.Checkbutton(
            box,
            text=(
                "Brightness "
                f"({AUGMENT_BRIGHTNESS_RANGE[0]:.2f}–{AUGMENT_BRIGHTNESS_RANGE[1]:.2f})"
            ),
            variable=brightness_var,
        ).grid(row=0, column=1, sticky="w", padx=(0, 14))

        ttk.Checkbutton(
            box,
            text=(
                "Contrast "
                f"({AUGMENT_CONTRAST_RANGE[0]:.2f}–{AUGMENT_CONTRAST_RANGE[1]:.2f})"
            ),
            variable=contrast_var,
        ).grid(row=0, column=2, sticky="w", padx=(0, 14))

        ttk.Checkbutton(
            box,
            text="Shuffle each epoch",
            variable=shuffle_var,
        ).grid(row=0, column=3, sticky="w")

        ttk.Label(
            box,
            text=(
                "Flip geometry: x_center -> 1 - x_center, then the 16x16 target "
                "is re-encoded. Brightness/contrast do not change labels."
            ),
        ).grid(
            row=1,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(5, 0),
        )

        box.columnconfigure(3, weight=1)
        return box

    def add_loss_profile_controls(
        self,
        parent,
        row,
        preset_var,
        overlap_var,
        hnm_var,
        spatial_var,
        crowd_var,
        min_object_px_var,
        summary_var,
    ):
        box = ttk.LabelFrame(
            parent,
            text="Object characteristics / loss preset",
            padding=8,
        )
        box.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(12, 0),
        )

        def update_summary():
            terms = ["L_pos"]
            if hnm_var.get():
                terms.append(f"{BACKGROUND_LOSS_WEIGHT:g} L_hard")
            if spatial_var.get():
                terms.extend([
                    f"{BOX_FOOTPRINT_NEGATIVE_WEIGHT:g} L_foot",
                    f"{BOX_HALO_NEGATIVE_WEIGHT:g} L_halo",
                ])
            text = "Effective: L_obj = " + " + ".join(terms)
            if crowd_var.get() and spatial_var.get():
                text += "  (spatial terms crowd-weighted)"

            overlap_label = (
                "IoU"
                if overlap_var.get() == BOX_OVERLAP_LOSS_IOU
                else "CIoU"
            )
            text += (
                f"  |  L = L_obj + {BOX_WEIGHT:g} L_box "
                f"+ {IOU_WEIGHT:g} L_{overlap_label}"
            )

            try:
                min_px = max(0.0, float(min_object_px_var.get()))
                text += f"  |  min object @128x128: {min_px:g}px"
            except ValueError:
                text += "  |  min object size: INVALID"

            summary_var.set(text)

        def apply_preset():
            value = preset_var.get()
            if value == LOSS_PRESET_COMPLEX:
                hnm_var.set(True)
                spatial_var.set(False)
                crowd_var.set(False)
                min_object_px_var.set(
                    f"{COMPLEX_MIN_OBJECT_SIZE_PX:g}"
                )
            elif value == LOSS_PRESET_PRIMITIVE:
                hnm_var.set(True)
                spatial_var.set(True)
                crowd_var.set(True)
                min_object_px_var.set(
                    f"{PRIMITIVE_MIN_OBJECT_SIZE_PX:g}"
                )
            update_summary()

        def manual_change(source):
            if source == "spatial" and not spatial_var.get():
                crowd_var.set(False)
            elif source == "crowd" and crowd_var.get():
                spatial_var.set(True)
            preset_var.set(LOSS_PRESET_CUSTOM)
            update_summary()

        def manual_min_size_change(_event=None):
            preset_var.set(LOSS_PRESET_CUSTOM)
            update_summary()

        ttk.Radiobutton(
            box,
            text=(
                "Complex / distinctive — faces, vehicles, animals; "
                "rich visual structure, usually fewer confusing repeats"
            ),
            variable=preset_var,
            value=LOSS_PRESET_COMPLEX,
            command=apply_preset,
        ).grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(0, 3),
        )

        ttk.Radiobutton(
            box,
            text=(
                "Primitive / repetitive — circles, blobs, simple shapes; "
                "many similar nearby objects may form pseudo-objects"
            ),
            variable=preset_var,
            value=LOSS_PRESET_PRIMITIVE,
            command=apply_preset,
        ).grid(
            row=1,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(0, 5),
        )

        ttk.Radiobutton(
            box,
            text="Custom — use the switches below",
            variable=preset_var,
            value=LOSS_PRESET_CUSTOM,
            command=update_summary,
        ).grid(
            row=2,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(0, 6),
        )

        ttk.Checkbutton(
            box,
            text="Hard-negative mining",
            variable=hnm_var,
            command=lambda: manual_change("hnm"),
        ).grid(
            row=3,
            column=0,
            sticky="w",
            padx=(18, 18),
        )

        ttk.Checkbutton(
            box,
            text="Box footprint + halo negatives",
            variable=spatial_var,
            command=lambda: manual_change("spatial"),
        ).grid(
            row=3,
            column=1,
            sticky="w",
            padx=(0, 18),
        )

        ttk.Checkbutton(
            box,
            text="Crowd-aware weighting",
            variable=crowd_var,
            command=lambda: manual_change("crowd"),
        ).grid(
            row=3,
            column=2,
            sticky="w",
        )

        ttk.Label(
            box,
            text="Minimum object size at 128x128 input (px)",
        ).grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
            padx=(18, 8),
            pady=(7, 0),
        )

        min_size_entry = ttk.Entry(
            box,
            textvariable=min_object_px_var,
            width=8,
        )
        min_size_entry.grid(
            row=4,
            column=2,
            sticky="w",
            pady=(7, 0),
        )
        min_size_entry.bind(
            "<FocusOut>",
            manual_min_size_change,
        )
        min_size_entry.bind(
            "<Return>",
            manual_min_size_change,
        )

        ttk.Label(
            box,
            text="Box overlap loss",
        ).grid(
            row=5,
            column=0,
            sticky="w",
            padx=(18, 8),
            pady=(7, 0),
        )

        ttk.Radiobutton(
            box,
            text="IoU (plain 1 - IoU; legacy face recipe)",
            variable=overlap_var,
            value=BOX_OVERLAP_LOSS_IOU,
            command=update_summary,
        ).grid(
            row=5,
            column=1,
            sticky="w",
            pady=(7, 0),
        )

        ttk.Radiobutton(
            box,
            text="CIoU (adds center-distance + aspect-ratio penalties)",
            variable=overlap_var,
            value=BOX_OVERLAP_LOSS_CIOU,
            command=update_summary,
        ).grid(
            row=5,
            column=2,
            columnspan=2,
            sticky="w",
            pady=(7, 0),
        )

        ttk.Label(
            box,
            textvariable=summary_var,
        ).grid(
            row=6,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(7, 0),
        )

        box.columnconfigure(3, weight=1)
        apply_preset()
        return box

    def add_folder_row(
        self,
        parent,
        row,
        label,
        variable,
        command,
    ):
        self.add_file_row(
            parent,
            row,
            label,
            variable,
            command,
        )

    def add_number(
        self,
        parent,
        row,
        column,
        label,
        variable,
    ):
        ttk.Label(
            parent,
            text=label,
        ).grid(
            row=row,
            column=column,
            sticky="w",
            padx=(0, 6),
            pady=4,
        )

        ttk.Entry(
            parent,
            textvariable=variable,
            width=12,
        ).grid(
            row=row,
            column=column + 1,
            sticky="w",
            padx=(0, 18),
            pady=4,
        )

    def choose_zip(self, title):
        return filedialog.askopenfilename(
            title=title,
            filetypes=[
                ("ZIP files", "*.zip"),
                ("All files", "*.*"),
            ],
        )

    def browse_train_dataset(self):
        path = self.choose_zip(
            "Choose training dataset ZIP"
        )

        if path:
            self.train_dataset_var.set(
                path
            )

    def browse_train_output(self):
        path = filedialog.askdirectory(
            title="Choose training output folder"
        )

        if path:
            self.train_output_var.set(
                path
            )

    def browse_adapt_base(self):
        path = self.choose_zip(
            "Choose base model export ZIP"
        )

        if path:
            self.adapt_base_var.set(
                path
            )

    def browse_adapt_original(self):
        path = self.choose_zip(
            "Choose original training dataset ZIP"
        )

        if path:
            self.adapt_original_var.set(
                path
            )

    def browse_adapt_real(self):
        path = self.choose_zip(
            "Choose real labeled dataset ZIP"
        )

        if path:
            self.adapt_real_var.set(
                path
            )

    def browse_adapt_output(self):
        path = filedialog.askdirectory(
            title="Choose adaptation output folder"
        )

        if path:
            self.adapt_output_var.set(
                path
            )

    def browse_deploy_cute(self):
        path = filedialog.askopenfilename(
            title="Choose Cute-YOLO .cute model",
            filetypes=[
                ("Cute-YOLO model", "*.cute"),
                ("All files", "*.*"),
            ],
        )

        if path:
            self.deploy_cute_var.set(
                path
            )

    def browse_converter(self):
        path = filedialog.askopenfilename(
            title="Choose zip_to_cute.py",
            filetypes=[
                ("Python files", "*.py"),
                ("All files", "*.*"),
            ],
        )

        if path:
            self.converter_var.set(
                path
            )

    # ========================================================
    # Config construction
    # ========================================================

    def make_converter_path(self):
        text = (
            self.converter_var
            .get()
            .strip()
        )

        return (
            Path(text)
            if text
            else None
        )

    def build_train_config(self):
        dataset_text = (
            self.train_dataset_var
            .get()
            .strip()
        )

        output_text = (
            self.train_output_var
            .get()
            .strip()
        )

        if not dataset_text:
            raise ValueError(
                "Choose a training dataset ZIP."
            )

        if not output_text:
            raise ValueError(
                "Choose an output folder."
            )

        val_fraction = (
            float(
                self.train_val_pct_var.get()
            )
            / 100.0
        )

        if not (
            0.0
            < val_fraction
            < 1.0
        ):
            raise ValueError(
                "Validation percentage must be between 0 and 100."
            )

        min_object_size_px = float(
            self.train_min_object_px_var.get()
        )
        if not (0.0 <= min_object_size_px <= IMG_SIZE):
            raise ValueError(
                f"Minimum object size must be between 0 and {IMG_SIZE} pixels."
            )

        box_overlap_loss = self.train_box_overlap_var.get().strip().lower()
        if box_overlap_loss not in {BOX_OVERLAP_LOSS_IOU, BOX_OVERLAP_LOSS_CIOU}:
            raise ValueError("Box overlap loss must be IoU or CIoU.")

        return TrainConfig(
            dataset_zip=Path(
                dataset_text
            ),
            output_dir=Path(
                output_text
            ),
            val_fraction=
                val_fraction,
            epochs=int(
                self.train_epochs_var.get()
            ),
            learning_rate=float(
                self.train_lr_var.get()
            ),
            batch_size=int(
                self.train_batch_var.get()
            ),
            make_cute=
                self.train_make_cute_var.get(),
            converter_path=
                self.make_converter_path(),
            box_overlap_loss=
                box_overlap_loss,
            loss_preset=
                self.train_loss_preset_var.get(),
            hard_negative_mining_enabled=
                self.train_hnm_var.get(),
            box_aware_negative_enabled=
                self.train_spatial_neg_var.get(),
            crowd_aware_negative_enabled=(
                self.train_crowd_var.get()
                and self.train_spatial_neg_var.get()
            ),
            min_object_size_px=
                min_object_size_px,
            augment_horizontal_flip=
                self.train_aug_flip_var.get(),
            augment_brightness=
                self.train_aug_brightness_var.get(),
            augment_contrast=
                self.train_aug_contrast_var.get(),
            shuffle_each_epoch=
                self.train_shuffle_var.get(),
        )

    def build_adapt_config(self):
        base_text = (
            self.adapt_base_var
            .get()
            .strip()
        )

        original_text = (
            self.adapt_original_var
            .get()
            .strip()
        )

        real_text = (
            self.adapt_real_var
            .get()
            .strip()
        )

        output_text = (
            self.adapt_output_var
            .get()
            .strip()
        )

        if not base_text:
            raise ValueError(
                "Choose a base model export ZIP."
            )

        if not original_text:
            raise ValueError(
                "Choose the original training dataset ZIP."
            )

        if not real_text:
            raise ValueError(
                "Choose the real labeled dataset ZIP."
            )

        if not output_text:
            raise ValueError(
                "Choose an output folder."
            )

        source_pct = float(
            self.adapt_source_pct_var.get()
        )

        target_pct = float(
            self.adapt_target_pct_var.get()
        )

        if not np.isclose(
            source_pct + target_pct,
            100.0,
        ):
            raise ValueError(
                "Original data % + Real data % must equal 100."
            )

        val_fraction = (
            float(
                self.adapt_val_pct_var.get()
            )
            / 100.0
        )

        if not (
            0.0
            < val_fraction
            < 1.0
        ):
            raise ValueError(
                "Real validation percentage must be between 0 and 100."
            )

        min_object_size_px = float(
            self.adapt_min_object_px_var.get()
        )
        if not (0.0 <= min_object_size_px <= IMG_SIZE):
            raise ValueError(
                f"Minimum object size must be between 0 and {IMG_SIZE} pixels."
            )

        box_overlap_loss = self.adapt_box_overlap_var.get().strip().lower()
        if box_overlap_loss not in {BOX_OVERLAP_LOSS_IOU, BOX_OVERLAP_LOSS_CIOU}:
            raise ValueError("Box overlap loss must be IoU or CIoU.")

        cross_loss_weight = float(
            self.adapt_cross_loss_var.get()
        )
        if not (0.0 <= cross_loss_weight <= 1.0):
            raise ValueError(
                "Cross-loss weight must be between 0 and 1."
            )

        return AdaptConfig(
            base_model_zip=Path(
                base_text
            ),
            original_data_zip=Path(
                original_text
            ),
            real_data_zip=Path(
                real_text
            ),
            output_dir=Path(
                output_text
            ),
            source_ratio=
                source_pct / 100.0,
            target_ratio=
                target_pct / 100.0,
            target_val_fraction=
                val_fraction,
            epochs=int(
                self.adapt_epochs_var.get()
            ),
            learning_rate=float(
                self.adapt_lr_var.get()
            ),
            batch_size=int(
                self.adapt_batch_var.get()
            ),
            make_cute=
                self.adapt_make_cute_var.get(),
            converter_path=
                self.make_converter_path(),
            box_overlap_loss=
                box_overlap_loss,
            specialized_finetune=
                self.adapt_specialized_var.get(),
            cross_loss_weight=
                cross_loss_weight,
            loss_preset=
                self.adapt_loss_preset_var.get(),
            hard_negative_mining_enabled=
                self.adapt_hnm_var.get(),
            box_aware_negative_enabled=
                self.adapt_spatial_neg_var.get(),
            crowd_aware_negative_enabled=(
                self.adapt_crowd_var.get()
                and self.adapt_spatial_neg_var.get()
            ),
            min_object_size_px=
                min_object_size_px,
            augment_horizontal_flip=
                self.adapt_aug_flip_var.get(),
            augment_brightness=
                self.adapt_aug_brightness_var.get(),
            augment_contrast=
                self.adapt_aug_contrast_var.get(),
            shuffle_each_epoch=
                self.adapt_shuffle_var.get(),
        )

    def build_deploy_config(self):
        cute_text = (
            self.deploy_cute_var
            .get()
            .strip()
        )

        address_text = (
            self.deploy_address_var
            .get()
            .strip()
        )

        if not cute_text:
            raise ValueError(
                "Choose a .cute model package."
            )

        cute_path = Path(
            cute_text
        )

        if cute_path.suffix.lower() != ".cute":
            raise ValueError(
                "Deployment now accepts a .cute model package directly."
            )

        return {
            "cute_path":
                cute_path,
            "address":
                (
                    address_text
                    if address_text
                    else None
                ),
        }

    # ========================================================
    # Worker handling
    # ========================================================

    def set_busy(self, busy: bool):
        state = (
            "disabled"
            if busy
            else "normal"
        )

        self.train_run_button.configure(
            state=state
        )

        self.adapt_run_button.configure(
            state=state
        )

        self.deploy_run_button.configure(
            state=state
        )

    def start_training(self):
        if (
            self.worker
            and self.worker.is_alive()
        ):
            return

        try:
            cfg = self.build_train_config()
        except Exception as exc:
            messagebox.showerror(
                "Invalid training settings",
                str(exc),
            )
            return

        self.last_output_dir = (
            cfg.output_dir
        )

        self.start_worker(
            "training",
            cfg,
        )

    def start_adaptation(self):
        if (
            self.worker
            and self.worker.is_alive()
        ):
            return

        try:
            cfg = self.build_adapt_config()
        except Exception as exc:
            messagebox.showerror(
                "Invalid adaptation settings",
                str(exc),
            )
            return

        self.last_output_dir = (
            cfg.output_dir
        )

        self.start_worker(
            "adaptation",
            cfg,
        )

    def start_deploy(self):
        if (
            self.worker
            and self.worker.is_alive()
        ):
            return

        try:
            cfg = self.build_deploy_config()
        except Exception as exc:
            messagebox.showerror(
                "Invalid deployment settings",
                str(exc),
            )
            return

        self.last_output_dir = (
            cfg["cute_path"].parent
        )

        self.start_worker(
            "deploy",
            cfg,
        )

    def start_worker(
        self,
        mode,
        cfg,
    ):
        self.progress_var.set(
            0.0
        )

        self.status_var.set(
            {
                "training":
                    "Training...",
                "adaptation":
                    "Adapting...",
                "deploy":
                    "Deploying...",
            }.get(
                mode,
                "Working...",
            )
        )

        self.set_busy(
            True
        )

        self.append_log(
            ""
        )

        self.append_log(
            "=" * 64
        )

        self.append_log(
            {
                "training":
                    "Starting base training",
                "adaptation":
                    "Starting domain adaptation",
                "deploy":
                    "Starting .cute -> BLE deployment",
            }.get(
                mode,
                "Starting operation",
            )
        )

        self.worker = threading.Thread(
            target=self.worker_main,
            args=(
                mode,
                cfg,
            ),
            daemon=True,
        )

        self.worker.start()

    def worker_main(
        self,
        mode,
        cfg,
    ):
        def logger(text):
            self.messages.put(
                (
                    "log",
                    str(text),
                )
            )

        def progress(value):
            self.messages.put(
                (
                    "progress",
                    float(value),
                )
            )

        try:
            if mode == "training":
                result = run_training(
                    cfg,
                    logger,
                    progress,
                )

            elif mode == "adaptation":
                result = run_adaptation(
                    cfg,
                    logger,
                    progress,
                )

            elif mode == "deploy":
                result = run_deploy_cute(
                    cfg["cute_path"],
                    cfg["address"],
                    logger,
                    progress,
                )

            else:
                raise ValueError(
                    f"Unknown worker mode: {mode}"
                )

            self.messages.put(
                (
                    "done",
                    (
                        mode,
                        result,
                    ),
                )
            )

        except Exception:
            self.messages.put(
                (
                    "error",
                    (
                        mode,
                        traceback.format_exc(),
                    ),
                )
            )

    # ========================================================
    # Log / status
    # ========================================================

    def append_log(self, text):
        self.log_widget.configure(
            state="normal"
        )

        self.log_widget.insert(
            "end",
            str(text) + "\n",
        )

        self.log_widget.see(
            "end"
        )

        self.log_widget.configure(
            state="disabled"
        )

    def clear_log(self):
        self.log_widget.configure(
            state="normal"
        )

        self.log_widget.delete(
            "1.0",
            "end",
        )

        self.log_widget.configure(
            state="disabled"
        )

    def poll_messages(self):
        try:
            while True:
                kind, payload = (
                    self.messages.get_nowait()
                )

                if kind == "log":
                    self.append_log(
                        payload
                    )

                elif kind == "progress":
                    self.progress_var.set(
                        payload
                    )

                elif kind == "done":
                    mode, result = payload

                    self.progress_var.set(
                        100.0
                    )

                    self.status_var.set(
                        "Complete"
                    )

                    self.set_busy(
                        False
                    )

                    cute_path = result.get(
                        "cute"
                    )

                    title = {
                        "training":
                            "Training complete",
                        "adaptation":
                            "Adaptation complete",
                        "deploy":
                            "BLE deployment complete",
                    }.get(
                        mode,
                        "Operation complete",
                    )

                    if mode == "deploy":
                        message = (
                            f"{title}.\n\n"
                            f".cute package:\n"
                            f"{cute_path}"
                        )

                    else:
                        export_zip = result[
                            "export_zip"
                        ]

                        message = (
                            f"{title}.\n\n"
                            f"Export ZIP:\n"
                            f"{export_zip}"
                        )

                        if cute_path:
                            message += (
                                "\n\n.cute package:\n"
                                f"{cute_path}"
                            )

                    messagebox.showinfo(
                        title,
                        message,
                    )

                elif kind == "error":
                    mode, trace = payload

                    self.status_var.set(
                        "Failed"
                    )

                    self.set_busy(
                        False
                    )

                    self.append_log(
                        trace
                    )

                    last_line = (
                        trace.splitlines()[-1]
                        if trace.splitlines()
                        else "Unknown error"
                    )

                    messagebox.showerror(
                        {
                            "training":
                                "Training failed",
                            "adaptation":
                                "Adaptation failed",
                            "deploy":
                                "BLE deployment failed",
                        }.get(
                            mode,
                            "Operation failed",
                        ),
                        (
                            "See the log for details.\n\n"
                            + last_line
                        ),
                    )

        except queue.Empty:
            pass

        self.root.after(
            100,
            self.poll_messages,
        )

    def current_output_dir(self):
        tab_index = self.tabs.index(
            self.tabs.select()
        )

        if tab_index == 0:
            text = (
                self.train_output_var
                .get()
                .strip()
            )

        elif tab_index == 1:
            text = (
                self.adapt_output_var
                .get()
                .strip()
            )

        else:
            text = (
                self.deploy_cute_var
                .get()
                .strip()
            )

            if text:
                return Path(text).parent

            return self.last_output_dir

        return (
            Path(text)
            if text
            else self.last_output_dir
        )

    def open_current_output_folder(self):
        path = self.current_output_dir()

        if path is None:
            return

        path = path.resolve()

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            webbrowser.open(
                path.as_uri()
            )
        except Exception:
            messagebox.showinfo(
                "Output folder",
                str(path),
            )


def main():
    print(
        "Cute-YOLO Studio"
    )

    print(
        "TensorFlow:",
        tf.__version__,
    )

    print(
        "Devices:",
        tf.config.list_physical_devices(),
    )

    for gpu in tf.config.list_physical_devices(
        "GPU"
    ):
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except Exception:
            pass

    root = tk.Tk()

    CuteYOLOStudio(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()
