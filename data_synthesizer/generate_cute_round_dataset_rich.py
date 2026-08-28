#!/usr/bin/env python3
"""
Generate a rich single-class YOLO dataset of synthetic round objects.

Design goal:
- zero configuration except number of images
- automatic coverage from:
    * dark -> bright backgrounds
    * matte -> shiny objects
    * scarce -> very crowded scenes
    * tiny -> large objects
    * sparse -> packed / touching layouts
- automatic export in the Cute-YOLO folder format:
    <out>/
        classes.txt
        images/
        labels/
        labeled/
        metadata/
- automatic ZIP creation for direct use in the training notebook

Usage:
    python generate_cute_round_dataset_rich.py 5000

Dependencies:
    pip install numpy opencv-python pillow
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# Fixed configuration
# ============================================================

IMG_SIZE = 128
CLASS_NAME = "round_object"

# "Very small but still allowed"
MIN_RADIUS = 4

# Large but still useful
MAX_RADIUS = 28

# Dense stress case
MAX_OBJECTS = 32

NEGATIVE_IMAGE_PROBABILITY = 0.08

# Visible area filter for partially cropped objects
MIN_RETAINED_AREA = 0.45

# Output root name. A timestamp is appended automatically.
OUT_PREFIX = "round_object_synth"

# Preview/metadata are always written for inspectability.
SAVE_PREVIEW = True
SAVE_METADATA = True


# ============================================================
# Scenario definitions
# ============================================================

@dataclass
class Scenario:
    name: str
    probability: float
    bg_min: int
    bg_max: int
    count_min: int
    count_max: int
    radius_min: int
    radius_max: int
    shiny_min: float
    shiny_max: float
    crop_bias: float
    touching_bias: float
    packed_bias: float
    mixed_scale: bool


SCENARIOS = [
    Scenario(
        name="easy_sparse",
        probability=0.12,
        bg_min=35,
        bg_max=160,
        count_min=1,
        count_max=4,
        radius_min=8,
        radius_max=18,
        shiny_min=0.00,
        shiny_max=0.25,
        crop_bias=0.05,
        touching_bias=0.00,
        packed_bias=0.00,
        mixed_scale=False,
    ),
    Scenario(
        name="bright_background",
        probability=0.12,
        bg_min=170,
        bg_max=245,
        count_min=2,
        count_max=8,
        radius_min=7,
        radius_max=18,
        shiny_min=0.10,
        shiny_max=0.60,
        crop_bias=0.10,
        touching_bias=0.10,
        packed_bias=0.05,
        mixed_scale=True,
    ),
    Scenario(
        name="dark_background",
        probability=0.12,
        bg_min=5,
        bg_max=70,
        count_min=2,
        count_max=8,
        radius_min=7,
        radius_max=18,
        shiny_min=0.00,
        shiny_max=0.70,
        crop_bias=0.10,
        touching_bias=0.10,
        packed_bias=0.05,
        mixed_scale=True,
    ),
    Scenario(
        name="shiny_objects",
        probability=0.10,
        bg_min=60,
        bg_max=200,
        count_min=2,
        count_max=10,
        radius_min=7,
        radius_max=20,
        shiny_min=0.70,
        shiny_max=1.00,
        crop_bias=0.08,
        touching_bias=0.12,
        packed_bias=0.08,
        mixed_scale=True,
    ),
    Scenario(
        name="mixed_sizes",
        probability=0.10,
        bg_min=30,
        bg_max=220,
        count_min=4,
        count_max=12,
        radius_min=MIN_RADIUS,
        radius_max=MAX_RADIUS,
        shiny_min=0.00,
        shiny_max=1.00,
        crop_bias=0.10,
        touching_bias=0.12,
        packed_bias=0.08,
        mixed_scale=True,
    ),
    Scenario(
        name="tiny_object_stress",
        probability=0.10,
        bg_min=25,
        bg_max=220,
        count_min=8,
        count_max=20,
        radius_min=MIN_RADIUS,
        radius_max=9,
        shiny_min=0.00,
        shiny_max=0.80,
        crop_bias=0.08,
        touching_bias=0.18,
        packed_bias=0.08,
        mixed_scale=False,
    ),
    Scenario(
        name="large_object_stress",
        probability=0.10,
        bg_min=20,
        bg_max=220,
        count_min=1,
        count_max=6,
        radius_min=18,
        radius_max=MAX_RADIUS,
        shiny_min=0.10,
        shiny_max=1.00,
        crop_bias=0.20,
        touching_bias=0.20,
        packed_bias=0.10,
        mixed_scale=True,
    ),
    Scenario(
        name="touching_gap_confusing",
        probability=0.10,
        bg_min=30,
        bg_max=220,
        count_min=8,
        count_max=18,
        radius_min=7,
        radius_max=18,
        shiny_min=0.00,
        shiny_max=1.00,
        crop_bias=0.12,
        touching_bias=0.65,
        packed_bias=0.18,
        mixed_scale=True,
    ),
    Scenario(
        name="crowded",
        probability=0.08,
        bg_min=25,
        bg_max=220,
        count_min=14,
        count_max=24,
        radius_min=5,
        radius_max=16,
        shiny_min=0.00,
        shiny_max=1.00,
        crop_bias=0.12,
        touching_bias=0.45,
        packed_bias=0.35,
        mixed_scale=True,
    ),
    Scenario(
        name="packed32",
        probability=0.06,
        bg_min=30,
        bg_max=220,
        count_min=24,
        count_max=32,
        radius_min=4,
        radius_max=14,
        shiny_min=0.00,
        shiny_max=1.00,
        crop_bias=0.12,
        touching_bias=0.75,
        packed_bias=0.75,
        mixed_scale=True,
    ),
]

SCENARIO_NAMES = [s.name for s in SCENARIOS]
SCENARIO_PROBS = np.array([s.probability for s in SCENARIOS], dtype=np.float64)
SCENARIO_PROBS = SCENARIO_PROBS / SCENARIO_PROBS.sum()


# ============================================================
# Utility functions
# ============================================================

def clip_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def sigmoid01(x):
    return 1.0 / (1.0 + np.exp(-x))


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def yolo_line(box):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return f"0 {cx/IMG_SIZE:.6f} {cy/IMG_SIZE:.6f} {w/IMG_SIZE:.6f} {h/IMG_SIZE:.6f}"


def box_visible_fraction(cx, cy, r):
    full = math.pi * r * r
    if full <= 1e-6:
        return 0.0
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    return float(mask.sum() / 255.0) / full


def choose_scenario(rng):
    idx = rng.choice(len(SCENARIOS), p=SCENARIO_PROBS)
    return SCENARIOS[int(idx)]


# ============================================================
# Background generation
# ============================================================

def make_background(rng, bg_min, bg_max):
    base = rng.uniform(bg_min, bg_max)
    img = np.full((IMG_SIZE, IMG_SIZE), base, dtype=np.float32)

    # Illumination gradients
    gx = np.linspace(-1.0, 1.0, IMG_SIZE, dtype=np.float32)[None, :]
    gy = np.linspace(-1.0, 1.0, IMG_SIZE, dtype=np.float32)[:, None]

    img += gx * rng.uniform(-40, 40)
    img += gy * rng.uniform(-40, 40)

    # Large soft blobs
    for _ in range(int(rng.integers(1, 5))):
        cx = rng.uniform(0, IMG_SIZE - 1)
        cy = rng.uniform(0, IMG_SIZE - 1)
        sigma = rng.uniform(15, 45)
        amp = rng.uniform(-30, 30)
        xx, yy = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
        img += amp * blob.astype(np.float32)

    # Textured noise at multiple scales
    for sigma, amp in [(1.5, 5.0), (4.0, 8.0), (9.0, 11.0)]:
        noise = rng.normal(0.0, 1.0, size=(IMG_SIZE, IMG_SIZE)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
        img += noise * amp * rng.uniform(0.8, 1.2)

    # Occasional linear streaks / folds
    img8 = clip_uint8(img)
    for _ in range(int(rng.integers(1, 6))):
        p1 = (int(rng.integers(0, IMG_SIZE)), int(rng.integers(0, IMG_SIZE)))
        p2 = (int(rng.integers(0, IMG_SIZE)), int(rng.integers(0, IMG_SIZE)))
        color = int(np.clip(np.mean(img8) + rng.integers(-20, 21), 0, 255))
        thickness = int(rng.integers(1, 3))
        cv2.line(img8, p1, p2, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    img = img8.astype(np.float32)

    # Slight blur to look more photographic
    sigma = rng.uniform(0.0, 1.2)
    if sigma > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return clip_uint8(img)


# ============================================================
# Object rendering
# ============================================================

def render_round_object(canvas, rng, cx, cy, radius, shiny_strength, edge_softness, contrast_mode):
    """
    Render a synthetic round object into a grayscale canvas.
    """
    h, w = canvas.shape
    x0 = max(0, int(math.floor(cx - radius - 3)))
    x1 = min(w, int(math.ceil(cx + radius + 4)))
    y0 = max(0, int(math.floor(cy - radius - 3)))
    y1 = min(h, int(math.ceil(cy + radius + 4)))
    if x1 <= x0 or y1 <= y0:
        return

    patch = canvas[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    rr = np.sqrt(dx * dx + dy * dy)
    norm = rr / max(radius, 1e-6)

    # Soft circular mask
    soft = max(0.9, edge_softness)
    alpha = 1.0 - sigmoid01((norm - 1.0) * (7.5 / soft))
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    # Decide whether object is darker or brighter than local background.
    local_bg = float(np.mean(patch))
    if contrast_mode == "lighter":
        base_obj = local_bg + rng.uniform(35, 95)
    elif contrast_mode == "darker":
        base_obj = local_bg - rng.uniform(35, 95)
    else:
        # Mixed / difficult contrast
        sign = 1.0 if rng.random() < 0.5 else -1.0
        base_obj = local_bg + sign * rng.uniform(15, 70)

    # Matte body
    body = np.full_like(patch, base_obj, dtype=np.float32)

    # Subtle radial shading
    radial = (1.0 - np.clip(norm, 0.0, 1.2))
    body += radial * rng.uniform(-18, 18)

    # Directional lighting
    light_angle = rng.uniform(0.0, 2.0 * math.pi)
    light_dir_x = math.cos(light_angle)
    light_dir_y = math.sin(light_angle)
    directional = (dx / (radius + 1e-6)) * light_dir_x + (dy / (radius + 1e-6)) * light_dir_y
    body += directional * rng.uniform(-16, 16)

    # Fine texture
    tex = rng.normal(0.0, 1.0, size=patch.shape).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), sigmaX=rng.uniform(0.5, 1.5), sigmaY=rng.uniform(0.5, 1.5))
    body += tex * rng.uniform(1.0, 6.0)

    # Shiny highlight
    if shiny_strength > 0.02:
        hx = cx + rng.uniform(-0.35, 0.35) * radius
        hy = cy + rng.uniform(-0.35, 0.35) * radius
        hs = rng.uniform(0.16, 0.42) * radius
        highlight = np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2.0 * hs * hs))
        body += highlight.astype(np.float32) * rng.uniform(25, 105) * shiny_strength

        # Secondary small highlight
        if rng.random() < 0.55:
            hx2 = hx + rng.uniform(-0.18, 0.18) * radius
            hy2 = hy + rng.uniform(-0.18, 0.18) * radius
            hs2 = rng.uniform(0.08, 0.20) * radius
            highlight2 = np.exp(-((xx - hx2) ** 2 + (yy - hy2) ** 2) / (2.0 * hs2 * hs2))
            body += highlight2.astype(np.float32) * rng.uniform(15, 70) * shiny_strength

    # Rim / edge cue
    rim = np.exp(-((norm - rng.uniform(0.80, 0.95)) ** 2) / (2.0 * rng.uniform(0.01, 0.04)))
    body += rim.astype(np.float32) * rng.uniform(-16, 16)

    # Slight object shadow / local darkening outside object
    if rng.random() < 0.8:
        sx = cx + rng.uniform(-0.12, 0.12) * radius
        sy = cy + rng.uniform(-0.12, 0.12) * radius
        srad = radius * rng.uniform(0.95, 1.15)
        shadow = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * (0.70 * srad) ** 2))
        patch -= shadow.astype(np.float32) * rng.uniform(2.0, 8.0)

    merged = patch * (1.0 - alpha) + body * alpha
    canvas[y0:y1, x0:x1] = clip_uint8(merged)


# ============================================================
# Scene generation
# ============================================================

def sample_radius(rng, scenario):
    """Sample a radius without ever creating an invalid NumPy interval.

    Mixed-scale scenarios try tiny / medium / large sub-ranges, but some
    scenarios intentionally live entirely outside one of those bands
    (for example, large_object_stress starts at radius 18).  If the chosen
    band does not intersect the scenario's legal radius interval, fall back
    to the full scenario interval.
    """

    def sample_from_band(band_lo, band_hi):
        lo = max(int(scenario.radius_min), int(band_lo))
        hi = min(int(scenario.radius_max), int(band_hi))

        if lo > hi:
            lo = int(scenario.radius_min)
            hi = int(scenario.radius_max)

        return int(rng.integers(lo, hi + 1))

    if scenario.mixed_scale and rng.random() < 0.70:
        u = rng.random()

        # Encourages both tiny and large objects, not just middle sizes.
        if u < 0.25:
            return sample_from_band(MIN_RADIUS, 10)
        elif u < 0.55:
            return sample_from_band(8, 16)
        else:
            return sample_from_band(14, MAX_RADIUS)

    return int(
        rng.integers(
            int(scenario.radius_min),
            int(scenario.radius_max) + 1,
        )
    )


def sample_center(rng, radius, crop_bias):
    if rng.random() < crop_bias:
        # Allow partial objects near the borders.
        margin = radius * 0.75
        cx = rng.uniform(-margin, IMG_SIZE - 1 + margin)
        cy = rng.uniform(-margin, IMG_SIZE - 1 + margin)
    else:
        cx = rng.uniform(radius + 1, IMG_SIZE - radius - 2)
        cy = rng.uniform(radius + 1, IMG_SIZE - radius - 2)
    return cx, cy


def generate_objects(rng, scenario, n_objects):
    accepted = []
    objects = []

    # Precompute per-scene layout behavior.
    if scenario.packed_bias >= 0.5:
        base_gap_min = -0.20
        max_trials_per_object = 250
    elif scenario.touching_bias >= 0.45:
        base_gap_min = -0.08
        max_trials_per_object = 180
    else:
        base_gap_min = 1.0
        max_trials_per_object = 120

    trials = 0
    target = int(n_objects)

    while len(objects) < target and trials < target * max_trials_per_object:
        trials += 1

        radius = sample_radius(rng, scenario)
        cx, cy = sample_center(rng, radius, scenario.crop_bias)

        # Visibility gate for cropped objects
        visible_fraction = box_visible_fraction(cx, cy, radius)
        if visible_fraction < MIN_RETAINED_AREA:
            continue

        ok = True
        for acx, acy, ar in accepted:
            d = math.hypot(cx - acx, cy - acy)
            local_gap = d - (radius + ar)

            # For touching scenarios, some objects are encouraged to be almost touching.
            if scenario.touching_bias > 0 and rng.random() < scenario.touching_bias:
                required_gap = base_gap_min * min(radius, ar)
            else:
                required_gap = max(0.6, 0.04 * min(radius, ar))

            if local_gap < required_gap:
                ok = False
                break

        if not ok:
            continue

        shiny = float(rng.uniform(scenario.shiny_min, scenario.shiny_max))
        edge_softness = float(rng.uniform(0.9, 1.6))

        # Harder scenes sometimes use low contrast.
        local_contrast_roll = rng.random()
        if local_contrast_roll < 0.33:
            contrast_mode = "lighter"
        elif local_contrast_roll < 0.66:
            contrast_mode = "darker"
        else:
            contrast_mode = "mixed"

        x1 = max(0.0, cx - radius)
        y1 = max(0.0, cy - radius)
        x2 = min(float(IMG_SIZE), cx + radius)
        y2 = min(float(IMG_SIZE), cy + radius)

        if (x2 - x1) < 2 * MIN_RADIUS or (y2 - y1) < 2 * MIN_RADIUS:
            # Prevent degenerate tiny visible remnants.
            continue

        objects.append({
            "cx": float(cx),
            "cy": float(cy),
            "radius": float(radius),
            "bbox": [x1, y1, x2, y2],
            "shiny": shiny,
            "edge_softness": edge_softness,
            "contrast_mode": contrast_mode,
        })
        accepted.append((cx, cy, radius))

    return objects


def draw_preview(gray_image, boxes):
    image = Image.fromarray(gray_image, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
    return image


def make_sample(rng, scenario):
    negative = bool(rng.random() < NEGATIVE_IMAGE_PROBABILITY)

    bg = make_background(rng, scenario.bg_min, scenario.bg_max)
    image = bg.copy()

    if negative:
        # Hard negatives still receive object-like distractors.
        for _ in range(int(rng.integers(0, 5))):
            x1 = int(rng.integers(0, IMG_SIZE - 10))
            y1 = int(rng.integers(0, IMG_SIZE - 10))
            x2 = int(min(IMG_SIZE - 1, x1 + rng.integers(6, 24)))
            y2 = int(min(IMG_SIZE - 1, y1 + rng.integers(6, 24)))
            color = int(np.clip(np.mean(image) + rng.integers(-60, 61), 0, 255))
            if rng.random() < 0.5:
                cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=-1, lineType=cv2.LINE_AA)
            else:
                cv2.ellipse(
                    image,
                    ((x1 + x2) // 2, (y1 + y2) // 2),
                    (max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)),
                    angle=float(rng.uniform(0, 180)),
                    startAngle=0,
                    endAngle=360,
                    color=color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
        return image, [], {
            "scenario": scenario.name,
            "negative_image": True,
            "requested_objects": 0,
            "placed_objects": 0,
        }

    n_objects = int(rng.integers(scenario.count_min, scenario.count_max + 1))
    objects = generate_objects(rng, scenario, n_objects)

    # Draw larger objects first so small objects stay visible.
    objects_sorted = sorted(objects, key=lambda d: d["radius"], reverse=True)
    for obj in objects_sorted:
        render_round_object(
            image,
            rng,
            obj["cx"],
            obj["cy"],
            obj["radius"],
            obj["shiny"],
            obj["edge_softness"],
            obj["contrast_mode"],
        )

    # Global post effects
    if rng.random() < 0.8:
        sigma = float(rng.uniform(0.0, 1.0))
        if sigma > 0.05:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if rng.random() < 0.6:
        # Sensor-like noise
        noise = rng.normal(0.0, rng.uniform(1.0, 6.0), size=image.shape).astype(np.float32)
        image = clip_uint8(image.astype(np.float32) + noise)

    if rng.random() < 0.4:
        # Slight exposure scaling
        image = clip_uint8(image.astype(np.float32) * rng.uniform(0.88, 1.12) + rng.uniform(-8, 8))

    boxes = [obj["bbox"] for obj in objects]
    metadata = {
        "scenario": scenario.name,
        "negative_image": False,
        "requested_objects": n_objects,
        "placed_objects": len(objects),
        "radius_min": float(min((o["radius"] for o in objects), default=0.0)),
        "radius_max": float(max((o["radius"] for o in objects), default=0.0)),
        "mean_shiny": float(np.mean([o["shiny"] for o in objects])) if objects else 0.0,
    }

    return image, boxes, metadata


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_cute_round_dataset_rich.py <num_images>")
        raise SystemExit(1)

    try:
        count = int(sys.argv[1])
    except ValueError:
        raise SystemExit("num_images must be an integer")

    if count <= 0:
        raise SystemExit("num_images must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(f"{OUT_PREFIX}_{timestamp}")
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    labeled_dir = out_root / "labeled"
    metadata_dir = out_root / "metadata"

    ensure_clean_dir(out_root)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_PREVIEW:
        labeled_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_METADATA:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    (out_root / "classes.txt").write_text(CLASS_NAME + "\n", encoding="utf-8")

    rng = np.random.default_rng(123)

    stats = {
        "total_boxes": 0,
        "negative_images": 0,
        "scenario_histogram": {name: 0 for name in SCENARIO_NAMES},
    }

    digits = max(6, len(str(count)))

    for idx in range(count):
        scenario = choose_scenario(rng)
        stats["scenario_histogram"][scenario.name] += 1

        image, boxes, meta = make_sample(rng, scenario)

        if not boxes:
            stats["negative_images"] += 1
        stats["total_boxes"] += len(boxes)

        stem = f"{idx + 1:0{digits}d}"

        image_path = images_dir / f"{stem}.png"
        label_path = labels_dir / f"{stem}.txt"
        preview_path = labeled_dir / f"{stem}_labeled.png"
        meta_path = metadata_dir / f"{stem}.json"

        Image.fromarray(image, mode="L").save(image_path)

        label_text = "\n".join(yolo_line(b) for b in boxes)
        if label_text:
            label_text += "\n"
        label_path.write_text(label_text, encoding="utf-8")

        if SAVE_PREVIEW:
            draw_preview(image, boxes).save(preview_path)

        if SAVE_METADATA:
            meta.update({
                "image": str(image_path.name),
                "label": str(label_path.name),
                "num_boxes": len(boxes),
                "boxes_xyxy": boxes,
            })
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if (idx + 1) % max(1, min(500, count // 10 if count >= 10 else 1)) == 0 or idx == count - 1:
            print(f"[{idx+1}/{count}] generated")

    zip_path = out_root.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in out_root.rglob("*"):
            zf.write(path, arcname=path.relative_to(out_root.parent))

    print("\nDone.")
    print(f"Dataset folder : {out_root}")
    print(f"ZIP file       : {zip_path}")
    print(f"Class          : {CLASS_NAME}")
    print(f"Images         : {count}")
    print(f"Boxes total    : {stats['total_boxes']}")
    print(f"Negative imgs  : {stats['negative_images']}")
    print("\nScenario histogram:")
    for name, value in stats["scenario_histogram"].items():
        print(f"  {name:24s} : {value}")


if __name__ == "__main__":
    main()
