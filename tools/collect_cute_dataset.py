#!/usr/bin/env python3
"""
Collect real-world Cute-YOLO detector inputs over USB serial.

The ESP32 sends ONE clean 128x128 GRAY8 image plus predicted boxes.
This script then saves:

    <out>/images/         clean PNG used for training
    <out>/labeled/        receiver-drawn preview
    <out>/pseudo_labels/  YOLO-format predicted boxes
    <out>/metadata/       JSON with confidences/model/timing

The predicted boxes are pseudo-labels, NOT verified ground truth.
Review/correct them before using them as final labels.

Usage:
    python tools/collect_cute_dataset.py /dev/ttyACM0

Dependencies:
    pip install pyserial pillow numpy
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import serial


HEARTBEAT = b"RDYSAMPLE\n"
STOP = b"STOPSAMPLE\n"


def read_exact(ser: serial.Serial, n: int) -> bytes:
    out = bytearray()
    deadline = time.monotonic() + 10.0

    while len(out) < n:
        chunk = ser.read(n - len(out))
        if chunk:
            out.extend(chunk)
            deadline = time.monotonic() + 10.0
            continue

        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Serial binary timeout: got {len(out)} / {n} bytes"
            )

    return bytes(out)


def sanitize(text: str) -> str:
    keep = []
    for c in text:
        if c.isalnum() or c in "-_.":
            keep.append(c)
        else:
            keep.append("_")
    return "".join(keep).strip("_.") or "object"


def draw_preview(gray: np.ndarray, boxes: list[dict], label: str) -> Image.Image:
    image = Image.fromarray(gray, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    w, h = image.size

    # Distinct high-contrast colors; receiver-side only.
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 160, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for i, box in enumerate(boxes):
        x1 = int(round(box["x1"] * (w - 1)))
        y1 = int(round(box["y1"] * (h - 1)))
        x2 = int(round(box["x2"] * (w - 1)))
        y2 = int(round(box["y2"] * (h - 1)))

        color = colors[i % len(colors)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

        #caption = f"{label} {box['confidence']:.2f}"
        #caption = f"{box['confidence']:.2f}"
        caption = f"{100 * box['confidence']:.0f}%"
        try:
            bbox = draw.textbbox((0, 0), caption, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(caption, font=font)

        ty = max(0, y1 - th - 2)
        draw.rectangle((x1, ty, x1 + tw + 2, ty + th + 2), fill=color)
        draw.text((x1 + 1, ty + 1), caption, fill=(0, 0, 0), font=font)

    return image


def yolo_line(box: dict) -> str:
    cx = 0.5 * (box["x1"] + box["x2"])
    cy = 0.5 * (box["y1"] + box["y2"])
    bw = max(0.0, box["x2"] - box["x1"])
    bh = max(0.0, box["y2"] - box["y1"])

    # Single-class model => class id 0.
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--out", type=Path, default=Path("cute_real_dataset"))
    parser.add_argument("--count", type=int, default=0, help="0 = collect forever")
    parser.add_argument("--heartbeat", type=float, default=1.0)
    args = parser.parse_args()

    dirs = {
        "images": args.out / "images",
        "labeled": args.out / "labeled",
        "pseudo": args.out / "pseudo_labels",
        "metadata": args.out / "metadata",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.10,
        write_timeout=1.0,
    ) as ser:
        time.sleep(0.4)

        print(f"Connected to {args.port} @ {args.baud}")
        print("Sending RDYSAMPLE heartbeat.")
        print("Press ESP32 BOOT for each sample.")
        print("Raw image + receiver-drawn preview will both be saved.")

        last_heartbeat = 0.0
        saved = 0

        try:
            while args.count == 0 or saved < args.count:
                now = time.monotonic()

                if now - last_heartbeat >= args.heartbeat:
                    ser.write(HEARTBEAT)
                    ser.flush()
                    last_heartbeat = now

                raw_line = ser.readline()
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if not line.startswith("SAMPLE "):
                    print(line)
                    continue

                # SAMPLE <id> <label> <inference_us> <count>
                parts = line.split()
                if len(parts) != 5:
                    print(f"Malformed SAMPLE header: {line}")
                    continue

                sample_id = int(parts[1])
                label = parts[2]
                inference_us = int(parts[3])
                expected_count = int(parts[4])

                raw_header = ser.readline().decode("utf-8", errors="replace").strip()
                raw_parts = raw_header.split()

                if len(raw_parts) != 5 or raw_parts[0] != "RAW":
                    raise RuntimeError(f"Expected RAW header, got: {raw_header}")

                width = int(raw_parts[1])
                height = int(raw_parts[2])
                encoding = raw_parts[3]
                nbytes = int(raw_parts[4])

                if encoding != "GRAY8":
                    raise RuntimeError(f"Unsupported encoding: {encoding}")

                if nbytes != width * height:
                    raise RuntimeError(
                        f"GRAY8 byte count mismatch: {nbytes} vs {width*height}"
                    )

                raw = read_exact(ser, nbytes)

                # Consume fixed binary delimiter newline.
                delimiter = read_exact(ser, 1)
                if delimiter != b"\n":
                    raise RuntimeError(f"Unexpected binary delimiter: {delimiter!r}")

                det_header = ser.readline().decode("utf-8", errors="replace").strip()
                det_parts = det_header.split()

                if len(det_parts) != 2 or det_parts[0] != "DET":
                    raise RuntimeError(f"Expected DET header, got: {det_header}")

                det_count = int(det_parts[1])
                boxes = []

                for _ in range(det_count):
                    box_line = ser.readline().decode("utf-8", errors="replace").strip()
                    box_parts = box_line.split()

                    if len(box_parts) != 6 or box_parts[0] != "BOX":
                        raise RuntimeError(f"Malformed BOX: {box_line}")

                    boxes.append({
                        "confidence": float(box_parts[1]),
                        "x1": float(box_parts[2]),
                        "y1": float(box_parts[3]),
                        "x2": float(box_parts[4]),
                        "y2": float(box_parts[5]),
                    })

                end_line = ser.readline().decode("utf-8", errors="replace").strip()
                if end_line != "END":
                    raise RuntimeError(f"Expected END, got: {end_line}")

                gray = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                stem = f"{sample_id:06d}_{stamp}_{sanitize(label)}"

                raw_path = dirs["images"] / f"{stem}.png"
                labeled_path = dirs["labeled"] / f"{stem}_labeled.png"
                pseudo_path = dirs["pseudo"] / f"{stem}.txt"
                metadata_path = dirs["metadata"] / f"{stem}.json"

                Image.fromarray(gray, mode="L").save(raw_path)
                draw_preview(gray, boxes, label).save(labeled_path)

                # Empty file is intentional for a zero-detection negative frame.
                pseudo_text = "\n".join(yolo_line(box) for box in boxes)
                if pseudo_text:
                    pseudo_text += "\n"
                pseudo_path.write_text(pseudo_text, encoding="utf-8")

                metadata = {
                    "sample_id": sample_id,
                    "label": label,
                    "inference_us": inference_us,
                    "inference_ms": inference_us / 1000.0,
                    "width": width,
                    "height": height,
                    "expected_detection_count": expected_count,
                    "received_detection_count": det_count,
                    "boxes": boxes,
                    "pseudo_label": True,
                    "raw_image": str(raw_path),
                    "labeled_preview": str(labeled_path),
                }

                metadata_path.write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )

                saved += 1

                print(
                    f"Saved #{saved}: {raw_path.name}  "
                    f"N={det_count}  T={inference_us/1000.0:.1f} ms"
                )

                # Re-arm immediately after successful receive.
                ser.write(HEARTBEAT)
                ser.flush()
                last_heartbeat = time.monotonic()

        except KeyboardInterrupt:
            print("\nStopping collector.")
            try:
                ser.write(STOP)
                ser.flush()
            except Exception:
                pass


if __name__ == "__main__":
    main()
