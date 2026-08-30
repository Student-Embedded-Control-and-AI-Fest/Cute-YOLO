#!/usr/bin/env python3
"""
Collect the exact rendered Cute-YOLO TFT framebuffer over USB serial.

The PC does NOT redraw boxes or text. The ESP32 sends its final 160x128
RGB565 shadow framebuffer after the color preview, red detector guide,
detection boxes/confidence captions, and N/T status line are rendered.

Output:
    <out>/display/    rendered TFT PNG
    <out>/metadata/   timing + model + detection metadata

Usage:
    python tools/collect_cute_display.py /dev/ttyACM0

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
from PIL import Image
import serial


HEARTBEAT = b"RDYDISPLAY\n"
STOP = b"STOPDISPLAY\n"


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


def rgb565le_to_rgb888(raw: bytes, width: int, height: int) -> np.ndarray:
    expected = width * height * 2

    if len(raw) != expected:
        raise ValueError(
            f"RGB565 byte count mismatch: {len(raw)} vs {expected}"
        )

    pixels = np.frombuffer(raw, dtype="<u2").reshape(height, width)

    r5 = (pixels >> 11) & 0x1F
    g6 = (pixels >> 5) & 0x3F
    b5 = pixels & 0x1F

    r8 = ((r5 << 3) | (r5 >> 2)).astype(np.uint8)
    g8 = ((g6 << 2) | (g6 >> 4)).astype(np.uint8)
    b8 = ((b5 << 3) | (b5 >> 2)).astype(np.uint8)

    return np.stack((r8, g8, b8), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cute_display_dump"),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="0 = collect forever",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="nearest-neighbor PNG enlargement, e.g. 4",
    )
    args = parser.parse_args()

    dirs = {
        "display": args.out / "display",
        "metadata": args.out / "metadata",
    }

    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.10,
        write_timeout=1.0,
    ) as ser:
        time.sleep(0.4)

        print(f"Connected to {args.port} @ {args.baud}")
        print("Sending RDYDISPLAY heartbeat.")
        print("Press ESP32 BOOT for each inference.")
        print("Saving exact firmware-rendered TFT frame.")
        print("No PC-side boxes or text are added.")

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

                line = raw_line.decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                if not line:
                    continue

                if not line.startswith("DISPLAY "):
                    print(line)
                    continue

                # DISPLAY <id> <label> <inference_us> <count>
                parts = line.split()

                if len(parts) != 5:
                    print(f"Malformed DISPLAY header: {line}")
                    continue

                frame_id = int(parts[1])
                label = parts[2]
                inference_us = int(parts[3])
                expected_count = int(parts[4])

                frame_header = ser.readline().decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                frame_parts = frame_header.split()

                if len(frame_parts) != 5 or frame_parts[0] != "FRAME":
                    raise RuntimeError(
                        f"Expected FRAME header, got: {frame_header}"
                    )

                width = int(frame_parts[1])
                height = int(frame_parts[2])
                encoding = frame_parts[3]
                nbytes = int(frame_parts[4])

                if encoding != "RGB565LE":
                    raise RuntimeError(
                        f"Unsupported encoding: {encoding}"
                    )

                if nbytes != width * height * 2:
                    raise RuntimeError(
                        f"RGB565 byte count mismatch: "
                        f"{nbytes} vs {width * height * 2}"
                    )

                raw = read_exact(ser, nbytes)

                delimiter = read_exact(ser, 1)

                if delimiter != b"\n":
                    raise RuntimeError(
                        f"Unexpected binary delimiter: {delimiter!r}"
                    )

                det_header = ser.readline().decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                det_parts = det_header.split()

                if len(det_parts) != 2 or det_parts[0] != "DET":
                    raise RuntimeError(
                        f"Expected DET header, got: {det_header}"
                    )

                det_count = int(det_parts[1])
                boxes = []

                for _ in range(det_count):
                    box_line = ser.readline().decode(
                        "utf-8",
                        errors="replace",
                    ).strip()

                    box_parts = box_line.split()

                    if len(box_parts) != 6 or box_parts[0] != "BOX":
                        raise RuntimeError(
                            f"Malformed BOX: {box_line}"
                        )

                    boxes.append({
                        "confidence": float(box_parts[1]),
                        "x1": float(box_parts[2]),
                        "y1": float(box_parts[3]),
                        "x2": float(box_parts[4]),
                        "y2": float(box_parts[5]),
                    })

                end_line = ser.readline().decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                if end_line != "ENDDISPLAY":
                    raise RuntimeError(
                        f"Expected ENDDISPLAY, got: {end_line}"
                    )

                rgb = rgb565le_to_rgb888(raw, width, height)

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                stem = f"{frame_id:06d}_{stamp}_{sanitize(label)}"

                display_path = dirs["display"] / f"{stem}.png"
                metadata_path = dirs["metadata"] / f"{stem}.json"

                image = Image.fromarray(rgb, mode="RGB")

                if args.scale > 1:
                    image = image.resize(
                        (width * args.scale, height * args.scale),
                        Image.Resampling.NEAREST,
                    )

                image.save(display_path)

                metadata = {
                    "frame_id": frame_id,
                    "label": label,
                    "inference_us": inference_us,
                    "inference_ms": inference_us / 1000.0,
                    "display_width": width,
                    "display_height": height,
                    "display_encoding": encoding,
                    "expected_detection_count": expected_count,
                    "received_detection_count": det_count,
                    "boxes": boxes,
                    "display_png": str(display_path),
                    "pc_redraw": False,
                }

                metadata_path.write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )

                saved += 1

                print(
                    f"Saved #{saved}: {display_path.name}  "
                    f"N={det_count}  T={inference_us/1000.0:.1f} ms"
                )

                ser.write(HEARTBEAT)
                ser.flush()
                last_heartbeat = time.monotonic()

        except KeyboardInterrupt:
            print("\nStopping exact-display collector.")

            try:
                ser.write(STOP)
                ser.flush()
            except Exception:
                pass


if __name__ == "__main__":
    main()
