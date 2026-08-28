#!/usr/bin/env python3
"""
Convert a Cute-YOLO TensorFlow/Colab export ZIP into one .cute model blob.

Typical use:
    python cute_zip_to_cute.py cute_yolo_circle_tensorflow_fixed_8_24_export.zip

Optional:
    python cute_zip_to_cute.py model_export.zip --out face.cute
    python cute_zip_to_cute.py model_export.zip --label face
    python cute_zip_to_cute.py model_export.zip --confidence 0.75 --nms 0.35

Expected files inside the ZIP:
    export/model_weights_int8.h
    export/runtime_config.json
    export/*_deployment_manifest.json

The resulting .cute file is intended for the fixed Cute-YOLO ESP32-S3
firmware using BLE + raw A/B model partitions + esp_partition_mmap().
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import zipfile
import zlib
from pathlib import Path


# ---------------------------------------------------------------------
# Fixed Cute-YOLO architecture contract
# ---------------------------------------------------------------------

HEADER_BYTES = 1024
FORMAT_VERSION = 1

ARCH_NAME = "cute_yolo_fixed_dualcore_hybrid_8_24"
ARCH_ID = zlib.crc32(ARCH_NAME.encode("ascii")) & 0xFFFFFFFF

# 3 stems split across cores = 6 parameter groups
# 5 hybrid blocks x (Conv + DW + PW) = 15
# head = 1
LAYER_COUNT = 22

BASE_FMT = "<4sHHIIIHH24sffffB3sfifi40sI"
RECORD_FMT = "<IIIIIIffii"

assert struct.calcsize(BASE_FMT) == 128
assert struct.calcsize(RECORD_FMT) == 40
assert 128 + LAYER_COUNT * 40 + 16 == HEADER_BYTES


# ---------------------------------------------------------------------
# Parameter groups in the exact order expected by the firmware
# ---------------------------------------------------------------------

LAYER_SPECS = [
    ("s1a", "w01a", "b01a", "m01a", "s01a",
     "CUTE_YOLO_INPUT_SCALE", "CUTE_YOLO_INPUT_ZERO_POINT",
     "CUTE_YOLO_STEM1_OUTPUT_SCALE", "CUTE_YOLO_STEM1_OUTPUT_ZERO_POINT"),

    ("s1b", "w01b", "b01b", "m01b", "s01b",
     "CUTE_YOLO_INPUT_SCALE", "CUTE_YOLO_INPUT_ZERO_POINT",
     "CUTE_YOLO_STEM1_OUTPUT_SCALE", "CUTE_YOLO_STEM1_OUTPUT_ZERO_POINT"),

    ("s2a", "w02a", "b02a", "m02a", "s02a",
     "CUTE_YOLO_STEM1_OUTPUT_SCALE", "CUTE_YOLO_STEM1_OUTPUT_ZERO_POINT",
     "CUTE_YOLO_STEM2_OUTPUT_SCALE", "CUTE_YOLO_STEM2_OUTPUT_ZERO_POINT"),

    ("s2b", "w02b", "b02b", "m02b", "s02b",
     "CUTE_YOLO_STEM1_OUTPUT_SCALE", "CUTE_YOLO_STEM1_OUTPUT_ZERO_POINT",
     "CUTE_YOLO_STEM2_OUTPUT_SCALE", "CUTE_YOLO_STEM2_OUTPUT_ZERO_POINT"),

    ("s3a", "w03a", "b03a", "m03a", "s03a",
     "CUTE_YOLO_STEM2_OUTPUT_SCALE", "CUTE_YOLO_STEM2_OUTPUT_ZERO_POINT",
     "CUTE_YOLO_STEM3_OUTPUT_SCALE", "CUTE_YOLO_STEM3_OUTPUT_ZERO_POINT"),

    ("s3b", "w03b", "b03b", "m03b", "s03b",
     "CUTE_YOLO_STEM2_OUTPUT_SCALE", "CUTE_YOLO_STEM2_OUTPUT_ZERO_POINT",
     "CUTE_YOLO_STEM3_OUTPUT_SCALE", "CUTE_YOLO_STEM3_OUTPUT_ZERO_POINT"),
]

for i in range(1, 6):
    prev_scale = (
        "CUTE_YOLO_STEM3_OUTPUT_SCALE"
        if i == 1 else
        f"CUTE_YOLO_H{i-1}_OUTPUT_SCALE"
    )
    prev_zp = (
        "CUTE_YOLO_STEM3_OUTPUT_ZERO_POINT"
        if i == 1 else
        f"CUTE_YOLO_H{i-1}_OUTPUT_ZERO_POINT"
    )

    LAYER_SPECS.extend([
        (f"h{i}a",
         f"w_h{i}a", f"b_h{i}a", f"m_h{i}a", f"s_h{i}a",
         prev_scale, prev_zp,
         f"CUTE_YOLO_H{i}_OUTPUT_SCALE",
         f"CUTE_YOLO_H{i}_OUTPUT_ZERO_POINT"),

        (f"h{i}dw",
         f"w_h{i}dw", f"b_h{i}dw", f"m_h{i}dw", f"s_h{i}dw",
         prev_scale, prev_zp,
         f"CUTE_YOLO_H{i}_DW_OUTPUT_SCALE",
         f"CUTE_YOLO_H{i}_DW_OUTPUT_ZERO_POINT"),

        (f"h{i}pw",
         f"w_h{i}pw", f"b_h{i}pw", f"m_h{i}pw", f"s_h{i}pw",
         f"CUTE_YOLO_H{i}_DW_OUTPUT_SCALE",
         f"CUTE_YOLO_H{i}_DW_OUTPUT_ZERO_POINT",
         f"CUTE_YOLO_H{i}_OUTPUT_SCALE",
         f"CUTE_YOLO_H{i}_OUTPUT_ZERO_POINT"),
    ])

LAYER_SPECS.append(
    ("head",
     "w_head", "b_head", "m_head", "s_head",
     "CUTE_YOLO_H5_OUTPUT_SCALE", "CUTE_YOLO_H5_OUTPUT_ZERO_POINT",
     "CUTE_YOLO_OUTPUT_SCALE", "CUTE_YOLO_OUTPUT_ZERO_POINT")
)

assert len(LAYER_SPECS) == LAYER_COUNT


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

def cstr(text: str, n: int) -> bytes:
    raw = text.encode("utf-8")[:n - 1]
    return raw + b"\0" * (n - len(raw))


def safe_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "model"


def parse_macro(text: str, name: str) -> str:
    m = re.search(
        rf"^\s*#define\s+{re.escape(name)}\s+([^\n]+)",
        text,
        re.MULTILINE,
    )
    if not m:
        raise KeyError(f"Macro not found: {name}")
    return m.group(1).split("//", 1)[0].strip()


def parse_macro_float(text: str, name: str) -> float:
    return float(parse_macro(text, name).rstrip("fF"))


def parse_macro_int(text: str, name: str) -> int:
    return int(parse_macro(text, name), 0)


def parse_array(text: str, name: str, ctype: str) -> list[int]:
    pattern = (
        rf"static\s+const\s+{re.escape(ctype)}\s+"
        rf"{re.escape(name)}\[(\d+)\]\s+PROGMEM\s*=\s*"
        rf"\{{(.*?)\}};"
    )

    m = re.search(pattern, text, re.DOTALL)

    if not m:
        raise KeyError(f"Array not found: {name}")

    expected = int(m.group(1))

    values = [
        int(x)
        for x in re.findall(r"[-+]?\d+", m.group(2))
    ]

    if len(values) != expected:
        raise ValueError(
            f"{name}: parsed {len(values)} values, expected {expected}"
        )

    return values


def pack_i8(values: list[int]) -> bytes:
    for value in values:
        if not -128 <= value <= 127:
            raise ValueError(f"INT8 value out of range: {value}")

    return struct.pack(f"<{len(values)}b", *values)


def pack_i32(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def find_member(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]

    if not matches:
        raise FileNotFoundError(
            f"ZIP member ending with {suffix!r} not found"
        )

    # Prefer export/ when multiple copies exist.
    export_matches = [
        name for name in matches
        if "/export/" in f"/{name}"
    ]

    if len(export_matches) == 1:
        return export_matches[0]

    return matches[0]


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_export(runtime: dict, manifest: dict) -> None:
    architecture = manifest.get("architecture_id")

    if architecture != ARCH_NAME:
        raise ValueError(
            "Wrong Cute-YOLO architecture.\n"
            f"  ZIP:      {architecture!r}\n"
            f"  Expected: {ARCH_NAME!r}"
        )

    runtime_arch = runtime.get("architecture_id")
    if runtime_arch and runtime_arch != ARCH_NAME:
        raise ValueError(
            f"runtime_config architecture mismatch: {runtime_arch!r}"
        )

    expected_input = [1, 128, 128]
    if manifest.get("input") != expected_input:
        raise ValueError(
            f"Expected input {expected_input}, got {manifest.get('input')}"
        )

    if manifest.get("output_firmware_chw") != [5, 16, 16]:
        raise ValueError(
            "Expected firmware output [5,16,16], got "
            f"{manifest.get('output_firmware_chw')}"
        )

    if manifest.get("hybrid_normal_out") != 8:
        raise ValueError("Expected 8-channel conventional branch")

    if manifest.get("hybrid_efficient_out") != 24:
        raise ValueError("Expected 24-channel DW/PW branch")

    if manifest.get("hybrid_blocks") != 5:
        raise ValueError("Expected exactly five hybrid blocks")

    if manifest.get("concat_directly_noodle_compatible") is False:
        raise ValueError(
            "TFLite concat quantization is not directly Noodle-compatible"
        )

    required_runtime = [
        "confidence_threshold",
        "nms_iou_threshold",
        "min_box_w",
        "min_box_h",
        "max_detections",
    ]

    for key in required_runtime:
        if key not in runtime:
            raise KeyError(f"runtime_config missing {key!r}")


# ---------------------------------------------------------------------
# Build the .cute blob
# ---------------------------------------------------------------------

def build_blob(header_text: str, runtime: dict, manifest: dict) -> bytes:
    validate_export(runtime, manifest)

    payload = bytearray()
    records = []

    def align_payload(alignment: int) -> None:
        while (HEADER_BYTES + len(payload)) % alignment:
            payload.append(0)

    for (
        layer_name,
        w_name,
        b_name,
        m_name,
        s_name,
        input_scale_name,
        input_zp_name,
        output_scale_name,
        output_zp_name,
    ) in LAYER_SPECS:

        weights = parse_array(header_text, w_name, "int8_t")
        bias = parse_array(header_text, b_name, "int32_t")
        multiplier = parse_array(header_text, m_name, "int32_t")
        shift = parse_array(header_text, s_name, "int32_t")

        if not (
            len(bias) == len(multiplier) == len(shift)
        ):
            raise ValueError(
                f"{layer_name}: bias/multiplier/shift lengths differ"
            )

        # Keep each weight region aligned, then keep all int32 arrays
        # naturally aligned for direct memory-mapped dereferencing.
        align_payload(16)

        weight_offset = HEADER_BYTES + len(payload)
        payload.extend(pack_i8(weights))

        align_payload(4)

        bias_offset = HEADER_BYTES + len(payload)
        payload.extend(pack_i32(bias))

        multiplier_offset = HEADER_BYTES + len(payload)
        payload.extend(pack_i32(multiplier))

        shift_offset = HEADER_BYTES + len(payload)
        payload.extend(pack_i32(shift))

        records.append((
            weight_offset,
            len(weights),
            bias_offset,
            len(bias),
            multiplier_offset,
            shift_offset,
            parse_macro_float(header_text, input_scale_name),
            parse_macro_float(header_text, output_scale_name),
            parse_macro_int(header_text, input_zp_name),
            parse_macro_int(header_text, output_zp_name),
        ))

    total_bytes = HEADER_BYTES + len(payload)
    payload_crc32 = zlib.crc32(payload) & 0xFFFFFFFF

    label = str(runtime.get(
        "label",
        manifest.get("label", "object"),
    ))

    base_args = dict(
        magic=b"CUTE",
        version=FORMAT_VERSION,
        header_bytes=HEADER_BYTES,
        arch_id=ARCH_ID,
        total_bytes=total_bytes,
        payload_crc=payload_crc32,
        layer_count=LAYER_COUNT,
        reserved=0,
        label=cstr(label, 24),
        confidence=float(runtime["confidence_threshold"]),
        nms=float(runtime["nms_iou_threshold"]),
        min_w=float(runtime["min_box_w"]),
        min_h=float(runtime["min_box_h"]),
        max_det=int(runtime["max_detections"]),
        pad=b"\0\0\0",
        input_scale=float(manifest["input_scale"]),
        input_zp=int(manifest["input_zero_point"]),
        output_scale=float(manifest["output_scale"]),
        output_zp=int(manifest["output_zero_point"]),
        arch_name=cstr(ARCH_NAME, 40),
    )

    def pack_base(header_crc: int) -> bytes:
        return struct.pack(
            BASE_FMT,
            base_args["magic"],
            base_args["version"],
            base_args["header_bytes"],
            base_args["arch_id"],
            base_args["total_bytes"],
            base_args["payload_crc"],
            base_args["layer_count"],
            base_args["reserved"],
            base_args["label"],
            base_args["confidence"],
            base_args["nms"],
            base_args["min_w"],
            base_args["min_h"],
            base_args["max_det"],
            base_args["pad"],
            base_args["input_scale"],
            base_args["input_zp"],
            base_args["output_scale"],
            base_args["output_zp"],
            base_args["arch_name"],
            header_crc,
        )

    # Build header with CRC field zero.
    header = bytearray(pack_base(0))

    for record in records:
        header.extend(struct.pack(RECORD_FMT, *record))

    header.extend(b"\0" * (HEADER_BYTES - len(header)))

    header_crc32 = zlib.crc32(header) & 0xFFFFFFFF

    # Replace first 128 bytes with CRC-bearing base header.
    header[:128] = pack_base(header_crc32)

    return bytes(header) + bytes(payload)


# ---------------------------------------------------------------------
# Read Colab export ZIP
# ---------------------------------------------------------------------

def load_export(export_zip: Path):
    with zipfile.ZipFile(export_zip, "r") as zf:
        names = zf.namelist()

        header_member = find_member(
            names,
            "model_weights_int8.h",
        )
        runtime_member = find_member(
            names,
            "runtime_config.json",
        )

        manifest_candidates = [
            name for name in names
            if name.endswith("_deployment_manifest.json")
        ]

        if not manifest_candidates:
            raise FileNotFoundError(
                "No *_deployment_manifest.json found in ZIP"
            )

        manifest_member = manifest_candidates[0]

        header_text = zf.read(header_member).decode("utf-8")
        runtime = json.loads(zf.read(runtime_member))
        manifest = json.loads(zf.read(manifest_member))

    return header_text, runtime, manifest


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Cute-YOLO TensorFlow export ZIP "
            "into a memory-mappable .cute model."
        )
    )

    parser.add_argument(
        "export_zip",
        type=Path,
        help="ZIP produced by the Cute-YOLO TensorFlow notebook",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .cute file (default: based on model label)",
    )

    parser.add_argument(
        "--label",
        default=None,
        help="Override task label stored in the model",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override confidence threshold",
    )

    parser.add_argument(
        "--nms",
        type=float,
        default=None,
        help="Override NMS IoU threshold",
    )

    parser.add_argument(
        "--min-box-w",
        type=float,
        default=None,
        help="Override minimum normalized box width",
    )

    parser.add_argument(
        "--min-box-h",
        type=float,
        default=None,
        help="Override minimum normalized box height",
    )

    parser.add_argument(
        "--max-detections",
        type=int,
        default=None,
        help="Override maximum detections",
    )

    args = parser.parse_args()

    if not args.export_zip.is_file():
        raise FileNotFoundError(args.export_zip)

    header_text, runtime, manifest = load_export(args.export_zip)

    # Optional deployment-only overrides.
    if args.label is not None:
        runtime["label"] = args.label

    if args.confidence is not None:
        runtime["confidence_threshold"] = args.confidence

    if args.nms is not None:
        runtime["nms_iou_threshold"] = args.nms

    if args.min_box_w is not None:
        runtime["min_box_w"] = args.min_box_w

    if args.min_box_h is not None:
        runtime["min_box_h"] = args.min_box_h

    if args.max_detections is not None:
        runtime["max_detections"] = args.max_detections

    # Basic range checks.
    if not 0.0 <= float(runtime["confidence_threshold"]) <= 1.0:
        raise ValueError("confidence threshold must be in [0,1]")

    if not 0.0 <= float(runtime["nms_iou_threshold"]) <= 1.0:
        raise ValueError("NMS IoU threshold must be in [0,1]")

    if not 1 <= int(runtime["max_detections"]) <= 64:
        raise ValueError("max_detections must be between 1 and 64")

    blob = build_blob(
        header_text,
        runtime,
        manifest,
    )

    label = str(runtime.get(
        "label",
        manifest.get("label", "model"),
    ))

    out_path = (
        args.out
        if args.out is not None
        else args.export_zip.with_name(
            safe_filename(label) + ".cute"
        )
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_bytes(blob)

    # Read back key header fields as a small self-check.
    magic = blob[:4]
    if magic != b"CUTE":
        raise RuntimeError("Internal packer error: invalid output magic")

    print()
    print("Cute-YOLO model package created")
    print("--------------------------------")
    print(f"Input ZIP:     {args.export_zip}")
    print(f"Output:        {out_path}")
    print(f"Bytes:         {len(blob):,}")
    print(f"Architecture:  {ARCH_NAME}")
    print(f"Architecture ID: 0x{ARCH_ID:08X}")
    print(f"Label:         {label}")
    print(
        "Operating point: "
        f"confidence={float(runtime['confidence_threshold']):.3f}, "
        f"NMS={float(runtime['nms_iou_threshold']):.3f}"
    )
    print(
        "Box filter:      "
        f"w>={float(runtime['min_box_w']):.3f}, "
        f"h>={float(runtime['min_box_h']):.3f}"
    )
    print(
        f"Max detections: {int(runtime['max_detections'])}"
    )
    print("Status:        OK")


if __name__ == "__main__":
    main()
