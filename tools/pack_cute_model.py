#!/usr/bin/env python3
"""
Pack a TensorFlow Cute-YOLO export into one fixed-format .cute model blob.

Accepted input:
  --export-zip cute_yolo_circle_tensorflow_fixed_8_24_export.zip

The ZIP must contain:
  export/model_weights_int8.h
  export/runtime_config.json
  export/cute_yolo_circle_deployment_manifest.json  (or any *_deployment_manifest.json)

The resulting .cute file contains:
  - 1024-byte protected model/config header
  - fixed layer directory for the 8+24 architecture
  - INT8 weights
  - INT32 biases
  - INT32 requantization multipliers
  - INT32 requantization shifts

It is intended to be written to the ESP32-S3 raw cuteA/cuteB data partitions
and memory-mapped directly by the fixed firmware.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import tempfile
import zipfile
import zlib
from pathlib import Path

HEADER_BYTES = 1024
FORMAT_VERSION = 1
ARCH_NAME = "cute_yolo_fixed_dualcore_hybrid_8_24"
ARCH_ID = zlib.crc32(ARCH_NAME.encode("ascii")) & 0xFFFFFFFF
LAYER_COUNT = 22

BASE_FMT = "<4sHHIIIHH24sffffB3sfifi40sI"
RECORD_FMT = "<IIIIIIffii"

assert struct.calcsize(BASE_FMT) == 128
assert struct.calcsize(RECORD_FMT) == 40
assert 128 + LAYER_COUNT * 40 + 16 == HEADER_BYTES


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
        (f"h{i}a", f"w_h{i}a", f"b_h{i}a", f"m_h{i}a", f"s_h{i}a",
         prev_scale, prev_zp,
         f"CUTE_YOLO_H{i}_OUTPUT_SCALE",
         f"CUTE_YOLO_H{i}_OUTPUT_ZERO_POINT"),

        (f"h{i}dw", f"w_h{i}dw", f"b_h{i}dw", f"m_h{i}dw", f"s_h{i}dw",
         prev_scale, prev_zp,
         f"CUTE_YOLO_H{i}_DW_OUTPUT_SCALE",
         f"CUTE_YOLO_H{i}_DW_OUTPUT_ZERO_POINT"),

        (f"h{i}pw", f"w_h{i}pw", f"b_h{i}pw", f"m_h{i}pw", f"s_h{i}pw",
         f"CUTE_YOLO_H{i}_DW_OUTPUT_SCALE",
         f"CUTE_YOLO_H{i}_DW_OUTPUT_ZERO_POINT",
         f"CUTE_YOLO_H{i}_OUTPUT_SCALE",
         f"CUTE_YOLO_H{i}_OUTPUT_ZERO_POINT"),
    ])

LAYER_SPECS.append(
    ("head", "w_head", "b_head", "m_head", "s_head",
     "CUTE_YOLO_H5_OUTPUT_SCALE", "CUTE_YOLO_H5_OUTPUT_ZERO_POINT",
     "CUTE_YOLO_OUTPUT_SCALE", "CUTE_YOLO_OUTPUT_ZERO_POINT")
)

assert len(LAYER_SPECS) == LAYER_COUNT


def cstr(text: str, n: int) -> bytes:
    raw = text.encode("utf-8")[: n - 1]
    return raw + b"\0" * (n - len(raw))


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
    for v in values:
        if not -128 <= v <= 127:
            raise ValueError(f"INT8 value out of range: {v}")
    return struct.pack(f"<{len(values)}b", *values)


def pack_i32(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def find_member(names: list[str], suffix: str) -> str:
    matches = [n for n in names if n.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"ZIP member ending with {suffix!r} not found")
    if len(matches) > 1:
        # Prefer a file under export/
        export = [n for n in matches if "/export/" in f"/{n}"]
        if len(export) == 1:
            return export[0]
    return matches[0]


def build_blob(header_text: str, runtime: dict, manifest: dict) -> bytes:
    if manifest.get("architecture_id") != ARCH_NAME:
        raise ValueError(
            "Architecture mismatch: "
            f"{manifest.get('architecture_id')!r} != {ARCH_NAME!r}"
        )

    payload = bytearray()
    records: list[tuple] = []

    def align_payload(alignment: int) -> None:
        while (HEADER_BYTES + len(payload)) % alignment:
            payload.append(0)

    for (
        layer_name,
        w_name,
        b_name,
        m_name,
        s_name,
        in_scale_name,
        in_zp_name,
        out_scale_name,
        out_zp_name,
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

        # Start each layer on a cache-friendly boundary.
        align_payload(16)
        weight_offset = HEADER_BYTES + len(payload)
        payload.extend(pack_i8(weights))

        # Noodle dereferences these as int32_t*, therefore maintain 4-byte
        # alignment even though the model partition itself is byte-addressable.
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
            parse_macro_float(header_text, in_scale_name),
            parse_macro_float(header_text, out_scale_name),
            parse_macro_int(header_text, in_zp_name),
            parse_macro_int(header_text, out_zp_name),
        ))

    total_bytes = HEADER_BYTES + len(payload)
    payload_crc32 = zlib.crc32(payload) & 0xFFFFFFFF

    label = runtime.get("label", manifest.get("label", "object"))

    base = struct.pack(
        BASE_FMT,
        b"CUTE",
        FORMAT_VERSION,
        HEADER_BYTES,
        ARCH_ID,
        total_bytes,
        payload_crc32,
        LAYER_COUNT,
        0,
        cstr(label, 24),
        float(runtime["confidence_threshold"]),
        float(runtime["nms_iou_threshold"]),
        float(runtime["min_box_w"]),
        float(runtime["min_box_h"]),
        int(runtime["max_detections"]),
        b"\0\0\0",
        float(manifest["input_scale"]),
        int(manifest["input_zero_point"]),
        float(manifest["output_scale"]),
        int(manifest["output_zero_point"]),
        cstr(ARCH_NAME, 40),
        0,  # header CRC is filled after the complete directory is assembled
    )

    header = bytearray(base)

    for record in records:
        header.extend(struct.pack(RECORD_FMT, *record))

    header.extend(b"\0" * (HEADER_BYTES - len(header)))

    # Header CRC is calculated with its own field zero.
    header_crc32 = zlib.crc32(header) & 0xFFFFFFFF

    base_with_crc = struct.pack(
        BASE_FMT,
        b"CUTE",
        FORMAT_VERSION,
        HEADER_BYTES,
        ARCH_ID,
        total_bytes,
        payload_crc32,
        LAYER_COUNT,
        0,
        cstr(label, 24),
        float(runtime["confidence_threshold"]),
        float(runtime["nms_iou_threshold"]),
        float(runtime["min_box_w"]),
        float(runtime["min_box_h"]),
        int(runtime["max_detections"]),
        b"\0\0\0",
        float(manifest["input_scale"]),
        int(manifest["input_zero_point"]),
        float(manifest["output_scale"]),
        int(manifest["output_zero_point"]),
        cstr(ARCH_NAME, 40),
        header_crc32,
    )

    header[:128] = base_with_crc

    return bytes(header) + bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-zip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with zipfile.ZipFile(args.export_zip, "r") as zf:
        names = zf.namelist()

        header_member = find_member(names, "model_weights_int8.h")
        runtime_member = find_member(names, "runtime_config.json")

        manifest_candidates = [
            n for n in names
            if n.endswith("_deployment_manifest.json")
        ]
        if not manifest_candidates:
            raise FileNotFoundError("Deployment manifest not found in ZIP")

        manifest_member = manifest_candidates[0]

        header_text = zf.read(header_member).decode("utf-8")
        runtime = json.loads(zf.read(runtime_member))
        manifest = json.loads(zf.read(manifest_member))

    blob = build_blob(
        header_text,
        runtime,
        manifest,
    )

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.out.write_bytes(blob)

    print(f"Wrote: {args.out}")
    print(f"Bytes: {len(blob)}")
    print(f"Architecture: {ARCH_NAME}")
    print(f"Architecture CRC32: 0x{ARCH_ID:08X}")
    print(f"Label: {runtime.get('label')}")
    print(
        "Operating point: "
        f"conf={runtime['confidence_threshold']}, "
        f"nms={runtime['nms_iou_threshold']}"
    )


if __name__ == "__main__":
    main()
