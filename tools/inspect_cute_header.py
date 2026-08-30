#!/usr/bin/env python3
import argparse
import struct
import zlib
from pathlib import Path

BASE_FMT = "<4sHHIIIHH24sffffB3sfifi40sI"
HEADER_BYTES = 1024
HEADER_CRC_OFFSET = 124

ap = argparse.ArgumentParser()
ap.add_argument("model", type=Path)
args = ap.parse_args()

blob = args.model.read_bytes()
if len(blob) < HEADER_BYTES:
    raise SystemExit("File shorter than 1024-byte Cute-YOLO header")

fields = struct.unpack(BASE_FMT, blob[:struct.calcsize(BASE_FMT)])

magic = fields[0]
version = fields[1]
header_bytes = fields[2]
architecture_id = fields[3]
total_bytes = fields[4]
payload_crc_stored = fields[5]
layer_count = fields[6]
label = fields[8].split(b"\0", 1)[0].decode("utf-8", errors="replace")
confidence = fields[9]
nms = fields[10]
min_box_w = fields[11]
min_box_h = fields[12]
max_detections = fields[13]
header_crc_stored = fields[-1]

header = bytearray(blob[:HEADER_BYTES])
struct.pack_into("<I", header, HEADER_CRC_OFFSET, 0)
header_crc_calc = zlib.crc32(header) & 0xFFFFFFFF
payload_crc_calc = zlib.crc32(blob[HEADER_BYTES:total_bytes]) & 0xFFFFFFFF

print(f"path              : {args.model}")
print(f"file bytes        : {len(blob)}")
print(f"magic             : {magic!r}")
print(f"version           : {version}")
print(f"header bytes      : {header_bytes}")
print(f"architecture ID   : 0x{architecture_id:08X}")
print(f"total bytes       : {total_bytes}")
print(f"layer count       : {layer_count}")
print(f"label             : {label}")
print(f"confidence        : {confidence:.6f}")
print(f"NMS IoU           : {nms:.6f}")
print(f"min box           : {min_box_w:.6f}, {min_box_h:.6f}")
print(f"max detections    : {max_detections}")
print()
print(f"header CRC stored : 0x{header_crc_stored:08X}")
print(f"header CRC calc   : 0x{header_crc_calc:08X}")
print("header CRC        :", "OK" if header_crc_stored == header_crc_calc else "FAIL")
print(f"payload CRC stored: 0x{payload_crc_stored:08X}")
print(f"payload CRC calc  : 0x{payload_crc_calc:08X}")
print("payload CRC       :", "OK" if payload_crc_stored == payload_crc_calc else "FAIL")
print("size              :", "OK" if total_bytes == len(blob) else "FAIL")
