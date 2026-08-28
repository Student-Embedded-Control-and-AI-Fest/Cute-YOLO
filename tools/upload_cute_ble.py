#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from bleak import BleakClient, BleakScanner

DEVICE_NAME = "Cute-YOLO"
SERVICE_UUID = "7f1d0001-9f3b-4c2b-8f3e-5b51c0de0001"
CTRL_UUID    = "7f1d0002-9f3b-4c2b-8f3e-5b51c0de0001"
DATA_UUID    = "7f1d0003-9f3b-4c2b-8f3e-5b51c0de0001"
STATUS_UUID  = "7f1d0004-9f3b-4c2b-8f3e-5b51c0de0001"

async def find_target(address: str | None):
    if address:
        return address
    print(f"Scanning for {DEVICE_NAME}...")
    dev = await BleakScanner.find_device_by_filter(
        lambda d, ad: (
            d.name == DEVICE_NAME
            or ad.local_name == DEVICE_NAME
            or SERVICE_UUID.lower() in [str(x).lower() for x in ad.service_uuids]
        ),
        timeout=12.0,
    )
    if dev is None:
        raise RuntimeError("Cute-YOLO BLE device not found")
    print(f"Found {dev.name} @ {dev.address}")
    return dev

async def upload(path: Path, address: str | None):
    blob = path.read_bytes()
    if len(blob) < 1024 or blob[:4] != b"CUTE":
        raise ValueError("Not a valid .cute model")

    target = await find_target(address)
    q: asyncio.Queue[str] = asyncio.Queue()

    def on_status(_sender, data: bytearray):
        text = bytes(data).decode("utf-8", errors="replace")
        print(f"[ESP32] {text}")
        q.put_nowait(text)

    async with BleakClient(target) as client:
        print("BLE connected")
        await client.start_notify(STATUS_UUID, on_status)
        await client.write_gatt_char(CTRL_UUID, b"INFO", response=True)

        mtu = getattr(client, "mtu_size", 23) or 23
        chunk = max(20, min(240, mtu - 3))
        print(f"Uploading {len(blob)} bytes with chunk={chunk}")

        await client.write_gatt_char(
            CTRL_UUID,
            f"BEGIN {len(blob)}".encode(),
            response=True,
        )

        # Wait for erase + READY if notification arrives.
        ready = False
        for _ in range(8):
            try:
                text = await asyncio.wait_for(q.get(), timeout=2.0)
            except asyncio.TimeoutError:
                break
            if text.startswith("ERR"):
                raise RuntimeError(text)
            if text.startswith("READY "):
                ready = True
                break
        if not ready:
            print("READY notification not observed; continuing")

        sent = 0
        while sent < len(blob):
            part = blob[sent:sent + chunk]
            await client.write_gatt_char(DATA_UUID, part, response=True)
            sent += len(part)
            if sent == len(blob) or sent % 4096 < chunk:
                print(f"  {sent}/{len(blob)} ({100.0*sent/len(blob):.1f}%)")

        await client.write_gatt_char(CTRL_UUID, b"END", response=True)

        # Wait for validation and activation.
        end_time = asyncio.get_running_loop().time() + 12.0
        while asyncio.get_running_loop().time() < end_time:
            try:
                text = await asyncio.wait_for(q.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            if text.startswith("ERR"):
                raise RuntimeError(text)
            if text.startswith("ACTIVE "):
                print("Model activated")
                break
        else:
            print("Activation notification not seen; requesting INFO")
            await client.write_gatt_char(CTRL_UUID, b"INFO", response=True)
            await asyncio.sleep(0.5)

        await client.stop_notify(STATUS_UUID)
        print("Done")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--address")
    args = ap.parse_args()
    asyncio.run(upload(args.model, args.address))

if __name__ == "__main__":
    main()
