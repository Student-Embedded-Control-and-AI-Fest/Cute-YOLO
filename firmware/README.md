# Cute-YOLO fixed firmware: BLE + raw flash model + mmap

This project keeps the **Cute-YOLO 8+24 topology and firmware fixed**. A task-specific `.cute` model is uploaded over BLE into one of two raw flash partitions and then memory-mapped directly. NoodleQ receives normal parameter pointers into mapped flash; the weights are **not copied to PSRAM**.

## Target used here

- PlatformIO / Arduino
- ESP32-S3 DevKitC-1 compatible target
- camera pin map from the supplied `camera_pins.h`
- 16 MB flash / 8 MB OPI PSRAM camera environment
- ST7735 TFT wiring preserved from the working firmware
- built-in **BOOT button = GPIO0** as inference trigger
- on-board RGB LED **GPIO48 forced off**
- camera darkening preserved:
  - exposure control on
  - gain control on
  - AE level = -2
  - brightness = -2
  - contrast = 0
  - saturation = 0
  - vflip = 1

## Flash layout

Two 256 KiB raw data partitions are reserved at the end of flash:

```text
cuteA   0x40000 bytes
cuteB   0x40000 bytes
```

The active model remains untouched while a new model is written to the inactive slot. After header/architecture/CRC validation, activation is deferred to the main Arduino loop and the new partition is mapped using `esp_partition_mmap(..., ESP_PARTITION_MMAP_DATA, ...)`.

The existing FFat partition is retained, only reduced by 512 KiB.

## IMPORTANT after changing the partition table

Erase flash once before the first upload with this new partition table:

```bash
pio run -e esp32-s3-camera-cute-ble -t erase
pio run -e esp32-s3-camera-cute-ble -t upload
pio device monitor -b 921600
```

At first boot there is intentionally no valid model. The TFT should show `BLE: UPLOAD`, and the board advertises as `Cute-YOLO`.

## Build

The included `platformio.ini` uses:

```ini
[env:esp32-s3-camera-cute-ble]
```

Noodle is still expected at:

```text
/home/auralius/works/noodle
```

The only new library dependency is NimBLE-Arduino.

## Make the circle model package

A ready-to-upload model is included as:

```text
round_object.cute
```

It was generated from the successful TensorFlow circle export and is 28,412 bytes. Its runtime metadata includes:

```text
label      = round_object
confidence = 0.50
NMS IoU    = 0.35
```

To regenerate it later from any compatible TensorFlow export ZIP:

```bash
python tools/pack_cute_model.py \
  --export-zip cute_yolo_circle_tensorflow_fixed_8_24_export.zip \
  --out round_object.cute
```

The `.cute` format has a protected 1024-byte header plus 22 fixed parameter records (split stems, five Conv/DW/PW hybrid blocks, and head). Each record points to INT8 weights and aligned INT32 bias/multiplier/shift arrays.

## Upload over BLE

Install Bleak on the PC:

```bash
python -m pip install bleak
```

Then:

```bash
python tools/upload_cute_ble.py round_object.cute
```

The uploader scans for `Cute-YOLO`, erases the inactive raw partition, writes the model in BLE chunks, asks the firmware to validate it, and waits for activation.

Typical status flow:

```text
ERASING
READY A 28412
RX ...
VERIFY
VALID A round_object
ACTIVE A round_object
```

Subsequent uploads automatically go to the other slot.

## BOOT button behavior

GPIO0 is used only after normal boot as the inference trigger. Holding BOOT while resetting/powering the ESP32-S3 still invokes the normal ROM download mode, as expected for a strapping pin.

## Runtime path

```text
BLE -> inactive cuteA/cuteB raw partition
    -> validate header + fixed architecture + CRC
    -> switch active slot
    -> esp_partition_mmap()
    -> const pointers into flash
    -> existing Noodle ConvMem/DWConv execution
```

Only activations/tensor buffers live in RAM/PSRAM. Model parameters remain in mapped flash.

## First-test sequence

1. Erase flash because the partition table changed.
2. Upload the fixed firmware.
3. Confirm serial shows BLE advertising and `NO MODEL`.
4. Run `upload_cute_ble.py round_object.cute`.
5. Confirm serial reports `ACTIVE A round_object` (or B).
6. Press the built-in BOOT button once to run inference.
7. Upload another compatible `.cute` model later; no firmware rebuild is required.

## Note

This package was structurally generated and checked here, but it was not compiled against your local PlatformIO/NimBLE installation. If your installed NimBLE-Arduino major version exposes a small API difference, send the first compiler error and it should be a very small compatibility fix.
