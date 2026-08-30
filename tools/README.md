# Cute-YOLO Tools

This directory contains the host-side utilities used to **train, adapt, package, deploy, collect, and review data** for Cute-YOLO.

The tools fall into four groups:

```text
TRAIN / ADAPT
    cute_yolo_studio.py

DATA COLLECTION / REVIEW
    collect_cute_dataset.py
    collect_cute_display.py
    cute_label_gui.py

MODEL PACKAGING
    zip_to_cute.py
    pack_cute_model.py

DEPLOYMENT
    upload_cute_ble.py
```

The normal end-to-end workflow is:

```text
source YOLO dataset
        │
        ▼
cute_yolo_studio.py
        │
        ├── base training
        ├── full INT8 conversion
        ├── INT8 confidence/NMS sweep
        └── .cute packaging
        │
        ▼
upload_cute_ble.py
        │
        ▼
ESP32-S3
        │
        ├── collect_cute_dataset.py ──► clean real-domain samples
        │                                  │
        │                                  ▼
        │                            cute_label_gui.py
        │                                  │
        │                                  ▼
        │                            verified YOLO ZIP
        │                                  │
        └──────────────────────────────────┤
                                           ▼
                                  Cute-YOLO Studio
                                  domain adaptation
                                           │
                                           ▼
                                     adapted .cute
                                           │
                                           ▼
                                         BLE

Optional documentation path:

ESP32-S3 TFT
     │
     ▼
collect_cute_display.py
     │
     ▼
exact rendered display PNGs
```

---

## Requirements

A complete environment for all tools can be installed with:

```bash
python -m pip install \
    tensorflow \
    numpy \
    opencv-python \
    matplotlib \
    pillow \
    pyserial \
    bleak
```

Cute-YOLO Studio and the labeling GUI use Tkinter.

On Ubuntu/Debian, install it if necessary:

```bash
sudo apt install python3-tk
```

For GPU TensorFlow, use the TensorFlow/CUDA environment appropriate for your machine.

---

# 1. `cute_yolo_studio.py`

## Purpose

`cute_yolo_studio.py` is the main host application for Cute-YOLO.

It provides the graphical workflow for:

```text
Train
Adapt to Real Data
Deploy via BLE
Log
```

The fixed architecture handled by the Studio is:

```text
cute_yolo_fixed_dualcore_hybrid_8_24
```

with:

```text
Input:       128 × 128 × 1
Output:       16 × 16 × 5
Parameters:   23,925
Conv MACs:    6,995,968
Hybrid:       5 × [Conv8 || DW/PW24]
```

## Run

```bash
python tools/cute_yolo_studio.py
```

## Train tab

The Train tab accepts a single-class YOLO dataset ZIP and performs:

```text
dataset loading
    ↓
center-square crop
    ↓
128×128 grayscale preprocessing
    ↓
target encoding
    ↓
training
    ↓
validation
    ↓
full INT8 conversion
    ↓
INT8 confidence/NMS operating-point sweep
    ↓
export ZIP
    ↓
optional .cute package
```

The default output directory is:

```text
trained_models/
```

Typical generated files are:

```text
trained_models/
├── <class>_base_export.zip
├── <class>_base.cute
└── <class>_base_debug/
    ├── training_dataset_preview.png
    ├── validation_dataset_preview.png
    ├── training_loss.png
    ├── training_loss_components.png
    ├── training_learning_rate.png
    ├── validation_operating_points.png
    └── ...
```

The `.cute` file is produced automatically when **Also create `<class>_base.cute`** is enabled and `zip_to_cute.py` is available.

## Training presets

The Studio supports:

```text
Complex / distinctive
Primitive / repetitive
Custom
```

Typical complex targets include faces, vehicles, animals, and tools.

Primitive/repetitive mode can enable additional footprint, halo, and crowd-aware negative supervision for repeated simple objects.

The box-overlap objective can be selected as:

```text
IoU
CIoU
```

Training-only augmentation includes:

```text
horizontal flip
brightness
contrast
shuffle
```

Horizontal flipping also transforms and re-encodes the bounding-box targets.

## Adapt to Real Data tab

Domain adaptation uses:

```text
base model export ZIP
+
original/source YOLO dataset
+
verified real deployment-domain YOLO ZIP
```

The default mix is:

```text
75% source/original data
25% real/target data
```

The default output directory is:

```text
adapted_models/
```

Typical outputs are:

```text
adapted_models/
├── <class>_adapted_export.zip
├── <class>_adapted.cute
└── <class>_adapted_debug/
    └── ...
```

In specialized fine-tuning mode:

```text
Stem 1/2/3     frozen
Head           frozen

Hybrid A       trainable
Hybrid B       trainable
```

The two branch parameter groups use complementary objectives:

```text
Branch A:
L_A = L_loc + α L_obj

Branch B:
L_B = L_obj + α L_loc

α = 0.25
```

where:

```text
L_loc = 5 L_box + 2 L_overlap
```

The losses are evaluated from the **common final detector output**. The specialization is implemented by applying different objectives to different parameter groups during backpropagation; it does not create separate inference outputs.

## Deploy via BLE tab

The Deploy tab uploads an already-created `.cute` file directly to an ESP32-S3 advertising as:

```text
Cute-YOLO
```

An explicit BLE address can also be supplied.

The Studio performs:

```text
scan
  ↓
connect
  ↓
upload
  ↓
verify
  ↓
activate
```

The Python package `bleak` is required.

---

# 2. `collect_cute_dataset.py`

## Purpose

`collect_cute_dataset.py` is the **training-oriented real-data collector**.

The ESP32-S3 sends:

```text
one clean 128×128 GRAY8 detector input
+
predicted detection boxes
+
timing/model metadata
```

The clean image is the image used for later labeling and adaptation.

The PC draws a separate convenience preview; it does **not** modify the clean training image.

## Run

```bash
python tools/collect_cute_dataset.py /dev/ttyACM0
```

Optional arguments:

```bash
python tools/collect_cute_dataset.py /dev/ttyACM0 \
    --baud 921600 \
    --out cute_real_dataset \
    --count 50 \
    --heartbeat 1.0
```

`--count 0` means collect indefinitely.

## Serial protocol

The collector periodically sends:

```text
RDYSAMPLE
```

The ESP32 responds after a triggered inference with a packet conceptually containing:

```text
SAMPLE <id> <label> <inference_us> <count>
RAW 128 128 GRAY8 16384
<binary grayscale image>
DET <count>
BOX <confidence> <x1> <y1> <x2> <y2>
...
END
```

## Generated directory

By default the script creates:

```text
cute_real_dataset/
├── images/
├── labeled/
├── pseudo_labels/
└── metadata/
```

### `images/`

Clean 128×128 grayscale detector inputs.

These are the files that should ultimately be used for verified training labels.

### `labeled/`

Receiver-drawn previews containing the current model's boxes and confidence percentages.

These are convenience images only.

### `pseudo_labels/`

Current model predictions converted to single-class YOLO format:

```text
0 x_center y_center width height
```

### `metadata/`

JSON files containing information such as:

```text
sample id
model label
inference time
image dimensions
detection count
confidence
box coordinates
```

## Important

The files in `pseudo_labels/` are **not verified ground truth**.

```text
model prediction ≠ ground truth
```

Review and correct them with `cute_label_gui.py` before using them for final domain adaptation.

---

# 3. `collect_cute_display.py`

## Purpose

`collect_cute_display.py` is the **deployment-visualization / documentation collector**.

It is intentionally different from `collect_cute_dataset.py`.

Instead of collecting the clean 128×128 training image, this tool receives the ESP32-S3's final:

```text
160 × 128 RGB565 TFT framebuffer
```

The PC does **not** redraw boxes or text.

The saved frame therefore represents the firmware-rendered display, including the elements provided by the compatible firmware variant, such as:

```text
color camera preview
red detector guide
detection boxes
confidence captions
N=<count>
T=<inference time>
```

This tool is intended for:

```text
paper figures
README screenshots
documentation
demo captures
on-device visualization checks
```

It is **not** the preferred source of training images.

## Run

```bash
python tools/collect_cute_display.py /dev/ttyACM0
```

For a paper-friendly nearest-neighbor enlargement:

```bash
python tools/collect_cute_display.py /dev/ttyACM0 --scale 4
```

Optional arguments:

```text
--baud
--out
--count
--heartbeat
--scale
```

The default output directory is:

```text
cute_display_dump/
```

with:

```text
cute_display_dump/
├── display/
└── metadata/
```

### `display/`

PNG conversion of the received firmware-rendered RGB565 framebuffer.

### `metadata/`

JSON containing:

```text
frame id
model label
inference time
display dimensions
detection count
box coordinates
```

The metadata explicitly records that the PC did not redraw the detections.

## Serial protocol

The tool sends:

```text
RDYDISPLAY
```

and expects:

```text
DISPLAY <id> <label> <inference_us> <count>
FRAME 160 128 RGB565LE 40960
<40960 binary bytes>
DET <count>
BOX ...
...
ENDDISPLAY
```

This requires a firmware build that supports the exact-TFT shadow framebuffer protocol.

---

# 4. `cute_label_gui.py`

## Purpose

`cute_label_gui.py` converts collected real-domain samples into a **verified single-class YOLO dataset**.

It understands the directory produced by `collect_cute_dataset.py`.

If:

```text
labels/<image>.txt
```

already exists, the verified label is loaded.

Otherwise, if:

```text
pseudo_labels/<image>.txt
```

exists, the model prediction is loaded as a starting point.

Saving always writes the corrected result to:

```text
labels/
```

## Run

```bash
python tools/cute_label_gui.py cute_real_dataset
```

If no dataset path is supplied, the default is:

```text
cute_real_dataset
```

## GUI controls

```text
Left-drag              add bounding box
Right-click in box     delete bounding box
A / D                  previous / next
← / →                  previous / next
S                      save
Enter                  save and next
Z                      undo
Delete / Backspace     delete last box
```

An image containing no target object is valid. Clear all boxes and save it to create an empty YOLO label file.

## Generated files

The labeler adds:

```text
cute_real_dataset/
├── labels/
├── reviewed/
└── classes.txt
```

### `labels/`

Verified YOLO ground truth.

### `reviewed/`

Preview images showing the human-reviewed boxes.

### `classes.txt`

The single Cute-YOLO class name.

## Export ZIP

The **Export ZIP** button creates a standard dataset:

```text
<class>_labeled.zip
└── <dataset>/
    ├── classes.txt
    ├── images/
    └── labels/
```

Every image receives a matching label file. Negative images receive an intentionally empty `.txt` file.

The exported ZIP can be selected directly in the Studio's **Adapt to Real Data** tab.

---

# 5. `zip_to_cute.py`

## Purpose

`zip_to_cute.py` is the recommended command-line converter for turning a Cute-YOLO TensorFlow/Studio export ZIP into the deployable `.cute` model package.

It validates the fixed architecture contract before packaging.

Expected ZIP contents include:

```text
export/model_weights_int8.h
export/runtime_config.json
export/*_deployment_manifest.json
```

The expected architecture is:

```text
cute_yolo_fixed_dualcore_hybrid_8_24
```

with:

```text
input                    [1,128,128]
firmware output          [5,16,16]
hybrid conventional      8 channels
hybrid efficient         24 channels
hybrid blocks            5
```

## Run

```bash
python tools/zip_to_cute.py model_export.zip
```

The output filename defaults to the model label.

Example:

```text
face.cute
```

## Optional deployment overrides

```bash
python tools/zip_to_cute.py model_export.zip \
    --out face.cute \
    --label face \
    --confidence 0.60 \
    --nms 0.25 \
    --min-box-w 0.05 \
    --min-box-h 0.05 \
    --max-detections 32
```

These options change deployment metadata; they do not retrain the network.

## `.cute` contents

The package contains the data required by the fixed firmware:

```text
1024-byte header
fixed 22-entry layer directory
INT8 weights
INT32 biases
INT32 requantization multipliers
INT32 requantization shifts
activation quantization metadata
class label
confidence threshold
NMS threshold
box filters
maximum detections
architecture identifier
header CRC
payload CRC
```

The network topology itself remains in the fixed ESP32-S3 firmware.

---

# 6. `pack_cute_model.py`

## Purpose

`pack_cute_model.py` is a lower-level fixed-format packer for the same 8+24 architecture.

It accepts an export ZIP using explicit command-line arguments:

```bash
python tools/pack_cute_model.py \
    --export-zip model_export.zip \
    --out model.cute
```

Like `zip_to_cute.py`, it extracts:

```text
model_weights_int8.h
runtime_config.json
*_deployment_manifest.json
```

and builds a memory-mappable `.cute` blob containing the fixed layer directory and quantized parameter arrays.

## When to use it

For normal users, prefer:

```text
zip_to_cute.py
```

because it performs additional architecture checks and supports convenient deployment metadata overrides.

`pack_cute_model.py` is useful as:

```text
a minimal/reference packer
a lower-level packaging utility
a format-debugging tool
```

Both utilities target the same fixed Cute-YOLO 8+24 firmware contract.

---

# 7. `upload_cute_ble.py`

## Purpose

`upload_cute_ble.py` is the lightweight command-line BLE deployment utility.

It uploads an already-created `.cute` file to an ESP32-S3 running the compatible Cute-YOLO firmware.

## Dependency

```bash
python -m pip install bleak
```

## Run

```bash
python tools/upload_cute_ble.py face.cute
```

By default, the script scans for a BLE device named:

```text
Cute-YOLO
```

An address may be supplied explicitly:

```bash
python tools/upload_cute_ble.py face.cute \
    --address XX:XX:XX:XX:XX:XX
```

## Deployment sequence

The utility checks the local model magic first:

```text
CUTE
```

and then performs:

```text
BLE scan
   ↓
connect
   ↓
INFO
   ↓
BEGIN <bytes>
   ↓
upload chunks
   ↓
END
   ↓
firmware validation
   ↓
activation
```

The uploader listens for firmware status notifications such as:

```text
READY
ERR ...
ACTIVE ...
```

The firmware writes the incoming model into the inactive raw-flash model slot and activates it only after validation.

---

# Which tool should I use?

| Goal | Tool |
|---|---|
| Train a new detector | `cute_yolo_studio.py` |
| Adapt a model to real ESP32 camera data | `cute_yolo_studio.py` |
| Deploy from the GUI | `cute_yolo_studio.py` |
| Collect clean detector inputs for retraining | `collect_cute_dataset.py` |
| Capture the exact rendered TFT result | `collect_cute_display.py` |
| Correct pseudo-labels / create ground truth | `cute_label_gui.py` |
| Convert an export ZIP to `.cute` manually | `zip_to_cute.py` |
| Low-level/reference `.cute` packaging | `pack_cute_model.py` |
| Upload a `.cute` from the command line | `upload_cute_ble.py` |

---

# Two collectors: do not confuse them

The two serial collectors intentionally serve different purposes.

| | `collect_cute_dataset.py` | `collect_cute_display.py` |
|---|---|---|
| Primary purpose | training/adaptation | documentation/demo |
| ESP32 payload | clean detector input | final TFT framebuffer |
| Resolution | 128×128 | 160×128 |
| Encoding | GRAY8 | RGB565LE |
| Boxes drawn by | PC convenience preview | ESP32 firmware |
| Saves pseudo-labels | yes | no |
| Suitable as clean training input | **yes** | no |
| Suitable as exact device screenshot | no | **yes** |
| Heartbeat | `RDYSAMPLE` | `RDYDISPLAY` |

A useful rule is:

> **Dataset collector = what the neural network saw. Display collector = what the user saw.**

---

# Recommended real-domain adaptation workflow

```text
1. Deploy a base .cute model
             │
             ▼
2. Run collect_cute_dataset.py
             │
             ▼
   cute_real_dataset/
             │
             ▼
3. Run cute_label_gui.py
             │
             ▼
   verify / correct all boxes
             │
             ▼
4. Export labeled YOLO ZIP
             │
             ▼
5. Open Cute-YOLO Studio
             │
             ▼
   Adapt to Real Data
             │
             ▼
6. Produce <class>_adapted.cute
             │
             ▼
7. Deploy via Studio or upload_cute_ble.py
```

For screenshots or paper figures, run `collect_cute_display.py` separately.

---

# Dataset format

Cute-YOLO uses one class per model.

A standard dataset ZIP contains:

```text
dataset/
├── classes.txt
├── images/
│   ├── image001.png
│   └── ...
└── labels/
    ├── image001.txt
    └── ...
```

Each line of a label file is:

```text
0 x_center y_center width height
```

All coordinates are normalized to `[0,1]`.

An empty label file represents a valid negative image containing no target object.

---

# Model artifact flow

The host-side model path is:

```text
TensorFlow/Keras model
        ↓
strict INT8 TFLite conversion
        ↓
model_weights_int8.h
runtime_config.json
deployment manifest
        ↓
export ZIP
        ↓
zip_to_cute.py
        ↓
.cute
        ↓
BLE
        ↓
ESP32-S3 raw flash
        ↓
header / architecture / CRC validation
        ↓
memory mapping
        ↓
Noodle INT8 inference
```

The fixed firmware and replaceable model are intentionally separate:

```text
firmware = how inference runs
.cute    = what the device detects
```

---

# Typical commands

```bash
# Main GUI
python tools/cute_yolo_studio.py

# Collect clean real deployment-domain data
python tools/collect_cute_dataset.py /dev/ttyACM0

# Verify/correct pseudo-labels
python tools/cute_label_gui.py cute_real_dataset

# Capture exact rendered TFT frames
python tools/collect_cute_display.py /dev/ttyACM0 --scale 4

# Convert an export manually
python tools/zip_to_cute.py face_base_export.zip

# Upload a model manually
python tools/upload_cute_ble.py face.cute
```

---

# Notes

- Cute-YOLO is currently a **single-class** detector.
- The fixed deployed architecture is the dual-core **8+24 hybrid** model.
- Pseudo-labels produced by the detector must not be treated automatically as verified ground truth.
- The display collector requires compatible firmware that implements `RDYDISPLAY` and the RGB565 shadow-frame protocol.
- Confidence and NMS thresholds stored in `.cute` are deployment parameters.
- Training and domain-adaptation loss terms do not change the fixed embedded inference topology.
