# Cute-YOLO Simple Label GUI

A minimal bounding-box annotation tool for the single-class Cute-YOLO workflow.

It produces the same simple dataset layout as the ESP32 face dataset:

```text
dataset/
├── classes.txt
├── images/
│   ├── sample_001.png
│   └── ...
└── labels/
    ├── sample_001.txt
    └── ...
```

Each label file uses standard YOLO detection format:

```text
<class_id> <center_x> <center_y> <width> <height>
```

For Cute-YOLO, the model is single-class, therefore:

```text
class_id = 0
```

Example:

```text
0 0.535677 0.452431 0.181771 0.242361
```

All coordinates are normalized to `[0,1]`.

## Install

```bash
pip install pillow
```

Tkinter normally ships with Python. On Ubuntu/Debian, if needed:

```bash
sudo apt install python3-tk
```

## Run

For the serial collector dataset:

```bash
python cute_label_gui.py cute_real_dataset
```

or for any dataset with an `images/` directory:

```bash
python cute_label_gui.py my_dataset
```

## Controls

- **Left-click + drag**: add a bounding box
- **Right-click inside a box**: delete the box
- **Undo** button or `Z`
- **Clear**: mark image as having no objects, then Save
- **Previous / Next** or `A / D`
- `S`: save
- `Enter`: save and next
- **Export ZIP**: create a notebook-ready package containing only:
  - `classes.txt`
  - `images/`
  - `labels/`

## Pseudo-label workflow

If the serial collector created:

```text
pseudo_labels/
metadata/
```

the GUI uses them automatically.

Loading priority:

1. `labels/<stem>.txt` — verified human labels
2. `pseudo_labels/<stem>.txt` — model suggestions
3. no boxes

Pseudo-labels are shown as editable starting boxes. Press Save to write the
human-verified labels into `labels/`.

## Class name

The GUI reads the single class from:

```text
classes.txt
```

If that file does not yet exist, it tries the serial collector JSON metadata.

For example:

```text
round_object
```

or later:

```text
face
```

The class name may change, but the YOLO class ID remains `0` because each
Cute-YOLO model is a one-class detector.

## Negative images

A verified image with no object is represented by an **empty `.txt` file**.
That is valid YOLO detection-dataset behavior and is useful for hard-negative
training.
