#!/usr/bin/env python3
"""
Cute-YOLO Simple Dataset Labeler
================================

Produces the same simple dataset layout used by the ESP32 face dataset:

dataset/
├── classes.txt
├── images/
│   ├── image_001.png
│   └── ...
└── labels/
    ├── image_001.txt
    └── ...

YOLO label format:
    <class_id> <cx> <cy> <w> <h>

For Cute-YOLO single-class training:
    class_id = 0

Coordinates are normalized to [0,1].

If the dataset was produced by collect_cute_dataset.py, this GUI also detects:
    pseudo_labels/
    metadata/

Pseudo-labels are shown as a starting point if no verified label exists yet.
Saving always writes the final verified labels to labels/.

Dependencies:
    pip install pillow

Run:
    python cute_label_gui.py cute_real_dataset
"""

from __future__ import annotations

import argparse
import json
import shutil
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
import zipfile

from PIL import Image, ImageTk, ImageDraw, ImageFont


DISPLAY_W = 640
DISPLAY_H = 640
MIN_DRAG_PX = 5
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def clean(self) -> "Box":
        x1 = min(self.x1, self.x2)
        y1 = min(self.y1, self.y2)
        x2 = max(self.x1, self.x2)
        y2 = max(self.y1, self.y2)

        return Box(
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        )

    def to_yolo(self) -> str:
        b = self.clean()

        cx = (b.x1 + b.x2) * 0.5
        cy = (b.y1 + b.y2) * 0.5
        w = b.x2 - b.x1
        h = b.y2 - b.y1

        return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    @staticmethod
    def from_yolo(line: str) -> "Box | None":
        parts = line.strip().split()

        if len(parts) != 5:
            return None

        try:
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            return None

        # Cute-YOLO is currently single-class. Ignore nonzero classes.
        if class_id != 0:
            return None

        return Box(
            cx - w * 0.5,
            cy - h * 0.5,
            cx + w * 0.5,
            cy + h * 0.5,
        ).clean()


class CuteLabelGUI:
    def __init__(self, root: tk.Tk, dataset_root: Path):
        self.root = root
        self.dataset_root = dataset_root.resolve()

        self.images_dir = self.dataset_root / "images"
        self.labels_dir = self.dataset_root / "labels"
        self.pseudo_dir = self.dataset_root / "pseudo_labels"
        self.metadata_dir = self.dataset_root / "metadata"
        self.reviewed_dir = self.dataset_root / "reviewed"

        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.reviewed_dir.mkdir(parents=True, exist_ok=True)

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Missing image directory:\n{self.images_dir}"
            )

        self.image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGES
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No images found in:\n{self.images_dir}"
            )

        self.class_name = self.detect_class_name()
        self.write_classes_txt()

        self.index = 0
        self.image: Image.Image | None = None
        self.display_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None

        self.boxes: list[Box] = []
        self.undo_stack: list[list[Box]] = []

        self.source = "EMPTY"
        self.dirty = False

        self.drag_start: tuple[int, int] | None = None
        self.drag_rect_id: int | None = None

        self.build_ui()
        self.bind_keys()
        self.load_index(0)

    # ---------------------------------------------------------
    # Dataset metadata
    # ---------------------------------------------------------

    def detect_class_name(self) -> str:
        classes_file = self.dataset_root / "classes.txt"

        if classes_file.exists():
            lines = [
                x.strip()
                for x in classes_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()
                if x.strip()
            ]

            if lines:
                return lines[0]

        # Collector metadata fallback.
        if self.metadata_dir.exists():
            for path in sorted(self.metadata_dir.glob("*.json")):
                try:
                    data = json.loads(
                        path.read_text(encoding="utf-8")
                    )
                    label = str(data.get("label", "")).strip()
                    if label:
                        return label
                except Exception:
                    pass

        return "object"

    def write_classes_txt(self):
        (self.dataset_root / "classes.txt").write_text(
            self.class_name.strip() + "\n",
            encoding="utf-8",
        )

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def build_ui(self):
        self.root.title("Cute-YOLO Simple YOLO Labeler")

        header = tk.Frame(self.root)
        header.pack(fill=tk.X, padx=8, pady=(8, 3))

        self.file_label = tk.Label(
            header,
            text="",
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.counter_label = tk.Label(header, text="")
        self.counter_label.pack(side=tk.RIGHT)

        class_row = tk.Frame(self.root)
        class_row.pack(fill=tk.X, padx=8, pady=(0, 5))

        tk.Label(
            class_row,
            text="Single class:",
        ).pack(side=tk.LEFT)

        self.class_var = tk.StringVar(value=self.class_name)

        self.class_entry = tk.Entry(
            class_row,
            textvariable=self.class_var,
            width=22,
        )
        self.class_entry.pack(side=tk.LEFT, padx=(5, 5))

        tk.Button(
            class_row,
            text="Set Class",
            command=self.set_class_name,
        ).pack(side=tk.LEFT)

        self.source_label = tk.Label(
            class_row,
            text="",
            width=16,
            anchor="w",
        )
        self.source_label.pack(side=tk.LEFT, padx=(18, 0))

        self.box_count_label = tk.Label(
            class_row,
            text="Boxes: 0",
        )
        self.box_count_label.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            self.root,
            width=DISPLAY_W,
            height=DISPLAY_H,
            bg="#202020",
            cursor="crosshair",
            highlightthickness=1,
            highlightbackground="#808080",
        )
        self.canvas.pack(padx=8, pady=3)

        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(
            controls,
            text="◀ Previous",
            width=11,
            command=self.previous,
        ).pack(side=tk.LEFT)

        tk.Button(
            controls,
            text="Next ▶",
            width=11,
            command=self.next,
        ).pack(side=tk.LEFT, padx=(4, 12))

        tk.Button(
            controls,
            text="Undo",
            width=8,
            command=self.undo,
        ).pack(side=tk.LEFT)

        tk.Button(
            controls,
            text="Clear",
            width=8,
            command=self.clear_boxes,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            controls,
            text="Save",
            width=9,
            command=self.save,
        ).pack(side=tk.LEFT, padx=(12, 4))

        tk.Button(
            controls,
            text="Save && Next",
            width=12,
            command=self.save_next,
        ).pack(side=tk.LEFT)

        tk.Button(
            controls,
            text="Export ZIP",
            width=11,
            command=self.export_zip,
        ).pack(side=tk.RIGHT)

        help_row = tk.Label(
            self.root,
            text=(
                "Left-drag: add box    Right-click inside box: delete    "
                "A/D or ←/→: navigate    S: save    Enter: save+next"
            ),
            anchor="w",
        )
        help_row.pack(fill=tk.X, padx=8)

        self.status_label = tk.Label(
            self.root,
            text="",
            anchor="w",
            relief=tk.SUNKEN,
        )
        self.status_label.pack(
            fill=tk.X,
            padx=8,
            pady=(4, 8),
        )

        self.canvas.bind("<ButtonPress-1>", self.drag_start_event)
        self.canvas.bind("<B1-Motion>", self.drag_move_event)
        self.canvas.bind("<ButtonRelease-1>", self.drag_end_event)
        self.canvas.bind("<Button-3>", self.right_click_event)

    def bind_keys(self):
        self.root.bind("<Left>", lambda _e: self.previous())
        self.root.bind("<Right>", lambda _e: self.next())
        self.root.bind("a", lambda _e: self.previous())
        self.root.bind("d", lambda _e: self.next())
        self.root.bind("s", lambda _e: self.save())
        self.root.bind("<Return>", lambda _e: self.save_next())
        self.root.bind("z", lambda _e: self.undo())
        self.root.bind("<Delete>", lambda _e: self.delete_last())
        self.root.bind("<BackSpace>", lambda _e: self.delete_last())

    # ---------------------------------------------------------
    # Current sample
    # ---------------------------------------------------------

    @property
    def image_path(self) -> Path:
        return self.image_paths[self.index]

    @property
    def stem(self) -> str:
        return self.image_path.stem

    def verified_label_path(self) -> Path:
        return self.labels_dir / f"{self.stem}.txt"

    def pseudo_label_path(self) -> Path:
        return self.pseudo_dir / f"{self.stem}.txt"

    def reviewed_path(self) -> Path:
        return self.reviewed_dir / f"{self.stem}_verified.png"

    # ---------------------------------------------------------
    # YOLO I/O
    # ---------------------------------------------------------

    def read_boxes(self, path: Path) -> list[Box]:
        if not path.exists():
            return []

        boxes = []

        for line in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if not line.strip():
                continue

            box = Box.from_yolo(line)

            if box is not None:
                boxes.append(box)

        return boxes

    def load_index(self, index: int):
        self.index = index % len(self.image_paths)

        self.image = Image.open(self.image_path).convert("RGB")

        verified = self.verified_label_path()
        pseudo = self.pseudo_label_path()

        if verified.exists():
            self.boxes = self.read_boxes(verified)
            self.source = "VERIFIED"
        elif pseudo.exists():
            self.boxes = self.read_boxes(pseudo)
            self.source = "PSEUDO"
        else:
            self.boxes = []
            self.source = "EMPTY"

        self.undo_stack.clear()
        self.dirty = False

        self.file_label.config(text=self.image_path.name)
        self.counter_label.config(
            text=f"{self.index + 1} / {len(self.image_paths)}"
        )

        self.class_var.set(self.class_name)

        self.refresh()

        if self.source == "PSEUDO":
            self.set_status(
                "Pseudo-labels loaded. Correct them and Save to verify."
            )
        elif self.source == "VERIFIED":
            self.set_status("Verified YOLO labels loaded.")
        else:
            self.set_status(
                "No boxes. Draw objects, or Save empty to verify a negative frame."
            )

    def save(self):
        self.set_class_name(silent=True)

        path = self.verified_label_path()

        lines = [
            box.clean().to_yolo()
            for box in self.boxes
        ]

        content = "\n".join(lines)

        if content:
            content += "\n"

        # Empty file is a legitimate zero-object YOLO label.
        path.write_text(content, encoding="utf-8")

        self.save_reviewed_preview()

        self.source = "VERIFIED"
        self.dirty = False
        self.refresh_labels_only()

        self.set_status(
            f"Saved {len(self.boxes)} box(es) → labels/{path.name}"
        )

    def save_reviewed_preview(self):
        if self.image is None:
            return

        preview = self.image.copy()
        draw = ImageDraw.Draw(preview)

        w, h = preview.size
        font = ImageFont.load_default()

        for i, box in enumerate(self.boxes):
            b = box.clean()

            x1 = int(round(b.x1 * (w - 1)))
            y1 = int(round(b.y1 * (h - 1)))
            x2 = int(round(b.x2 * (w - 1)))
            y2 = int(round(b.y2 * (h - 1)))

            draw.rectangle(
                (x1, y1, x2, y2),
                outline=(255, 0, 0),
                width=max(1, round(min(w, h) / 128)),
            )

            draw.text(
                (x1 + 2, max(0, y1 - 11)),
                f"{self.class_name} #{i+1}",
                fill=(255, 0, 0),
                font=font,
            )

        preview.save(self.reviewed_path())

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    def refresh(self):
        if self.image is None:
            return

        shown = self.image.resize(
            (DISPLAY_W, DISPLAY_H),
            Image.Resampling.NEAREST,
        )

        self.tk_image = ImageTk.PhotoImage(shown)

        self.canvas.delete("all")
        self.canvas.create_image(
            0,
            0,
            anchor=tk.NW,
            image=self.tk_image,
        )

        self.draw_boxes()
        self.refresh_labels_only()

    def refresh_labels_only(self):
        self.source_label.config(
            text=f"Source: {self.source}"
        )

        self.box_count_label.config(
            text=f"Boxes: {len(self.boxes)}"
        )

    def draw_boxes(self):
        for i, box in enumerate(self.boxes):
            b = box.clean()

            x1 = b.x1 * DISPLAY_W
            y1 = b.y1 * DISPLAY_H
            x2 = b.x2 * DISPLAY_W
            y2 = b.y2 * DISPLAY_H

            color = (
                "#ffd400"
                if self.source == "PSEUDO" and not self.dirty
                else "#00ff66"
            )

            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=color,
                width=3,
                tags=("box",),
            )

            self.canvas.create_text(
                x1 + 4,
                y1 + 4,
                text=f"{i+1}: {self.class_name}",
                fill=color,
                anchor=tk.NW,
                font=("TkDefaultFont", 9, "bold"),
                tags=("box",),
            )

    def set_status(self, text: str):
        self.status_label.config(text=text)

    # ---------------------------------------------------------
    # Editing
    # ---------------------------------------------------------

    def snapshot(self):
        self.undo_stack.append(
            [Box(b.x1, b.y1, b.x2, b.y2) for b in self.boxes]
        )

        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    @staticmethod
    def clamp(v: int, maximum: int) -> int:
        return max(0, min(maximum - 1, v))

    def drag_start_event(self, event):
        x = self.clamp(event.x, DISPLAY_W)
        y = self.clamp(event.y, DISPLAY_H)

        self.drag_start = (x, y)

        if self.drag_rect_id is not None:
            self.canvas.delete(self.drag_rect_id)

        self.drag_rect_id = self.canvas.create_rectangle(
            x,
            y,
            x,
            y,
            outline="#00ffff",
            width=2,
            dash=(5, 3),
        )

    def drag_move_event(self, event):
        if self.drag_start is None or self.drag_rect_id is None:
            return

        x = self.clamp(event.x, DISPLAY_W)
        y = self.clamp(event.y, DISPLAY_H)

        x0, y0 = self.drag_start

        self.canvas.coords(
            self.drag_rect_id,
            x0,
            y0,
            x,
            y,
        )

    def drag_end_event(self, event):
        if self.drag_start is None:
            return

        x0, y0 = self.drag_start
        x1 = self.clamp(event.x, DISPLAY_W)
        y1 = self.clamp(event.y, DISPLAY_H)

        self.drag_start = None

        if self.drag_rect_id is not None:
            self.canvas.delete(self.drag_rect_id)
            self.drag_rect_id = None

        if (
            abs(x1 - x0) < MIN_DRAG_PX
            or abs(y1 - y0) < MIN_DRAG_PX
        ):
            self.set_status("Ignored tiny drag.")
            return

        self.snapshot()

        self.boxes.append(
            Box(
                x0 / DISPLAY_W,
                y0 / DISPLAY_H,
                x1 / DISPLAY_W,
                y1 / DISPLAY_H,
            ).clean()
        )

        self.source = "EDITED"
        self.dirty = True

        self.refresh()

        self.set_status(
            f"Added box #{len(self.boxes)}. Save when finished."
        )

    def right_click_event(self, event):
        if not self.boxes:
            return

        nx = self.clamp(event.x, DISPLAY_W) / DISPLAY_W
        ny = self.clamp(event.y, DISPLAY_H) / DISPLAY_H

        hits = []

        for i, box in enumerate(self.boxes):
            b = box.clean()

            if b.x1 <= nx <= b.x2 and b.y1 <= ny <= b.y2:
                area = (b.x2 - b.x1) * (b.y2 - b.y1)
                hits.append((area, i))

        if not hits:
            self.set_status(
                "Right-click inside an existing box to delete it."
            )
            return

        _, index = min(hits)

        self.snapshot()
        del self.boxes[index]

        self.source = "EDITED"
        self.dirty = True

        self.refresh()
        self.set_status(f"Deleted box #{index + 1}.")

    def undo(self):
        if not self.undo_stack:
            self.set_status("Nothing to undo.")
            return

        self.boxes = self.undo_stack.pop()
        self.source = "EDITED"
        self.dirty = True

        self.refresh()
        self.set_status("Undo.")

    def clear_boxes(self):
        if self.boxes:
            self.snapshot()

        self.boxes = []
        self.source = "EDITED"
        self.dirty = True

        self.refresh()

        self.set_status(
            "Boxes cleared. Save to confirm this is a zero-object image."
        )

    def delete_last(self):
        if not self.boxes:
            return

        self.snapshot()
        self.boxes.pop()

        self.source = "EDITED"
        self.dirty = True

        self.refresh()
        self.set_status("Deleted last box.")

    # ---------------------------------------------------------
    # Navigation / class / export
    # ---------------------------------------------------------

    def previous(self):
        self.load_index(self.index - 1)

    def next(self):
        self.load_index(self.index + 1)

    def save_next(self):
        self.save()
        self.load_index(self.index + 1)

    def set_class_name(self, silent: bool = False):
        value = self.class_var.get().strip()

        if not value:
            if not silent:
                messagebox.showwarning(
                    "Cute-YOLO",
                    "Class name cannot be empty.",
                )
            return

        # Single token is safer for serial/model metadata.
        value = value.replace(" ", "_")

        self.class_name = value
        self.class_var.set(value)
        self.write_classes_txt()

        self.refresh()

        if not silent:
            self.set_status(
                f"Single class set to '{self.class_name}' (YOLO class id 0)."
            )

    def export_zip(self):
        self.set_class_name(silent=True)

        default_name = f"{self.class_name}_labeled.zip"

        output = filedialog.asksaveasfilename(
            title="Export labeled YOLO dataset",
            defaultextension=".zip",
            initialfile=default_name,
            filetypes=[("ZIP archive", "*.zip")],
        )

        if not output:
            return

        output_path = Path(output)

        root_name = output_path.stem

        with zipfile.ZipFile(
            output_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as z:
            classes_path = self.dataset_root / "classes.txt"

            z.write(
                classes_path,
                f"{root_name}/classes.txt",
            )

            for image_path in self.image_paths:
                z.write(
                    image_path,
                    f"{root_name}/images/{image_path.name}",
                )

                label_path = (
                    self.labels_dir
                    / f"{image_path.stem}.txt"
                )

                # Ensure every image has a matching label file.
                # Missing labels become empty negative-frame labels.
                if not label_path.exists():
                    label_path.write_text(
                        "",
                        encoding="utf-8",
                    )

                z.write(
                    label_path,
                    f"{root_name}/labels/{label_path.name}",
                )

        self.set_status(
            f"Exported standard YOLO dataset: {output_path.name}"
        )

        messagebox.showinfo(
            "Cute-YOLO",
            (
                "Dataset exported.\n\n"
                f"{output_path}\n\n"
                "Contains classes.txt, images/, and labels/."
            ),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Cute-YOLO single-class standard-YOLO labeling GUI"
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default="cute_real_dataset",
        type=Path,
        help="dataset root containing images/",
    )

    args = parser.parse_args()

    root = tk.Tk()

    try:
        CuteLabelGUI(root, args.dataset)
    except Exception as exc:
        messagebox.showerror(
            "Cute-YOLO Labeler",
            str(exc),
        )
        root.destroy()
        raise SystemExit(1)

    root.mainloop()


if __name__ == "__main__":
    main()
