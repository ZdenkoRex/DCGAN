#!/usr/bin/env python3
from pathlib import Path
from PIL import Image

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------

# Root folder that contains "Screenshots_Fridtjof" and "Screenshots_Marie"
ROOT = Path(".")

# Input folders with your screenshots
IN_FOLDERS = {
    "Fridtjof": ROOT / "Screenshots_Fridtjof",
    "Marie": ROOT / "Screenshots_Marie",
}

# Output root folder
OUT_ROOT = ROOT / "processed"

# If True, only print actions (no files written)
DRY_RUN = False

# Target height after preprocessing; width will be kept unchanged
TARGET_HEIGHT = 512  # adjust if you prefer another height

# Optional: crop a few pixels from top/bottom before resizing
CROP_TOP = 0         # e.g. 10–20 if you later discover artefacts at the edges
CROP_BOTTOM = 0


# ----------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------

def process_folder(label: str, in_dir: Path):
    """
    label:    'Fridtjof' or 'Marie' (used in output filename prefix)
    in_dir:   path to folder containing original screenshots
    """
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # All png files that start with "Screenshot"
    files = sorted(in_dir.glob("Screenshot*.png"))

    print(f"\nProcessing {label}: found {len(files)} files in {in_dir}")

    for idx, src_path in enumerate(files, start=1):
        # New filename: e.g. Fridtjof_0001.png
        new_name = f"{label}_{idx:04d}.png"
        dst_path = out_dir / new_name

        print(f"  {src_path.name} -> {dst_path.relative_to(ROOT)}")

        if DRY_RUN:
            continue

        # --- Load image ---
        img = Image.open(src_path)

        # Convert brown RGB image to grayscale
        img = img.convert("L")

        # Optional crop
        if CROP_TOP > 0 or CROP_BOTTOM > 0:
            w, h = img.size
            top = CROP_TOP
            bottom = h - CROP_BOTTOM
            img = img.crop((0, top, w, bottom))  # (left, top, right, bottom)

        # Resize only vertically to TARGET_HEIGHT, keep width
        w, h = img.size
        if h != TARGET_HEIGHT:
            img = img.resize((w, TARGET_HEIGHT), Image.BICUBIC)

        # Save processed image
        img.save(dst_path)


def main():
    OUT_ROOT.mkdir(exist_ok=True)
    for label, in_dir in IN_FOLDERS.items():
        if not in_dir.is_dir():
            print(f"WARNING: input folder does not exist: {in_dir}")
            continue
        process_folder(label, in_dir)


if __name__ == "__main__":
    main()
