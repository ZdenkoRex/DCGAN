#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
import random

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------

ROOT = Path(".")
PROCESSED_ROOT = ROOT / "processed"

IN_FOLDERS = [
    PROCESSED_ROOT / "Fridtjof",
    PROCESSED_ROOT / "Marie",
]

OUT_TRAIN = ROOT / "datasets" / "sonar" / "trainB"
OUT_TEST  = ROOT / "datasets" / "sonar" / "testB"

PATCH_SIZE = 256
STRIDE = 128          # overlap; smaller stride = more patches
TEST_RATIO = 0.10     # ~10% of patches to testB

random.seed(42)       # reproducible train/test split


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def make_patches_from_image(img: Image.Image, prefix: str):
    """Split image into left/right and generate patches."""
    w, h = img.size

    # Split into left and right side
    mid = w // 2
    left = img.crop((0, 0, mid, h))
    right = img.crop((mid, 0, w, h))

    sides = [("L", left), ("R", right)]

    patch_index = 0
    for side_tag, side_img in sides:
        sw, sh = side_img.size

        for y in range(0, sh - PATCH_SIZE + 1, STRIDE):
            for x in range(0, sw - PATCH_SIZE + 1, STRIDE):
                patch = side_img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

                # Decide whether this patch goes to train or test
                if random.random() < TEST_RATIO:
                    out_dir = OUT_TEST
                else:
                    out_dir = OUT_TRAIN

                out_dir.mkdir(parents=True, exist_ok=True)

                filename = f"B_{prefix}_{side_tag}_{patch_index:05d}.png"
                patch.save(out_dir / filename)

                patch_index += 1


def main():
    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_TEST.mkdir(parents=True, exist_ok=True)

    all_images = []
    for folder in IN_FOLDERS:
        all_images.extend(sorted(folder.glob("*.png")))

    print(f"Found {len(all_images)} images in processed/")

    for idx, path in enumerate(all_images, start=1):
        print(f"Processing {idx}/{len(all_images)}: {path}")
        img = Image.open(path).convert("L")  # just to be safe

        # prefix encodes original file stem
        prefix = path.stem
        make_patches_from_image(img, prefix)

    print("Done. Patches written to:")
    print(f"  {OUT_TRAIN}")
    print(f"  {OUT_TEST}")


if __name__ == "__main__":
    main()
