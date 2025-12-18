#!/usr/bin/env python3
from pathlib import Path
from PIL import Image

ROOT = Path(".")
IN_ROOT = ROOT / "processed"
OUT_ROOT = ROOT / "full_512x128"

TARGET_W = 512
TARGET_H = 128

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    in_folders = [
        IN_ROOT / "Fridtjof",
        IN_ROOT / "Marie",   # remove this line if you really only want Fridtjof
    ]

    all_files = []
    for folder in in_folders:
        if not folder.is_dir():
            continue
        all_files.extend(sorted(folder.glob("*.png")))

    print(f"Found {len(all_files)} images in processed/")

    for i, path in enumerate(all_files, start=1):
        img = Image.open(path).convert("L")   # grayscale
        img_resized = img.resize((TARGET_W, TARGET_H), Image.BICUBIC)

        out_name = OUT_ROOT / path.name      # keep same filename
        img_resized.save(out_name)

        print(f"[{i}/{len(all_files)}] {path} -> {out_name}")

    print("Done. Resized images in:", OUT_ROOT)


if __name__ == "__main__":
    main()
