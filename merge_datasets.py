"""
merge_datasets.py — Merge Roboflow datasets into a unified YOLO dataset
=========================================================================
Combines two Roboflow exports into one dataset with 2 classes:
  Class 0: Helmet
  Class 1: No Helmet

Dataset 1 (roboflow_helmet):
  - Original: nc=1, names=['Helmet']
  - Class 0 (Helmet) → maps to OUR Class 0 (Helmet)

Dataset 2 (roboflow_no_helmet):
  - Original: nc=6, names=['1-2-helmet', '3-4-helmet', 'Bald', 'Cap', 'Face and Hair', 'Full-face-helmet']
  - Class 0 (1-2-helmet)       → OUR Class 0 (Helmet)
  - Class 1 (3-4-helmet)       → OUR Class 0 (Helmet)
  - Class 2 (Bald)             → OUR Class 1 (No Helmet)
  - Class 3 (Cap)              → OUR Class 1 (No Helmet)
  - Class 4 (Face and Hair)    → OUR Class 1 (No Helmet)
  - Class 5 (Full-face-helmet) → OUR Class 0 (Helmet)
"""

import os
import shutil
from pathlib import Path

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ── Config ──────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_DIR / "dataset"

# Dataset 1: roboflow_helmet (1 class: Helmet)
DS1_DIR = PROJECT_DIR / "roboflow_helmet"
DS1_CLASS_MAP = {
    0: 0,   # Helmet → Helmet
}

# Dataset 2: roboflow_no_helmet (6 classes)
DS2_DIR = PROJECT_DIR / "roboflow_no_helmet"
DS2_CLASS_MAP = {
    0: 0,   # 1-2-helmet       → Helmet
    1: 0,   # 3-4-helmet       → Helmet
    2: 1,   # Bald             → No Helmet
    3: 1,   # Cap              → No Helmet
    4: 1,   # Face and Hair    → No Helmet
    5: 0,   # Full-face-helmet → Helmet
}


def remap_labels(src_label_path, dst_label_path, class_map):
    """Read a YOLO label file, remap class IDs, and write to destination."""
    lines = []
    with open(src_label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_class = int(parts[0])
            if old_class in class_map:
                new_class = class_map[old_class]
                parts[0] = str(new_class)
                lines.append(' '.join(parts))

    with open(dst_label_path, 'w') as f:
        f.write('\n'.join(lines))
        if lines:
            f.write('\n')


def copy_dataset(src_dir, class_map, prefix, stats):
    """Copy images and remapped labels from a source dataset into the output."""

    # Roboflow uses 'valid' folder name, we use 'val'
    split_mapping = {
        'train': 'train',
        'valid': 'val',
        'test': 'val',   # Put test images into val set
    }

    for src_split, dst_split in split_mapping.items():
        src_images = src_dir / src_split / "images"
        src_labels = src_dir / src_split / "labels"

        if not src_images.exists():
            print(f"  ⚠️  Skipping {src_split} (not found)")
            continue

        dst_images = OUTPUT_DIR / dst_split / "images"
        dst_labels = OUTPUT_DIR / dst_split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        image_files = [f for f in os.listdir(src_images)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

        copied = 0
        skipped = 0
        for img_file in image_files:
            # Add prefix to avoid filename collisions between datasets
            new_name = f"{prefix}_{img_file}"
            label_file = os.path.splitext(img_file)[0] + '.txt'
            new_label = f"{prefix}_{os.path.splitext(img_file)[0]}.txt"

            src_img_path = src_images / img_file
            src_lbl_path = src_labels / label_file
            dst_img_path = dst_images / new_name
            dst_lbl_path = dst_labels / new_label

            # Skip if already exists
            if dst_img_path.exists():
                skipped += 1
                continue

            # Copy image
            shutil.copy2(src_img_path, dst_img_path)

            # Copy and remap label
            if src_lbl_path.exists():
                remap_labels(src_lbl_path, dst_lbl_path, class_map)
            else:
                # Create empty label file (no objects in image)
                dst_lbl_path.touch()

            copied += 1

        print(f"  {src_split} → {dst_split}: {copied} images copied, {skipped} skipped")
        stats[dst_split] = stats.get(dst_split, 0) + copied


def main():
    print("=" * 60)
    print("🔀 MERGING ROBOFLOW DATASETS")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"\nTarget classes:")
    print(f"  0 = Helmet")
    print(f"  1 = No Helmet")
    print()

    stats = {}

    # ── Dataset 1 ──
    if DS1_DIR.exists():
        print(f"📦 Dataset 1: {DS1_DIR.name}")
        print(f"   Classes: Helmet (→0)")
        copy_dataset(DS1_DIR, DS1_CLASS_MAP, "ds1", stats)
    else:
        print(f"❌ Dataset 1 not found: {DS1_DIR}")

    print()

    # ── Dataset 2 ──
    if DS2_DIR.exists():
        print(f"📦 Dataset 2: {DS2_DIR.name}")
        print(f"   Classes: 1-2-helmet(→0), 3-4-helmet(→0), Bald(→1), Cap(→1), Face&Hair(→1), Full-face(→0)")
        copy_dataset(DS2_DIR, DS2_CLASS_MAP, "ds2", stats)
    else:
        print(f"❌ Dataset 2 not found: {DS2_DIR}")

    # ── Summary ──
    print()
    print("=" * 60)
    print("✅ MERGE COMPLETE!")
    print("=" * 60)

    for split, count in stats.items():
        split_dir = OUTPUT_DIR / split / "images"
        total = len([f for f in os.listdir(split_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]) if split_dir.exists() else 0
        print(f"  {split}: {total} total images")

    # Verify matching labels
    print("\n📋 Verification:")
    for split in ['train', 'val']:
        img_dir = OUTPUT_DIR / split / "images"
        lbl_dir = OUTPUT_DIR / split / "labels"
        if img_dir.exists() and lbl_dir.exists():
            images = set(os.path.splitext(f)[0] for f in os.listdir(img_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')))
            labels = set(os.path.splitext(f)[0] for f in os.listdir(lbl_dir)
                         if f.endswith('.txt'))
            missing = images - labels
            if missing:
                print(f"  ⚠️ {split}: {len(missing)} images missing labels")
            else:
                print(f"  ✅ {split}: All {len(images)} images have labels")

    print(f"\n🎯 Next step: python train.py")


if __name__ == "__main__":
    main()
