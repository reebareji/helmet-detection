"""
prepare_dataset.py — Dataset Preparation & Merging Utility
===========================================================
This script helps you:
  1. Merge multiple dataset sources into one unified dataset
  2. Remap class IDs from different datasets to your standard
  3. Split images into train/val sets
  4. Validate that every image has a matching label file

Usage:
  # Merge a folder of images+labels into the dataset
  python prepare_dataset.py --source ./my_images --labels ./my_labels

  # Merge + remap class IDs (e.g., old class 2 → new class 0)
  python prepare_dataset.py --source ./downloaded_data/images --labels ./downloaded_data/labels --remap "2:0,3:1"

  # Just split existing data into train/val
  python prepare_dataset.py --split-only --ratio 0.8

  # Validate dataset integrity
  python prepare_dataset.py --validate
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
DATASET_DIR = Path("dataset")
TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"
VAL_IMAGES = DATASET_DIR / "val" / "images"
VAL_LABELS = DATASET_DIR / "val" / "labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_dirs():
    """Create the dataset directory structure."""
    for d in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ Dataset directories created.")


def remap_labels(label_dir: Path, class_map: dict):
    """
    Remap class IDs in YOLO label files.
    
    Args:
        label_dir: Path to directory containing .txt label files
        class_map: Dictionary mapping old_id (str) → new_id (str)
                   Example: {"2": "0", "3": "1"}
    """
    count = 0
    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        modified = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] in class_map:
                parts[0] = class_map[parts[0]]
                modified = True
            new_lines.append(" ".join(parts))

        if modified:
            with open(label_file, "w") as f:
                f.write("\n".join(new_lines) + "\n")
            count += 1

    print(f"✅ Remapped class IDs in {count} label files.")


def merge_data(source_images: Path, source_labels: Path, dest_images: Path, dest_labels: Path):
    """
    Copy images and their matching labels from source to destination.
    Handles filename conflicts by adding a suffix.
    """
    copied = 0
    skipped = 0

    for img_file in source_images.iterdir():
        if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_file = source_labels / (img_file.stem + ".txt")

        # Handle filename conflicts
        dest_img = dest_images / img_file.name
        dest_lbl = dest_labels / (img_file.stem + ".txt")

        if dest_img.exists():
            # Add a random suffix to avoid overwriting
            suffix = f"_{random.randint(1000, 9999)}"
            new_name = f"{img_file.stem}{suffix}{img_file.suffix}"
            dest_img = dest_images / new_name
            dest_lbl = dest_labels / f"{img_file.stem}{suffix}.txt"

        shutil.copy2(img_file, dest_img)

        if label_file.exists():
            shutil.copy2(label_file, dest_lbl)
        else:
            # Create empty label file (image with no objects)
            dest_lbl.touch()
            skipped += 1

        copied += 1

    print(f"✅ Merged {copied} images ({skipped} had no label file).")


def split_dataset(ratio: float = 0.8):
    """
    Split all images in train/ into train/ and val/ directories.
    
    Args:
        ratio: Fraction of data to keep in training set (default 0.8 = 80%)
    """
    all_images = [f for f in TRAIN_IMAGES.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

    if not all_images:
        print("❌ No images found in train/images/. Add images first.")
        return

    random.shuffle(all_images)
    split_idx = int(len(all_images) * ratio)

    val_images = all_images[split_idx:]
    moved = 0

    for img in val_images:
        label = TRAIN_LABELS / (img.stem + ".txt")

        # Move image
        shutil.move(str(img), str(VAL_IMAGES / img.name))

        # Move label
        if label.exists():
            shutil.move(str(label), str(VAL_LABELS / label.name))

        moved += 1

    print(f"✅ Split complete: {split_idx} train / {moved} val images.")


def validate_dataset():
    """
    Check dataset integrity:
      - Every image should have a matching .txt label
      - Every label file should have valid YOLO format
      - Report class distribution
    """
    print("\n🔍 Validating dataset...\n")
    issues = []
    class_counts = {}

    for split_name, img_dir, lbl_dir in [
        ("train", TRAIN_IMAGES, TRAIN_LABELS),
        ("val", VAL_IMAGES, VAL_LABELS),
    ]:
        images = [f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS] if img_dir.exists() else []
        labels = [f for f in lbl_dir.iterdir() if f.suffix == ".txt"] if lbl_dir.exists() else []

        image_stems = {f.stem for f in images}
        label_stems = {f.stem for f in labels}

        # Images without labels
        missing_labels = image_stems - label_stems
        if missing_labels:
            issues.append(f"⚠️  {split_name}: {len(missing_labels)} images have no label file")
            for name in list(missing_labels)[:5]:
                issues.append(f"     → {name}")

        # Labels without images
        orphan_labels = label_stems - image_stems
        if orphan_labels:
            issues.append(f"⚠️  {split_name}: {len(orphan_labels)} labels have no matching image")

        # Validate label format & count classes
        for lbl_file in labels:
            with open(lbl_dir / lbl_file.name, "r") as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue
                    if len(parts) != 5:
                        issues.append(
                            f"❌ {split_name}/{lbl_file.name} line {line_num}: "
                            f"expected 5 values, got {len(parts)}"
                        )
                        continue

                    class_id = parts[0]
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

                    # Check if coordinates are valid floats in [0, 1]
                    try:
                        coords = [float(x) for x in parts[1:]]
                        for val in coords:
                            if val < 0 or val > 1:
                                issues.append(
                                    f"⚠️  {split_name}/{lbl_file.name} line {line_num}: "
                                    f"coordinate out of range [0,1]: {val}"
                                )
                    except ValueError:
                        issues.append(
                            f"❌ {split_name}/{lbl_file.name} line {line_num}: "
                            f"non-numeric coordinates"
                        )

        print(f"📂 {split_name}: {len(images)} images, {len(labels)} labels")

    # Print class distribution
    print("\n📊 Class distribution:")
    for cls_id in sorted(class_counts.keys()):
        label = {0: "Helmet", 1: "No Helmet"}.get(int(cls_id), f"Unknown({cls_id})")
        print(f"   Class {cls_id} ({label}): {class_counts[cls_id]} instances")

    # Print issues
    if issues:
        print(f"\n⚠️  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ No issues found! Dataset looks good.")


def parse_remap(remap_str: str) -> dict:
    """Parse remap string like '2:0,3:1' into dict {'2':'0', '3':'1'}."""
    mapping = {}
    for pair in remap_str.split(","):
        old, new = pair.strip().split(":")
        mapping[old.strip()] = new.strip()
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="🛡️ Helmet Detection Dataset Preparation Tool"
    )
    parser.add_argument(
        "--source", type=str,
        help="Path to source images directory"
    )
    parser.add_argument(
        "--labels", type=str,
        help="Path to source labels directory"
    )
    parser.add_argument(
        "--remap", type=str, default=None,
        help='Remap class IDs, e.g., "2:0,3:1" maps old class 2→0 and 3→1'
    )
    parser.add_argument(
        "--split-only", action="store_true",
        help="Only perform train/val split on existing data"
    )
    parser.add_argument(
        "--ratio", type=float, default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate dataset integrity"
    )

    args = parser.parse_args()

    # Always create directory structure
    create_dirs()

    if args.validate:
        validate_dataset()
        return

    if args.split_only:
        split_dataset(args.ratio)
        return

    if args.source and args.labels:
        source_images = Path(args.source)
        source_labels = Path(args.labels)

        if not source_images.exists():
            print(f"❌ Source images directory not found: {source_images}")
            sys.exit(1)
        if not source_labels.exists():
            print(f"❌ Source labels directory not found: {source_labels}")
            sys.exit(1)

        # Remap class IDs if requested
        if args.remap:
            class_map = parse_remap(args.remap)
            print(f"🔄 Remapping classes: {class_map}")
            remap_labels(source_labels, class_map)

        # Merge into train set (you can split after)
        merge_data(source_images, source_labels, TRAIN_IMAGES, TRAIN_LABELS)

        # Ask if user wants to split
        print(f"\n💡 Run with --split-only --ratio {args.ratio} to split into train/val.")
    else:
        parser.print_help()
        print("\n📌 Examples:")
        print("  python prepare_dataset.py --source ./raw_images --labels ./raw_labels")
        print('  python prepare_dataset.py --source ./data/images --labels ./data/labels --remap "2:0,3:1"')
        print("  python prepare_dataset.py --split-only --ratio 0.8")
        print("  python prepare_dataset.py --validate")


if __name__ == "__main__":
    main()
