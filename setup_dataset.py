"""
setup_dataset.py — One-time script to remap the downloaded dataset
===================================================================
Remaps the 6-class Roboflow dataset to our 2-class format:
  0: Helmet     (was: 0=1-2-helmet, 1=3-4-helmet, 5=Full-face-helmet)
  1: No Helmet  (was: 2=Bald, 3=Cap, 4=Face and Hair)

Then copies everything into the dataset/ folder ready for training.
"""

import os
import shutil
from pathlib import Path


# Class mapping: old_class_id → new_class_id
# Old: 0=1-2-helmet, 1=3-4-helmet, 2=Bald, 3=Cap, 4=Face and Hair, 5=Full-face-helmet
# New: 0=Helmet, 1=No Helmet
CLASS_MAP = {
    "0": "0",  # 1-2-helmet → Helmet
    "1": "0",  # 3-4-helmet → Helmet
    "2": "1",  # Bald → No Helmet
    "3": "1",  # Cap → No Helmet
    "4": "1",  # Face and Hair → No Helmet
    "5": "0",  # Full-face-helmet → Helmet
}

# Paths
SOURCE = Path("downloaded_dataset_v2")
DEST = Path("dataset")


def remap_and_copy_labels(src_label_dir: Path, dest_label_dir: Path):
    """Remap class IDs in label files and copy to destination."""
    dest_label_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for label_file in src_label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] in CLASS_MAP:
                parts[0] = CLASS_MAP[parts[0]]
                new_lines.append(" ".join(parts))

        # Write remapped label
        dest_file = dest_label_dir / label_file.name
        with open(dest_file, "w") as f:
            f.write("\n".join(new_lines) + "\n" if new_lines else "")
        count += 1

    return count


def copy_images(src_img_dir: Path, dest_img_dir: Path):
    """Copy images to destination."""
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for img_file in src_img_dir.iterdir():
        if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            shutil.copy2(img_file, dest_img_dir / img_file.name)
            count += 1

    return count


def main():
    print("=" * 50)
    print("🛡️  DATASET SETUP — Remapping 6 → 2 classes")
    print("=" * 50)
    print()
    print("Class mapping:")
    print("  0 (1-2-helmet)       → 0 (Helmet)")
    print("  1 (3-4-helmet)       → 0 (Helmet)")
    print("  2 (Bald)             → 1 (No Helmet)")
    print("  3 (Cap)              → 1 (No Helmet)")
    print("  4 (Face and Hair)    → 1 (No Helmet)")
    print("  5 (Full-face-helmet) → 0 (Helmet)")
    print()

    # Process train split
    print("📂 Processing train split...")
    n_img = copy_images(SOURCE / "train" / "images", DEST / "train" / "images")
    n_lbl = remap_and_copy_labels(SOURCE / "train" / "labels", DEST / "train" / "labels")
    print(f"   ✅ {n_img} images, {n_lbl} labels")

    # Process valid → val
    print("📂 Processing validation split...")
    n_img = copy_images(SOURCE / "valid" / "images", DEST / "val" / "images")
    n_lbl = remap_and_copy_labels(SOURCE / "valid" / "labels", DEST / "val" / "labels")
    print(f"   ✅ {n_img} images, {n_lbl} labels")

    # Process test (merge into val for more validation data)
    print("📂 Processing test split (merging into val)...")
    n_img = copy_images(SOURCE / "test" / "images", DEST / "val" / "images")
    n_lbl = remap_and_copy_labels(SOURCE / "test" / "labels", DEST / "val" / "labels")
    print(f"   ✅ {n_img} images, {n_lbl} labels")

    # Count final class distribution
    helmet_count = 0
    no_helmet_count = 0
    for split in ["train", "val"]:
        label_dir = DEST / split / "labels"
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        if parts[0] == "0":
                            helmet_count += 1
                        elif parts[0] == "1":
                            no_helmet_count += 1

    print()
    print("=" * 50)
    print("✅ DATASET READY!")
    print("=" * 50)
    print(f"  📊 Helmet:     {helmet_count} instances")
    print(f"  📊 No Helmet:  {no_helmet_count} instances")
    print(f"  📂 Train:      {len(list((DEST / 'train' / 'images').glob('*')))} images")
    print(f"  📂 Val:        {len(list((DEST / 'val' / 'images').glob('*')))} images")
    print()
    print("  Next step: python train.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
