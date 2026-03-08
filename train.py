"""
train.py — YOLOv8 Model Training Script
=========================================
Trains a YOLOv8 model on the helmet detection dataset.

Usage:
  # Train with default settings (recommended for first run)
  python train.py

  # Train with custom settings
  python train.py --epochs 100 --batch 32 --model yolov8s.pt --imgsz 640

  # Resume training from last checkpoint
  python train.py --resume
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO

# Resolve all paths relative to this script's directory
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="🛡️ Train YOLOv8 Helmet Detection Model"
    )
    parser.add_argument(
        "--model", type=str, default=str(PROJECT_DIR / "yolov8n.pt"),
        help="Pretrained model to start from (default: yolov8n.pt — fastest). "
             "Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt"
    )
    parser.add_argument(
        "--data", type=str, default=str(PROJECT_DIR / "data.yaml"),
        help="Path to data.yaml config file (default: data.yaml)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (default: 16, reduce to 8 if you get memory errors)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to train on: 'cpu', '0' (GPU 0), etc. Auto-detects by default."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--name", type=str, default="helmet_detector",
        help="Name for this training run (default: helmet_detector)"
    )

    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────
    # Validate that the data.yaml and dataset exist
    # ──────────────────────────────────────────────────────────
    # Change working directory to project dir so YOLO resolves paths correctly
    os.chdir(PROJECT_DIR)
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Data config file not found: {data_path}")
        print("   Make sure data.yaml exists and paths inside it are correct.")
        return

    print("=" * 60)
    print("🛡️  HELMET DETECTION — MODEL TRAINING")
    print("=" * 60)
    print(f"  Model     : {args.model}")
    print(f"  Data      : {args.data}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch Size: {args.batch}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Device    : {args.device or 'auto'}")
    print(f"  Run Name  : {args.name}")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # Load model & train
    # ──────────────────────────────────────────────────────────
    if args.resume:
        # Resume from last checkpoint
        last_checkpoint = Path(f"runs/detect/{args.name}/weights/last.pt")
        if not last_checkpoint.exists():
            print(f"❌ No checkpoint found at {last_checkpoint}")
            print("   Start a fresh training run first.")
            return
        model = YOLO(str(last_checkpoint))
        print(f"\n🔄 Resuming training from: {last_checkpoint}")
    else:
        # Start fresh from pretrained model
        model = YOLO(args.model)
        print(f"\n📦 Loaded pretrained model: {args.model}")

    # Start training
    print("\n🚀 Starting training...\n")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        patience=20,         # Early stopping: stop if no improvement for 20 epochs
        save=True,           # Save checkpoints
        save_period=10,      # Save checkpoint every 10 epochs
        plots=True,          # Generate training plots
        verbose=True,
    )

    # ──────────────────────────────────────────────────────────
    # Post-training summary
    # ──────────────────────────────────────────────────────────
    best_model = Path(f"runs/detect/{args.name}/weights/best.pt")

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Best model saved to : {best_model}")
    print(f"  Training results in : runs/detect/{args.name}/")
    print()
    print("  📊 Check these files for training metrics:")
    print(f"     • runs/detect/{args.name}/results.png")
    print(f"     • runs/detect/{args.name}/confusion_matrix.png")
    print()
    print("  🎯 To run detection:")
    print(f"     python detect.py --model {best_model} --source your_image.jpg")
    print(f"     python detect.py --model {best_model} --source 0  (webcam)")
    print("=" * 60)


if __name__ == "__main__":
    main()
