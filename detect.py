"""
detect.py — Helmet Detection Inference Script
===============================================
Run detection on images, videos, or live webcam feed.
Draws bounding boxes with "Helmet" (green) and "No Helmet" (red) labels.

Usage:
  # Detect on an image
  python detect.py --source path/to/image.jpg

  # Detect on a video file
  python detect.py --source path/to/video.mp4

  # Detect on webcam (live demo)
  python detect.py --source 0

  # With custom model and confidence threshold
  python detect.py --source 0 --model runs/detect/helmet_detector/weights/best.pt --conf 0.5

  # Save output to file
  python detect.py --source path/to/video.mp4 --save
"""

import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────
# Visual styling for bounding boxes
# ──────────────────────────────────────────────────────────────
# Colors in BGR format
COLORS = {
    0: (0, 200, 0),      # Helmet → Green
    1: (0, 0, 220),      # No Helmet → Red
}

CLASS_NAMES = {
    0: "Helmet",
    1: "No Helmet",
}

# Box styling
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
LABEL_PADDING = 5


def draw_detections(frame, results):
    """
    Draw bounding boxes and labels on the frame.
    Green for Helmet, Red for No Helmet.
    
    Args:
        frame: The image/frame (numpy array, BGR format)
        results: YOLO prediction results
        
    Returns:
        frame: Annotated frame
        counts: Dict with detection counts {'Helmet': N, 'No Helmet': M}
    """
    counts = {"Helmet": 0, "No Helmet": 0}

    for result in results:
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get class and confidence
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())

            # Get color and label
            color = COLORS.get(class_id, (255, 255, 255))
            label = CLASS_NAMES.get(class_id, f"Class {class_id}")
            display_text = f"{label} {confidence:.0%}"

            # Update counts
            if label in counts:
                counts[label] += 1

            # Draw filled rectangle behind label text for readability
            text_size = cv2.getTextSize(display_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
            label_bg_x2 = x1 + text_size[0] + LABEL_PADDING * 2
            label_bg_y1 = y1 - text_size[1] - LABEL_PADDING * 2

            # Make sure label doesn't go above frame
            if label_bg_y1 < 0:
                label_bg_y1 = y1
                text_y = y1 + text_size[1] + LABEL_PADDING
            else:
                text_y = y1 - LABEL_PADDING

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            # Draw label background
            cv2.rectangle(
                frame,
                (x1, label_bg_y1),
                (label_bg_x2, y1),
                color,
                -1  # Filled
            )

            # Draw label text (white on colored background)
            cv2.putText(
                frame, display_text,
                (x1 + LABEL_PADDING, text_y),
                FONT, FONT_SCALE,
                (255, 255, 255),  # White text
                FONT_THICKNESS
            )

    return frame, counts


def draw_stats_overlay(frame, counts, fps=None):
    """
    Draw a semi-transparent stats overlay in the top-left corner.
    Shows detection counts and FPS.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Background rectangle
    stats_h = 100 if fps else 80
    cv2.rectangle(overlay, (10, 10), (280, stats_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Title
    cv2.putText(frame, "Detection Stats", (20, 35), FONT, 0.6, (255, 255, 255), 1)

    # Helmet count (green)
    cv2.putText(
        frame, f"Helmet: {counts.get('Helmet', 0)}",
        (20, 58), FONT, 0.55, (0, 200, 0), 1
    )

    # No Helmet count (red)
    cv2.putText(
        frame, f"No Helmet: {counts.get('No Helmet', 0)}",
        (150, 58), FONT, 0.55, (0, 0, 220), 1
    )

    # FPS
    if fps is not None:
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (20, 85), FONT, 0.55, (255, 255, 0), 1
        )

    return frame


def detect_image(model, source: str, conf: float, save: bool):
    """Run detection on a single image."""
    print(f"\n🖼️  Detecting on image: {source}")

    frame = cv2.imread(source)
    if frame is None:
        print(f"❌ Could not read image: {source}")
        return

    # Run inference
    results = model.predict(source=frame, conf=conf, verbose=False)

    # Draw results
    frame, counts = draw_detections(frame, results)
    frame = draw_stats_overlay(frame, counts)

    print(f"   ✅ Helmet: {counts['Helmet']}  |  ❌ No Helmet: {counts['No Helmet']}")

    if save:
        out_path = f"output_{Path(source).name}"
        cv2.imwrite(out_path, frame)
        print(f"   💾 Saved to: {out_path}")

    # Show image
    cv2.imshow("Helmet Detection", frame)
    print("   Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_video(model, source, conf: float, save: bool):
    """Run detection on video file or webcam stream."""
    # Determine if webcam or video file
    if str(source).isdigit():
        source = int(source)
        is_webcam = True
        print(f"\n📹 Starting webcam (device {source})...")
    else:
        is_webcam = False
        print(f"\n🎬 Detecting on video: {source}")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Could not open video source: {source}")
        return

    # Get video properties
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

    # Setup video writer if saving
    writer = None
    if save:
        out_path = "output_detection.mp4" if is_webcam else f"output_{Path(str(source)).name}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_video, (frame_w, frame_h))
        print(f"   💾 Saving output to: {out_path}")

    print("   Press 'Q' to quit.\n")

    frame_count = 0
    fps = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_webcam:
                print("❌ Lost webcam feed.")
            else:
                print("✅ Video processing complete.")
            break

        frame_count += 1

        # Calculate FPS
        current_time = time.time()
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        # Run inference
        results = model.predict(source=frame, conf=conf, verbose=False)

        # Draw results
        frame, counts = draw_detections(frame, results)
        frame = draw_stats_overlay(frame, counts, fps=fps)

        # Save frame
        if writer:
            writer.write(frame)

        # Display
        cv2.imshow("Helmet Detection - Press Q to quit", frame)

        # Quit on 'Q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 Quitting...")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="🛡️ Helmet Detection — Run Inference"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Input source: image path, video path, or '0' for webcam"
    )
    parser.add_argument(
        "--model", type=str, default="runs/detect/helmet_detector2/weights/best.pt",
        help="Path to trained model weights (default: runs/detect/helmet_detector2/weights/best.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save output to file"
    )

    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────
    # Load model
    # ──────────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Train a model first with: python train.py")
        print("   Or specify a model path with: --model path/to/best.pt")
        return

    print("=" * 60)
    print("🛡️  HELMET DETECTION SYSTEM")
    print("=" * 60)
    print(f"  Model      : {args.model}")
    print(f"  Source      : {args.source}")
    print(f"  Confidence : {args.conf}")
    print(f"  Save Output: {args.save}")
    print("=" * 60)

    model = YOLO(args.model)

    # ──────────────────────────────────────────────────────────
    # Determine input type and run detection
    # ──────────────────────────────────────────────────────────
    source = args.source

    if source.isdigit():
        # Webcam
        detect_video(model, source, args.conf, args.save)
    elif Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        # Image
        detect_image(model, source, args.conf, args.save)
    elif Path(source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}:
        # Video
        detect_video(model, source, args.conf, args.save)
    else:
        print(f"❌ Unsupported source type: {source}")
        print("   Supported: image (.jpg/.png), video (.mp4/.avi), or webcam ('0')")


if __name__ == "__main__":
    main()
