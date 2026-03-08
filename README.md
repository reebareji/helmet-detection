# 🛡️ Helmet Detection System using YOLOv8

An AI-powered real-time helmet detection system built with **YOLOv8**, **OpenCV**, and **PyTorch**.
Detects whether a person is wearing a **Helmet** or **No Helmet** in images, videos, and live webcam feeds — displaying accurate bounding boxes with color-coded labels.

> Built for the **AI-Based Helmet Detection System using YOLO** competition at Saintgits.

---

## 📁 Project Structure

```
helmet-detection/
├── data.yaml              # Dataset configuration (classes & paths)
├── requirements.txt       # Python dependencies
├── train.py               # Model training script (with resume support)
├── detect.py              # Detection on images / video / webcam
├── prepare_dataset.py     # Dataset preparation & merging utility
├── merge_datasets.py      # Merge multiple dataset sources
├── setup_dataset.py       # Initial dataset setup helper
├── dataset/               # YOLO-format dataset
│   ├── train/
│   │   ├── images/        # 5,578 training images
│   │   └── labels/        # Corresponding YOLO label files
│   └── val/
│       ├── images/        # 1,190 validation images
│       └── labels/        # Corresponding YOLO label files
└── runs/                  # Training outputs (auto-created)
    └── detect/
        └── helmet_detector2/
            └── weights/
                └── best.pt  # Best trained model
```

## 🏷️ Classes

| ID | Class     | Color  | Description                    |
|----|-----------|--------|--------------------------------|
| 0  | Helmet    | 🟢 Green | Person wearing a safety helmet |
| 1  | No Helmet | 🔴 Red   | Person NOT wearing a helmet    |

## 📊 Training Results

| Metric | Value |
|--------|-------|
| **Model** | YOLOv8n (nano) |
| **Dataset** | 6,768 images (5,578 train / 1,190 val) |
| **Epochs Trained** | 9 |
| **mAP50** | 0.766 |
| **mAP50-95** | 0.488 |
| **Precision** | 0.768 |
| **Recall** | 0.686 |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Train with default settings
python train.py

# Train with custom settings
python train.py --epochs 100 --batch 16 --imgsz 640

# Resume from last checkpoint
python train.py --resume
```

### 3. Run Detection
```bash
# Detect on an image
python detect.py --source path/to/image.jpg

# Detect on a video file
python detect.py --source path/to/video.mp4

# Live webcam detection
python detect.py --source 0

# Custom confidence threshold + save output
python detect.py --source path/to/image.jpg --conf 0.4 --save
```

## 🔧 Tech Stack

- **Python 3.10+**
- **Ultralytics YOLOv8** — object detection framework
- **PyTorch** — deep learning backend
- **OpenCV** — image/video processing and visualization
- **Roboflow** — dataset sourcing

## 👩‍💻 Author

Built for the **AI-Based Helmet Detection System using YOLO** competition at Saintgits.
