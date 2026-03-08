"""
run_webcam.py — Helmet Detection on Webcam (Optimized for CPU)
==============================================================
Open this file in Python IDLE and press F5 to run.
- Green box = Helmet detected
- Red box = No Helmet detected
- Output video saved to: output_webcam.mp4
Press 'Q' to quit and save.
"""

import cv2
import os
import threading
import time
from ultralytics import YOLO

# Change to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the trained model
print("Loading model... please wait...")
model = YOLO("runs/detect/helmet_detector2/weights/best.pt")
print("Model loaded! Starting webcam...")

# Colors and labels
GREEN = (0, 200, 0)
RED = (0, 0, 220)
CLASS_NAMES = {0: "Helmet", 1: "No Helmet"}
CLASS_COLORS = {0: GREEN, 1: RED}

# ----------------------------------------------------------------------
# Threaded Camera Class to prevent lag/buffering
# ----------------------------------------------------------------------
class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
        # Setup video writer
        self.frame_w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Start threaded video stream
video_getter = VideoGet(0).start()
time.sleep(1.0) # Let camera warm up

if not video_getter.grabbed:
    print("ERROR: Could not open webcam!")
    video_getter.stop()
    exit()

save_path = "output_webcam.mp4"
writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (video_getter.frame_w, video_getter.frame_h))

print("Webcam running! Recording to: " + save_path)
print("Press 'Q' to quit and save.")

# Detection logic variables
last_boxes = []
helmet_count = 0
no_helmet_count = 0
frames_since_detect = 0

# PERFORMANCE TUNING:
# Detect every N frames (higher = less lag, but more choppy boxes)
DETECT_EVERY = 3       
# How long to keep boxes visible when no detection runs
KEEP_BOXES_FOR = int(DETECT_EVERY * 1.5)

frame_count = 0
prev_time = time.time()
fps = 0

while True:
    if video_getter.stopped:
        break
        
    frame = video_getter.read().copy()
    frame_count += 1

    # Calculate approximate FPS
    current_time = time.time()
    if current_time - prev_time >= 1.0:
        fps = frame_count / (current_time - prev_time)
        frame_count = 0
        prev_time = current_time

    # Run detection only on every Nth frame
    if frame_count % DETECT_EVERY == 0:
        # imgsz=256 is much faster on CPU than 640
        results = model.predict(source=frame, conf=0.35, verbose=False, imgsz=256)

        new_boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                new_boxes.append((x1, y1, x2, y2, class_id, confidence))

        if len(new_boxes) > 0:
            last_boxes = new_boxes
            frames_since_detect = 0
        else:
            frames_since_detect += 1

    # Clear old boxes if they haven't been updated in a while
    if frames_since_detect > KEEP_BOXES_FOR:
        last_boxes = []
        helmet_count = 0
        no_helmet_count = 0

    # Count current detections
    current_helmet = 0
    current_no_helmet = 0

    # Draw the boxes
    for (x1, y1, x2, y2, class_id, confidence) in last_boxes:
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        label = CLASS_NAMES.get(class_id, "Unknown")
        text = f"{label} {confidence:.0%}"

        if class_id == 0:
            current_helmet += 1
        else:
            current_no_helmet += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 5, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Smooth the display counts (so it doesn't flicker immediately to 0)
    helmet_count = current_helmet if len(last_boxes) > 0 else helmet_count
    no_helmet_count = current_no_helmet if len(last_boxes) > 0 else no_helmet_count

    # Draw UI Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 110), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, "Detection Count", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Helmet: {helmet_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
    cv2.putText(frame, f"No Helmet: {no_helmet_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    
    # Save frame
    writer.write(frame)

    # Display frame
    cv2.imshow("Helmet Detection - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\nStopping webcam and wrapping up...")
video_getter.stop()
writer.release()
cv2.destroyAllWindows()
print("Video saved to: " + os.path.abspath(save_path))
print("Done!")
