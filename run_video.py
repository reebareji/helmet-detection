"""
run_video.py — Helmet Detection on Videos
===========================================
Open this file in Python IDLE and press F5 to run.
It will open a pop-up asking you to choose a video file.
"""

import os
from tkinter import Tk, filedialog
from ultralytics import YOLO

# Hide the main tkinter window
root = Tk()
root.withdraw()

# Ask user to select a video
print("Please select a video file from the pop-up window...")
video_path = filedialog.askopenfilename(
    title="Select a Video to Test",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    print("No video selected. Exiting.")
    exit()

print(f"\nAnalyzing video: {video_path}")
print("This may take some time depending on the video length...")

# Change to project directory so YOLO finds the model
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load model
model = YOLO("runs/detect/helmet_detector2/weights/best.pt")

# Run prediction
# show=True will display the video as it processes
# save=True will save the final video
results = model.predict(source=video_path, conf=0.35, save=True, show=True)

print("\n✅ Done! The resulting video has been saved in the 'runs/detect/predict' folder.")
