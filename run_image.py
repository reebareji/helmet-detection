"""
run_image.py — Helmet Detection on Images
===========================================
Open this file in Python IDLE and press F5 to run.
It will open a pop-up asking you to choose an image.
"""

import cv2
import os
from tkinter import Tk, filedialog
from ultralytics import YOLO

# Hide the main tkinter window
root = Tk()
root.withdraw()

# Ask user to select an image
print("Please select an image file from the pop-up window...")
image_path = filedialog.askopenfilename(
    title="Select an Image to Test",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("No image selected. Exiting.")
    exit()

print(f"\nAnalyzing image: {image_path}")

# Change to project directory so YOLO finds the model
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load model and run detection
model = YOLO("runs/detect/helmet_detector2/weights/best.pt")

# Run prediction with show=True to display it, and save=True to save the result
results = model.predict(source=image_path, conf=0.35, save=True, show=True)

print("\n✅ Done! The resulting image has been saved in the 'runs/detect/predict' folder.")
print("Press any key on your keyboard while selecting the image window to close it.")

# Wait for a key press to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
