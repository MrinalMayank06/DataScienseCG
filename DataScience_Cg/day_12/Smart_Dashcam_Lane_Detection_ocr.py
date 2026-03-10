# Scenario: Smart Dashcam Lane Detection
# Imagine you’re building a smart dashcam system for cars that helps drivers stay in their lanes. The dashcam captures frames of
#  the road, and your algorithm processes them step by step:
# - Step 1 – Capture the road scene
# The dashcam takes a snapshot (road.jpg). This is the raw input, just like a driver’s eye view.
# - Step 2 – Focus on essentials
# Convert the image to grayscale. Colors aren’t important for lane detection; what matters are contrasts and shapes.
# - Step 3 – Smooth out distractions
# Apply a Gaussian blur to reduce noise. Think of it as filtering out small pebbles or shadows that could confuse the system.
# - Step 4 – Spot the lane boundaries
# Use Canny edge detection to highlight sharp changes in intensity—these are likely lane markings.
# - Step 5 – Define the driver’s view
# Create a region of interest (ROI) shaped like a trapezoid, covering the part of the road where lanes usually appear. This prevents
#  the system from wasting effort on irrelevant areas like the sky or nearby buildings.
# - Step 6 – Overlay results for feedback
# Combine the detected edges with the original frame. The driver (or tester) now sees lane boundaries highlighted directly on the
# road image.

# 🎯 Teaching Angle
# This scenario shows how computer vision can be applied to real-world problems. Instead of just running code, learners can imagine
# themselves designing a lane-assist feature for autonomous vehicles. Each step connects to a practical need: clarity, focus, safety.

import cv2
import numpy as np
 

# ── Step 1: Load dashcam frame ───────────────────────────
frame = cv2.imread('road.jpeg')

# Check if image was loaded successfully
if frame is None:
    print("Error: Could not load image. Please ensure 'road.jpg' exists in the environment and the path is correct.")
    exit()

# ── Step 2: Convert to grayscale ─────────────────────────
gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ── Step 3: Reduce noise with Gaussian blur ───────────────
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ── Step 4: Canny edge detection ─────────────────────────
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# ── Step 5: Define region of interest (ROI) ───────────────
h, w  = edges.shape
mask  = np.zeros_like(edges)
# Ensure points are integers for cv2.fillPoly
pts   = np.array([[0,h],[w,h],[w*0.6,h*0.6],[w*0.4,h*0.6]], dtype=np.int32)
cv2.fillPoly(mask, [pts], 255)
roi   = cv2.bitwise_and(edges, mask)

# ── Step 6: Overlay on original ──────────────────────────
edges_col = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
result    = cv2.addWeighted(frame, 0.8, edges_col, 1.0, 0)

cv2.imshow("Lane Detection Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()