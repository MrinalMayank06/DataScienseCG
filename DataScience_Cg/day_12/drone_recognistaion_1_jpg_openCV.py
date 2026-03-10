# 🎬 Scenario: You received a JPG from a drone camera. Before processing, understand its
# dimensions, color space, and pixel type.

# Verify installation
import cv2
print(cv2.__version__)

import numpy as np

# ── READING ───────────────────────────────────────────────────
img = cv2.imread('drone_photo.jpg.jpeg')
   # reads as BGR (NOT RGB — OpenCV's historical quirk)

print(img.shape)   # (height, width, channels)
print(img.dtype)   # uint8

# ── COLOR CONVERSION ──────────────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ── RESIZING ──────────────────────────────────────────────────
resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

# half size
half = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

# ── NORMALIZATION ─────────────────────────────────────────────
img_float = resized.astype(np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
normalized = (img_float - mean) / std

# ── CROPPING ──────────────────────────────────────────────────
roi = img[100:300, 50:250]

# ── ROTATION ──────────────────────────────────────────────────
h, w = img.shape[:2]
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rot = cv2.warpAffine(img, M, (w, h))

# ── FLIPPING ──────────────────────────────────────────────────
flip_h = cv2.flip(img, 1)
flip_v = cv2.flip(img, 0)

# ── DISPLAY ───────────────────────────────────────────────────
cv2.imshow("Original", img)
cv2.imshow("Resized", resized)
cv2.imshow("Half", half)
cv2.imshow("ROI", roi)
cv2.imshow("Rotated", rot)
cv2.imshow("Flip Horizontal", flip_h)
cv2.imshow("Flip Vertical", flip_v)
cv2.imshow("Gray", gray)
cv2.imshow("HSV", hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()