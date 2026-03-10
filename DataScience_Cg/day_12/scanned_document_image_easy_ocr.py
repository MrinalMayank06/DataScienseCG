# 🎬 Scenario:
# A scanned document (report/form) often contains noise, shadows, and uneven lighting.
# Before running OCR (Optical Character Recognition), we clean and enhance the image
# so the text becomes clearer and easier for OCR engines to detect.

# Import OpenCV for image processing
import cv2

# Import NumPy for numerical operations and kernel creation
import numpy as np

# Import cv2_imshow for displaying images in Google Colab
# (cv2.imshow() does not work inside Colab notebooks)
 


# ─────────────────────────────────────────────────────────
# Step 1: Read the image from disk
# ─────────────────────────────────────────────────────────

# cv2.imread loads the scanned document image
# Ensure 'report.jpg' exists in the working directory
img = cv2.imread('report.jpg.jpeg')


# ─────────────────────────────────────────────────────────
# Step 2: Verify image loaded successfully
# ─────────────────────────────────────────────────────────

# If the file path is incorrect, imread returns None
if img is None:
    print("Error: Could not load image. Please ensure 'report.jpg' exists and the path is correct.")

else:

    # ─────────────────────────────────────────────────────────
    # Step 3: Convert image to grayscale
    # ─────────────────────────────────────────────────────────

    # OCR works better on grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # =========================================================
    # Smoothing / Denoising
    # =========================================================

    # Scanned documents may contain:
    # scanner noise, dust particles, uneven lighting


    # Gaussian Blur
    # Smooths image using Gaussian distribution
    blur_g = cv2.GaussianBlur(gray, (5, 5), 0)


    # Median Blur
    # Very effective for removing salt-and-pepper noise
    blur_m = cv2.medianBlur(gray, 5)


    # Bilateral Filter
    # Reduces noise while preserving edges
    blur_b = cv2.bilateralFilter(img, 9, 75, 75)


    # =========================================================
    # Sharpening using Kernel
    # =========================================================

    # Kernel used to enhance edges
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    # Apply sharpening filter
    sharp = cv2.filter2D(gray, -1, kernel)


    # =========================================================
    # Thresholding
    # =========================================================

    # Convert grayscale image to binary image


    # Global Threshold
    _, thresh = cv2.threshold(
        blur_g,
        127,
        255,
        cv2.THRESH_BINARY
    )


    # Adaptive Threshold
    adap_thresh = cv2.adaptiveThreshold(
        blur_g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )


    # =========================================================
    # Morphological Operations
    # =========================================================

    # Create morphological kernel
    kernel2 = np.ones((3, 3), np.uint8)


    # Erosion
    eroded = cv2.erode(
        thresh,
        kernel2,
        iterations=1
    )


    # Dilation
    dilated = cv2.dilate(
        thresh,
        kernel2,
        iterations=1
    )


    # Opening
    opened = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel2
    )


    # Closing
    closed = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel2
    )


    # =========================================================
    # Image Information
    # =========================================================

    print('Size :', img.size)


    # =========================================================
    # Convert BGR to RGB
    # =========================================================

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # =========================================================
    # Display Images
    # =========================================================

    # =========================================================
# Display Images
# =========================================================

# Original scanned document
cv2.imshow("Original", img)

# Grayscale image
cv2.imshow("Gray", gray)

# Gaussian blurred image
cv2.imshow("Gaussian Blur", blur_g)

# Median blurred image
cv2.imshow("Median Blur", blur_m)

# Sharpened image
cv2.imshow("Sharpened", sharp)

# Global threshold result
cv2.imshow("Threshold", thresh)

# Adaptive threshold result
cv2.imshow("Adaptive Threshold", adap_thresh)

# Eroded image
cv2.imshow("Eroded", eroded)

# Dilated image
cv2.imshow("Dilated", dilated)

# Opening operation result
cv2.imshow("Opening", opened)

# Closing operation result
cv2.imshow("Closing", closed)

cv2.waitKey(0)
cv2.destroyAllWindows()