import cv2
import matplotlib.pyplot as plt

# read image
img = cv2.imread('images.jpg.jpeg')

if img is None:
    print("Image not found. Check file name or path.")
    exit()

# resize image
resized = cv2.resize(img, (224, 224))
half = cv2.resize(img, None, fx=0.5, fy=0.5)

# crop region of interest
roi = img[100:400, 200:600]

# rotate image
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
rot = cv2.warpAffine(img, M, (w, h))

# flip operations
flip_h = cv2.flip(img, 1)
flip_v = cv2.flip(img, 0)

# color conversions
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# show using opencv
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

# show using matplotlib
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Resized")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))
plt.title("Rotated")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(cv2.cvtColor(flip_h, cv2.COLOR_BGR2RGB))
plt.title("Flip H")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(gray, cmap="gray")
plt.title("Gray")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB))
plt.title("HSV")
plt.axis("off")

plt.show()