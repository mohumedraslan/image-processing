import cv2
import matplotlib.pyplot as plt
import numpy as np

# Function to display images using matplotlib
# Converts BGR to RGB for correct visualization
# Supports grayscale images

def matshow(title="image", image=None, gray=False):
    if len(image.shape) == 2:  # Grayscale image
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')  # Hide axis
    plt.show()

# Read an image (color mode)
image = cv2.imread(r"C:\Users\moras\OneDrive\Desktop\projects\com\projects\image processing\asmaa_glal_test.jpg", 1)

# Display the original image
matshow("Original Image", image)

# Print image details
type(image), image.shape

# Save the image copy
cv2.imwrite('asmaa_glal_test.jpg', image)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
matshow("Grayscale Image", gray_image, gray=True)

# Convert BGR to RGB for correct visualization
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
matshow("RGB Image", rgb_image)

# Function to translate an image
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# Apply translation
matshow("Shifted Down", translate(image, 0, 50))
matshow("Shifted Left", translate(image, -100, 0))
matshow("Shifted Right and Down", translate(image, 50, 100))

# Linear grayscale transformation (inversion)
def linear_trans(image, k, b=0):
    trans_table = np.array([np.float32(x) * k + b for x in range(256)])
    trans_table = np.clip(trans_table, 0, 255).astype(np.uint8)
    return cv2.LUT(image, trans_table)

# Invert grayscale image
matshow("Inverted Image", linear_trans(gray_image, -1, 255), gray=True)

# Histogram equalization
matshow("Equalized Image", cv2.equalizeHist(gray_image), gray=True)

# Display histogram comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(gray_image.ravel(), 256, [0, 256], label='Original')
plt.legend()
plt.subplot(1, 2, 2)
plt.hist(cv2.equalizeHist(gray_image).ravel(), 256, [0, 256], label='Equalized')
plt.legend()
plt.show()

# Apply different blurring techniques
matshow("Median Blurred", cv2.medianBlur(image, 5))
matshow("Mean Blurred", cv2.blur(image, (5, 5)))
matshow("Gaussian Blurred", cv2.GaussianBlur(image, (5, 5), 0))

# Sharpening filter
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
matshow("Sharpened Image", cv2.filter2D(image, -1, sharpen_kernel))
