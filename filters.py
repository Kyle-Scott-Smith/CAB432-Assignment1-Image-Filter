import cv2
import numpy as np

def gaussian_blur(image):
    # Kernel size 15x15 (larger = more CPU)
    return cv2.GaussianBlur(image, (15, 15), 0)

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    # Convert back to 3 channels for consistency
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def emboss(image):
    kernel = np.array([[ -2, -1, 0],
                       [ -1, 1, 1],
                       [  0, 1, 2]])
    return cv2.filter2D(image, -1, kernel)

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image path")

    # Apply filters sequentially
    image = gaussian_blur(image)
    image = edge_detection(image)
    image = emboss(image)
    return image
