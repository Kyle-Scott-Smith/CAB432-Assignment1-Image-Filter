import cv2
import numpy as np

def gaussian_blur(image, intensity):
    """Apply strong visible Gaussian blur"""
    if intensity <= 0:
        return image.copy()
    
    print("DEBUG: Using alternative median blur approach")
    
    # Use median blur which is very visible
    ksize = 25 + (intensity // 2)  # 25 to 50 kernel size
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    
    blurred = cv2.medianBlur(image, ksize)
    print(f"DEBUG: Applied medianBlur with kernel size {ksize}")
    
    return blurred

def edge_detection(image, intensity):
    """Apply clear edge detection"""
    if intensity <= 0:
        return image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Strong edge detection with clear thresholds
    threshold1 = 50
    threshold2 = 150
    
    # Detect edges
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    # Create pure white edges on pure black background
    result = np.zeros_like(image)
    result[edges > 0] = [255, 255, 255]  # White edges
    
    return result

def emboss(image, intensity):
    """Apply strong emboss effect"""
    if intensity <= 0:
        return image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Strong emboss kernel
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]]) * 2.0  # Stronger effect
    
    # Apply convolution
    embossed = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    
    # Normalize and add offset for classic emboss look
    embossed = embossed + 128
    embossed = np.clip(embossed, 0, 255)
    
    # Convert to color
    embossed = embossed.astype(np.uint8)
    return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)

def sharpen(image, intensity):
    """Apply strong sharpening"""
    if intensity <= 0:
        return image.copy()
    
    # Strong sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 13, -1],
                       [-1, -1, -1]], dtype=np.float32)
    
    # Apply sharpening
    sharpened = cv2.filter2D(image.astype(np.float32), -1, kernel)
    sharpened = np.clip(sharpened, 0, 255)
    
    return sharpened.astype(np.uint8)

def grayscale(image, intensity):
    """Convert to grayscale"""
    if intensity <= 0:
        return image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def sepia(image, intensity):
    """Apply sepia tone filter"""
    if intensity <= 0:
        return image.copy()
    
    # Sepia transformation matrix
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    
    # Apply sepia effect
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255)
    
    return sepia_image.astype(np.uint8)

def invert(image, intensity):
    """Invert colors"""
    if intensity <= 0:
        return image.copy()
    
    return 255 - image

def apply_filter(input_path, filter_name, output_path, intensity=0):
    """Apply the specified filter to an image"""
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Apply the selected filter
    if filter_name == "blur":
        processed = gaussian_blur(img, intensity)
    elif filter_name == "edges":
        processed = edge_detection(img, intensity)
    elif filter_name == "emboss":
        processed = emboss(img, intensity)
    elif filter_name == "sharpen":
        processed = sharpen(img, intensity)
    elif filter_name == "grayscale":
        processed = grayscale(img, intensity)
    elif filter_name == "sepia":
        processed = sepia(img, intensity)
    elif filter_name == "invert":
        processed = invert(img, intensity)
    else:
        raise ValueError(f"Unknown filter: {filter_name}")
    
    # Save processed image
    cv2.imwrite(output_path, processed)