import cv2
import numpy as np
import requests
import os
import random

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

def fetch_random_image():
    """Fetch a high-resolution random image with multiple fallback options"""
    try:
        print("DEBUG: Attempting to fetch high-resolution random image...")
        
        # Try to get the maximum dimensions that Picsum supports
        # Picsum allows up to 5000px dimensions according to their docs
        max_width, max_height = 3840, 2160  # 4K resolution
        
        # Option 1: Try multiple Picsum Photos URLs with high resolution
        image_sources = [
            f'https://picsum.photos/id/{img_id}/{max_width}/{max_height}'
            for img_id in [1, 10, 100, 1000, 1001, 1015, 1018, 1020, 1025]
        ]
        
        # Add some landscape-specific IDs known to work well
        landscape_ids = [1018, 1015, 1020, 1025, 1039, 1040, 1043, 1047]
        for img_id in landscape_ids:
            image_sources.append(f'https://picsum.photos/id/{img_id}/{max_width}/{max_height}')
        
        # Shuffle to try different sources randomly
        random.shuffle(image_sources)
        
        for source in image_sources:
            try:
                print(f"DEBUG: Trying high-res source: {source}")
                response = requests.get(source, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if image is not None:
                        print(f"DEBUG: Successfully fetched high-res image: {image.shape}")
                        return image
            except Exception as e:
                print(f"DEBUG: Failed with {source}: {e}")
                continue
        
        # Option 2: If high-res fails, try standard resolution
        print("DEBUG: High-res sources failed, trying standard resolution...")
        standard_sources = [
            'https://picsum.photos/1920/1080',
            'https://picsum.photos/1200/800',
            'https://picsum.photos/1024/768'
        ]
        
        for source in standard_sources:
            try:
                response = requests.get(source, timeout=8)
                if response.status_code == 200:
                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if image is not None:
                        print(f"DEBUG: Successfully fetched standard image: {image.shape}")
                        return image
            except Exception as e:
                print(f"DEBUG: Failed with standard source {source}: {e}")
                continue
        
        # Option 3: If all external sources fail, generate a high-resolution test pattern
        print("DEBUG: All external sources failed, generating high-res test pattern")
        return generate_high_res_test_image()
        
    except Exception as e:
        print(f"DEBUG: Error in fetch_random_image: {e}")
        import traceback
        traceback.print_exc()
        return generate_high_res_test_image()

def generate_high_res_test_image():
    """Generate a high-resolution test pattern image"""
    # Use 4K resolution (3840x2160) or the largest that OpenCV can handle efficiently
    width, height = 3840, 2160
    
    # For systems with limited memory, use a more manageable size
    try:
        image = np.zeros((height, width, 3), dtype=np.uint8)
    except MemoryError:
        print("DEBUG: Memory error with 4K, using Full HD instead")
        width, height = 1920, 1080
        image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a detailed colorful gradient pattern
    for y in range(0, height, 2):  # Step by 2 for performance
        for x in range(0, width, 2):
            image[y, x] = [
                int(255 * x / width),           # Red gradient
                int(255 * y / height),          # Green gradient  
                int(255 * (x+y)/(width+height)) # Blue gradient
            ]
    
    # Add multiple shapes for visual interest
    cv2.rectangle(image, (100, 100), (1000, 1000), (255, 0, 0), 10)
    cv2.circle(image, (1920, 1080), 500, (0, 255, 0), 10)
    cv2.line(image, (500, 100), (3000, 2000), (0, 0, 255), 10)
    
    # Add a grid pattern
    for i in range(0, width, 200):
        cv2.line(image, (i, 0), (i, height), (200, 200, 200), 1)
    for i in range(0, height, 200):
        cv2.line(image, (0, i), (width, i), (200, 200, 200), 1)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'HIGH-RESOLUTION TEST IMAGE', (500, 500), font, 2, (255, 255, 255), 4)
    cv2.putText(image, f'{width}x{height}px', (500, 600), font, 1.5, (255, 255, 255), 3)
    
    print(f"DEBUG: Generated high-res test pattern: {image.shape}")
    return image

# Keep your existing filter functions (gaussian_blur, edge_detection, etc.) here
# [Your existing filter functions remain unchanged]

def apply_filter(input_path, filter_name, output_path, intensity=0):
    """Apply the specified filter to an image with CPU-intensive options"""
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    total_pixels = height * width
    
    # Make processing more CPU-intensive for larger images
    iterations = 1
    if total_pixels > 1000000:  # More than 1MP
        iterations = 2
    if total_pixels > 4000000:  # More than 4MP
        iterations = 3
    
    # Apply the selected filter multiple times for CPU intensity
    processed = img.copy()
    for i in range(iterations):
        if filter_name == "blur":
            processed = gaussian_blur(processed, intensity)
        elif filter_name == "edges":
            processed = edge_detection(processed, intensity)
        elif filter_name == "emboss":
            processed = emboss(processed, intensity)
        elif filter_name == "sharpen":
            processed = sharpen(processed, intensity)
        elif filter_name == "grayscale":
            processed = grayscale(processed, intensity)
        elif filter_name == "sepia":
            processed = sepia(processed, intensity)
        elif filter_name == "invert":
            processed = invert(processed, intensity)
        else:
            raise ValueError(f"Unknown filter: {filter_name}")
    
    # Save processed image
    success = cv2.imwrite(output_path, processed)
    if not success:
        raise ValueError(f"Failed to save image to {output_path}")
    
    print(f"DEBUG: Applied {filter_name} with {iterations} iterations on {width}x{height} image")