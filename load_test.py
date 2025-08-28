import os
import requests
from multiprocessing import Pool, cpu_count
import time

# Config
URL = "http://127.0.0.1:5000/process-multiple"  # Update if your app uses a different endpoint
IMAGE_FOLDER = "test_images"
NUM_WORKERS = cpu_count()  # Use all CPU cores
BATCH_SIZE = 5  # Number of requests per worker in each round

# Get all image paths
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]

if not image_files:
    raise Exception(f"No images found in folder {IMAGE_FOLDER}!")

def create_form_data():
    files = []
    for img_path in image_files:
        files.append(("images", (os.path.basename(img_path), open(img_path, "rb"), "image/jpeg")))
    data = {
        "intensity": "25",
        "scale": "3"
    }
    return files, data

def send_request(_):
    files, data = create_form_data()
    try:
        response = requests.post(URL, files=files, data=data)
        print(f"Request done: {response.status_code}")
    except Exception as e:
        print("Request failed:", e)
    finally:
        for f in files:
            f[1][1].close()  # Close file handles

def worker(_):
    while True:  # Continuous load
        for _ in range(BATCH_SIZE):
            send_request(_)
        print("Round completed. Sending next batch...")

if __name__ == "__main__":
    print(f"Starting load test with {NUM_WORKERS} workers (CPU cores)...")
    with Pool(NUM_WORKERS) as pool:
        pool.map(worker, range(NUM_WORKERS))
