import os
import requests
from multiprocessing import Pool, cpu_count
import time

# Config
URL = "http://13.239.65.250:8080/process-multiple"  # must update eachtime ec2 reboots
IMAGE_FOLDER = "test_images"
NUM_WORKERS = cpu_count()
BATCH_SIZE = 10 # Number of requests per worker in each round

# Get all image paths
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith((".jpeg", ".jpg"))]

if not image_files:
    raise Exception(f"No images found in folder {IMAGE_FOLDER}!")

def create_form_data():
    files = []
    for img_path in image_files:
        files.append(("images", (os.path.basename(img_path), open(img_path, "rb"), "image/jpeg")))
    data = {
        "intensity": "50",
        "scale": "5"
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
        print("Round completed. Sending next batch")

if __name__ == "__main__":
    print(f"Starting load test with {NUM_WORKERS} workers")
    with Pool(NUM_WORKERS) as pool:
        pool.map(worker, range(NUM_WORKERS))
