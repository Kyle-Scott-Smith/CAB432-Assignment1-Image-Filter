import os
import time
import shutil

UPLOADS_FOLDER = "uploads"  # adjust if your folder path is different
WAIT_TIME = 3  # seconds to wait before cleanup (optional)

def clear_uploads():
    if os.path.exists(UPLOADS_FOLDER):
        for filename in os.listdir(UPLOADS_FOLDER):
            file_path = os.path.join(UPLOADS_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove directory
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        print("Uploads folder cleared.")
    else:
        print("Uploads folder does not exist.")

if __name__ == "__main__":
    # optional delay to allow processing to finish
    print(f"Waiting {WAIT_TIME} seconds before cleanup...")
    time.sleep(WAIT_TIME)
    clear_uploads()