import os
import time
from flask import Flask, request, send_file, session, redirect, url_for, render_template
from functools import wraps
from filters import apply_filter
from auth import auth_bp

app = Flask(__name__)
app.secret_key = "supersecret"
app.register_blueprint(auth_bp)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALL_FILTERS = ["blur", "edges", "emboss", "sharpen", "grayscale", "sepia", "invert"]

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return wrapper

@app.route("/")
def home():
    return render_template("login.html")

@app.route("/filter-page")
@login_required
def filter_page():
    return render_template("upload.html")

@app.route("/filter", methods=["POST"])
@login_required
def filter_image():
    try:
        if "image" not in request.files:
            return render_template("error.html", message="No image uploaded"), 400

        image = request.files["image"]
        if image.filename == '':
            return render_template("error.html", message="No image selected"), 400

        # Validate file type
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return render_template("error.html", message="Invalid image format"), 400

        filename = image.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(input_path)

        # Read intensity from form
        intensity = int(request.form.get("intensity", 0))
        intensity = max(0, min(50, intensity))
        
        print(f"DEBUG: Processing image {filename} with intensity {intensity}")

        output_files = []
        for filter_name in ALL_FILTERS:
            try:
                output_filename = f"{os.path.splitext(filename)[0]}_{filter_name}.png"
                output_path = os.path.join(UPLOAD_FOLDER, output_filename)
                
                print(f"DEBUG: Applying {filter_name} filter...")
                apply_filter(input_path, filter_name, output_path, intensity=intensity)
                
                # Verify the output file was created
                if not os.path.exists(output_path):
                    raise ValueError(f"Filter {filter_name} did not create output file")
                
                output_files.append((filter_name, output_filename))
                print(f"DEBUG: {filter_name} filter completed successfully")
                
            except Exception as e:
                print(f"ERROR in {filter_name}: {str(e)}")
                # Continue with other filters instead of failing completely
                continue

        if not output_files:
            return render_template("error.html", message="All filters failed to process the image"), 500

        timestamp = int(time.time())
        return render_template("display.html", 
                             output_files=output_files, 
                             intensity=intensity,
                             timestamp=timestamp)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template("error.html", message=f"Error processing image: {str(e)}"), 500

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)