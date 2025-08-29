import os
import time
from flask import Flask, request, send_file, session, redirect, url_for, render_template, jsonify
from functools import wraps
from filters import apply_filter
from auth import auth_bp, token_required
import cv2
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template("error.html", message="Invalid image format"), 400

        filename = image.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(input_path)

        # Read intensity and scale from form
        intensity = int(request.form.get("intensity", 0))
        intensity = max(0, min(50, intensity))
        
        scale = int(request.form.get("scale", 1))
        scale = max(1, min(10, scale))  # Limit scale to 1-10
        
        print(f"Processing image {filename} with intensity {intensity}, scale {scale}")

        # Apply scaling to image if needed
        if scale > 1:
            scaled_path = os.path.join(UPLOAD_FOLDER, f"scaled_{filename}")
            scale_image(input_path, scaled_path, scale)
            input_path = scaled_path

        output_files = []
        for filter_name in ALL_FILTERS:
            try:
                output_filename = f"{os.path.splitext(filename)[0]}_{filter_name}.png"
                output_path = os.path.join(UPLOAD_FOLDER, output_filename)
                
                print(f"Applying {filter_name} filter")
                apply_filter(input_path, filter_name, output_path, intensity=intensity)
                
                # Verify the output file was created
                if not os.path.exists(output_path):
                    raise ValueError(f"Filter {filter_name} did not create output file")
                
                output_files.append((filter_name, output_filename))
                print(f"{filter_name} filter completed successfully")
                
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
                             scale=scale,
                             timestamp=timestamp)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template("error.html", message=f"Error processing image: {str(e)}"), 500

@app.route("/use-random-image", methods=["POST"])
@login_required
def use_random_image():
    """Handle form submission with random image"""
    try:
        from filters import fetch_random_image
        
        # Fetch a random image
        random_img = fetch_random_image()
        if random_img is None:
            return render_template("error.html", message="Failed to fetch random image"), 500
        
        # Generate a filename
        filename = f"random_{session['user']}_{int(time.time())}.jpg"
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the image
        cv2.imwrite(input_path, random_img)
        
        # Read intensity and scale from form
        intensity = int(request.form.get("intensity", 25))
        intensity = max(0, min(50, intensity))
        
        scale = int(request.form.get("scale", 1))
        scale = max(1, min(10, scale))
        
        print(f"Processing random image {filename} with intensity {intensity}, scale {scale}")

        # Apply scaling to image if needed
        if scale > 1:
            scaled_path = os.path.join(UPLOAD_FOLDER, f"scaled_{filename}")
            scale_image(input_path, scaled_path, scale)
            input_path = scaled_path  # Use the scaled image for processing

        # Process the image with all filters
        output_files = []
        for filter_name in ALL_FILTERS:
            try:
                output_filename = f"{os.path.splitext(filename)[0]}_{filter_name}.png"
                output_path = os.path.join(UPLOAD_FOLDER, output_filename)
                
                print(f"Applying {filter_name} filter")
                apply_filter(input_path, filter_name, output_path, intensity=intensity)
                
                if not os.path.exists(output_path):
                    raise ValueError(f"Filter {filter_name} did not create output file")
                
                output_files.append((filter_name, output_filename))
                print(f"{filter_name} filter completed successfully")
                
            except Exception as e:
                print(f"ERROR in {filter_name}: {str(e)}")
                continue

        if not output_files:
            return render_template("error.html", message="All filters failed to process the image"), 500

        timestamp = int(time.time())
        return render_template("display.html", 
                             output_files=output_files, 
                             intensity=intensity,
                             scale=scale,
                             timestamp=timestamp)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template("error.html", message=f"Error processing image: {str(e)}"), 500

def scale_image(input_path, output_path, scale_factor):
    print(f"Scaling image by factor {scale_factor}")
    
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Calculate new dimensions
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    print(f"Scaling from {width}x{height} to {new_width}x{new_height}")
    
    # Use different interpolation methods based on scale factor
    if scale_factor > 1:
        # For upscaling - CPU intensive!
        interpolation = cv2.INTER_CUBIC  # More CPU intensive than INTER_LINEAR
    else:
        # For downscaling
        interpolation = cv2.INTER_AREA
    
    # Perform the scaling - this is CPU intensive
    scaled_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    
    # Save the scaled image
    cv2.imwrite(output_path, scaled_img)
    print(f"Image scaled successfully to {new_width}x{new_height}")

@app.route("/api/random-image", methods=["GET"])
@login_required
def random_image():
    try:
        from filters import fetch_random_image
        random_img = fetch_random_image()
        if random_img is not None:
            filename = f"random_{session['user']}_{int(time.time())}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            cv2.imwrite(filepath, random_img)
            return jsonify({"success": True, "filename": filename, "message": "Random image fetched successfully"})
        return jsonify({"success": False, "error": "Failed to fetch random image"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/filters", methods=["GET"])
def get_filters():
    """Return available filters"""
    return jsonify({"filters": ALL_FILTERS})

@app.route("/api/apply-filter", methods=["POST"])
@token_required
def api_apply_filter():
    """API endpoint to apply a specific filter"""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        image = request.files["image"]
        filter_name = request.form.get("filter")
        intensity = int(request.form.get("intensity", 25))
        
        if filter_name not in ALL_FILTERS:
            return jsonify({"error": "Invalid filter name"}), 400
        
        filename = image.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(input_path)
        
        output_filename = f"{os.path.splitext(filename)[0]}_{filter_name}.png"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        apply_filter(input_path, filter_name, output_path, intensity=intensity)
        
        return jsonify({
            "original": filename,
            "filtered": output_filename,
            "filter": filter_name,
            "intensity": intensity
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process-multiple", methods=["POST"])
@login_required
def process_multiple_images():
    try:
        if "images" not in request.files:
            return render_template("error.html", message="No images uploaded"), 400

        images = request.files.getlist("images")
        if not images or images[0].filename == '':
            return render_template("error.html", message="No images selected"), 400

        # Limit to 10 images as more are not possible with ec2 cpu
        if len(images) > 10:
            images = images[:10]
            print(f"Limited to 10 images for performance")

        intensity = int(request.form.get("intensity", 25))
        intensity = max(0, min(50, intensity))

        scale = int(request.form.get("scale", 1))
        scale = max(1, min(10, scale))

        username = session.get("user", "anon")

        print(f"Processing {len(images)} images in parallel with intensity {intensity}, scale {scale}")

        # Prepare list of saved input files
        saved_inputs = []
        for image_file in images:
            original_name = image_file.filename
            if not original_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping invalid format: {original_name}")
                continue

            unique_id = uuid.uuid4().hex[:8]
            safe_filename = f"{username}_{unique_id}_{original_name}"
            input_path = os.path.join(UPLOAD_FOLDER, safe_filename)
            # Save the uploaded file to disk immediately
            image_file.save(input_path)
            saved_inputs.append({
                "original_name": original_name,
                "safe_filename": safe_filename,
                "input_path": input_path
            })

        if not saved_inputs:
            return render_template("error.html", message="No valid images to process"), 400

        results = []
        # worker uses only file paths and simple primitives
        def process_saved_image(item):
            safe_filename = item["safe_filename"]
            input_path = item["input_path"]
            original_name = item["original_name"]

            try:
                # If scaling requested, create scaled version next to it and use that
                work_input = input_path
                if scale > 1:
                    scaled_path = os.path.join(UPLOAD_FOLDER, f"scaled_{safe_filename}")
                    scale_image(input_path, scaled_path, scale)
                    work_input = scaled_path

                filter_results = []
                for filter_name in ALL_FILTERS:
                    try:
                        output_filename = f"{os.path.splitext(safe_filename)[0]}_{filter_name}.png"
                        output_path = os.path.join(UPLOAD_FOLDER, output_filename)

                        apply_filter(work_input, filter_name, output_path, intensity=intensity)

                        if os.path.exists(output_path):
                            filter_results.append((filter_name, output_filename))
                        else:
                            # Some filters might not write, treat as failure
                            print(f"filter {filter_name} produced no output for {safe_filename}")

                    except Exception as e:
                        print(f"ERROR in image {safe_filename}, filter {filter_name}: {str(e)}")
                        continue

                return {
                    "original": safe_filename,
                    "original_name": original_name,
                    "filters": filter_results,
                    "success": True if filter_results else False,
                    "error": None if filter_results else "No filters produced output"
                }

            except Exception as e:
                print(f"ERROR processing saved image {safe_filename}: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    "original": safe_filename,
                    "original_name": original_name,
                    "filters": [],
                    "success": False,
                    "error": str(e)
                }

        # Determine reasonable worker count
        cpu_count = os.cpu_count() or 1
        max_workers = min(1, cpu_count, len(saved_inputs))

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(process_saved_image, item): item for item in saved_inputs}

            for future in as_completed(future_to_item):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    item = future_to_item[future]
                    print(f"worker failed for {item.get('safe_filename')}: {str(e)}")
                    results.append({
                        "original": item.get("safe_filename"),
                        "original_name": item.get("original_name"),
                        "filters": [],
                        "success": False,
                        "error": str(e)
                    })

        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        print(f"Parallel processing completed - {successful} successful, {failed} failed")

        timestamp = int(time.time())
        return render_template("display.html",
                               multi_results=results,
                               intensity=intensity,
                               scale=scale,
                               timestamp=timestamp)

    except Exception as e:
        print(f"ERROR in parallel processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template("error.html", message=f"Error processing multiple images: {str(e)}"), 500
    
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
