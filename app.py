from flask import Flask, request, send_file
from filters import process_image
import cv2
import numpy as np
import tempfile

app = Flask(__name__)
@app.route('/filter', methods=['POST'])
@app.route('/filter', methods=['POST'])
def filter_image():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}
    
    file = request.files["Image"]
    
    temporary_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    file.save(temporary_file.name)
    
    output = process_image(temporary_file.name)
    
    temporary_output = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temporary_output.name, output)
    
    return send_file(temporary_output.name, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug = True)