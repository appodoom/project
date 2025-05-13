from flask import Flask, request, jsonify, send_file
import time
from flask_cors import CORS
import os
import uuid
from urllib.parse import unquote


app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '../uploads'))
GENERATED_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '../generated'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post('/files')
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400


    if file and file.filename.endswith('.wav'):
        file_id = str(uuid.uuid4())
        extension = os.path.splitext(file.filename)[1]  # keep .wav
        filename = f"{file_id}{extension}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({"message": "File saved successfully", "id": file_id}), 200

    return jsonify({"error": "Invalid file type. Only .wav allowed"}), 400

def wait_for_complete_file(filepath, min_size=10000, stable_secs=2, timeout=30):
    start = time.time()
    last_size = -1
    stable_since = None

    while time.time() - start < timeout:
        if not os.path.exists(filepath):
            time.sleep(0.2)
            continue

        current_size = os.path.getsize(filepath)

        if current_size >= min_size:
            if current_size == last_size:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_secs:
                    return True  # File is stable
            else:
                stable_since = None  # Reset if size changed

        last_size = current_size
        time.sleep(0.2)

    return False  # Timeout

@app.get('/files/<id>')
def get_file(id):
    id = unquote(id)
    generated_path = os.path.abspath(f"../generated/{id}_synthesised.wav")

    if wait_for_complete_file(generated_path, min_size=10000, stable_secs=2, timeout=30):

        return send_file(
            generated_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f"{id}_synthesised.wav"
        )

    return '', 204  # Still not ready after timeout

if __name__ == '__main__':
    app.run(debug=True, port=5000)
