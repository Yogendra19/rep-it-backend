import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from app.processing import process_workout_video
from app.utils import allowed_file

app = Flask(__name__)

# Configure upload folder and allowed file types
app.config['UPLOAD_FOLDER'] = 'app/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return "Welcome to the Rep-it Backend!"

@app.route('/upload', methods=['POST'])
def upload_workout_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    exercise_mode = request.form.get('exercise_mode')

    if not exercise_mode:
        return jsonify({"error": "Exercise mode not specified"}), 400

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        try:
            result = process_workout_video(video_path, exercise_mode)
            return jsonify(result), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Allowed types are: mp4, avi, mov"}), 400
