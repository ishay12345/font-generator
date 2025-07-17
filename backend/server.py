from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import subprocess

# נתיבים
UPLOAD_FOLDER = 'backend/uploads'
SPLIT_OUTPUT_FOLDER = 'backend/split_letters_output'
BW_FOLDER = 'backend/bw_letters'
SVG_FOLDER = 'backend/svg_letters'
EXPORT_FONT_FOLDER = 'exports'  # תוקן כאן

# הגדרות Flask
app = Flask(__name__, template_folder='../frontend/templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# ודא שכל התיקיות קיימות
for folder in [UPLOAD_FOLDER, SPLIT_OUTPUT_FOLDER, BW_FOLDER, SVG_FOLDER, EXPORT_FONT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # שלב 1: פיצול אותיות
    subprocess.run(['python3', 'backend/split_letters.py', filepath])

    # שלב 2: המרה לשחור-לבן
    subprocess.run(['python3', 'backend/bw_converter.py'])

    # שלב 3: המרה ל־SVG
    subprocess.run(['python3', 'backend/svg_converter.py'])

    # שלב 4: יצירת פונט
    subprocess.run(['python3', 'backend/generate_font.py'])

    font_path = os.path.join(EXPORT_FONT_FOLDER, 'handwriting_font.ttf')

    if os.path.exists(font_path):
        return send_file(font_path, as_attachment=True)
    else:
        return jsonify({'error': 'Font generation failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))

