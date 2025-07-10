from flask import Flask, request, jsonify, send_from_directory
import os
from split_letters import split_letters
from bw_converter import convert_to_bw
from svg_converter import convert_to_svg
from generate_font import generate_ttf

app = Flask(__name__, static_folder='../frontend')

# נתיבי תיקיות
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
SPLIT_FOLDER = os.path.join(BASE_DIR, 'split_letters_output')
BW_FOLDER = os.path.join(BASE_DIR, 'bw_letters')
SVG_FOLDER = os.path.join(BASE_DIR, 'svg_letters')
EXPORT_FOLDER = os.path.join(BASE_DIR, 'exports')

# ודא שהתיקיות קיימות
for folder in [UPLOAD_FOLDER, SPLIT_FOLDER, BW_FOLDER, SVG_FOLDER, EXPORT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def serve_home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'לא התקבל קובץ'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'שם הקובץ ריק'}), 400

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    try:
        # 1. חיתוך האותיות
        split_letters(image_path, SPLIT_FOLDER)

        # 2. המרה לשחור־לבן
        convert_to_bw(SPLIT_FOLDER, BW_FOLDER)

        # 3. המרה ל־SVG
        convert_to_svg(BW_FOLDER, SVG_FOLDER)

        # 4. יצירת פונט TTF
        ttf_path = os.path.join(EXPORT_FOLDER, 'my_font.ttf')
        generate_ttf(SVG_FOLDER, ttf_path)

        return jsonify({'message': '✅ הפונט נוצר בהצלחה! אפשר להוריד אותו עכשיו'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"שגיאה בתהליך: {e}")
        return jsonify({'error': '⚠ שגיאה במהלך יצירת הפונט'}), 500


@app.route('/download-font')
def download_font():
    return send_from_directory(EXPORT_FOLDER, 'my_font.ttf', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
