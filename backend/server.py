from flask import Flask, request, jsonify, send_from_directory
import os
from split_letters import split_letters
from bw_converter import convert_to_bw
from svg_converter import convert_to_svg
from generate_font import generate_ttf

# הגדרת אפליקציה עם תיקיית סטטיים
app = Flask(__name__, static_folder='../frontend')
BASE = os.path.dirname(__file__)

# נתיבי תיקיות
UPLOAD = os.path.join(BASE, 'uploads')
SPLIT  = os.path.join(BASE, 'split_letters_output')
BW     = os.path.join(BASE, 'bw_letters')
SVG    = os.path.join(BASE, 'svg_letters')
EXPORT = os.path.join(BASE, 'exports')

# יצירת כל התיקיות אם לא קיימות
for d in [UPLOAD, SPLIT, BW, SVG, EXPORT]:
    os.makedirs(d, exist_ok=True)

# מסך הבית - מחזיר את index.html
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# אם מנסים לגשת לקבצים סטטיים כמו style.css
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# API - העלאת קובץ
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error':'קובץ לא הועלה'}),400

    img_path = os.path.join(UPLOAD, file.filename)
    file.save(img_path)

    try:
        split_letters(img_path, SPLIT)
        convert_to_bw(SPLIT, BW)
        convert_to_svg(BW, SVG)
        ttf = os.path.join(EXPORT, 'my_font.ttf')
        generate_ttf(SVG, ttf)
        return jsonify({'download':'/download-font'}),200

    except Exception as e:
        print("Error pipeline:", e)
        return jsonify({'error':'⚠ שגיאה ביצירת הפונט'}),500

# API - הורדת הפונט
@app.route('/download-font')
def download_font():
    return send_from_directory(EXPORT, 'my_font.ttf', as_attachment=True)

# הרצת השרת
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

