from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np

app = Flask(__name__, static_folder='../frontend')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'split_letters_output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:
        split_letters(path)
        return jsonify({'message': '✅ האותיות נחתכו בהצלחה'})
    except Exception as e:
        print(f"שגיאה בפיצול האותיות: {e}")
        return jsonify({'error': 'שגיאה בפיצול האותיות'}), 500

def split_letters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    clean = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]

    boxes.sort(key=lambda b: b[1])  # לפי גובה

    rows = []
    for b in boxes:
        x, y, w, h = b
        found_row = False
        for row in rows:
            if abs(row[0][1] - y) < h:
                row.append(b)
                found_row = True
                break
        if not found_row:
            rows.append([b])

    rows.sort(key=lambda r: r[0][1])
    ordered = []
    for row in rows:
        row.sort(key=lambda b: -b[0])
        ordered.extend(row)

    padding = 10
    for i, (x, y, w, h) in enumerate(ordered[:27]):
        x1, y1 = max(x - padding, 0), max(y - padding, 0)
        x2, y2 = min(x + w + padding, img.shape[1]), min(y + h + padding, img.shape[0])
        crop = img[y1:y2, x1:x2]
        filename = f"{i:02d}_000.png"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), crop)

    print(f"✅ נחתכו {min(27, len(ordered))} אותיות ונשמרו בתיקייה:\n{OUTPUT_FOLDER}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
