from flask import Flask, request, jsonify, send_from_directory
import os
from split_letters import split_letters

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
        split_letters(path, OUTPUT_FOLDER)
        return jsonify({'message': '✅ האותיות נחתכו בהצלחה'})
    except Exception as e:
        print(f"שגיאה בפיצול האותיות: {e}")
        return jsonify({'message': '⚠ לא נחתכו האותיות, אך לא נכשלה השליחה'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
