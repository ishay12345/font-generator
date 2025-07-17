from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import subprocess

# הבסיס – תיקיית הקובץ הנוכחי (backend/)
BASE = os.path.dirname(os.path.abspath(__file__))

# תיקיות עבודה מוחלטות
UPLOAD_FOLDER       = os.path.join(BASE, 'uploads')
SPLIT_OUTPUT_FOLDER = os.path.join(BASE, 'split_letters_output')
BW_FOLDER           = os.path.join(BASE, 'bw_letters')
SVG_FOLDER          = os.path.join(BASE, 'svg_letters')
EXPORT_FONT_FOLDER  = os.path.join(BASE, '..', 'exports')  # exports/ בגיט ראשי

# ודא כל התיקיות קיימות
for folder in [UPLOAD_FOLDER, SPLIT_OUTPUT_FOLDER, BW_FOLDER, SVG_FOLDER, EXPORT_FONT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# אתחול Flask – templates בתיקיית frontend/templates
app = Flask(
    __name__,
    template_folder=os.path.join(BASE, '..', 'frontend', 'templates'),
    static_folder=os.path.join(BASE, '..', 'frontend', 'static'),
)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return 'No file part', 400

        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # שלב א': פילוח האותיות
            from split_letters import split_letters
            split_letters(file_path, output_folder='split_letters_output')

            # שלב ב': המרה לשחור-לבן
            from bw_converter import convert_to_bw
            convert_to_bw(input_dir='split_letters_output', output_dir='bw_letters')

            # שלב ג': המרה ל-SVG
            from svg_converter import convert_to_svg
            convert_to_svg(input_dir='bw_letters', output_dir='svg_letters')

            # שלב ד': יצירת הפונט
            from generate_font import generate_font_from_svgs
            generate_font_from_svgs(input_dir='svg_letters', output_path='fonts/output.ttf')

            return 'File uploaded and font generated successfully', 200
    except Exception as e:
        import traceback
        traceback.print_exc()  # ידפיס את כל השגיאה לשרת
        return f'Internal Server Error: {str(e)}', 500


    file = request.files['file']
    if not file or file.filename == '':
        return jsonify(error='No selected file'), 400

    # שמירת הקובץ
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # נקיון תיקיות לפני ריצה
    for folder in [SPLIT_OUTPUT_FOLDER, BW_FOLDER, SVG_FOLDER]:
        subprocess.run(['rm', '-rf', folder], cwd=BASE)
        os.makedirs(folder, exist_ok=True)

    # 1. פיצול אותיות
    proc = subprocess.run(
        ['python3', os.path.join(BASE, 'split_letters.py'), filepath],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        app.logger.error(proc.stderr)
        return jsonify(error='Error in split_letters'), 500

    # 2. המרה לשחור-לבן
    proc = subprocess.run(
        ['python3', os.path.join(BASE, 'bw_converter.py'), SPLIT_OUTPUT_FOLDER, BW_FOLDER],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        app.logger.error(proc.stderr)
        return jsonify(error='Error in bw_converter'), 500

    # 3. המרה ל‑SVG
    proc = subprocess.run(
        ['python3', os.path.join(BASE, 'svg_converter.py'), BW_FOLDER, SVG_FOLDER],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        app.logger.error(proc.stderr)
        return jsonify(error='Error in svg_converter'), 500

    # 4. יצירת הפונט דרך fontTools (generate_font.py)
    font_output = os.path.join(EXPORT_FONT_FOLDER, 'my_font.ttf')
    proc = subprocess.run(
        ['python3', os.path.join(BASE, 'generate_font.py'), SVG_FOLDER, font_output],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        app.logger.error(proc.stderr)
        return jsonify(error='Error in generate_font'), 500

    if os.path.exists(font_output):
        return send_file(font_output, as_attachment=True)
    else:
        return jsonify(error='Font not found'), 500


@app.route('/download/<path:filename>', methods=['GET'])
def download_font(filename):
    return send_file(os.path.join(EXPORT_FONT_FOLDER, filename), as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

