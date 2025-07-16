import subprocess

def generate_ttf(svg_folder, output_path):
    # הרצת FontForge עם הקובץ PE המתוקן
    try:
        subprocess.run([
            "fontforge",
            "-script",
            "backend/generate_font.pe",
            svg_folder,
            output_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FontForge failed with return code {e.returncode}")
