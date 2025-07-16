import subprocess

def generate_ttf(svg_folder, output_path):
    try:
        result = subprocess.run(
            ["fontforge", "-script", "backend/generate_font.pe", svg_folder, output_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("❌ FontForge Error Output:")
        print(e.stdout)
        print(e.stderr)
        raise RuntimeError(f"FontForge failed with return code {e.returncode}")
