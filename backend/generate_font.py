import subprocess
import json
import os

def generate_ttf(svg_folder, output_path):
    letter_map = {
        "alef": 0x05D0, "bet": 0x05D1, "gimel": 0x05D2, "dalet": 0x05D3,
        "he": 0x05D4, "vav": 0x05D5, "zayin": 0x05D6, "het": 0x05D7,
        "tet": 0x05D8, "yod": 0x05D9, "kaf": 0x05DB, "lamed": 0x05DC,
        "mem": 0x05DE, "nun": 0x05E0, "samekh": 0x05E1, "ayin": 0x05E2,
        "pe": 0x05E4, "tsadi": 0x05E6, "qof": 0x05E7, "resh": 0x05E8,
        "shin": 0x05E9, "tav": 0x05EA,
        "final_kaf": 0x05DA, "final_mem": 0x05DD, "final_nun": 0x05DF,
        "final_pe": 0x05E3, "final_tsadi": 0x05E5
    }

    map_path = os.path.join("backend", "letter_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(letter_map, f)

    try:
        result = subprocess.run(
            [
                "fontforge",
                "-script",
                "backend/generate_font.pe",
                svg_folder,
                output_path,
                map_path
            ],
            capture_output=True,
            text=True,
            check=True
        )

        print("✅ FontForge STDOUT:\n", result.stdout)
        print("📛 FontForge STDERR:\n", result.stderr)

    except subprocess.CalledProcessError as e:
        print("❌ FontForge failed!")
        print("🔧 STDOUT:\n", e.stdout)
        print("🚨 STDERR:\n", e.stderr)
        raise RuntimeError(f"FontForge failed with return code {e.returncode}")
