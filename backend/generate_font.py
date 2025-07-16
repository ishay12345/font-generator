import os
from defcon import Font
from ufo2ft import compileTTF

# מיפוי שם האות לעקוד יוניקוד
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


def generate_ttf(svg_folder, output_path):
    print("🚀 Generating font with defcon + ufo2ft")
    print("📁 SVG folder:", svg_folder)
    print("📄 Output path:", output_path)

    font = Font()
    font.info.familyName = "HebrewFont"
    font.info.styleName = "Regular"
    font.info.unitsPerEm = 1000
    font.info.ascender = 800
    font.info.descender = -200

    glyph_count = 0

    for filename in sorted(os.listdir(svg_folder)):
        if not filename.endswith(".svg"):
            continue

        name = os.path.splitext(filename)[0]
        if "_" not in name:
            continue

        # דוגמה לשם: 01_alef → alef
        parts = name.split("_")
        if len(parts) != 2:
            continue

        letter_name = parts[1]
        unicode_val = letter_map.get(letter_name)
        if unicode_val is None:
            continue

        glyph = font.newGlyph(letter_name)
        glyph.unicodes = [unicode_val]
        glyph.width = 600

        # טען את קובץ ה-SVG
        glyph.importOutlines(os.path.join(svg_folder, filename))
        glyph_count += 1

    if glyph_count == 0:
        raise ValueError("❌ No valid glyphs were loaded.")

    # יצירת TTF
    ttf = compileTTF(font)
    with open(output_path, "wb") as f:
        ttf.save(f)

    print(f"✅ Font created successfully: {output_path}")
