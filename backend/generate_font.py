import os
from ufoLib2 import Font
from fontTools.ttLib import TTFont
from fontTools.ufoLib import toFile
from cu2qu import cu2qu

def generate_ttf(svg_folder: str, output_path: str):
    # 1. יצירת UFO
    uf = Font()
    uf.info.unitsPerEm = 1000
    uf.info.ascender    = 800
    uf.info.descender   = -200

    # 2. מיפוי שם→יוניקוד
    letter_map = {
      "alef":0x05D0, "bet":0x05D1, "gimel":0x05D2, "dalet":0x05D3,
      "he":0x05D4, "vav":0x05D5, "zayin":0x05D6, "het":0x05D7,
      "tet":0x05D8, "yod":0x05D9, "kaf":0x05DB, "lamed":0x05DC,
      "mem":0x05DE, "nun":0x05E0, "samekh":0x05E1, "ayin":0x05E2,
      "pe":0x05E4, "tsadi":0x05E6, "qof":0x05E7, "resh":0x05E8,
      "shin":0x05E9, "tav":0x05EA,
      "final_kaf":0x05DA,"final_mem":0x05DD,"final_nun":0x05DF,
      "final_pe":0x05E3,"final_tsadi":0x05E5
    }

    # 3. עבור כל SVG, הוסף glyph ל‑UFO
    for fname in os.listdir(svg_folder):
        if not fname.endswith('.svg'): continue
        name = fname.split('_',1)[1].rsplit('.svg',1)[0]
        code = letter_map.get(name)
        if not code: continue

        glyph = uf.newGlyph(name)
        glyph.lib['public.openTypeGlyphName'] = name
        # ייבוא קווים מקובץ SVG:
        from fontTools.pens.svgPathPen import SVGPathPen
        from fontTools.pens.ttGlyphPen import TTGlyphPen
        from xml.dom import minidom

        doc = minidom.parse(os.path.join(svg_folder, fname))
        path = doc.getElementsByTagName('path')[0].getAttribute('d')
        pen = TTGlyphPen(None)
        svgPen = SVGPathPen(pen)
        svgPen.path(path)
        glyph.coordinates, glyph.endPts, glyph.flags = pen.getCoordinates()

    # 4. שמירת UFO זמני + המרתו ל‑TTF
    tmp_ufo = os.path.join(svg_folder, 'tmp_font.ufo')
    uf.save(tmp_ufo)
    tt = TTFont()
    toFile(tmp_ufo, tt)
    tt.save(output_path)
    # מחיקת קבצים זמניים
    os.remove(tmp_ufo)
