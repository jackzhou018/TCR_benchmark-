"""Figure 1: what was truncated, class I (A) and class II (B).

Composes the two ChimeraX panels into one figure at native resolution: white
margins trimmed (which is what raises the effective dpi in the PDF), both panels
scaled to a common width, and a labelled header strip added above each.
"""
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib

OUT = "/14TBDrive/6TBDrive1_backup/benchmark_fresh/paper/figures/outputs"
SRC = [("A", "Class I", "panel_truncation_classI.png"),
       ("B", "Class II", "panel_truncation_classII.png")]
PAD, GAP, HDR = 24, 46, 92
FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")


def trim(im, thresh=248):
    """Drop the white border so the content itself carries more pixels per inch."""
    g = im.convert("L").point(lambda v: 0 if v > thresh else 255)
    box = g.getbbox()
    return im.crop(box) if box else im


panels = []
for letter, cls, name in SRC:
    im = Image.open(f"{OUT}/{name}")
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
    panels.append((letter, cls, trim(bg)))

W = max(p.width for _, _, p in panels)
panels = [(l, c, p if p.width == W else p.resize((W, round(p.height * W / p.width)), Image.LANCZOS))
          for l, c, p in panels]

H = sum(HDR + p.height for _, _, p in panels) + GAP + 2 * PAD
out = Image.new("RGB", (W + 2 * PAD, H), "white")
d = ImageDraw.Draw(out)
big, small = ImageFont.truetype(FONT, 62), ImageFont.truetype(FONT, 52)

y = PAD
for letter, cls, p in panels:
    d.text((PAD, y + 8), letter, font=big, fill="black")
    d.text((PAD + 78, y + 16), cls, font=small, fill="black")
    out.paste(p, (PAD, y + HDR))
    y += HDR + p.height + GAP

out.save(f"{OUT}/Figure_truncation.png")
print("wrote Figure_truncation.png", out.size)
