"""Figure 1: what was truncated, class I (A) and class II (B).

Composes the two ChimeraX panels into one figure at native resolution: white
margins trimmed (which is what raises the effective dpi in the PDF), both panels
scaled to a common width, and a labelled header strip added above each.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

OUT = "/14TBDrive/6TBDrive1_backup/benchmark_fresh/paper/figures/outputs"
# The entry each ChimeraX panel was rendered from. The panels arrive as finished PNGs with
# no accession drawn on them, so this is the only record of it -- and it is printed on the
# figure, so it has to be edited here if a panel is ever re-rendered from a different entry.
SRC = [("A", "Class I", "7RTR", "panel_truncation_classI.png"),
       ("B", "Class II", "8PJG", "panel_truncation_classII.png")]
PAD, GAP, HDR = 24, 46, 92
FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")


def drop_annotation_box(im, grey=118, tol=6, min_run=120):
    """Remove the rectangle drawn around the class II 'Excluded TCR constant domains'
    label. The box is a flat 2 px line of one grey; rows/columns carrying a long run
    of it are the edges, and those pixels are inpainted by diffusion from their
    unmasked neighbours, which restores the ribbon where the box crossed it.
    No-ops on a panel that has no such box."""
    a = np.asarray(im.convert("RGBA")).astype(float)
    line = np.abs(np.asarray(im.convert("L")).astype(int) - grey) <= tol
    rows = np.flatnonzero(line.sum(1) > min_run)
    cols = np.flatnonzero(line.sum(0) > min_run)
    if not len(rows) or not len(cols):
        return im
    y0, y1, x0, x1 = rows.min(), rows.max(), cols.min(), cols.max()
    m = np.zeros(line.shape, bool)
    m[rows[:, None], np.arange(x0, x1 + 1)[None, :]] = True
    m[np.arange(y0, y1 + 1)[:, None], cols[None, :]] = True
    m &= line                                   # only the drawn pixels, never the text

    out, todo = a.copy(), m.copy()
    for _ in range(20):
        if not todo.any():
            break
        known = ~todo
        acc = np.zeros_like(out); cnt = np.zeros(out.shape[:2])
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            acc += np.roll(np.roll(out, dy, 0), dx, 1) * np.roll(np.roll(known, dy, 0), dx, 1)[..., None]
            cnt += np.roll(np.roll(known, dy, 0), dx, 1)
        fill = todo & (cnt > 0)
        out[fill] = acc[fill] / cnt[fill][:, None]
        todo &= ~fill
    assert not todo.any(), "inpaint did not converge"
    return Image.fromarray(out.round().astype("uint8"), "RGBA")


def trim(im, thresh=248):
    """Drop the white border so the content itself carries more pixels per inch."""
    g = im.convert("L").point(lambda v: 0 if v > thresh else 255)
    box = g.getbbox()
    return im.crop(box) if box else im


panels = []
for letter, cls, pdb, name in SRC:
    im = drop_annotation_box(Image.open(f"{OUT}/{name}"))
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
    panels.append((letter, cls, pdb, trim(bg)))

W = max(p.width for *_, p in panels)
panels = [(l, c, i, p if p.width == W else p.resize((W, round(p.height * W / p.width)),
                                                    Image.LANCZOS))
          for l, c, i, p in panels]

H = sum(HDR + p.height for *_, p in panels) + GAP + 2 * PAD
out = Image.new("RGB", (W + 2 * PAD, H), "white")
d = ImageDraw.Draw(out)
big, small = ImageFont.truetype(FONT, 62), ImageFont.truetype(FONT, 52)

y = PAD
for letter, cls, pdb, p in panels:
    d.text((PAD, y + 8), letter, font=big, fill="black")
    d.text((PAD + 78, y + 16), cls, font=small, fill="black")
    # the accession sits after the class name, in the same strip
    x = PAD + 78 + d.textlength(cls, font=small)
    d.text((x + 26, y + 16), f"(PDB {pdb})", font=small, fill="black")
    out.paste(p, (PAD, y + HDR))
    y += HDR + p.height + GAP

out.save(f"{OUT}/Figure_truncation.png")
print("wrote Figure_truncation.png", out.size)
