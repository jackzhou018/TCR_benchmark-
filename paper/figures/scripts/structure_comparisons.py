#!/usr/bin/env python
"""Figure_structure_comparisons.png -- one row per complex, four renders across.

    python paper/figures/scripts/structure_comparisons.py

Input is the hand-made slides in paper/benchmark_fresh/figures/Figure_structure_comparisons/,
one SVG per complex holding four embedded renders. Their order inside the file is not the order
they appear in, so panels are sorted by rendered x. All four panels of a row are cropped to the
same rectangle -- the union of their content boxes -- so a TCR that sits lower in the overlap
panel still lines up with itself in the single-model panels.
"""
import base64, io, os, re
import xml.etree.ElementTree as ET

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = f"{ROOT}/benchmark_fresh/figures/Figure_structure_comparisons"
OUT = f"{ROOT}/benchmark_fresh/figures/Figure_structure_comparisons.png"

# (row label, svg) -- 8RYO and 9GV7 are the two failures, 43RH the success
ROWS = [("8RYO", "8RYO_analysis.svg"), ("9GV7", "9GV7.svg"), ("43RH", "43RH_analysis.svg")]
COLS = [("Experimental", "#5A5A5A"), ("AlphaFold3", "#B22222"),
        ("ESMFold2", "#008080"), ("Overlap", "#1A1A1A")]
PAD = 24          # px of white kept around the row's content box


def renders(svg):
    """The four embedded images, left to right as the slide lays them out."""
    root = ET.parse(svg).getroot()
    found = []
    for parent in root.iter():
        for img in list(parent):
            if not img.tag.endswith("image"):
                continue
            n = re.findall(r"[-\d.eE]+", parent.get("transform") or img.get("transform") or "")
            sx, tx = (float(n[0]), float(n[4])) if len(n) >= 6 else (1.0, 0.0)
            href = img.get("{http://www.w3.org/1999/xlink}href") or img.get("href")
            found.append((float(img.get("x", 0)) * sx + tx,
                          Image.open(io.BytesIO(base64.b64decode(href.split(",", 1)[1])))))
    assert len(found) == 4, f"{svg}: {len(found)} images"
    return [im.convert("RGB") for _, im in sorted(found, key=lambda z: z[0])]


def box(im):
    nz = np.where((np.asarray(im) < 245).any(2))
    return nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()


rows = []
for label, svg in ROWS:
    ims = renders(f"{SRC}/{svg}")
    b = [box(i) for i in ims]
    x0, y0 = min(v[0] for v in b) - PAD, min(v[1] for v in b) - PAD
    x1, y1 = max(v[2] for v in b) + PAD, max(v[3] for v in b) + PAD
    rows.append((label, [i.crop((x0, y0, x1, y1)) for i in ims]))

# each row's cell takes that row's own aspect, so every render fills its cell instead of
# being letterboxed into a common one -- 8RYO is tall and narrow, the other two are not
ar = [r[1][0].height / r[1][0].width for r in rows]
fig_w = 9.0
cell_w = fig_w / 4.25
fig, axes = plt.subplots(3, 4, figsize=(fig_w, cell_w * sum(ar) * 1.10),
                         gridspec_kw=dict(wspace=0.02, hspace=0.05, height_ratios=ar))
for r, (label, ims) in enumerate(rows):
    for c, im in enumerate(ims):
        ax = axes[r][c]
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 0:
            ax.set_title(COLS[c][0], color=COLS[c][1], fontsize=11, fontweight="bold", pad=8)
        if c == 0:
            ax.set_ylabel(label, fontsize=12, fontweight="bold", rotation=0,
                          ha="right", va="center", labelpad=14)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(OUT)
