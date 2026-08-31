"""Side-by-side composites: Figure_accuracy (1A|1B) and Figure_determinants (2A|2C).

Panels are padded to a common height, never rescaled, so text is the same size in
every panel of a figure -- which is only true because the panel scripts already
render at one dpi and one figure height.
"""
from PIL import Image

OUT = "/14TBDrive/6TBDrive1_backup/benchmark_fresh/paper/figures/outputs"
GAP = 40
FIGS = {"Figure_accuracy": ["Figure_1A", "Figure_1B"],
        # 2B (resolution) moved to Figure 6
        "Figure_determinants": ["Figure_2A", "Figure_2C"]}

for name, panels in FIGS.items():
    ims = [Image.open(f"{OUT}/{p}.png").convert("RGB") for p in panels]
    h = max(i.height for i in ims)
    assert h - min(i.height for i in ims) < 0.12 * h, \
        f"{name}: panels differ by more than 12% in height " + str([i.size for i in ims])
    out = Image.new("RGB", (sum(i.width for i in ims) + GAP * (len(ims) - 1), h), "white")
    x = 0
    for im in ims:
        out.paste(im, (x, (h - im.height) // 2))
        x += im.width + GAP
    out.save(f"{OUT}/{name}.png", dpi=(200, 200))
    print("wrote", name, out.size, [i.size for i in ims])
