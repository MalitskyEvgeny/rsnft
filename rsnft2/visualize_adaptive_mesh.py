"""Render the final adaptive mesh with temperature fill and refinement levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visualize_results import colour_map, font


LEVEL_COLOURS = [(75, 85, 99), (37, 99, 235), (220, 38, 38), (147, 51, 234), (5, 150, 105)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="Adaptive run directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run
    mesh = np.load(run_dir / "final_adaptive_mesh.npz")
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    x0, x1, z0, z1 = mesh["x0"], mesh["x1"], mesh["z0"], mesh["z1"]
    level, active, temperature = mesh["level"], mesh["active"], mesh["temperature_c"]
    finite = temperature[active & np.isfinite(temperature)]
    low, high = float(finite.min()), float(finite.max())
    colours = colour_map(temperature, low, high)

    width, height = 1500, 900
    margin_left, margin_top, plot_width, plot_height = 90, 95, 1280, 700
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small_font = font(24), font(17), font(14)
    xmin, xmax, zmin, zmax = float(x0.min()), float(x1.max()), float(z0.min()), float(z1.max())

    def px(x: float) -> int:
        return margin_left + int((x - xmin) / (xmax - xmin) * plot_width)

    def py(z: float) -> int:
        return margin_top + int((z - zmin) / (zmax - zmin) * plot_height)

    draw.text((40, 22),
              f"Adaptive FVM mesh · run {meta['run_id']} · {meta['final_cells']} cells",
              fill="black", font=title_font)
    for i in range(len(x0)):
        box = (px(float(x0[i])), py(float(z0[i])), px(float(x1[i])), py(float(z1[i])))
        fill = tuple(int(v) for v in colours[i]) if active[i] else (244, 244, 245)
        outline = LEVEL_COLOURS[int(level[i]) % len(LEVEL_COLOURS)]
        draw.rectangle(box, fill=fill, outline=outline, width=1 if level[i] < 2 else 2)
    draw.rectangle((margin_left, margin_top, margin_left + plot_width, margin_top + plot_height), outline="black", width=2)
    draw.text((margin_left + plot_width // 2 - 20, 835), "x, m", fill="black", font=label_font)
    draw.text((22, margin_top + plot_height // 2), "z, m", fill="black", font=label_font)
    for fraction in np.linspace(0.0, 1.0, 5):
        xx = margin_left + int(fraction * plot_width)
        yy = margin_top + int(fraction * plot_height)
        draw.text((xx - 15, 805), f"{xmin + fraction * (xmax - xmin):.0f}", fill="black", font=small_font)
        draw.text((margin_left - 62, yy - 8), f"{zmin + fraction * (zmax - zmin):.0f}", fill="black", font=small_font)
    legend_x = 1050
    for lev in sorted(set(int(v) for v in level)):
        x = legend_x + lev * 105
        draw.rectangle((x, 43, x + 20, 63), outline=LEVEL_COLOURS[lev % len(LEVEL_COLOURS)], width=3)
        draw.text((x + 27, 44), f"level {lev}", fill="black", font=small_font)
    draw.text((90, 62),
              f"min cell {meta['minimum_dx_m']:.1f} × {meta['minimum_dz_m']:.1f} m · "
              f"temperature {low:.1f}…{high:.1f} °C",
              fill=(70, 70, 70), font=small_font)
    output = args.output or run_dir / "adaptive_mesh_0Ma.png"
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
