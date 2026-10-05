"""Create quick PNG visualisations from a cached basin_fvm run (Pillow only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = np.array([
    [0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106],
    [186, 54, 85], [227, 89, 51], [249, 140, 10], [249, 201, 50],
    [252, 255, 164],
], dtype=float)


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def colour_map(values: np.ndarray, low: float, high: float) -> np.ndarray:
    finite = np.isfinite(values)
    scaled = np.where(
        finite,
        np.clip((values - low) / max(high - low, 1.0e-12), 0.0, 1.0),
        0.0,
    )
    position = scaled * (len(PALETTE) - 1)
    left = np.floor(position).astype(int)
    right = np.minimum(left + 1, len(PALETTE) - 1)
    weight = position - left
    rgb = PALETTE[left] * (1.0 - weight[..., None]) + PALETTE[right] * weight[..., None]
    rgb[~finite] = 245
    return rgb.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="Run directory or result.npz")
    parser.add_argument("--age", type=float, default=0.0, help="Nearest saved geological age, Ma")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result_path = args.run / "result.npz" if args.run.is_dir() else args.run
    run_dir = result_path.parent
    data = np.load(result_path)
    ages = data["ages_ma"]
    index = int(np.argmin(np.abs(ages - args.age)))
    temperature = data["temperature_c"][index]
    x, z = data["x"], data["z"]
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    solver = metadata["solver"]
    if metadata.get("mesh_type") == "balanced_quadtree":
        mesh_label = f"adaptive {solver['base_nx']}x{solver['base_nz']} · level {solver['max_level']}"
    else:
        mesh_label = f"{solver['nx']}x{solver['nz']}"

    finite = temperature[np.isfinite(temperature)]
    low, high = float(finite.min()), float(finite.max())
    canvas = Image.new("RGB", (1400, 620), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small_font = font(22), font(16), font(13)
    draw.text((40, 18),
        f"FVM run {metadata['run_id']} | {mesh_label} | dt <= {solver['max_dt_ma']} Ma",
        fill="black", font=title_font)

    left, top, width, height = 70, 85, 820, 450
    heat = Image.fromarray(colour_map(temperature, low, high), mode="RGB")
    heat = heat.resize((width, height), Image.Resampling.NEAREST)
    canvas.paste(heat, (left, top))
    draw.rectangle((left, top, left + width, top + height), outline="black", width=2)
    draw.text((left, 56), f"Temperature at {ages[index]:g} Ma", fill="black", font=label_font)
    draw.text((left + width // 2 - 20, top + height + 30), "x, m", fill="black", font=label_font)
    draw.text((12, top + height // 2), "z, m", fill="black", font=label_font)
    for fraction in np.linspace(0.0, 1.0, 5):
        px = left + int(fraction * width)
        pz = top + int(fraction * height)
        draw.text((px - 15, top + height + 7), f"{x[0] + fraction * (x[-1] - x[0]):.0f}", fill="black", font=small_font)
        draw.text((left - 55, pz - 8), f"{z[0] + fraction * (z[-1] - z[0]):.0f}", fill="black", font=small_font)

    bar_x = left + width + 18
    gradient = np.linspace(high, low, height)[:, None]
    bar = Image.fromarray(colour_map(gradient, low, high), mode="RGB").resize((28, height))
    canvas.paste(bar, (bar_x, top))
    draw.rectangle((bar_x, top, bar_x + 28, top + height), outline="black")
    draw.text((bar_x + 36, top - 6), f"{high:.1f} °C", fill="black", font=small_font)
    draw.text((bar_x + 36, top + height - 12), f"{low:.1f} °C", fill="black", font=small_font)

    plot_left, plot_top, plot_width, plot_height = 1040, 85, 300, 450
    draw.rectangle((plot_left, plot_top, plot_left + plot_width, plot_top + plot_height), outline="black", width=2)
    centre = len(x) // 2
    valid = np.isfinite(temperature[:, centre])
    tv, zv = temperature[valid, centre], z[valid]
    points = []
    for temp, depth in zip(tv, zv):
        px = plot_left + int((temp - low) / max(high - low, 1.0e-12) * plot_width)
        py = plot_top + int((depth - z[0]) / max(z[-1] - z[0], 1.0e-12) * plot_height)
        points.append((px, py))
    for fraction in np.linspace(0.0, 1.0, 5):
        px = plot_left + int(fraction * plot_width)
        py = plot_top + int(fraction * plot_height)
        draw.line((px, plot_top, px, plot_top + plot_height), fill=(225, 225, 225))
        draw.line((plot_left, py, plot_left + plot_width, py), fill=(225, 225, 225))
        draw.text((px - 12, plot_top + plot_height + 7), f"{low + fraction * (high - low):.0f}", fill="black", font=small_font)
    if len(points) > 1:
        draw.line(points, fill=(205, 45, 35), width=3)
    draw.text((plot_left, 56), f"Profile at x={x[centre]:.1f} m", fill="black", font=label_font)
    draw.text((plot_left + 80, plot_top + plot_height + 30), "Temperature, °C", fill="black", font=label_font)

    output = args.output or run_dir / f"temperature_{ages[index]:g}Ma.png"
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
