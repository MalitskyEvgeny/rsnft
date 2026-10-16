"""Compare final temperature fields from two cached runs without SciPy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def interpolate_field(x0: np.ndarray, z0: np.ndarray, field: np.ndarray,
                      x1: np.ndarray, z1: np.ndarray) -> np.ndarray:
    horizontal = np.full((len(z0), len(x1)), np.nan)
    for j in range(len(z0)):
        valid = np.isfinite(field[j])
        if np.count_nonzero(valid) >= 2:
            horizontal[j] = np.interp(x1, x0[valid], field[j, valid])
    result = np.full((len(z1), len(x1)), np.nan)
    for i in range(len(x1)):
        valid = np.isfinite(horizontal[:, i])
        if np.count_nonzero(valid) >= 2:
            result[:, i] = np.interp(z1, z0[valid], horizontal[valid, i])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("--output", type=Path, default=Path("comparison.json"))
    args = parser.parse_args()
    coarse_path = args.coarse / "result.npz" if args.coarse.is_dir() else args.coarse
    fine_path = args.fine / "result.npz" if args.fine.is_dir() else args.fine
    coarse, fine = np.load(coarse_path), np.load(fine_path)
    coarse_final = coarse["temperature_c"][-1]
    fine_final = fine["temperature_c"][-1]
    mapped = interpolate_field(coarse["x"], coarse["z"], coarse_final, fine["x"], fine["z"])
    mask = np.isfinite(mapped) & np.isfinite(fine_final)
    difference = mapped[mask] - fine_final[mask]
    report = {
        "coarse": str(coarse_path), "fine": str(fine_path),
        "compared_cells": int(mask.sum()),
        "rms_difference_c": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(fine_final[mask])),
        "max_abs_difference_c": float(np.max(np.abs(difference))),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
