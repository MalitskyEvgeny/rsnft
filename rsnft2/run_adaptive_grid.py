"""Run several adaptive meshes and create a convergence comparison report."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_basin_fvm import DEFAULT_DATA_DIR
from compare_runs import interpolate_field


DEFAULT_CASES = ("16,16,1", "16,16,2", "16,16,3")


def parse_case(value: str) -> tuple[int, int, int]:
    try:
        parts = value.split(",")
        if len(parts) != 3:
            raise ValueError
        base_nx, base_nz, max_level = (int(part.strip()) for part in parts)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "case must have format BASE_NX,BASE_NZ,MAX_LEVEL"
        ) from error
    if base_nx < 2 or base_nz < 2 or not 0 <= max_level <= 5:
        raise argparse.ArgumentTypeError(
            "BASE_NX and BASE_NZ must be at least 2; MAX_LEVEL must be between 0 and 5"
        )
    return base_nx, base_nz, max_level


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.stdout


def run_id_from_output(output: str) -> str:
    match = re.search(r"(?:RUN|CACHE_HIT)\s+([0-9a-f]{16})", output)
    if match is None:
        raise RuntimeError("adaptive solver did not report a run id")
    return match.group(1)


def comparison(run_dir: Path, reference_dir: Path) -> dict[str, float | int]:
    with np.load(run_dir / "result.npz") as current, np.load(
        reference_dir / "result.npz"
    ) as reference:
        mapped = interpolate_field(
            current["x"], current["z"], current["temperature_c"][-1],
            reference["x"], reference["z"],
        )
        reference_final = reference["temperature_c"][-1]
        mask = np.isfinite(mapped) & np.isfinite(reference_final)
        if not np.any(mask):
            raise ValueError(f"no overlapping finite cells for {run_dir} and {reference_dir}")
        difference = mapped[mask] - reference_final[mask]
        return {
            "compared_cells": int(mask.sum()),
            "rms_difference_c": float(np.sqrt(np.mean(difference**2))),
            "relative_l2": float(
                np.linalg.norm(difference) / np.linalg.norm(reference_final[mask])
            ),
            "max_abs_difference_c": float(np.max(np.abs(difference))),
        }


def format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def write_reports(rows: list[dict[str, Any]], reference_id: str, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"reference_run_id": reference_id, "runs": rows}
    (report_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fieldnames = list(rows[0])
    with (report_dir / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Сравнение адаптивных сеток",
        "",
        f"Эталон для попарных отклонений: `{reference_id}` (самая мелкая ячейка).",
        "",
        "| Сетка | Уровень | Ячеек | Мин. ячейка, м | Время, с | L2 | RMS, °C | Max, °C | Сходимость |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        run_link = f"../{row['run_id']}"
        mesh = f"{row['base_nx']}×{row['base_nz']}"
        min_cell = (
            f"{format_number(row['minimum_dx_m'], 5)}×"
            f"{format_number(row['minimum_dz_m'], 5)}"
        )
        lines.append(
            f"| [{mesh}]({run_link}) | {row['max_level']} | {row['final_cells']} | "
            f"{min_cell} | {format_number(row['elapsed_seconds'], 5)} | "
            f"{format_number(row['relative_l2'], 5)} | "
            f"{format_number(row['rms_difference_c'], 5)} | "
            f"{format_number(row['max_abs_difference_c'], 5)} | "
            f"{'да' if row['nonlinear_converged'] else 'нет'} |"
        )
    lines.extend(["", "## Изображения", ""])
    for row in rows:
        lines.extend([
            f"### {row['base_nx']}×{row['base_nz']}, уровень {row['max_level']}",
            "",
            f"[Температура](../{row['run_id']}/temperature_0Ma.png) · "
            f"[Адаптивная сетка](../{row['run_id']}/adaptive_mesh_0Ma.png)",
            "",
        ])
    (report_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a series of adaptive FVM meshes and compare them"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=Path("adaptive_results"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument(
        "--case", action="append", type=parse_case, default=[], metavar="NX,NZ,LEVEL",
        help="repeatable; defaults: 16,16,1; 16,16,2; 16,16,3",
    )
    parser.add_argument("--max-dt-ma", type=float, default=2.5)
    parser.add_argument("--interface-band", type=float, default=0.55)
    parser.add_argument("--picard-iterations", type=int, default=8)
    parser.add_argument("--picard-tolerance-c", type=float, default=1.0e-3)
    parser.add_argument("--linear-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    cases = args.case or [parse_case(value) for value in DEFAULT_CASES]
    if len(set(cases)) != len(cases):
        parser.error("duplicate --case values are not allowed")
    if args.max_dt_ma <= 0:
        parser.error("--max-dt-ma must be positive")

    solver = Path(__file__).with_name("adaptive_basin_fvm.py")
    temperature_visualizer = Path(__file__).with_name("visualize_results.py")
    mesh_visualizer = Path(__file__).with_name("visualize_adaptive_mesh.py")
    run_dirs: list[Path] = []

    for number, (base_nx, base_nz, max_level) in enumerate(cases, start=1):
        print(f"\n[{number}/{len(cases)}] base={base_nx}x{base_nz}, level={max_level}")
        command = [
            sys.executable, str(solver), "--data-dir", str(args.data_dir),
            "--cache-dir", str(args.cache_dir), "--base-nx", str(base_nx),
            "--base-nz", str(base_nz), "--max-level", str(max_level),
            "--max-dt-ma", str(args.max_dt_ma), "--interface-band", str(args.interface_band),
            "--picard-iterations", str(args.picard_iterations),
            "--picard-tolerance-c", str(args.picard_tolerance_c),
            "--linear-tolerance", str(args.linear_tolerance),
        ]
        if args.force:
            command.append("--force")
        run_id = run_id_from_output(run_command(command))
        run_dir = args.cache_dir / run_id
        run_dirs.append(run_dir)
        if not args.skip_images:
            run_command([sys.executable, str(temperature_visualizer), str(run_dir), "--age", "0"])
            run_command([sys.executable, str(mesh_visualizer), str(run_dir)])

    metadata = [
        json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    reference_index = min(
        range(len(metadata)),
        key=lambda index: (
            metadata[index]["minimum_dx_m"] * metadata[index]["minimum_dz_m"],
            -metadata[index]["final_cells"],
        ),
    )
    reference_dir = run_dirs[reference_index]
    reference_id = metadata[reference_index]["run_id"]

    rows: list[dict[str, Any]] = []
    for run_dir, item in zip(run_dirs, metadata):
        metrics = comparison(run_dir, reference_dir)
        solver_config = item["solver"]
        rows.append({
            "run_id": item["run_id"],
            "is_reference": item["run_id"] == reference_id,
            "base_nx": solver_config["base_nx"],
            "base_nz": solver_config["base_nz"],
            "max_level": solver_config["max_level"],
            "final_cells": item["final_cells"],
            "final_active_cells": item["final_active_cells"],
            "minimum_dx_m": item["minimum_dx_m"],
            "minimum_dz_m": item["minimum_dz_m"],
            "elapsed_seconds": item["elapsed_seconds"],
            "temperature_min_c": item["final_temperature_min_c"],
            "temperature_max_c": item["final_temperature_max_c"],
            "nonlinear_converged": item.get("nonlinear_converged", False),
            "unconverged_time_steps": item.get("unconverged_time_steps"),
            **metrics,
        })

    report_dir = args.report_dir or args.cache_dir / "grid_comparison"
    write_reports(rows, reference_id, report_dir)
    print(f"\nREFERENCE {reference_id}")
    print(f"REPORT {report_dir / 'README.md'}")
    print(f"CSV {report_dir / 'comparison.csv'}")
    print(f"JSON {report_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
