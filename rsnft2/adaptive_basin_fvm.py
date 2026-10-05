from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from basin_fvm import (
    BasinDataset,
    SECONDS_PER_MA,
    harmonic,
    interpolate_state,
    pcg_solve,
    sekiguchi_conductivity,
    update_manifest,
    waples_heat_capacity,
)


DEFAULT_DATA_DIR = Path(r"C:\Users\solo\Desktop\МЭИ\3 мага\роснефть\data1")


@dataclass(frozen=True)
class AdaptiveConfig:
    base_nx: int = 16
    base_nz: int = 32
    max_level: int = 5
    interface_band: float = 0.55
    max_dt_ma: float = 2.5
    picard_iterations: int = 8
    picard_tolerance_c: float = 1.0e-3
    linear_tolerance: float = 1.0e-8
    fluid_lambda: float = 0.60
    fluid_density: float = 1000.0
    fluid_heat_capacity: float = 4180.0
    initial_lambda: float = 2.5
    preview_nx: int = 160
    preview_nz: int = 240

    def validate(self) -> None:
        if self.base_nx < 2 or self.base_nz < 2:
            raise ValueError("base_nx and base_nz must be at least 2")
        if not 0 <= self.max_level <= 5:
            raise ValueError("max_level must be between 0 and 5")
        if self.max_dt_ma <= 0 or self.interface_band < 0:
            raise ValueError("time step must be positive and interface_band non-negative")
        if self.picard_iterations < 1:
            raise ValueError("picard_iterations must be positive")
        if self.picard_tolerance_c <= 0 or self.linear_tolerance <= 0:
            raise ValueError("solver tolerances must be positive")
        if self.preview_nx < 2 or self.preview_nz < 2:
            raise ValueError("preview_nx and preview_nz must be at least 2")


class Cell(NamedTuple):
    ix0: int
    ix1: int
    iz0: int
    iz1: int
    level: int


class AdaptiveMesh:
    def __init__(self, cells: list[Cell], bounds: tuple[float, float, float, float],
                 config: AdaptiveConfig):
        self.cells = cells
        self.bounds = bounds
        self.config = config
        self.scale = 2 ** config.max_level
        self.nxf = config.base_nx * self.scale
        self.nzf = config.base_nz * self.scale
        x0, x1, z0, z1 = bounds
        self.dxf = (x1 - x0) / self.nxf
        self.dzf = (z1 - z0) / self.nzf
        self.x0 = np.array([x0 + c.ix0 * self.dxf for c in cells])
        self.x1 = np.array([x0 + c.ix1 * self.dxf for c in cells])
        self.z0 = np.array([z0 + c.iz0 * self.dzf for c in cells])
        self.z1 = np.array([z0 + c.iz1 * self.dzf for c in cells])
        self.xc = 0.5 * (self.x0 + self.x1)
        self.zc = 0.5 * (self.z0 + self.z1)
        self.width = self.x1 - self.x0
        self.height = self.z1 - self.z0
        self.area = self.width * self.height
        self.level = np.array([c.level for c in cells], dtype=np.int8)
        self.lookup = np.empty((self.nzf, self.nxf), dtype=np.int32)
        for index, cell in enumerate(cells):
            self.lookup[cell.iz0:cell.iz1, cell.ix0:cell.ix1] = index
        self.pairs = face_pairs(cells, self.dxf, self.dzf, bounds)

    def locate(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x_min, x_max, z_min, z_max = self.bounds
        ix = np.clip(((x - x_min) / (x_max - x_min) * self.nxf).astype(int), 0, self.nxf - 1)
        iz = np.clip(((z - z_min) / (z_max - z_min) * self.nzf).astype(int), 0, self.nzf - 1)
        return self.lookup[iz, ix]


def state_at(data: BasinDataset, older_age: float, younger_age: float,
             fraction: float, x: np.ndarray) -> dict[str, Any]:
    older = data.exact_state(older_age, x)
    if older_age == younger_age:
        return older
    younger = data.exact_state(younger_age, x)
    return interpolate_state(older, younger, fraction)


def cell_intersects_refinement_zone(cell: Cell, probe_x: np.ndarray,
                                    state: dict[str, Any], bounds: tuple[float, float, float, float],
                                    config: AdaptiveConfig) -> bool:
    x_min, x_max, z_min, z_max = bounds
    scale = 2 ** config.max_level
    dxf = (x_max - x_min) / (config.base_nx * scale)
    dzf = (z_max - z_min) / (config.base_nz * scale)
    xa, xb = x_min + cell.ix0 * dxf, x_min + cell.ix1 * dxf
    za, zb = z_min + cell.iz0 * dzf, z_min + cell.iz1 * dzf
    indices = np.flatnonzero((probe_x >= xa - 1.0e-10) & (probe_x <= xb + 1.0e-10))
    if len(indices) == 0:
        indices = np.array([int(np.argmin(np.abs(probe_x - 0.5 * (xa + xb))))])
    band = config.interface_band * (zb - za)
    for layer in state["layers"].values():
        for key in ("top", "bot"):
            boundary = layer[key][indices]
            if np.nanmin(boundary) <= zb + band and np.nanmax(boundary) >= za - band:
                return True
        lith = layer["lith"][indices]
        if len(lith) > 1 and np.any(lith[1:] != lith[:-1]):
            lo = np.minimum(layer["top"][indices], layer["bot"][indices])
            hi = np.maximum(layer["top"][indices], layer["bot"][indices])
            if np.nanmin(lo) <= zb + band and np.nanmax(hi) >= za - band:
                return True
    return False


def split_cell(cell: Cell) -> list[Cell]:
    ixm = (cell.ix0 + cell.ix1) // 2
    izm = (cell.iz0 + cell.iz1) // 2
    level = cell.level + 1
    return [
        Cell(cell.ix0, ixm, cell.iz0, izm, level),
        Cell(ixm, cell.ix1, cell.iz0, izm, level),
        Cell(cell.ix0, ixm, izm, cell.iz1, level),
        Cell(ixm, cell.ix1, izm, cell.iz1, level),
    ]


def _integer_face_pairs(cells: list[Cell]) -> list[tuple[int, int, int, int, int]]:
    """Return (left/top cell, right/bottom cell, overlap0, overlap1, orientation)."""
    left: dict[int, list[int]] = {}
    right: dict[int, list[int]] = {}
    top: dict[int, list[int]] = {}
    bottom: dict[int, list[int]] = {}
    for i, c in enumerate(cells):
        left.setdefault(c.ix0, []).append(i); right.setdefault(c.ix1, []).append(i)
        top.setdefault(c.iz0, []).append(i); bottom.setdefault(c.iz1, []).append(i)
    result: list[tuple[int, int, int, int, int]] = []
    for coordinate in right.keys() & left.keys():
        for a in right[coordinate]:
            for b in left[coordinate]:
                lo, hi = max(cells[a].iz0, cells[b].iz0), min(cells[a].iz1, cells[b].iz1)
                if hi > lo:
                    result.append((a, b, lo, hi, 0))
    for coordinate in bottom.keys() & top.keys():
        for a in bottom[coordinate]:
            for b in top[coordinate]:
                lo, hi = max(cells[a].ix0, cells[b].ix0), min(cells[a].ix1, cells[b].ix1)
                if hi > lo:
                    result.append((a, b, lo, hi, 1))
    return result


def balance_cells(cells: list[Cell], max_level: int) -> list[Cell]:
    for _ in range(max_level + 1):
        marked: set[int] = set()
        for a, b, _, _, _ in _integer_face_pairs(cells):
            if abs(cells[a].level - cells[b].level) > 1:
                marked.add(a if cells[a].level < cells[b].level else b)
        if not marked:
            return cells
        updated: list[Cell] = []
        for i, cell in enumerate(cells):
            if i in marked and cell.level < max_level:
                updated.extend(split_cell(cell))
            else:
                updated.append(cell)
        cells = updated
    return cells


def face_pairs(cells: list[Cell], dxf: float, dzf: float,
               bounds: tuple[float, float, float, float]) -> dict[str, np.ndarray]:
    raw = _integer_face_pairs(cells)
    p, q, length, distance, face_x, face_z, orientation = [], [], [], [], [], [], []
    x_min, _, z_min, _ = bounds
    for a, b, lo, hi, axis in raw:
        ca, cb = cells[a], cells[b]
        p.append(a); q.append(b); orientation.append(axis)
        if axis == 0:
            length.append((hi - lo) * dzf)
            distance.append(0.5 * ((ca.ix1 - ca.ix0) + (cb.ix1 - cb.ix0)) * dxf)
            face_x.append(x_min + ca.ix1 * dxf)
            face_z.append(z_min + 0.5 * (lo + hi) * dzf)
        else:
            length.append((hi - lo) * dxf)
            distance.append(0.5 * ((ca.iz1 - ca.iz0) + (cb.iz1 - cb.iz0)) * dzf)
            face_x.append(x_min + 0.5 * (lo + hi) * dxf)
            face_z.append(z_min + ca.iz1 * dzf)
    return {
        "p": np.asarray(p, int), "q": np.asarray(q, int),
        "length": np.asarray(length, float), "distance": np.asarray(distance, float),
        "face_x": np.asarray(face_x, float), "face_z": np.asarray(face_z, float),
        "orientation": np.asarray(orientation, np.int8),
    }


def build_mesh(data: BasinDataset, older_age: float, younger_age: float,
               fraction: float, config: AdaptiveConfig) -> AdaptiveMesh:
    x_min, x_max, _, _ = data.bounds
    probe_count = max(257, config.base_nx * (2 ** config.max_level) * 4 + 1)
    probe_x = np.linspace(x_min, x_max, probe_count)
    probe_state = state_at(data, older_age, younger_age, fraction, probe_x)
    scale = 2 ** config.max_level
    cells: list[Cell] = []

    def refine(cell: Cell) -> None:
        if cell.level < config.max_level and cell_intersects_refinement_zone(
            cell, probe_x, probe_state, data.bounds, config
        ):
            for child in split_cell(cell):
                refine(child)
        else:
            cells.append(cell)

    for j in range(config.base_nz):
        for i in range(config.base_nx):
            refine(Cell(i * scale, (i + 1) * scale, j * scale, (j + 1) * scale, 0))
    cells = balance_cells(cells, config.max_level)
    return AdaptiveMesh(cells, data.bounds, config)


def rasterize_cells(state: dict[str, Any], mesh: AdaptiveMesh) -> dict[str, Any]:
    n = len(mesh.cells)
    layer_index = np.full(n, -1, int)
    lith = np.full(n, -1, int)
    phi = np.full(n, np.nan)
    names = sorted(state["layers"])
    for number, name in enumerate(names):
        layer = state["layers"][name]
        lo = np.minimum(layer["top"], layer["bot"])
        hi = np.maximum(layer["top"], layer["bot"])
        mask = (mesh.zc >= lo) & (mesh.zc <= hi) & ((hi - lo) > 1.0e-8)
        layer_index[mask] = number
        lith[mask] = layer["lith"][mask]
        phi[mask] = layer["phi"][mask]
    return {"active": layer_index >= 0, "layer_index": layer_index,
            "lith": lith, "phi": phi, "names": names}


def adaptive_properties(temperature_c: np.ndarray, grid: dict[str, Any],
                        properties: dict[int, dict[str, float | str]],
                        config: AdaptiveConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lam = np.full_like(temperature_c, np.nan)
    rho_c = np.full_like(temperature_c, np.nan)
    source = np.full_like(temperature_c, np.nan)
    tk = temperature_c + 273.15
    for code in np.unique(grid["lith"][grid["active"]]):
        if int(code) not in properties:
            raise KeyError(f"Unknown lithotype {code}")
        prop = properties[int(code)]
        mask = grid["active"] & (grid["lith"] == code)
        phi = grid["phi"][mask]
        lam_s = sekiguchi_conductivity(tk[mask], float(prop["lambda20"]))
        c_s = waples_heat_capacity(tk[mask], float(prop["c20"]))
        lam[mask] = lam_s ** (1.0 - phi) * config.fluid_lambda ** phi
        rho_c[mask] = ((1.0 - phi) * float(prop["rho_s"]) * c_s
                       + phi * config.fluid_density * config.fluid_heat_capacity)
        source[mask] = (1.0 - phi) * float(prop["As"])
    return lam, rho_c, source


def initialise(mesh: AdaptiveMesh, state: dict[str, Any], grid: dict[str, Any],
               config: AdaptiveConfig) -> np.ndarray:
    depth = np.maximum(mesh.zc - state["surface"], 0.0)
    result = state["tsurf"] + state["qbasal"] * depth / config.initial_lambda
    result[~grid["active"]] = np.nan
    return result


def material_remap(old_mesh: AdaptiveMesh, old_temperature: np.ndarray,
                   old_grid: dict[str, Any], old_state_on_new_x: dict[str, Any],
                   new_mesh: AdaptiveMesh, new_state: dict[str, Any],
                   new_grid: dict[str, Any]) -> np.ndarray:
    result = np.full(len(new_mesh.cells), np.nan)
    active_old = old_grid["active"] & np.isfinite(old_temperature)
    for number, name in enumerate(new_grid["names"]):
        mask = new_grid["active"] & (new_grid["layer_index"] == number)
        if not np.any(mask):
            continue
        new_layer = new_state["layers"][name]
        old_layer = old_state_on_new_x["layers"].get(name)
        if old_layer is None:
            result[mask] = new_state["tsurf"][mask]
            continue
        new_lo = np.minimum(new_layer["top"], new_layer["bot"])
        new_hi = np.maximum(new_layer["top"], new_layer["bot"])
        old_lo = np.minimum(old_layer["top"], old_layer["bot"])
        old_hi = np.maximum(old_layer["top"], old_layer["bot"])
        valid_thickness = (old_hi - old_lo) > 1.0e-8
        local = mask & valid_thickness
        eta = np.zeros(len(new_mesh.cells))
        eta[local] = np.clip(
            (new_mesh.zc[local] - new_lo[local]) / np.maximum(new_hi[local] - new_lo[local], 1.0e-12),
            0.0, 1.0,
        )
        target_z = old_lo[local] + eta[local] * (old_hi[local] - old_lo[local])
        old_cell = old_mesh.locate(new_mesh.xc[local], target_z)
        values = old_temperature[old_cell]
        good = active_old[old_cell]
        target_indices = np.flatnonzero(local)
        result[target_indices[good]] = values[good]
        result[target_indices[~good]] = old_state_on_new_x["tsurf"][target_indices[~good]]
        result[mask & ~valid_thickness] = new_state["tsurf"][mask & ~valid_thickness]
    missing = new_grid["active"] & ~np.isfinite(result)
    result[missing] = new_state["tsurf"][missing]
    return result


def adaptive_step(transported: np.ndarray, mesh: AdaptiveMesh, state: dict[str, Any],
                  grid: dict[str, Any], dt_seconds: float,
                  properties: dict[int, dict[str, float | str]],
                  config: AdaptiveConfig) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    active = grid["active"]
    local_index = np.full(len(mesh.cells), -1, int)
    local_index[active] = np.arange(np.count_nonzero(active))
    trial = transported.copy()
    pairs = mesh.pairs
    last_linear_iterations = 0
    last_linear_residual = math.inf
    max_change = math.inf

    for nonlinear_iteration in range(config.picard_iterations):
        lam, rho_c, source = adaptive_properties(trial, grid, properties, config)
        diagonal = rho_c[active] * mesh.area[active] / dt_seconds
        rhs = diagonal * transported[active] + source[active] * mesh.area[active]
        pp: list[int] = []; qq: list[int] = []; gg: list[float] = []
        for face in range(len(pairs["p"])):
            a, b = pairs["p"][face], pairs["q"][face]
            a_active, b_active = active[a], active[b]
            if a_active and b_active:
                g = harmonic(lam[a], lam[b]) * pairs["length"][face] / pairs["distance"][face]
                pa, pb = local_index[a], local_index[b]
                diagonal[pa] += g; diagonal[pb] += g
                pp.append(pa); qq.append(pb); gg.append(g)
            elif a_active != b_active:
                cell = a if a_active else b
                p = local_index[cell]
                face_z = pairs["face_z"][face]
                if pairs["orientation"][face] == 0:
                    continue
                if abs(face_z - state["surface"][cell]) <= abs(face_z - state["base"][cell]):
                    distance = (0.5 * mesh.width[cell] if pairs["orientation"][face] == 0
                                else 0.5 * mesh.height[cell])
                    g = lam[cell] * pairs["length"][face] / distance
                    diagonal[p] += g
                    rhs[p] += g * state["tsurf"][cell]
                else:
                    rhs[p] += state["qbasal"][cell] * pairs["length"][face]

        for cell in np.flatnonzero(active):
            p = local_index[cell]
            descriptor = mesh.cells[cell]
            if descriptor.iz0 == 0:
                g = lam[cell] * mesh.width[cell] / (0.5 * mesh.height[cell])
                diagonal[p] += g
                rhs[p] += g * state["tsurf"][cell]
            if descriptor.iz1 == mesh.nzf:
                rhs[p] += state["qbasal"][cell] * mesh.width[cell]

        pp_array = np.asarray(pp, int); qq_array = np.asarray(qq, int); gg_array = np.asarray(gg, float)
        solution, last_linear_iterations, last_linear_residual = pcg_solve(
            diagonal, pp_array, qq_array, gg_array, rhs, trial[active],
            tolerance=config.linear_tolerance,
            max_iterations=max(800, 6 * int(math.sqrt(len(diagonal)))),
        )
        updated = np.full_like(trial, np.nan); updated[active] = solution
        max_change = float(np.nanmax(np.abs(updated - trial)))
        trial = updated
        if max_change <= config.picard_tolerance_c:
            break
    return trial, {
        "picard_iterations": nonlinear_iteration + 1,
        "picard_change_c": max_change,
        "nonlinear_converged": max_change <= config.picard_tolerance_c,
        "linear_iterations": last_linear_iterations,
        "linear_relative_residual": last_linear_residual,
    }


def mesh_to_regular(mesh: AdaptiveMesh, values: np.ndarray, nx: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, x1, z0, z1 = mesh.bounds
    dx, dz = (x1 - x0) / nx, (z1 - z0) / nz
    x = np.linspace(x0 + 0.5 * dx, x1 - 0.5 * dx, nx)
    z = np.linspace(z0 + 0.5 * dz, z1 - 0.5 * dz, nz)
    xx, zz = np.meshgrid(x, z)
    indices = mesh.locate(xx.ravel(), zz.ravel())
    return x, z, values[indices].reshape(nz, nx)


def run_adaptive(data: BasinDataset, config: AdaptiveConfig) -> dict[str, Any]:
    config.validate()
    ages = data.ages
    mesh = build_mesh(data, ages[0], ages[0], 0.0, config)
    state = state_at(data, ages[0], ages[0], 0.0, mesh.xc)
    grid = rasterize_cells(state, mesh)
    temperature = initialise(mesh, state, grid, config)
    px, pz, preview = mesh_to_regular(mesh, temperature, config.preview_nx, config.preview_nz)
    saved = [preview]
    cell_counts = [len(mesh.cells)]
    diagnostics: list[dict[str, float | int | bool]] = []
    start = time.perf_counter()

    for older_age, younger_age in zip(ages[:-1], ages[1:]):
        span = older_age - younger_age
        steps = max(1, int(math.ceil(span / config.max_dt_ma)))
        previous_fraction = 0.0
        previous_older, previous_younger = older_age, younger_age
        for step in range(1, steps + 1):
            fraction = step / steps
            new_mesh = build_mesh(data, older_age, younger_age, fraction, config)
            new_state = state_at(data, older_age, younger_age, fraction, new_mesh.xc)
            new_grid = rasterize_cells(new_state, new_mesh)
            old_state_on_new_x = state_at(
                data, previous_older, previous_younger, previous_fraction, new_mesh.xc
            )
            transported = material_remap(
                mesh, temperature, grid, old_state_on_new_x, new_mesh, new_state, new_grid
            )
            temperature, diag = adaptive_step(
                transported, new_mesh, new_state, new_grid,
                span / steps * SECONDS_PER_MA, data.properties, config,
            )
            diagnostics.append({
                "age_ma": float(new_state["age"]), "cells": len(new_mesh.cells),
                "active_cells": int(np.count_nonzero(new_grid["active"])), **diag,
            })
            mesh, state, grid = new_mesh, new_state, new_grid
            previous_fraction = fraction
        _, _, preview = mesh_to_regular(mesh, temperature, config.preview_nx, config.preview_nz)
        saved.append(preview)
        cell_counts.append(len(mesh.cells))

    elapsed = time.perf_counter() - start
    final_lith = mesh_to_regular(mesh, grid["lith"].astype(float), config.preview_nx, config.preview_nz)[2]
    final_phi = mesh_to_regular(mesh, grid["phi"], config.preview_nx, config.preview_nz)[2]
    final_preview_state = data.exact_state(ages[-1], px)
    return {
        "x": px, "z": pz, "ages_ma": np.asarray(ages),
        "temperature_c": np.stack(saved), "final_lithology": final_lith,
        "final_porosity": final_phi, "mesh": mesh, "state": state,
        "final_surface_m": final_preview_state["surface"],
        "final_base_m": final_preview_state["base"],
        "grid": grid, "final_temperature_cells": temperature,
        "cell_counts": cell_counts, "diagnostics": diagnostics,
        "elapsed_seconds": elapsed,
    }


def adaptive_run_id(data: BasinDataset, config: AdaptiveConfig) -> str:
    payload = json.dumps({"input": data.fingerprint, "adaptive": asdict(config), "version": 3}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_adaptive(result: dict[str, Any], run_dir: Path, metadata: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    mesh: AdaptiveMesh = result["mesh"]
    state, grid = result["state"], result["grid"]
    np.savez_compressed(
        run_dir / "result.npz", x=result["x"], z=result["z"], ages_ma=result["ages_ma"],
        temperature_c=result["temperature_c"], final_lithology=result["final_lithology"],
        final_porosity=result["final_porosity"], final_surface_m=result["final_surface_m"],
        final_base_m=result["final_base_m"],
    )
    np.savez_compressed(
        run_dir / "final_adaptive_mesh.npz",
        x0=mesh.x0, x1=mesh.x1, z0=mesh.z0, z1=mesh.z1,
        level=mesh.level, temperature_c=result["final_temperature_cells"],
        active=grid["active"], lithology=grid["lith"], porosity=grid["phi"],
    )
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    defaults = AdaptiveConfig()
    parser = argparse.ArgumentParser(description="Adaptive quadtree FVM basin solver")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=Path("adaptive_results"))
    parser.add_argument("--base-nx", type=int, default=defaults.base_nx)
    parser.add_argument("--base-nz", type=int, default=defaults.base_nz)
    parser.add_argument("--max-level", type=int, default=defaults.max_level)
    parser.add_argument("--interface-band", type=float, default=defaults.interface_band)
    parser.add_argument("--max-dt-ma", type=float, default=defaults.max_dt_ma)
    parser.add_argument("--picard-iterations", type=int, default=defaults.picard_iterations,
                        help="maximum Picard iterations per time step")
    parser.add_argument("--picard-tolerance-c", type=float, default=defaults.picard_tolerance_c)
    parser.add_argument("--linear-tolerance", type=float, default=defaults.linear_tolerance)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = AdaptiveConfig(
        base_nx=args.base_nx, base_nz=args.base_nz, max_level=args.max_level,
        interface_band=args.interface_band, max_dt_ma=args.max_dt_ma,
        picard_iterations=args.picard_iterations,
        picard_tolerance_c=args.picard_tolerance_c,
        linear_tolerance=args.linear_tolerance,
    )
    config.validate()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    data = BasinDataset(args.data_dir, args.cache_dir)
    run_id = adaptive_run_id(data, config)
    run_dir = args.cache_dir / run_id
    if (run_dir / "result.npz").is_file() and not args.force:
        print(f"CACHE_HIT {run_id} {run_dir / 'result.npz'}")
        return
    print(f"RUN {run_id}: adaptive base={config.base_nx}x{config.base_nz}, level={config.max_level}")
    result = run_adaptive(data, config)
    final = result["final_temperature_cells"]
    finite = final[np.isfinite(final)]
    mesh: AdaptiveMesh = result["mesh"]
    unconverged_steps = sum(
        not bool(item["nonlinear_converged"]) for item in result["diagnostics"]
    )
    metadata = {
        "run_id": run_id, "status": "complete", "mesh_type": "balanced_quadtree",
        "input_directory": str(data.data_dir), "input_fingerprint": data.fingerprint,
        "solver": asdict(config), "elapsed_seconds": result["elapsed_seconds"],
        "cell_counts_at_saved_ages": result["cell_counts"],
        "final_cells": len(mesh.cells), "final_active_cells": int(np.count_nonzero(result["grid"]["active"])),
        "minimum_dx_m": float(mesh.width.min()), "minimum_dz_m": float(mesh.height.min()),
        "final_temperature_min_c": float(finite.min()), "final_temperature_max_c": float(finite.max()),
        "nonlinear_converged": unconverged_steps == 0,
        "unconverged_time_steps": unconverged_steps,
        "refinement": "layer top/bottom interfaces and lateral lithology transitions (fault/contact zones)",
        "diagnostics": result["diagnostics"],
    }
    save_adaptive(result, run_dir, metadata)
    update_manifest(args.cache_dir, metadata)
    print(f"SAVED {run_dir / 'result.npz'}")
    print(f"CELLS {len(mesh.cells)} ACTIVE {metadata['final_active_cells']} MIN_CELL {metadata['minimum_dx_m']:.3f}x{metadata['minimum_dz_m']:.3f} m")
    print(f"T_FINAL {finite.min():.3f} {finite.max():.3f} C")
    if unconverged_steps:
        print(f"WARNING NONLINEAR_NOT_CONVERGED {unconverged_steps} time steps")


if __name__ == "__main__":
    main()
