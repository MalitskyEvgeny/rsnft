"""2-D basin heat-flow forward solver using cell-centred finite volumes.

The solver reads the tournament XLSX files directly.  Geological time is
traversed from the oldest configuration to the present.  Between two dated
configurations the geometry is linearly interpolated and the temperature is
conservatively transported with the layers by a material-coordinate remap.
Diffusion and radiogenic heating are then advanced with backward Euler and
Picard iterations for temperature-dependent properties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SECONDS_PER_MA = 1.0e6 * 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class SolverConfig:
    nx: int = 60
    nz: int = 120
    max_dt_ma: float = 2.5
    picard_iterations: int = 3
    picard_tolerance_c: float = 1.0e-3
    fluid_lambda: float = 0.60
    fluid_density: float = 1000.0
    fluid_heat_capacity: float = 4180.0
    initial_lambda: float = 2.5

    def validate(self) -> None:
        if self.nx < 3 or self.nz < 3:
            raise ValueError("nx and nz must be at least 3")
        if self.max_dt_ma <= 0:
            raise ValueError("max_dt_ma must be positive")
        if self.picard_iterations < 1:
            raise ValueError("picard_iterations must be positive")


def waples_heat_capacity(temperature_k: np.ndarray, c20: np.ndarray) -> np.ndarray:
    tc = temperature_k - 273.15
    factor = 0.953 + 2.29e-3 * tc - 2.835e-6 * tc**2 + 1.191e-9 * tc**3
    return c20 * factor


def sekiguchi_conductivity(temperature_k: np.ndarray, lambda20: np.ndarray) -> np.ndarray:
    temperature_k = np.maximum(temperature_k, 150.0)
    return 1.84 + 358.0 * (1.0227 * lambda20 - 1.882) * (
        1.0 / temperature_k - 0.00068
    )


def harmonic(a: float, b: float) -> float:
    return 2.0 * a * b / max(a + b, 1.0e-30)


def pcg_solve(
    diagonal: np.ndarray,
    pair_p: np.ndarray,
    pair_q: np.ndarray,
    conductance: np.ndarray,
    rhs: np.ndarray,
    initial: np.ndarray,
    tolerance: float = 1.0e-9,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, int, float]:
    """Jacobi-preconditioned conjugate gradients for an FVM graph matrix."""

    def matvec(vector: np.ndarray) -> np.ndarray:
        result = diagonal * vector
        np.add.at(result, pair_p, -conductance * vector[pair_q])
        np.add.at(result, pair_q, -conductance * vector[pair_p])
        return result

    solution = initial.copy()
    residual = rhs - matvec(solution)
    scale = max(float(np.linalg.norm(rhs)), 1.0)
    relative = float(np.linalg.norm(residual) / scale)
    if relative <= tolerance:
        return solution, 0, relative
    z = residual / diagonal
    direction = z.copy()
    rz_old = float(np.dot(residual, z))
    for iteration in range(1, max_iterations + 1):
        applied = matvec(direction)
        denominator = float(np.dot(direction, applied))
        if denominator <= 0.0:
            raise RuntimeError("PCG lost positive definiteness")
        alpha = rz_old / denominator
        solution += alpha * direction
        residual -= alpha * applied
        relative = float(np.linalg.norm(residual) / scale)
        if relative <= tolerance:
            return solution, iteration, relative
        z = residual / diagonal
        rz_new = float(np.dot(residual, z))
        direction = z + (rz_new / rz_old) * direction
        rz_old = rz_new
    raise RuntimeError(f"PCG did not converge in {max_iterations} iterations (residual={relative:.3e})")


def _value_column(frame: pd.DataFrame, prefix: str = "Aver") -> str:
    for col in frame.columns:
        if str(col).startswith(prefix):
            return str(col)
    raise KeyError(f"No column starting with {prefix!r}")


def _series_from_frame(frame: pd.DataFrame) -> dict[float, dict[str, np.ndarray]]:
    value_col = _value_column(frame)
    result: dict[float, dict[str, np.ndarray]] = {}
    for age, group in frame.groupby("Time_Ma"):
        group = group.sort_values("X").drop_duplicates("X", keep="last")
        result[float(age)] = {
            "x": group["X"].to_numpy(float),
            "top": group["Z_Top [m]"].to_numpy(float),
            "bot": group["Z_Bot [m]"].to_numpy(float),
            "value": group[value_col].to_numpy(float),
        }
    return result


class BasinDataset:
    REQUIRED = (
        "Depth.xlsx",
        "Lithology.xlsx",
        "Porosity.xlsx",
        "Heat_Flow_Basal.xlsx",
        "Tsurf.xlsx",
        "lithotype_properties.xlsx",
    )

    def __init__(self, data_dir: Path, cache_dir: Path):
        self.data_dir = data_dir.resolve()
        missing = [name for name in self.REQUIRED if not (self.data_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing input files: {', '.join(missing)}")
        self.fingerprint = self._fingerprint()
        input_cache = cache_dir / "_input_cache"
        input_cache.mkdir(parents=True, exist_ok=True)
        cache_file = input_cache / f"{self.fingerprint}.pkl"
        if cache_file.is_file():
            with cache_file.open("rb") as stream:
                payload = pickle.load(stream)
        else:
            payload = self._read_all()
            with cache_file.open("wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.__dict__.update(payload)

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name in self.REQUIRED:
            path = self.data_dir / name
            stat = path.stat()
            digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return digest.hexdigest()[:16]

    @staticmethod
    def _read_workbook(path: Path) -> dict[str, dict[float, dict[str, np.ndarray]]]:
        book = pd.ExcelFile(path)
        return {sheet: _series_from_frame(pd.read_excel(book, sheet_name=sheet)) for sheet in book.sheet_names}

    def _read_all(self) -> dict[str, Any]:
        depth = self._read_workbook(self.data_dir / "Depth.xlsx")
        lith = self._read_workbook(self.data_dir / "Lithology.xlsx")
        porosity = self._read_workbook(self.data_dir / "Porosity.xlsx")
        heat = self._read_workbook(self.data_dir / "Heat_Flow_Basal.xlsx")
        tsurf = self._read_workbook(self.data_dir / "Tsurf.xlsx")

        prop = pd.read_excel(self.data_dir / "lithotype_properties.xlsx")
        properties: dict[int, dict[str, float | str]] = {}
        for _, row in prop.iterrows():
            properties[int(row["ID"])] = {
                "name": str(row["Name"]),
                "rho_s": float(row["ρs [kg/m³]"]),
                "lambda20": float(row["λ20 [W/(m·K)]"]),
                "c20": float(row["c20 [J/(kg·K)]"]),
                "As": float(row["As [µW/m³]"]) * 1.0e-6,
            }

        ages = sorted({age for layer in depth.values() for age in layer}, reverse=True)
        x_min = min(item["x"].min() for layer in depth.values() for item in layer.values())
        x_max = max(item["x"].max() for layer in depth.values() for item in layer.values())
        z_min = min(item["top"].min() for layer in depth.values() for item in layer.values())
        z_max = max(item["bot"].max() for layer in depth.values() for item in layer.values())
        return {
            "depth": depth,
            "lith": lith,
            "porosity": porosity,
            "heat": heat,
            "tsurf_data": tsurf,
            "properties": properties,
            "ages": ages,
            "bounds": (float(x_min), float(x_max), float(z_min), float(z_max)),
        }

    @staticmethod
    def _interp_record(record: dict[str, np.ndarray], x: np.ndarray, key: str) -> np.ndarray:
        return np.interp(x, record["x"], record[key], left=record[key][0], right=record[key][-1])

    def exact_state(self, age: float, x: np.ndarray) -> dict[str, Any]:
        layers: dict[str, dict[str, np.ndarray]] = {}
        for name, depth_series in self.depth.items():
            if age not in depth_series:
                continue
            d = depth_series[age]
            lith = self.lith[name][age]
            phi = self.porosity[name][age]
            layers[name] = {
                "top": self._interp_record(d, x, "top"),
                "bot": self._interp_record(d, x, "bot"),
                "lith": np.floor(self._interp_record(lith, x, "value") + 1.0e-8).astype(int),
                "phi": np.clip(self._interp_record(phi, x, "value") / 100.0, 0.0, 0.95),
            }
        if not layers:
            raise ValueError(f"No geometry at age {age} Ma")
        tops = np.vstack([v["top"] for v in layers.values()])
        bots = np.vstack([v["bot"] for v in layers.values()])
        return {
            "age": float(age),
            "layers": layers,
            "surface": np.min(tops, axis=0),
            "base": np.max(bots, axis=0),
            "tsurf": self.boundary_value(self.tsurf_data, age, x),
            "qbasal": self.boundary_value(self.heat, age, x) * 1.0e-3,
        }

    def boundary_value(
        self, workbook: dict[str, dict[float, dict[str, np.ndarray]]], age: float, x: np.ndarray
    ) -> np.ndarray:
        candidates = []
        for layer in workbook.values():
            if age in layer:
                rec = layer[age]
                candidates.append(self._interp_record(rec, x, "value"))
        if not candidates:
            raise ValueError(f"No boundary data at age {age} Ma")
        return np.nanmedian(np.vstack(candidates), axis=0)


def interpolate_state(old: dict[str, Any], young: dict[str, Any], fraction: float) -> dict[str, Any]:
    f = float(np.clip(fraction, 0.0, 1.0))
    layers: dict[str, dict[str, np.ndarray]] = {}
    names = set(old["layers"]) | set(young["layers"])
    for name in names:
        a = old["layers"].get(name)
        b = young["layers"].get(name)
        if a is None and b is not None:
            a = {"top": old["surface"], "bot": old["surface"], "lith": b["lith"], "phi": b["phi"]}
        elif b is None and a is not None:
            b = {"top": young["surface"], "bot": young["surface"], "lith": a["lith"], "phi": a["phi"]}
        assert a is not None and b is not None
        layers[name] = {
            "top": (1.0 - f) * a["top"] + f * b["top"],
            "bot": (1.0 - f) * a["bot"] + f * b["bot"],
            "lith": np.where(f < 0.5, a["lith"], b["lith"]).astype(int),
            "phi": (1.0 - f) * a["phi"] + f * b["phi"],
        }
    return {
        "age": (1.0 - f) * old["age"] + f * young["age"],
        "layers": layers,
        "surface": (1.0 - f) * old["surface"] + f * young["surface"],
        "base": (1.0 - f) * old["base"] + f * young["base"],
        "tsurf": (1.0 - f) * old["tsurf"] + f * young["tsurf"],
        "qbasal": (1.0 - f) * old["qbasal"] + f * young["qbasal"],
    }


def rasterize(state: dict[str, Any], z: np.ndarray) -> dict[str, np.ndarray]:
    nz, nx = len(z), len(state["surface"])
    layer_id = np.full((nz, nx), -1, dtype=int)
    lith = np.full((nz, nx), -1, dtype=int)
    phi = np.full((nz, nx), np.nan)
    names = sorted(state["layers"])
    for layer_number, name in enumerate(names):
        item = state["layers"][name]
        lo = np.minimum(item["top"], item["bot"])[None, :]
        hi = np.maximum(item["top"], item["bot"])[None, :]
        mask = (z[:, None] >= lo) & (z[:, None] <= hi) & ((hi - lo) > 1.0e-8)
        layer_id[mask] = layer_number
        lith[mask] = np.broadcast_to(item["lith"], (nz, nx))[mask]
        phi[mask] = np.broadcast_to(item["phi"], (nz, nx))[mask]
    return {"active": layer_id >= 0, "layer_id": layer_id, "lith": lith, "phi": phi, "names": np.array(names)}


def material_remap(
    temperature: np.ndarray,
    old_state: dict[str, Any],
    new_state: dict[str, Any],
    z: np.ndarray,
) -> np.ndarray:
    old_grid = rasterize(old_state, z)
    new_grid = rasterize(new_state, z)
    out = np.full_like(temperature, np.nan)
    for i in range(temperature.shape[1]):
        valid_column = old_grid["active"][:, i] & np.isfinite(temperature[:, i])
        if not np.any(valid_column):
            out[new_grid["active"][:, i], i] = new_state["tsurf"][i]
            continue
        z_valid = z[valid_column]
        t_valid = temperature[valid_column, i]
        for name in new_state["layers"]:
            new_layer = new_state["layers"][name]
            lo_new, hi_new = sorted((new_layer["top"][i], new_layer["bot"][i]))
            cells = (z >= lo_new) & (z <= hi_new) & new_grid["active"][:, i]
            if not np.any(cells):
                continue
            old_layer = old_state["layers"].get(name)
            if old_layer is None or abs(old_layer["bot"][i] - old_layer["top"][i]) < 1.0e-8:
                out[cells, i] = new_state["tsurf"][i]
                continue
            thickness_new = max(hi_new - lo_new, 1.0e-12)
            eta = np.clip((z[cells] - lo_new) / thickness_new, 0.0, 1.0)
            lo_old, hi_old = sorted((old_layer["top"][i], old_layer["bot"][i]))
            material_z = lo_old + eta * (hi_old - lo_old)
            out[cells, i] = np.interp(material_z, z_valid, t_valid)
    out[new_grid["active"] & ~np.isfinite(out)] = np.broadcast_to(
        new_state["tsurf"], out.shape
    )[new_grid["active"] & ~np.isfinite(out)]
    return out


def thermal_properties(
    temperature_c: np.ndarray,
    grid: dict[str, np.ndarray],
    lith_properties: dict[int, dict[str, float | str]],
    config: SolverConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lam = np.full_like(temperature_c, np.nan)
    rho_c = np.full_like(temperature_c, np.nan)
    source = np.full_like(temperature_c, np.nan)
    tk = temperature_c + 273.15
    for code in np.unique(grid["lith"][grid["active"]]):
        if int(code) not in lith_properties:
            raise KeyError(f"Unknown lithotype code {code}; available: {sorted(lith_properties)}")
        prop = lith_properties[int(code)]
        mask = grid["active"] & (grid["lith"] == code)
        phi = grid["phi"][mask]
        lam_s = sekiguchi_conductivity(tk[mask], float(prop["lambda20"]))
        c_s = waples_heat_capacity(tk[mask], float(prop["c20"]))
        lam[mask] = np.power(lam_s, 1.0 - phi) * config.fluid_lambda**phi
        rho_c[mask] = (
            (1.0 - phi) * float(prop["rho_s"]) * c_s
            + phi * config.fluid_density * config.fluid_heat_capacity
        )
        source[mask] = (1.0 - phi) * float(prop["As"])
    return lam, rho_c, source


def implicit_fvm_step(
    transported_c: np.ndarray,
    state: dict[str, Any],
    z: np.ndarray,
    dx: float,
    dz: float,
    dt_seconds: float,
    lith_properties: dict[int, dict[str, float | str]],
    config: SolverConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    grid = rasterize(state, z)
    active = grid["active"]
    indices = np.full(active.shape, -1, dtype=int)
    indices[active] = np.arange(np.count_nonzero(active))
    trial = transported_c.copy()
    max_change = math.inf
    linear_iterations = 0
    linear_residual = math.inf

    for iteration in range(config.picard_iterations):
        lam, rho_c, source = thermal_properties(trial, grid, lith_properties, config)
        count = np.count_nonzero(active)
        volume = dx * dz
        diagonal = rho_c[active] * volume / dt_seconds
        rhs = diagonal * transported_c[active] + source[active] * volume
        pair_p: list[int] = []
        pair_q: list[int] = []
        pair_g: list[float] = []
        for j, i in zip(*np.nonzero(active)):
            p = indices[j, i]

            # Each internal face is inserted once, from its west/north cell.
            for dj, di, area, distance in ((0, 1, dz, dx), (1, 0, dx, dz)):
                jj, ii = j + dj, i + di
                inside = 0 <= jj < active.shape[0] and 0 <= ii < active.shape[1]
                if inside and active[jj, ii]:
                    conductance = harmonic(lam[j, i], lam[jj, ii]) * area / distance
                    q = indices[jj, ii]
                    diagonal[p] += conductance
                    diagonal[q] += conductance
                    pair_p.append(p); pair_q.append(q); pair_g.append(conductance)

            # Physical top/bottom boundaries on the stair-step domain.
            for dj in (-1, 1):
                jj = j + dj
                inside = 0 <= jj < active.shape[0]
                if not inside or not active[jj, i]:
                    face_z = z[j] + 0.5 * dj * dz
                    surface_distance = abs(face_z - state["surface"][i])
                    base_distance = abs(face_z - state["base"][i])
                    if surface_distance <= base_distance:
                        conductance = lam[j, i] * dx / (0.5 * dz)
                        diagonal[p] += conductance
                        rhs[p] += conductance * state["tsurf"][i]
                    else:
                        rhs[p] += state["qbasal"][i] * dx

        pair_p_array = np.asarray(pair_p, dtype=int)
        pair_q_array = np.asarray(pair_q, dtype=int)
        pair_g_array = np.asarray(pair_g, dtype=float)
        solution, linear_iterations, linear_residual = pcg_solve(
            diagonal, pair_p_array, pair_q_array, pair_g_array, rhs,
            trial[active], tolerance=1.0e-8, max_iterations=max(500, 4 * config.nx),
        )
        updated = np.full_like(trial, np.nan)
        updated[active] = solution
        max_change = float(np.nanmax(np.abs(updated - trial)))
        trial = updated
        if max_change <= config.picard_tolerance_c:
            break
    return trial, {
        "picard_iterations": iteration + 1,
        "picard_change_c": max_change,
        "linear_iterations": linear_iterations,
        "linear_relative_residual": linear_residual,
    }


def initialise_temperature(
    state: dict[str, Any], z: np.ndarray, config: SolverConfig
) -> np.ndarray:
    grid = rasterize(state, z)
    depth = np.maximum(z[:, None] - state["surface"][None, :], 0.0)
    temperature = state["tsurf"][None, :] + state["qbasal"][None, :] * depth / config.initial_lambda
    temperature[~grid["active"]] = np.nan
    return temperature


def run_forward(data: BasinDataset, config: SolverConfig) -> dict[str, Any]:
    config.validate()
    x0, x1, z0, z1 = data.bounds
    dx = (x1 - x0) / config.nx
    dz = (z1 - z0) / config.nz
    x = np.linspace(x0 + 0.5 * dx, x1 - 0.5 * dx, config.nx)
    z = np.linspace(z0 + 0.5 * dz, z1 - 0.5 * dz, config.nz)
    exact = {age: data.exact_state(age, x) for age in data.ages}
    temperature = initialise_temperature(exact[data.ages[0]], z, config)
    saved_temperature = [temperature.copy()]
    diagnostics: list[dict[str, float]] = []

    start = time.perf_counter()
    for age_old, age_young in zip(data.ages[:-1], data.ages[1:]):
        span = age_old - age_young
        n_substeps = max(1, int(math.ceil(span / config.max_dt_ma)))
        previous_state = exact[age_old]
        for substep in range(1, n_substeps + 1):
            fraction = substep / n_substeps
            state = interpolate_state(exact[age_old], exact[age_young], fraction)
            transported = material_remap(temperature, previous_state, state, z)
            dt_seconds = (span / n_substeps) * SECONDS_PER_MA
            temperature, diag = implicit_fvm_step(
                transported, state, z, dx, dz, dt_seconds, data.properties, config
            )
            diagnostics.append({"age_ma": state["age"], **diag})
            previous_state = state
        saved_temperature.append(temperature.copy())

    final_state = exact[data.ages[-1]]
    final_grid = rasterize(final_state, z)
    elapsed = time.perf_counter() - start
    return {
        "x": x,
        "z": z,
        "ages_ma": np.asarray(data.ages),
        "temperature_c": np.stack(saved_temperature),
        "final_lithology": final_grid["lith"],
        "final_porosity": final_grid["phi"],
        "final_surface_m": final_state["surface"],
        "final_base_m": final_state["base"],
        "diagnostics": diagnostics,
        "elapsed_seconds": elapsed,
        "dx": dx,
        "dz": dz,
    }


def run_id_for(data: BasinDataset, config: SolverConfig) -> str:
    payload = json.dumps(
        {"input": data.fingerprint, "solver": asdict(config), "version": 1},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_result(result: dict[str, Any], run_dir: Path, metadata: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "result.npz",
        x=result["x"], z=result["z"], ages_ma=result["ages_ma"],
        temperature_c=result["temperature_c"],
        final_lithology=result["final_lithology"],
        final_porosity=result["final_porosity"],
        final_surface_m=result["final_surface_m"], final_base_m=result["final_base_m"],
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_manifest(cache_dir: Path, metadata: dict[str, Any]) -> None:
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"runs": {}}
    manifest["runs"][metadata["run_id"]] = metadata
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="2-D basin finite-volume forward solver")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("results"))
    parser.add_argument("--nx", type=int, default=60)
    parser.add_argument("--nz", type=int, default=120)
    parser.add_argument("--max-dt-ma", type=float, default=2.5)
    parser.add_argument("--picard-iterations", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    config = SolverConfig(
        nx=args.nx, nz=args.nz, max_dt_ma=args.max_dt_ma,
        picard_iterations=args.picard_iterations,
    )
    data = BasinDataset(args.data_dir, args.cache_dir)
    run_id = run_id_for(data, config)
    run_dir = args.cache_dir / run_id
    if (run_dir / "result.npz").is_file() and not args.force:
        print(f"CACHE_HIT {run_id} {run_dir / 'result.npz'}")
        return

    print(f"RUN {run_id}: grid={config.nx}x{config.nz}, dt<={config.max_dt_ma} Ma")
    result = run_forward(data, config)
    finite = result["temperature_c"][-1][np.isfinite(result["temperature_c"][-1])]
    metadata = {
        "run_id": run_id,
        "status": "complete",
        "input_directory": str(data.data_dir),
        "input_fingerprint": data.fingerprint,
        "solver": asdict(config),
        "elapsed_seconds": result["elapsed_seconds"],
        "dx_m": result["dx"], "dz_m": result["dz"],
        "final_temperature_min_c": float(finite.min()),
        "final_temperature_max_c": float(finite.max()),
        "assumptions": [
            "Initial temperature is a linear conductive profile based on the oldest surface temperature and basal heat flow.",
            "Fluid properties are constant and configurable.",
            "Layer motion is represented by material-coordinate remapping between interpolated geometries.",
            "EASY%Ro is not computed because kinetic constants are absent from the supplied files.",
        ],
        "diagnostics": result["diagnostics"],
    }
    save_result(result, run_dir, metadata)
    update_manifest(args.cache_dir, metadata)
    print(f"SAVED {run_dir / 'result.npz'}")
    print(f"T_FINAL {finite.min():.3f} {finite.max():.3f} C")


if __name__ == "__main__":
    main()
