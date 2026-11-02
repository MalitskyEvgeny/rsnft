# Inverse problem: basal heat flow q(x,t) from well data

Reconstruction of the unknown basal heat flow **q(x,t)** (piecewise in 4
epochs × 2–3 x-nodes) of the 2-D basin forward model (`basin_fvm.jl`) from
well observations: temperature profiles (σ_T) and vitrinite reflectance
(EASY%Ro, σ_Ro).

```
parameterized_data.jl   ThetaLayout + make_param_data: θ -> synthetic heat Workbook
ro_kinetics.jl          EASY%Ro post-processor over a forward run
observations.jl         WellPoint/WellSet loaders (organizers' XLSX, twin JSON) + load_wells_multi
objective.jl            J(θ): forward run on the working grid -> sampled T/Ro residuals + cache
criteria.jl             per-well RMSE, fractions within σ/2σ (criterion 18), E^L2
invert.jl               DE (BlackBoxOptim) -> multilevel NM polish -> ensemble
run_inverse.jl          CLI entry point  ->  results/<run>/solution.json + result.npz
make_twin.jl            twin observations from a known θ_true
report_inverse.jl       self-contained HTML report
```

## 1. Working grid and budget

The working grid was fixed by timing (step 0): **100×120**, forward ≈ 8.4 s
(200×240 ≈ 76 s, 300×360 ≈ 322 s). One J evaluation ≈ 1.2 s per worker.
Default budget: DE 4000–8000 evals, ensemble 4 starts, 4 workers ⇒ a full
run fits in ≈ 1–2 h. Use `--grid 50,60` for pipeline debugging (~1.2 s/run).

## 2. Twin verification (S1/S2 — run before touching real data)

```bash
# exact twin (sigma=0): theta_hat must reproduce theta_true, J -> 0
julia --project=. make_twin.jl --data-dir "../Данные для обратной задачи" \
      --wells "../Данные для обратной задачи/TemperatureWellData.xlsx" \
      --grid 50,60 --q 75,80,85,90 --sigma-T 0 --sigma-Ro 0 \
      --seed 42 --out ../data/twin_exact.json

# noisy twin (S2)
julia --project=. make_twin.jl ... --sigma-T 2 --sigma-Ro 0.05 \
      --seed 42 --out ../data/twin_noisy.json

# inversion on the twin (same grid, theta_true hidden from the optimizer)
julia --project=. run_inverse.jl --data-dir "../Данные для обратной задачи" \
      --wells ../data/twin_noisy.json --grid 50,60 \
      --ensemble 4 --workers 4 --de-budget 4000 --seed 1 \
      --out ../results/inverse_twin
julia --project=. report_inverse.jl ../results/inverse_twin
```

## 3. Inversion on the real observations

```bash
julia --project=. run_inverse.jl \
      --data-dir "../Данные для обратной задачи" \
      --wells "../Данные для обратной задачи/TemperatureWellData.xlsx;../Данные для обратной задачи/VitriniteWellData.xlsx" \
      --grid 100,120 --ensemble 4 --workers 4 \
      --de-budget 8000 --sigma-T 2 --sigma-Ro 0.05 --seed 1 \
      --out ../results/inverse_real
julia --project=. report_inverse.jl ../results/inverse_real
```

Key options: `--epochs "200-100,100-35,35-10,10-0"`, `--n-nodes 2|3`
(dim = epochs × nodes = 8|12), `--mid-x` (middle q-node), `--reg`
(spatial/epoch smoothness weight), `--weight-ro` (Ro contribution weight),
`--no-multilevel` (fit all epochs at once instead of young→old).

Outputs per run directory: `solution.json` (θ̂ ensemble, criteria, settings),
`result.npz` (fields of the best θ̂), `report.html`, `cache_main/` (J cache —
safe to delete; speeds up restarts of the same run).

## 4. How to interpret the results

1. **Criterion 18 (acceptance)**: the fractions "% of points within σ / 2σ".
   A run is acceptable when RMSE ≤ 2σ and ≥ 95 % of points fall within 2σ.
   If RMSE_T ≫ σ_T: the model does not explain temperature — look at where
   the misfit sits (upper profile → surface temperature/properties, lower → q).
2. **Ensemble θ̂ table (mean/std/min/max)**: std ≪ box width ⇒ the parameter
   is identified; std ≈ half the box ⇒ non-uniqueness. Ro integrates the full
   thermal history, so it struggles to separate "short and hot" from "long
   and moderate" — expect the old epochs to be the least constrained.
3. **Per-well profiles**: the western wells (A, B) sit on the thin wedge where
   Layer_1 does not cover the section; their T misfit weights the youngest
   strata. Systematic +Ro bias at depth (the constant-q prior) is exactly the
   inversion signal.
4. **J_reg** grows if the optimizer wants a rough q(x) — a large jump from the
   smooth prior means the data demand lateral variability; check the
   x-node columns of θ̂ for that.
5. **Grid convergence**: acceptance on the working grid; re-run the best θ̂
   on a 1.5× finer grid as a control (a change of grid is a change of model —
   report both).

## 5. Pitfalls

- The XLSX well sheets use depth positive **downwards**; the loaders already
  convert. Well A (x=400) has 2 points outside the sediment geometry — they
  are dropped at load time and counted in `solution.json`.
- The J cache is keyed by θ (SHA-256 of the rounded vector): identical θ from
  different starts are free. Cache format mismatch is handled on load.
- First run after `addprocs` pays per-worker JIT (~20–30 s); the budget
  estimates above refer to steady state.
- With `--sigma-T 0` (exact twin) residuals are unweighted; criterion 18
  fractions are reported as null.
