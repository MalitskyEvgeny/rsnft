# Юнит-тесты criteria.jl (B4): доли |res|<=kσ, RMSE, per-well статистика,
# l2_error на синтетических прогонах.
# Запуск: julia --project=. test_criteria.jl

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))
include(joinpath(@__DIR__, "observations.jl"))
include(joinpath(@__DIR__, "objective.jl"))
include(joinpath(@__DIR__, "criteria.jl"))

using Printf

failures = 0
function check(name, ok)
    global failures
    println((ok ? "PASS " : "FAIL "), name)
    ok || (failures += 1)
end

# --- 1. frac_within / rmse ---
check("frac_within: 2 of 4 within 1σ", frac_within([(0.5, 1.0), (-0.9, 1.0), (1.5, 1.0), (2.0, 1.0)], 1) == 0.5)
check("frac_within: all within 2σ", frac_within([(0.5, 1.0), (-0.9, 1.0), (1.5, 1.0), (2.0, 1.0)], 2) == 1.0)
check("frac_within: empty -> NaN", isnan(frac_within(Tuple{Float64,Float64}[], 1)))
check("rmse", abs(rmse([1.0, -1.0]) - 1.0) < 1e-12)

# --- 2. l2_error: трапеции на аналитике ---
base = (x = [0.0, 1.0], z = [0.0, 1.0], ages_ma = [200.0, 100.0, 0.0])
mk(T) = (x = base.x, z = base.z, ages_ma = base.ages_ma,
         temperature_c = T, elapsed_seconds = 0.0, dx = 1.0, diagnostics = [])
T_ref = reshape([10.0, 10.0, 20.0, 20.0, 30.0, 30.0], 3, 2, 1)  # (nt, nz, nx)
r_ref = mk(T_ref)
T_eq = copy(T_ref)
check("l2_error: identical runs -> 0", l2_error(mk(T_eq), r_ref) == 0.0)
T_off = copy(T_ref); T_off[:, :, 1] .+= 2.0     # ΔT = 2 на всех срезах
check("l2_error: constant offset -> 2", abs(l2_error(mk(T_off), r_ref) - 2.0) < 1e-12)
T_lin = copy(T_ref)
T_lin[1, :, 1] .+= 4.0; T_lin[2, :, 1] .+= 2.0; T_lin[3, :, 1] .+= 0.0  # ΔT 4 → 2 → 0
# e2: [16, 4, 0] (на срезе все ячейки конечны) → ∫e2 dt / span = средневзв. трапециями
# ages 200..100..0: трапеции (16+4)/2*100 + (4+0)/2*100 = 1200, span 200 → sqrt(6)
check("l2_error: linear decay -> sqrt(6)", abs(l2_error(mk(T_lin), r_ref) - sqrt(6.0)) < 1e-12)

# --- 3. criteria_report на реальном прогоне (50x60, θ = (75,80,85,90)) ---
const DATA = normpath(joinpath(@__DIR__, "..", "Данные для обратной задачи"))
d = load_dataset_derived(DATA, Workbook())
lay = make_q_layout(d)
wells = let
    wsT = load_wells_xlsx(joinpath(DATA, "TemperatureWellData.xlsx"), d; kind = :T)
    wsR = load_wells_xlsx(joinpath(DATA, "VitriniteWellData.xlsx"), d; kind = :Ro)
    WellSet(vcat(wsT.points, wsR.points), vcat(wsT.excluded, wsR.excluded),
            collect(union(wsT.excluded_wells, wsR.excluded_wells)))
end
cfg = SolverConfig(nx = 50, nz = 60)
pd = make_param_data(d, lay, theta_constant(lay, [75.0, 80.0, 85.0, 90.0]))
res = run_forward(pd, cfg)
rep = criteria_report(res, d, wells)
check("criteria: 5 wells", length(rep.wells) == 5)
check("criteria: overall rmse_T finite", isfinite(rep.rmse_T))
check("criteria: frac sums <= 1", rep.frac_1s_T <= rep.frac_2s_T <= 1.0)
println(@printf("  overall: rmse_T=%.2f rmse_Ro=%.3f, in 1σ: T=%.1f%% Ro=%.1f%%, in 2σ: T=%.1f%% Ro=%.1f%%",
    rep.rmse_T, rep.rmse_Ro, 100 * rep.frac_1s_T, 100 * rep.frac_1s_Ro,
    100 * rep.frac_2s_T, 100 * rep.frac_2s_Ro))
println(@printf("  per-well rmse_T: %s", join([@sprintf("%s=%.1f", w.well, w.rmse_T) for w in rep.wells], ", ")))

println()
if failures == 0
    println("ALL PASS")
else
    println(@printf("%d FAILURES", failures))
    exit(1)
end
