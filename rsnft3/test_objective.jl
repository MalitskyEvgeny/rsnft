# Юнит-тесты objective.jl (B3): билинейная выборка, кэш, регуляризация,
# исключение точек в неактивных ячейках, конечность J.
# Запуск: julia --project=. test_objective.jl

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))
include(joinpath(@__DIR__, "observations.jl"))
include(joinpath(@__DIR__, "objective.jl"))

using Printf

failures = 0
function check(name, ok)
    global failures
    println((ok ? "PASS " : "FAIL "), name)
    ok || (failures += 1)
end

# --- 1. Билинейная выборка на аналитическом поле T(x,z) = 2x + 0.1z ---
xg = collect(range(0.0, 100.0; length = 11))
zg = collect(range(0.0, 1000.0; length = 21))
field = [2.0 * x + 0.1 * z for z in zg, x in xg]
v = bilinear_sample(xg, zg, field, 25.0, 250.0)
check("bilinear: linear field reproduced exactly", abs(v - (2 * 25 + 0.1 * 250)) < 1e-9)
v2 = bilinear_sample(xg, zg, field, 27.5, 275.0)
check("bilinear: interpolation between nodes", abs(v2 - (2 * 27.5 + 0.1 * 275)) < 1e-9)
field_nan = copy(field); field_nan[6, 3] = NaN   # ячейка стенсила (25, 250)
check("bilinear: NaN stencil -> NaN", isnan(bilinear_sample(xg, zg, field_nan, 25.0, 250.0)))

# --- 2. make_objective на реальных данных (сетка 50x60) ---
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
cachedir = joinpath(tempdir(), "b3_cache_test")
rm(cachedir; force = true, recursive = true)
Jfun = make_objective(d, wells, cfg, lay; reg = 0.1, weight_ro = 1.0, cache_dir = cachedir)

check("T targets: 487 (499 − 12 в неактивных ячейках)", n_targets(Jfun, :T) == 487)
check("dropped T points == 12", Jfun.n_dropped_T == 12)
check("Ro targets: 49", n_targets(Jfun, :Ro) == 49)

θt = theta_constant(lay, [75.0, 80.0, 85.0, 90.0])
J1, info1 = Jfun(θt)
check("J finite and positive", isfinite(J1) && J1 > 0)
check("info breakdown consistent", abs(info1.J_T + Jfun.weight_ro * info1.J_Ro + info1.J_reg - J1) < 1e-6 * J1)
J2, info2 = Jfun(θt)
check("cache: second call cached", info2.cached && J2 == J1)
check("cache JSON dumped", isfile(joinpath(cachedir, "j_cache.json")))
check("evals counted once", info2.evals == 1)

# регуляризация: грубый θ (пила по эпохам) — больший J_reg
θrough = copy(θt)
θrough[1:3] .= 30.0
θrough[10:12] .= 120.0
_, ir = Jfun(θrough)
check("rough theta -> larger J_reg", ir.J_reg > info1.J_reg)

# симметрия регуляризации: константный θ_true → d_node = 0
_, ism = Jfun(theta_constant(lay, [75.0, 75.0, 75.0, 75.0]))
dn, de = theta_diffs(lay, theta_constant(lay, [75.0, 75.0, 75.0, 75.0]))
check("constant theta: d_node=0, J_reg = 0.1*sum(d_epoch^2)",
      all(dn .== 0.0) && abs(ism.J_reg - 0.1 * sum(abs2, de)) < 1e-9)

println()
if failures == 0
    println("ALL PASS")
else
    println(@printf("%d FAILURES", failures))
    exit(1)
end
