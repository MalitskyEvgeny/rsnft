# Верификация быстрого пути evaluate (precompute_geometry + qbasal_by_age):
# форвард через geometry должен быть бит-в-бит равен classic-путю
# (make_param_data + run_forward). Запуск:
#   julia --project=. test_fastpath.jl [data-dir]

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))
include(joinpath(@__DIR__, "observations.jl"))
include(joinpath(@__DIR__, "objective.jl"))

const DATA_DIR = length(ARGS) >= 1 ? ARGS[1] :
                 joinpath(dirname(@__DIR__), "Данные для обратной задачи")

d = load_dataset_derived(DATA_DIR, Workbook())
lay = make_q_layout(d)
cfg = SolverConfig(nx = 30, nz = 60, max_dt_ma = 5.0,
                   picard_iterations = 2, picard_tolerance_c = 1.0e-2)
geometry = precompute_geometry(d, cfg)

θs = [theta_constant(lay, [75.0, 80.0, 85.0, 90.0]),
      theta_constant(lay, [108.0, 120.0, 33.0, 56.0]),
      theta_constant(lay, [40.0, 90.0, 115.0, 70.0])]

for (i, θ) in enumerate(θs)
    pdata = make_param_data(d, lay, θ)
    r_old = run_forward(pdata, cfg)
    qba = qbasal_by_age(lay, θ, geometry.x, d.ages)
    r_new = run_forward(d, cfg; geometry = geometry, qbasal_by_age = qba)
    size(r_old.temperature_c) == size(r_new.temperature_c) || (println("FAIL: size"); exit(1))
    ΔT = maximum(abs.(r_old.temperature_c .- r_new.temperature_c))
    identical = isequal(r_old.temperature_c, r_new.temperature_c)   # NaN == NaN
    length(r_new.diagnostics) == length(r_old.diagnostics) ||
        (println("FAIL: diagnostics count"); exit(1))
    println("θ #", i, ": max |ΔT| = ", ΔT, "  bit-identical = ", identical)
    ΔT == 0.0 || identical ||
        (println("FAIL: classic и быстрый путь различаются"); exit(1))
end

"Ручной пересчёт J по classic-прогону (функция, чтобы избежать soft-scope)."
function classic_J(obj, r_old)
    temp0 = r_old.temperature_c[end, :, :]
    s = 0.0
    for t in obj.t_targets
        v = bilinear_sample(r_old.x, r_old.z, temp0, t.x, t.z)
        w = t.sigma > 0 ? 1.0 / t.sigma^2 : 1.0
        s += isfinite(v) ? w * (v - t.value)^2 : 1.0e6
    end
    for t in obj.ro_targets
        rc = particle_ro(r_old, obj.data, t.x, t.layer, t.eta)
        w = t.sigma > 0 ? 1.0 / t.sigma^2 : 1.0
        s += isfinite(rc) ? w * (rc - t.value)^2 : 1.0e6
    end
    return s
end

# Полная J: objective (быстрый) против ручного пересчёта по classic-прогону
using XLSX: Workbook
ws_T = load_wells_xlsx(joinpath(DATA_DIR, "TemperatureWellData.xlsx"), d; kind = :T)
ws_Ro = load_wells_xlsx(joinpath(DATA_DIR, "VitriniteWellData.xlsx"), d; kind = :Ro)
wells = WellSet(vcat(ws_T.points, ws_Ro.points), vcat(ws_T.excluded, ws_Ro.excluded), String[])
obj = make_objective(d, wells, cfg, lay; reg = 0.0, weight_ro = 1.0)

θ = theta_constant(lay, [108.0, 120.0, 33.0, 56.0])
J_fast, _ = obj(θ)

pdata = make_param_data(d, lay, θ)
r_old = run_forward(pdata, cfg)
J_classic = classic_J(obj, r_old)
rel = abs(J_fast - J_classic) / max(abs(J_classic), 1.0)
println("J_fast = ", J_fast, "  J_classic = ", J_classic, "  rel = ", rel)
rel < 1e-10 || (println("FAIL: J расходится"); exit(1))

# Тайминг
t0 = time(); for _ in 1:5; obj(θ .+ rand(length(θ)) * 0.5); end
t_fast = (time() - t0) / 5
t0 = time()
for _ in 1:5
    pd = make_param_data(d, lay, θ .+ rand(length(θ)) * 0.5)
    run_forward(pd, cfg)
end
t_classic = (time() - t0) / 5
@printf("с/оценку: fast = %.3f, classic = %.3f, ускорение = %.2fx\n",
        t_fast, t_classic, t_classic / t_fast)
println("PASS: test_fastpath")
