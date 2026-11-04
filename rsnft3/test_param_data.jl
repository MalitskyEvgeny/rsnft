# Юнит-тесты parameterized_data.jl (0.0/B1/B2): инжекция q, тождественность
# синтетического Workbook, ThetaLayout, ступенчатость по эпохам.
# Запуск: julia --project=. test_param_data.jl

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "observations.jl"))

using Printf, Statistics

failures = 0
function check(name, ok)
    global failures
    println((ok ? "PASS " : "FAIL "), name)
    ok || (failures += 1)
end

const D1 = normpath(joinpath(@__DIR__, "..", "Данные_для_прямой_задачи", "data1"))
cfg = SolverConfig(nx = 30, nz = 60)

# --- 1. Тождественность инжекции: синтетический Workbook (q(age) = медиана по
#        слоям файла, константа по x) воспроизводит прогон data1 до 1e-10 ---
base = read_all(D1)
r_ref = run_forward(BasinDataset(D1, "ref", base.depth, base.lith, base.porosity,
                                 base.heat, base.tsurf_data, base.properties, base.ages, base.bounds), cfg)
xs_syn = collect(range(base.bounds[1], base.bounds[2]; length = 5))
d = Dict{Float64,Record}()
for age in base.ages
    qs = Float64[]
    for (_, recs) in base.heat
        haskey(recs, age) || continue
        v = filter(isfinite, recs[age].value)
        isempty(v) || push!(qs, median(v))
    end
    q = isempty(qs) ? NaN : median(qs)
    d[age] = Record(xs_syn, fill(NaN, 5), fill(NaN, 5), fill(q, 5))
end
syn = BasinDataset(D1, "inj", base.depth, base.lith, base.porosity, Workbook("Injected" => d),
                   base.tsurf_data, base.properties, base.ages, base.bounds)
r_syn = run_forward(syn, cfg)
mask = isfinite.(r_ref.temperature_c[end, :, :]) .& isfinite.(r_syn.temperature_c[end, :, :])
dT = maximum(abs.(r_syn.temperature_c[end, :, :][mask] .- r_ref.temperature_c[end, :, :][mask]))
check("injection identity: max|dT_final| <= 1e-10 (got $(round(dT, digits=12)))", dT <= 1.0e-10)

# --- 2. make_param_data: ступенчатость по эпохам, интерполяция по X-узлам ---
d_new = load_dataset_derived(D1, Workbook())
lay = make_q_layout(d_new; mid_x = 750.0)
check("layout dim == 12", length(lay) == 12)
check("layout nodes == [0, 750, 1500] (границы профиля)", lay.nodes_x == [0.0, 750.0, 1500.0])
θt = theta_constant(lay, [75.0, 80.0, 85.0, 90.0])
pd = make_param_data(d_new, lay, θt)
qs = Dict{Float64,Float64}()
for age in [150.0, 55.0, 10.0, 0.0]
    qs[age] = boundary_value(pd.heat, age, [100.0])[1]
end
check("epoch steps: 150 Ma -> 75", qs[150.0] == 75.0)
check("epoch steps: 55 Ma -> 80", qs[55.0] == 80.0)
check("epoch steps: 10 Ma -> 85", qs[10.0] == 85.0)
check("epoch steps: 0 Ma -> 90", qs[0.0] == 90.0)
qx = boundary_value(pd.heat, 10.0, [0.0, 375.0, 750.0, 1125.0, 1500.0])
check("linear interp across nodes at fixed epoch", qx == [85.0, 85.0, 85.0, 85.0, 85.0])
check("non-constant theta interpolates in x",
      boundary_value(make_param_data(d_new, lay, [30.0, 60.0, 90.0, 60.0, 60.0, 60.0,
                                                    60.0, 60.0, 60.0, 60.0, 60.0, 60.0]).heat,
                     10.0, [0.0, 375.0, 750.0, 1125.0, 1500.0]) == [60.0, 60.0, 60.0, 60.0, 60.0])
θvar = Float64[30, 60, 90, 60, 60, 60, 60, 60, 60, 60, 60, 60]
θvar[3] = 90.0  # x=1500 на старой эпохе
qx2 = boundary_value(make_param_data(d_new, lay, θvar).heat, 150.0, [0.0, 375.0, 750.0, 1125.0, 1500.0])
check("x-interpolation between node values", qx2 == [30.0, 45.0, 60.0, 75.0, 90.0])

# --- 3. Ошибки размерности/боксов ---
try
    make_param_data(d_new, lay, fill(60.0, 11))
    check("wrong theta length raises", false)
catch
    check("wrong theta length raises", true)
end

println()
if failures == 0
    println("ALL PASS")
else
    println(@printf("%d FAILURES", failures))
    exit(1)
end
