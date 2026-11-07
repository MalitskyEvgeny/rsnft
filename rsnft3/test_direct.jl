# Проверка linear_solver=:direct (Холецкий) против :pcg:
# T и Ro совпадают в пределах линейной точности, диагностика согласована.
# Запуск: julia --project=. test_direct.jl [data-dir]

include(joinpath(@__DIR__, "basin_fvm.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))
include(joinpath(@__DIR__, "objective.jl"))

const DATA_DIR = length(ARGS) >= 1 ? ARGS[1] :
                 normpath(joinpath(dirname(@__DIR__), "Данные_для_прямой_задачи", "data1"))

data = load_dataset(DATA_DIR, normpath(joinpath(dirname(@__DIR__), "results")))
common = (nx = 30, nz = 60, max_dt_ma = 5.0, picard_iterations = 3)
cfg_pcg = SolverConfig(; common..., linear_solver = :pcg)
cfg_dir = SolverConfig(; common..., linear_solver = :direct)

geo_p = precompute_geometry(data, cfg_pcg)
geo_d = precompute_geometry(data, cfg_dir)
qba = Dict(age => zeros(length(geo_p.x)) for age in data.ages)

r_p = run_forward(data, cfg_pcg; geometry = geo_p, qbasal_by_age = qba)
r_d = run_forward(data, cfg_dir; geometry = geo_d, qbasal_by_age = qba)

fd_p = r_p.temperature_c[findall(isfinite.(r_p.temperature_c))]
fd_d = r_d.temperature_c[findall(isfinite.(r_d.temperature_c))]
dT = maximum(abs.(fd_p .- fd_d))
println("max |ΔT| direct-vs-pcg = ", dT, " C")
dT < 5.0e-3 || (println("FAIL: температура разошлась"); exit(1))

ro_p = particle_ro_grid(r_p, data)
ro_d = particle_ro_grid(r_d, data)
dRo = 0.0
for name in keys(ro_p)
    global dRo = max(dRo, maximum(abs.(ro_p[name] .- ro_d[name])))
end
println("max |ΔRo| = ", dRo)
dRo < 1.0e-4 || (println("FAIL: Ro разошлась"); exit(1))

# Валидация конфига
try
    validate(SolverConfig(linear_solver = :bogus))
    println("FAIL: невалидный linear_solver принят"); exit(1)
catch e
    println("validate ok: ", e)
end

println("PASS: test_direct")
