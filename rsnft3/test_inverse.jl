# Сводный раннер юнит-тестов инверсии (этапы A–B). Каждый модуль — отдельный
# процесс, чтобы JIT и side-effects не смешивались. Выход — сводка PASS/FAIL.
const TESTS = [
    ("ro_kinetics",  joinpath(@__DIR__, "test_ro_kinetics.jl")),
    ("observations", joinpath(@__DIR__, "test_observations.jl")),
    ("param_data",   joinpath(@__DIR__, "test_param_data.jl")),
    ("objective",    joinpath(@__DIR__, "test_objective.jl")),
    ("criteria",     joinpath(@__DIR__, "test_criteria.jl")),
    ("inverse_c2f",  joinpath(@__DIR__, "test_inverse_c2f.jl")),
]
const EXE = joinpath(Sys.BINDIR, Sys.iswindows() ? "julia.exe" : "julia")
results = Tuple{String,Bool}[]
for (name, path) in TESTS
    ok = success(run(`$EXE --project=$(@__DIR__) $path`; wait = true))
    push!(results, (name, ok))
end
println("\n=== SUMMARY ===")
fail = 0
for (name, ok) in results
    println(rpad(name, 14), ok ? "OK" : "FAIL")
    ok || (global fail += 1)
end
fail == 0 || exit(1)
