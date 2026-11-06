# Юнит-тесты coarse-to-fine инверсии (invert_c2f): согласованность стадий,
# перенос θ между сетками, поли-улучшение, переинициализация воркеров.
# Запуск: julia --project=. test_inverse_c2f.jl [data-dir]

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))
include(joinpath(@__DIR__, "observations.jl"))
include(joinpath(@__DIR__, "invert.jl"))

using Printf

failures = 0
function check(name, ok)
    global failures
    println((ok ? "PASS " : "FAIL "), name)
    ok || (failures += 1)
end

const DATA_DIR = length(ARGS) >= 1 ? ARGS[1] :
                 normpath(joinpath(@__DIR__, "..", "Данные для обратной задачи"))

d = load_dataset_derived(DATA_DIR, Workbook())
lay = make_q_layout(d)
wells = let
    wsT = load_wells_xlsx(joinpath(DATA_DIR, "TemperatureWellData.xlsx"), d; kind = :T)
    wsR = load_wells_xlsx(joinpath(DATA_DIR, "VitriniteWellData.xlsx"), d; kind = :Ro)
    WellSet(vcat(wsT.points, wsR.points), vcat(wsT.excluded, wsR.excluded),
            collect(union(wsT.excluded_wells, wsR.excluded_wells)))
end

cachedir = joinpath(tempdir(), "c2f_cache_test"); rm(cachedir; force = true, recursive = true)
obj_coarse = make_objective(d, wells, SolverConfig(nx = 12, nz = 20, max_dt_ma = 5.0,
                            picard_iterations = 2, picard_tolerance_c = 1.0e-2), lay;
                            reg = 0.1, cache_dir = joinpath(cachedir, "coarse"))
obj_fine = make_objective(d, wells, SolverConfig(nx = 30, nz = 60, max_dt_ma = 5.0,
                          picard_iterations = 2, picard_tolerance_c = 1.0e-2), lay;
                          reg = 0.1, cache_dir = joinpath(cachedir, "fine"))

# --- 1. de_stage: возврат конечных θ, J ---
stage = de_stage(obj_coarse; de_budget = 60, seed = 1)
check("de_stage: theta length matches layout", length(stage.theta) == length(lay))
check("de_stage: J finite positive", isfinite(stage.J) && stage.J > 0)
check("de_stage: de_evals counted", stage.de_evals > 0)

# --- 2. drift-check: J на двух сетках при одном θ конечен ---
Jc, _ = obj_coarse(stage.theta)
Jf, _ = obj_fine(stage.theta)
check("drift-check: J finite on both grids", isfinite(Jc) && isfinite(Jf))

# --- 3. invert_c2f: полировка не хуже старта на fine ---
best, stage2 = invert_c2f(obj_coarse, obj_fine; de_budget = 60, nm_iterations = 10, seed = 1)
Jfine_at_coarse, _ = obj_fine(stage2.theta)
check("invert_c2f: layout lengths match", length(best.theta) == length(obj_fine.layout))
check("invert_c2f: polished J <= J_fine at theta_coarse", best.J <= Jfine_at_coarse + 1e-9)
check("invert_c2f: best J finite", isfinite(best.J) && best.J > 0)
check("invert_c2f: nm polish happened (de_evals==0 in result)", best.de_evals == 0)

# --- 4. кэши стадий раздельны ---
check("separate caches: coarse cache dir exists", isfile(joinpath(cachedir, "coarse", "j_cache.json")))

# --- 5. воркер-режим: воркеры получают invert.jl, c2f_stage_init! ставит Objective, DE идёт через worker_J ---
if nprocs() > 1
    @everywhere include(joinpath($(@__DIR__), "invert.jl"))
    ccfg = SolverConfig(nx = 12, nz = 20, max_dt_ma = 5.0,
                        picard_iterations = 2, picard_tolerance_c = 1.0e-2)
    for w in workers()
        remotecall_fetch(worker_setup, w, obj_coarse.data, wells, ccfg, lay;
                         cache_dir = joinpath(cachedir, "w$w"))
    end
    st = de_stage(obj_coarse; de_budget = 40, seed = 1)
    check("worker mode: de_stage via worker_J works", isfinite(st.J) && st.J > 0)
else
    println("SKIP worker-mode test (run with julia -p 2)")
end

println()
if failures == 0
    println("ALL PASS")
else
    println(@printf("%d FAILURES", failures))
    exit(1)
end
