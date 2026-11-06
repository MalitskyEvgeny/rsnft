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

# --- 6. ансамбль параллельно: детерминизм (старт-целиком-на-воркере == старты
# по одному на воркере, побитово, включая de_evals) ---
if nprocs() > 1
    fine_cfg = SolverConfig(nx = 30, nz = 60, max_dt_ma = 5.0,
                            picard_iterations = 2, picard_tolerance_c = 1.0e-2)
    for w in workers()
        remotecall_fetch(worker_setup, w, obj_fine.data, wells, fine_cfg, lay;
                         cache_dir = joinpath(cachedir, "ens_w$w"))
    end
    ens_kwargs = (de_budget = 60, multilevel = false, nm_iterations = 5, seed0 = 7)
    par = invert_ensemble_parallel(obj_fine; ensemble = 2, ens_kwargs...)
    check("ensemble parallel: count", length(par) == 2)
    check("ensemble parallel: J finite", all(isfinite(r.J) && r.J > 0 for r in par))
    check("ensemble parallel: sorted by J", issorted(r.J for r in par))
    check("ensemble parallel: best is first", par[1].J <= par[2].J)
    # базовая линия: те же старты по одному на воркере 2, последовательно
    seq = [remotecall_fetch(invert_start_on_worker, workers()[1],
                            (de_budget = 60, multilevel = false,
                             seed = 7 + s, nm_iterations = 5,
                             trace = false, max_f_calls = 0))
           for s in 0:1]
    by_seed_par = Dict(r.seed => r for r in par)
    for r in seq
        b = get(by_seed_par, r.seed, nothing)
        check("ensemble parallel: seed $(r.seed) present", b !== nothing)
        b === nothing && continue
        check("ensemble parallel: seed $(r.seed) theta bit-identical",
              length(b.theta) == length(r.theta) &&
              all(b.theta .== r.theta))
        check("ensemble parallel: seed $(r.seed) J bit-identical", b.J == r.J)
        check("ensemble parallel: seed $(r.seed) de_evals match (no pmap from worker)",
              b.de_evals == r.de_evals)
    end
    # воспроизводимость: второй параллельный прогон — те же значения
    par2 = invert_ensemble_parallel(obj_fine; ensemble = 2, ens_kwargs...)
    check("ensemble parallel: reproducible",
          [r.J for r in par] == [r.J for r in par2] &&
          all(par[i].theta == par2[i].theta for i in 1:2))
else
    println("SKIP ensemble parallel tests (run with julia -p 2)")
end

# --- 7. c2f-ансамбль параллельно: воркер держит coarse + fine, старт целиком;
# результат каждого seed совпадает с одиночным запуском той же функции ---
if nprocs() > 1
    epochs_c2f = [(200.0, 100.0), (100.0, 35.0), (35.0, 10.0), (10.0, 0.0)]
    for w in workers()
        remotecall_fetch(worker_init_c2f!, w, DATA_DIR, wells, epochs_c2f,
                         0, 0.0;
                         nodes_x = lay.nodes_x,
                         coarse_nx = 12, coarse_nz = 20,
                         coarse_cache_dir = joinpath(cachedir, "c2f_ens_w_coarse$w"),
                         fine_nx = 30, fine_nz = 60,
                         fine_cache_dir = joinpath(cachedir, "c2f_ens_w_fine$w"),
                         reg = 0.1, max_dt_ma = 5.0,
                         picard_iterations = 2, picard_tolerance_c = 1.0e-2)
    end
    c2f_kwargs = (de_budget = 60, nm_iterations = 5, multilevel = false, seed0 = 11)
    results_ens, stages_ens = invert_ensemble_c2f(obj_coarse, obj_fine;
                                                  ensemble = 2, c2f_kwargs...)
    check("c2f ensemble: counts", length(results_ens) == 2 && length(stages_ens) == 2)
    check("c2f ensemble: seeds = seed0..seed0+E-1",
          Set(r.seed for r in results_ens) == Set(11:12))
    check("c2f ensemble: results sorted by J", issorted(r.J for r in results_ens))
    check("c2f ensemble: stages sorted like results",
          [s.seed for s in stages_ens] == [r.seed for r in results_ens])
    check("c2f ensemble: J finite", all(isfinite(r.J) && r.J > 0 for r in results_ens))
    best_res, best_stage = results_ens[1], stages_ens[1]
    Jf_c, _ = obj_fine(best_stage.theta)
    check("c2f ensemble: polished J <= J_fine at theta_coarse",
          best_res.J <= Jf_c + 1e-9)
    # базовая линия: одиночный c2f-старт на воркере тем же seed — побитово
    br, bs = remotecall_fetch(invert_start_c2f_on_worker, workers()[1],
                              (de_budget = 60, nm_iterations = 5,
                               multilevel = false, seed = 11,
                               trace = false, max_f_calls = 0))
    i11 = findfirst(r -> r.seed == 11, results_ens)
    check("c2f ensemble: seed 11 present", i11 !== nothing)
    i11 === nothing && (i11 = 1)
    cand, cand_stage = results_ens[i11], stages_ens[i11]
    check("c2f ensemble: seed 11 theta bit-identical to single start",
          all(cand.theta .== br.theta))
    check("c2f ensemble: seed 11 J bit-identical to single start", cand.J == br.J)
    check("c2f ensemble: seed 11 de_evals match (no pmap from worker)",
          cand_stage.de_evals == bs.de_evals)
else
    println("SKIP c2f ensemble tests (run with julia -p 2)")
end

println()
if failures == 0
    println("ALL PASS")
else
    println(@printf("%d FAILURES", failures))
    exit(1)
end
