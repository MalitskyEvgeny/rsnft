# C3: CLI запуска инверсии. Ансамбль DE→NM на воркерах (каждый — свой датасет),
# результат: solution.json (θ̂ ансамбля, breakdown, критерии, счётчики точек,
# время) + result.npz финального прогона + report-данные.
#
# Запуск:
#   julia --project=. run_inverse.jl --data-dir "../Данные для обратной задачи" \
#         --wells "../.../TemperatureWellData.xlsx;../.../VitriniteWellData.xlsx" \
#         --grid 50,60 --ensemble 4 --workers 4 --de-budget 4000 \
#         --sigma-T 2 --sigma-Ro 0.05 --seed 1 --out ../results/inverse_run1

using ArgParse
using Printf
using NPZ
using Distributed
using LinearAlgebra

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))
include(joinpath(@__DIR__, "observations.jl"))
include(joinpath(@__DIR__, "objective.jl"))
include(joinpath(@__DIR__, "criteria.jl"))
include(joinpath(@__DIR__, "invert.jl"))
include(joinpath(@__DIR__, "config_check.jl"))

include(joinpath(@__DIR__, "progress.jl"))
using .Progress

function parse_cli()
    s = ArgParseSettings(description = "Inverse problem: basal heat flow q(x,t) from well T and Ro")
    @add_arg_table s begin
        "--data-dir"      ; required = true
        "--wells"         ; required = true; help = "semicolon-separated XLSX/JSON paths"
        "--grid"          ; default = "50,60"; help = "nx,nz of the working grid"
        "--ensemble"      ; arg_type = Int; default = 4   # 0 = авто: Sys.CPU_THREADS
        "--workers"       ; arg_type = Int; default = 0   # 0 = авто: по числу ансамблей
        "--de-budget"     ; arg_type = Int; default = 4000
        "--nm-iterations" ; arg_type = Int; default = 200
        "--sigma-T"       ; arg_type = Float64; default = 2.0
        "--sigma-Ro"      ; arg_type = Float64; default = 0.05
        "--seed"          ; arg_type = Int; default = 1
        "--reg"           ; arg_type = Float64; default = 1.0
        "--weight-ro"     ; arg_type = Float64; default = 1.0
        "--n-nodes"       ; arg_type = Int; default = 3; help = "X-nodes of q: 2 or 3"
        "--mid-x"         ; arg_type = Float64; default = 23400.0
        "--node-xs"       ; default = ""; help = "comma X-nodes of q in meters, e.g. 0,7000,23400,40000 (overrides --n-nodes/--mid-x)"
        "--epochs"        ; default = "200-100,100-35,35-10,10-0"
        "--no-multilevel" ; action = :store_true
        "--theta0-file"   ; default = ""; help = "solution.json to polish from (skips DE unless --de-warm-start; grid-hierarchy fallback)"
        "--de-warm-start" ; action = :store_true; help = "with --theta0-file: run DE warm-started from theta0 (injected into the start population) + NM polish; de-budget > 0 required"
        "--q-true"        ; default = ""; help = "twin mode: theta_true as one q per epoch (like make_twin --q); adds theta_true vs theta_hat table to solution.json/report"
        "--max-dt-ma"     ; arg_type = Float64; default = 2.5; help = "solver max time step, Ma"
        "--picard-iters"  ; arg_type = Int; default = 3
        "--picard-tol"    ; arg_type = Float64; default = 1.0e-3
        "--pcg-tol"       ; arg_type = Float64; default = 1.0e-6; help = "PCG relative tolerance"
        "--max-f-calls"   ; arg_type = Int; default = -1; help = "bound NM by J-evaluations per epoch-level (-1 = dynamic: 2x nm-iterations; 0 = unbounded)"
        "--weight-ro-cal" ; default = "1"; help = "comma list: weight_ro calibration ladder (applied to the best stage-A theta)"
        "--adaptive-final"; action = :store_true; help = "control stage: evaluate criteria on the adaptive grid at the best theta"
        "--adaptive-search"; action = :store_true; help = "run the ENTIRE pipeline (search + criteria) on the adaptive grid; no uniform grid anywhere"
        "--adaptive-base" ; default = "16,32"; help = "adaptive base grid nx,nz (with --adaptive-final)"
        "--adaptive-level"; arg_type = Int; default = 2; help = "adaptive max refinement level (with --adaptive-final)"
        "--adaptive-preview"; default = "120,160"; help = "adaptive preview grid nx,nz for observation sampling"
        "--c2f"           ; action = :store_true; help = "coarse-to-fine: DE on a coarse grid, NM polish on the working grid"
        "--c2f-grid"      ; default = "20,40"; help = "coarse nx,nz (with --c2f)"
        "--c2f-de-budget" ; arg_type = Int; default = 0; help = "DE budget on the coarse grid (0 = 25% of --de-budget, recommended: stage A needs far fewer evals than polish)"
        "--adaptive-frozen"; action = :store_true; help = "build the adaptive mesh once on the final stratigraphy (no per-substep rebuild/remap)"
        "--out"           ; required = true
    end
    return parse_args(s)
end

function parse_epochs(spec::AbstractString)
    out = Tuple{Float64,Float64}[]
    for part in split(spec, ",")
        a, b = split(strip(part), "-")
        push!(out, (parse(Float64, a), parse(Float64, b)))
    end
    return out
end

"F7: NaN в логах -> n/a вместо NaN/nothing."
fmt_or_na(x) = isnan(x) ? "n/a" : @sprintf("%.2f", x)

"Явные ошибки на несовместимых комбинациях флагов вместо молчаливого поведения (3.2)."
function validate_flags(args)
    bad(msg) = throw(ErrorException(msg))
    args["weight-ro-cal"] != "1" && isempty(args["theta0-file"]) &&
        bad("--weight-ro-cal требует --theta0-file (лестница калибровки применяется к лучшему θ)")
    args["adaptive-search"] && args["c2f"] &&
        bad("--adaptive-search несовместим с --c2f (coarse-to-fine работает только на регулярных сетках)")
    any(args[k] for k in ("adaptive-final", "adaptive-search")) && !args["adaptive-search"] &&
        args["adaptive-base"] != "16,32" && isempty(args["theta0-file"]) &&
        @warn "--adaptive-base/--adaptive-level применяются только вместе с --adaptive-final или --adaptive-search"
    args["de-warm-start"] && isempty(args["theta0-file"]) &&
        bad("--de-warm-start требует --theta0-file (источник тёплого старта)")
    args["de-warm-start"] && args["de-budget"] <= 0 &&
        bad("--de-warm-start требует --de-budget > 0 (DE без бюджета не запускается)")
    return nothing
end

"7.1: подготовка общего контекста инверсии (данные, layout, воркеры, objective)."
function setup_inverse(args)
    nx, nz = parse.(Int, split(args["grid"], ","))
    out_dir = abspath(args["out"])
    validate_inputs(args["data-dir"], args["wells"], out_dir)
    t_start = time()

    data = load_dataset_derived(args["data-dir"], Workbook())
    wells = load_wells_multi(args["wells"], data; sigma_T = args["sigma-T"], sigma_Ro = args["sigma-Ro"])
    node_xs = isempty(args["node-xs"]) ? nothing :
              Float64.(parse.(Float64, split(args["node-xs"], ",")))
    layout = make_q_layout(data; epochs = parse_epochs(args["epochs"]),
                           n_nodes = args["n-nodes"], mid_x = args["mid-x"],
                           nodes_x = node_xs)
    cfg = SolverConfig(nx = nx, nz = nz, max_dt_ma = args["max-dt-ma"],
                       picard_iterations = args["picard-iters"],
                       picard_tolerance_c = args["picard-tol"],
                       pcg_tolerance = args["pcg-tol"])

    # авто-воркеры: не больше стартов, чем воркеров — воркеров ровно по ансамблю
    # (иначе серийный fallback invert_ensemble и лишние воркеры простаивают в памяти)
    if args["ensemble"] == 0
        args["ensemble"] = args["workers"] > 0 ? args["workers"] : Int(Sys.CPU_THREADS)
        args["workers"] = args["ensemble"]
    elseif args["workers"] == 0
        args["workers"] = args["ensemble"]
    end
    # процесс-параллелизм: по одному потоку на процесс, BLAS-потоки в 1 —
    # иначе W процессов × N BLAS-потоков переполняют ядра (oversubscription)
    LinearAlgebra.BLAS.set_num_threads(1)
    # воркеры: каждый со своим датасетом и кэшем
    if args["workers"] > 1 && nprocs() == 1
        addprocs(args["workers"] - 1)
        for f in ("parameterized_data.jl", "ro_kinetics.jl", "observations.jl",
                  "objective.jl", "criteria.jl", "progress.jl", "invert.jl")
            @everywhere include(joinpath($(@__DIR__), $f))
        end
        for pr in procs()
            remotecall_fetch(LinearAlgebra.BLAS.set_num_threads, pr, 1)
        end
        data_dir_abs = abspath(args["data-dir"])
        epochs = parse_epochs(args["epochs"])
        for w in workers()
            remotecall_fetch(worker_init!, w, data_dir_abs, wells, epochs,
                             args["n-nodes"], args["mid-x"], nx, nz;
                             reg = args["reg"], weight_ro = args["weight-ro"],
                             cache_dir = joinpath(out_dir, "cache_w$w"),
                             max_dt_ma = args["max-dt-ma"],
                             picard_iterations = args["picard-iters"],
                             picard_tolerance_c = args["picard-tol"],
                             pcg_tolerance = args["pcg-tol"],
                             nodes_x = node_xs)
        end
    end
    cache_dir = joinpath(out_dir, "cache_main")
    use_adaptive = args["adaptive-search"] || args["adaptive-final"]
    acfg = nothing
    if use_adaptive
        ab = parse.(Int, split(args["adaptive-base"], ","))
        ap = parse.(Int, split(args["adaptive-preview"], ","))
        acfg = AdaptiveConfig(base_nx = ab[1], base_nz = ab[2], max_level = args["adaptive-level"],
                              max_dt_ma = args["max-dt-ma"],
                              picard_iterations = args["picard-iters"],
                              picard_tolerance_c = args["picard-tol"],
                              preview_nx = ap[1], preview_nz = ap[2],
                              frozen_mesh = args["adaptive-frozen"])
        println("ADAPTIVE backend: base=$(ab[1])x$(ab[2]) level=$(args["adaptive-level"]) preview=$(ap[1])x$(ap[2]) search=$(args["adaptive-search"]) frozen=$(args["adaptive-frozen"])")
    end
    obj = make_objective(data, wells, cfg, layout; reg = args["reg"],
                         weight_ro = args["weight-ro"], cache_dir = cache_dir,
                         backend = args["adaptive-search"] ? :adaptive : :regular,
                         adaptive_cfg = acfg)
    _WORKER_OBJ[] = obj      # BlackBoxOptim вызывает fitness и на главном
    return (args = args, nx = nx, nz = nz, out_dir = out_dir, t_start = t_start,
            data = data, wells = wells, layout = layout, cfg = cfg, obj = obj,
            node_xs = node_xs, cache_dir = cache_dir, acfg = acfg)
end

"7.1: ветка --c2f. Возвращает (results, best, c2f_stage, c2f_stages, cde)."
function mode_c2f(ctx)
    args = ctx.args
    obj = ctx.obj
    data, wells, layout, out_dir = ctx.data, ctx.wells, ctx.layout, ctx.out_dir
    nx, nz = ctx.nx, ctx.nz
    node_xs = ctx.node_xs
    c2f_stage = nothing
    c2f_stages = C2fStage[]
    cde = 0
    begin
        # Stage A (coarse): DE на грубой regular-сетке; layout тот же.
        cnx, cnz = parse.(Int, split(args["c2f-grid"], ","))
        # 2.7: грубая стадия исторически в ~7 раз дольше полировки — дефолт 25%
        cde = args["c2f-de-budget"] > 0 ? args["c2f-de-budget"] : cld(args["de-budget"], 4)
        args["c2f-de-budget"] == args["de-budget"] && @warn "c2f: задан полный DE-бюджет для грубой сетки - обычно достаточно 25%"
        ccfg = SolverConfig(nx = cnx, nz = cnz, max_dt_ma = args["max-dt-ma"],
                            picard_iterations = args["picard-iters"],
                            picard_tolerance_c = args["picard-tol"],
                            pcg_tolerance = args["pcg-tol"])
        cobj = make_objective(data, wells, ccfg, layout; reg = args["reg"],
                              weight_ro = args["weight-ro"],
                              cache_dir = joinpath(cache_dir, "c2f_coarse"))
        println("C2F: coarse grid $(cnx)x$(cnz), DE budget $cde; fine $(nx)x$(nz)")
        # воркеры и главный — под грубую сетку (кэш воркеров — свой cache_dir)
        c2f_stages = C2fStage[]
        if nprocs() > 1 && args["ensemble"] >= nworkers()
            # стартов >= воркеров: старт целиком на воркере (coarse + fine за одну
            # инициализацию), все ядра заняты волнами, NM перекрыт
            # воркеры: coarse + fine Objective за один remotecall (кэши раздельны),
            # старт ансамбля выполняется на воркере целиком без переинициализаций
            data_dir_abs = abspath(args["data-dir"])
            epochs = parse_epochs(args["epochs"])
            for w in workers()
                remotecall_fetch(worker_init_c2f!, w, data_dir_abs, wells, epochs,
                                 args["n-nodes"], args["mid-x"];
                                 coarse_nx = cnx, coarse_nz = cnz,
                                 coarse_cache_dir = joinpath(out_dir, "cache_w_coarse$w"),
                                 fine_nx = nx, fine_nz = nz,
                                 fine_cache_dir = joinpath(out_dir, "cache_w$w"),
                                 nodes_x = node_xs,
                                 reg = args["reg"], weight_ro = args["weight-ro"],
                                 max_dt_ma = args["max-dt-ma"],
                                 picard_iterations = args["picard-iters"],
                                 picard_tolerance_c = args["picard-tol"],
                                 pcg_tolerance = args["pcg-tol"])
            end
            results, c2f_stages = invert_ensemble_c2f(cobj, obj;
                                                      ensemble = args["ensemble"],
                                                      de_budget = cde,
                                                      nm_iterations = args["nm-iterations"],
                                                      multilevel = !args["no-multilevel"],
                                                      seed0 = args["seed"],
                                                      max_f_calls = args["max-f-calls"])
            best = first(results)
            c2f_stage = first(c2f_stages)    # coarse-стадия best-старта
        elseif nprocs() > 1 && args["ensemble"] > 1
            # стартов меньше воркеров: DE каждого старта pmap'ит по всем воркерам,
            # старты последовательно (полное заполнение ядер в DE-фазе)
            results = InversionResult[]
            for st_i in 1:args["ensemble"]
                r, stg = invert_c2f(cobj, obj; de_budget = cde,
                                    nm_iterations = args["nm-iterations"],
                                    multilevel = !args["no-multilevel"],
                                    seed = args["seed"] + st_i - 1,
                                    max_f_calls = args["max-f-calls"])
                push!(results, r)
                push!(c2f_stages, stg)
                println(@sprintf("C2F ensemble start %d/%d (seed %d): J=%.1f",
                                 st_i, args["ensemble"], args["seed"] + st_i - 1, r.J))
            end
            sort!(results; by = r -> r.J)
            best = first(results)
            c2f_stage = c2f_stages[findfirst(st -> st.seed == best.seed, c2f_stages)]
        else
            if nprocs() > 1
                data_dir_abs = abspath(args["data-dir"])
                epochs = parse_epochs(args["epochs"])
                c2f_stage_init!(worker_init!, data_dir_abs, wells, epochs,
                                args["n-nodes"], args["mid-x"], cnx, cnz;
                                reg = args["reg"], weight_ro = args["weight-ro"],
                                cache_dir = joinpath(out_dir, "cache_w_coarse"),
                                max_dt_ma = args["max-dt-ma"],
                                picard_iterations = args["picard-iters"],
                                pcg_tolerance = args["pcg-tol"],
                                nodes_x = node_xs)
            end
            _WORKER_OBJ[] = cobj
            best, c2f_stage = invert_c2f(cobj, obj; de_budget = cde,
                                         nm_iterations = args["nm-iterations"],
                                         multilevel = !args["no-multilevel"],
                                         seed = args["seed"], max_f_calls = args["max-f-calls"])
            results = [best]
        end
        # drift-check: J на грубой и рабочей сетке при одном θ_coarse. Уровни J
        # напрямую несравнимы (смещение дискретизации) — индикатор расхождения.
        if c2f_stage !== nothing
            Jc, _ = cobj(c2f_stage.theta)
            println(@sprintf("C2F drift-check: J_coarse(theta_coarse)=%.1f (stage-A J=%.1f); J_fine(theta_coarse)=%.1f -> polished J=%.1f",
                             Jc, c2f_stage.J, obj(c2f_stage.theta)[1], best.J))
        end
        # воркеры обратно под рабочую сетку (финальные прогоны и adaptive-final
        # считают J на главном, но держим консистентность)
        if nprocs() > 1
            data_dir_abs = abspath(args["data-dir"])
            epochs = parse_epochs(args["epochs"])
            for w in workers()
                remotecall_fetch(worker_init!, w, data_dir_abs, wells, epochs,
                                 args["n-nodes"], args["mid-x"], nx, nz;
                                 reg = args["reg"], weight_ro = args["weight-ro"],
                                 cache_dir = joinpath(out_dir, "cache_w$w"),
                                 max_dt_ma = args["max-dt-ma"],
                                 picard_iterations = args["picard-iters"],
                                 picard_tolerance_c = args["picard-tol"],
                                 pcg_tolerance = args["pcg-tol"],
                                 nodes_x = node_xs)
            end
            _WORKER_OBJ[] = obj
        end
    end
    return (results = results, best = best, c2f_stage = c2f_stage,
            c2f_stages = c2f_stages, cde = cde)
end

"7.1: ветка --weight-ro-cal (лестница weight_ro на theta0)."
function mode_weight_ro_cal(ctx)
    args = ctx.args
    data, wells, cfg, layout, cache_dir = ctx.data, ctx.wells, ctx.cfg, ctx.layout, ctx.cache_dir
    src = JSON.parsefile(args["theta0-file"])
    theta0 = Float64.(src["best"]["theta"])
    ladder = parse.(Float64, split(args["weight-ro-cal"], ","))
    best = nothing; results = []
    for wr in ladder
        wobj = make_objective(data, wells, cfg, layout; reg = args["reg"],
                              weight_ro = wr, cache_dir = joinpath(cache_dir, "wr$wr"))
        r = polish_from(wobj, theta0; nm_iterations = args["nm-iterations"],
                        multilevel = !args["no-multilevel"],
                        max_f_calls = args["max-f-calls"])
        println(@sprintf("weight_ro=%.0f: J=%.1f (J_T=%.1f J_Ro=%.1f J_reg=%.1f)",
                         wr, r.J, r.J_T, r.J_Ro, r.J_reg))
        push!(results, r)
    end
    sort!(results; by = r -> r.J)
    best = first(results)
    return (results = results, best = best)
end

"7.1: ветка --theta0-file (polish-only или DE warm-start)."
function mode_theta0(ctx)
    args = ctx.args
    obj = ctx.obj
    layout = ctx.layout
    src = JSON.parsefile(args["theta0-file"])
    theta0 = Float64.(src["best"]["theta"])
    if args["de-warm-start"]
        println("DE warm-start from $(args["theta0-file"]): J=$(src["best"]["J"]), budget $(args["de-budget"])")
        length(theta0) == length(layout) || error("theta0 dimension mismatch: $(length(theta0)) vs layout $(length(layout))")
        clamp!(theta0, layout.lower, layout.upper)
        results = invert_ensemble(obj; ensemble = args["ensemble"], de_budget = args["de-budget"],
                                  multilevel = !args["no-multilevel"], seed0 = args["seed"],
                                  nm_iterations = args["nm-iterations"],
                                  max_f_calls = args["max-f-calls"], warm_start = theta0)
    else
        println("POLISH from $(args["theta0-file"]): J=$(src["best"]["J"])")
        results = [polish_from(obj, theta0; nm_iterations = args["nm-iterations"],
                               multilevel = !args["no-multilevel"],
                               max_f_calls = args["max-f-calls"])]
    end
    best = first(results)
    return (results = results, best = best)
end

"7.1: обычный режим - DE-ансамбль с нуля."
function mode_plain(ctx)
    args = ctx.args
    obj = ctx.obj
    results = invert_ensemble(obj; ensemble = args["ensemble"], de_budget = args["de-budget"],
                              multilevel = !args["no-multilevel"], seed0 = args["seed"],
                              nm_iterations = args["nm-iterations"],
                              max_f_calls = args["max-f-calls"])
    best = first(results)
    return (results = results, best = best)
end

"7.1: диспетчер режимов по флагам."
function run_mode(ctx)
    args = ctx.args
    if args["c2f"]
        Progress.stage_set(1, 2, "coarse-to-fine inversion")
        out = mode_c2f(ctx)
        Progress.stage_set(2, 2, "final forward & control")
        return out
    elseif !isempty(args["weight-ro-cal"]) && !isempty(args["theta0-file"])
        Progress.stage_set(1, 2, "weight_ro calibration ladder")
        out = mode_weight_ro_cal(ctx)
        Progress.stage_set(2, 2, "final forward & control")
        return out
    elseif !isempty(args["theta0-file"])
        Progress.stage_set(1, 2, "polish from theta0")
        out = mode_theta0(ctx)
        Progress.stage_set(2, 2, "final forward & control")
        return out
    else
        Progress.stage_set(1, 2, "DE inversion ensemble")
        out = mode_plain(ctx)
        Progress.stage_set(2, 2, "final forward & control")
        return out
    end
end

"7.1: финальный прогон при best.theta, критерии, adaptive-final контроль."
function final_forward(ctx, best)
    args = ctx.args
    data, wells, cfg, layout = ctx.data, ctx.wells, ctx.cfg, ctx.layout
    obj, acfg, out_dir = ctx.obj, ctx.acfg, ctx.out_dir
    # финальный прогон и критерии
    if args["adaptive-search"]
        bnds = data.bounds
        fpdx = (bnds[2] - bnds[1]) / acfg.preview_nx
        fxg = collect(range(bnds[1] + 0.5 * fpdx, bnds[2] - 0.5 * fpdx; length = acfg.preview_nx))
        result = run_adaptive(data, acfg; qbasal_by_age = qbasal_by_age(layout, best.theta, fxg, data.ages),
                              qbasal_nodes_x = layout.nodes_x, geometry = obj.geometry)
        mesh_png = save_mesh_drawing(result.mesh, result.grid, result.final_temperature_cells,
                                     joinpath(out_dir, "mesh_adaptive.png"))
        println("MESH_PNG ", mesh_png)
    else
        pdata = make_param_data(data, layout, best.theta)
        result = run_forward(pdata, cfg)
    end
    rep = criteria_report(result, data, wells)
    npzwrite(joinpath(out_dir, "result.npz"), Dict(
        "x" => result.x, "z" => result.z, "ages_ma" => result.ages_ma,
        "temperature_c" => result.temperature_c,
        "final_lithology" => result.final_lithology,
        "final_porosity" => result.final_porosity,
        "final_surface_m" => result.final_surface_m,
        "final_base_m" => result.final_base_m))

    adaptive_rep = nothing
    if args["adaptive-final"]
        println("ADAPTIVE control stage: base=$(args["adaptive-base"]) level=$(args["adaptive-level"]) preview=$(args["adaptive-preview"])")
        aobj = make_objective(data, wells, cfg, layout; reg = args["reg"],
                              weight_ro = args["weight-ro"],
                              backend = :adaptive, adaptive_cfg = acfg)
        aJ, ainfo = aobj(best.theta)
        bnds = data.bounds
        apdx = (bnds[2] - bnds[1]) / acfg.preview_nx
        axg = collect(range(bnds[1] + 0.5 * apdx, bnds[2] - 0.5 * apdx; length = acfg.preview_nx))
        aresult = run_adaptive(data, acfg;
                               qbasal_by_age = qbasal_by_age(layout, best.theta, axg, data.ages),
                               qbasal_nodes_x = layout.nodes_x,
                               geometry = aobj.geometry)
        adaptive_rep = criteria_report(aresult, data, wells)
        npzwrite(joinpath(out_dir, "result_adaptive.npz"), Dict(
            "x" => aresult.x, "z" => aresult.z, "ages_ma" => aresult.ages_ma,
            "temperature_c" => aresult.temperature_c,
            "final_lithology" => aresult.final_lithology,
            "final_porosity" => aresult.final_porosity,
            "final_surface_m" => aresult.final_surface_m,
            "final_base_m" => aresult.final_base_m))
        mesh_png = save_mesh_drawing(aresult.mesh, aresult.grid, aresult.final_temperature_cells,
                                     joinpath(out_dir, "mesh_adaptive.png"))
        println("MESH_PNG ", mesh_png)
        @printf("ADAPTIVE J=%.2f (J_T=%.2f J_Ro=%.2f)  RMSE_T=%s RMSE_Ro=%s  T in 2s: %s
",
                aJ, ainfo.J_T, ainfo.J_Ro, fmt_or_na(adaptive_rep.rmse_T),
                fmt_or_na(adaptive_rep.rmse_Ro), fmt_or_na(100 * adaptive_rep.frac_2s_T))
    end
    return (result = result, rep = rep, adaptive_rep = adaptive_rep)
end

"7.1: сборка solution.json из результатов режима."
function build_solution(ctx, best, results, ff, c2f_info, cde)
    args = ctx.args
    nx, nz = ctx.nx, ctx.nz
    wells = ctx.wells
    layout, obj, rep = ctx.layout, ctx.obj, ff.rep
    adaptive_rep = ff.adaptive_rep
    c2f_stage, c2f_stages = c2f_info === nothing ? (nothing, C2fStage[]) : c2f_info
    solution = OrderedDict{String,Any}(
        "status" => "complete",
        "data_dir" => abspath(args["data-dir"]),
        "wells_spec" => args["wells"],
        "theta_true" => isempty(args["q-true"]) ? nothing :
                        begin
                            qe = parse.(Float64, split(args["q-true"], ","))
                            length(qe) == length(layout.epochs) ||
                                error("--q-true: нужно одно значение на эпоху ($(length(layout.epochs)))")
                            OrderedDict("epoch_q_mwm2" => qe,
                                        "theta" => collect(theta_constant(layout, qe)))
                        end,
        "grid" => OrderedDict("nx" => nx, "nz" => nz,
                              "search_backend" => args["adaptive-search"] ? "adaptive" : "regular"),
        "layout" => OrderedDict("names" => layout.names, "nodes_x" => layout.nodes_x,
                                "epochs" => [collect(e) for e in layout.epochs],
                                "lower" => layout.lower, "upper" => layout.upper),
        "settings" => merge(
            OrderedDict(k => args[k] for k in
                ("ensemble", "workers", "de-budget", "nm-iterations", "sigma-T", "sigma-Ro",
                 "seed", "reg", "weight-ro", "no-multilevel")),
            OrderedDict("ensemble_mode" =>
                (nprocs() > 1 && args["ensemble"] >= nworkers()) ? "parallel" : "serial")),
        "observations" => wellset_summary(wells),
        "n_dropped_T_inactive" => obj.n_dropped_T,
        "n_artifact_T" => obj.n_artifact_T[],
        "n_artifact_Ro" => obj.n_artifact_Ro[],
        "ensemble" => [OrderedDict(
            "theta" => r.theta, "J" => r.J, "J_T" => r.J_T, "J_Ro" => r.J_Ro,
            "J_reg" => r.J_reg, "de_evals" => r.de_evals, "nm_evals" => r.nm_evals,
            "elapsed_s" => r.elapsed_s, "seed" => r.seed) for r in results],
        "best" => OrderedDict("theta" => best.theta, "J" => best.J,
                              "J_T" => best.J_T, "J_Ro" => best.J_Ro, "J_reg" => best.J_reg),
        "criteria" => OrderedDict(
            "rmse_T" => rep.rmse_T, "rmse_Ro" => rep.rmse_Ro,
            "frac_1s_T" => rep.frac_1s_T, "frac_2s_T" => rep.frac_2s_T,
            "frac_1s_Ro" => rep.frac_1s_Ro, "frac_2s_Ro" => rep.frac_2s_Ro,
            "l2" => rep.l2,
            "n_artifact" => rep.n_artifact, "n_kept" => rep.n_kept,
            "n_dropped" => rep.n_dropped, "passed" => rep.passed,
            "per_well" => [OrderedDict(
                "well" => w.well, "n_T" => w.n_T, "n_Ro" => w.n_Ro,
                "rmse_T" => w.rmse_T, "rmse_Ro" => w.rmse_Ro,
                "frac_1s_T" => w.frac_1s_T, "frac_2s_T" => w.frac_2s_T,
                "frac_1s_Ro" => w.frac_1s_Ro, "frac_2s_Ro" => w.frac_2s_Ro,
                "n_artifact" => w.n_artifact, "n_kept" => w.n_kept)
                for w in rep.wells]),
        "elapsed_total_s" => time() - ctx.t_start,
    )
    if c2f_stage !== nothing
        cnx, cnz = parse.(Int, split(args["c2f-grid"], ","))
        solution["c2f"] = OrderedDict(
            "coarse_grid" => OrderedDict("nx" => cnx, "nz" => cnz),
            "coarse_de_budget" => cde,
            "stage_A" => OrderedDict("theta" => c2f_stage.theta, "J" => c2f_stage.J,
                                     "de_evals" => c2f_stage.de_evals,
                                     "elapsed_s" => c2f_stage.elapsed_s,
                                     "seed" => c2f_stage.seed),
            "J_fine_at_theta_coarse" => obj(c2f_stage.theta)[1])
        if !isempty(c2f_stages)
            solution["c2f"]["stage_A_ensemble"] = [OrderedDict(
                "J" => st.J, "de_evals" => st.de_evals, "seed" => st.seed)
                for st in c2f_stages]
        end
    end
    if adaptive_rep !== nothing
        solution["adaptive"] = OrderedDict(
            "base" => args["adaptive-base"], "level" => args["adaptive-level"],
            "preview" => args["adaptive-preview"],
            "criteria" => OrderedDict(
                "rmse_T" => adaptive_rep.rmse_T, "rmse_Ro" => adaptive_rep.rmse_Ro,
                "frac_1s_T" => adaptive_rep.frac_1s_T, "frac_2s_T" => adaptive_rep.frac_2s_T,
                "frac_1s_Ro" => adaptive_rep.frac_1s_Ro, "frac_2s_Ro" => adaptive_rep.frac_2s_Ro,
                "l2" => adaptive_rep.l2))
    end
    open(joinpath(ctx.out_dir, "solution.json"), "w") do f
        JSON.print(f, nan2null(solution), 2)
    end
    return solution
end

"7.1: итоговая печать в консоль."
function print_summary(ctx, best, results, rep)
    println("ENSEMBLE J: ", join([round(r.J; digits = 1) for r in results], ", "))
    println("BEST θ̂: ", join([round(v; digits = 2) for v in best.theta], ", "))
    @printf("RMSE_T=%s RMSE_Ro=%s  T in 2σ: %s%%  Ro in 2σ: %s%%  E^L2=%s
",
            fmt_or_na(rep.rmse_T), fmt_or_na(rep.rmse_Ro), fmt_or_na(100 * rep.frac_2s_T),
            fmt_or_na(100 * rep.frac_2s_Ro), fmt_or_na(rep.l2))
    println("SAVED ", ctx.out_dir)
end

function main()
    args = parse_cli()
    Progress.init()
    # 4.4: динамический бюджет полировки по умолчанию — не «тихое удешевление»,
    # а ограничение сверху: NM обычно сходится раньше, лимит режет хвосты.
    args["max-f-calls"] == -1 &&
        (args["max-f-calls"] = 2 * args["nm-iterations"] * length(parse_epochs(args["epochs"])))
    try
        validate_flags(args)
    catch err
        println(stderr, "ОШИБКА КОНФИГУРАЦИИ: ", sprint(showerror, err))
        exit(1)
    end
    ctx = setup_inverse(args)
    out = run_mode(ctx)
    best, results = out.best, out.results
    c2f_info = haskey(out, :c2f_stage) ? (out.c2f_stage, out.c2f_stages) : nothing
    cde = haskey(out, :cde) ? out.cde : 0
    obj = ctx.obj
    (obj.n_artifact_T[] > 0 || obj.n_artifact_Ro[] > 0) &&
        @warn "Артефактные выборки в лучшем прогоне" n_artifact_T = obj.n_artifact_T[] n_artifact_Ro = obj.n_artifact_Ro[]
    cache_save(obj; force = true)          # финальная запись кэша (2.2)
    ff = final_forward(ctx, best)
    build_solution(ctx, best, results, ff, c2f_info, cde)
    print_summary(ctx, best, results, ff.rep)
    return nothing
end

try
    main()
catch err
    showerror(stderr, err)
    println(stderr)
    exit(1)
end

