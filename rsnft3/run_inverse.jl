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
using NPZ
using Distributed

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))
include(joinpath(@__DIR__, "observations.jl"))
include(joinpath(@__DIR__, "objective.jl"))
include(joinpath(@__DIR__, "criteria.jl"))
include(joinpath(@__DIR__, "invert.jl"))

function parse_cli()
    s = ArgParseSettings(description = "Inverse problem: basal heat flow q(x,t) from well T and Ro")
    @add_arg_table s begin
        "--data-dir"      ; required = true
        "--wells"         ; required = true; help = "semicolon-separated XLSX/JSON paths"
        "--grid"          ; default = "50,60"; help = "nx,nz of the working grid"
        "--ensemble"      ; arg_type = Int; default = 4
        "--workers"       ; arg_type = Int; default = 4
        "--de-budget"     ; arg_type = Int; default = 4000
        "--nm-iterations" ; arg_type = Int; default = 200
        "--sigma-T"       ; arg_type = Float64; default = 2.0
        "--sigma-Ro"      ; arg_type = Float64; default = 0.05
        "--seed"          ; arg_type = Int; default = 1
        "--reg"           ; arg_type = Float64; default = 1.0
        "--weight-ro"     ; arg_type = Float64; default = 1.0
        "--n-nodes"       ; arg_type = Int; default = 3; help = "X-nodes of q: 2 or 3"
        "--mid-x"         ; arg_type = Float64; default = 23400.0
        "--epochs"        ; default = "200-100,100-35,35-10,10-0"
        "--no-multilevel" ; action = :store_true
        "--theta0-file"   ; default = ""; help = "solution.json to polish from (skips DE; grid-hierarchy fallback)"
        "--max-dt-ma"     ; arg_type = Float64; default = 2.5; help = "solver max time step, Ma"
        "--picard-iters"  ; arg_type = Int; default = 3
        "--picard-tol"    ; arg_type = Float64; default = 1.0e-3
        "--pcg-tol"       ; arg_type = Float64; default = 1.0e-8; help = "PCG relative tolerance"
        "--max-f-calls"   ; arg_type = Int; default = 0; help = "bound NM by J-evaluations (0 = unbounded)"
        "--weight-ro-cal" ; default = "1"; help = "comma list: weight_ro calibration ladder (applied to the best stage-A theta)"
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

function main()
    args = parse_cli()
    nx, nz = parse.(Int, split(args["grid"], ","))
    out_dir = abspath(args["out"])
    mkpath(out_dir)
    t_start = time()

    data = load_dataset_derived(args["data-dir"], Workbook())
    wells = load_wells_multi(args["wells"], data; sigma_T = args["sigma-T"], sigma_Ro = args["sigma-Ro"])
    layout = make_q_layout(data; epochs = parse_epochs(args["epochs"]),
                           n_nodes = args["n-nodes"], mid_x = args["mid-x"])
    cfg = SolverConfig(nx = nx, nz = nz, max_dt_ma = args["max-dt-ma"],
                       picard_iterations = args["picard-iters"],
                       picard_tolerance_c = args["picard-tol"],
                       pcg_tolerance = args["pcg-tol"])

    # воркеры: каждый со своим датасетом и кэшем
    if args["workers"] > 1 && nprocs() == 1
        addprocs(args["workers"] - 1)
        for f in ("parameterized_data.jl", "ro_kinetics.jl", "observations.jl",
                  "objective.jl", "criteria.jl", "invert.jl")
            @everywhere include(joinpath($(@__DIR__), $f))
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
                             pcg_tolerance = args["pcg-tol"])
        end
    end
    cache_dir = joinpath(out_dir, "cache_main")
    obj = make_objective(data, wells, cfg, layout; reg = args["reg"],
                         weight_ro = args["weight-ro"], cache_dir = cache_dir)
    _WORKER_OBJ[] = obj      # BlackBoxOptim вызывает fitness и на главном

    if !isempty(args["weight-ro-cal"]) && !isempty(args["theta0-file"])
        src = JSON.parsefile(args["theta0-file"])
        theta0 = Float64.(src["best"]["theta"])
        ladder = parse.(Float64, split(args["weight-ro-cal"], ","))
        best = nothing; results = []
        for wr in ladder
            wobj = make_objective(data, wells, cfg, layout; reg = args["reg"],
                                  weight_ro = wr, cache_dir = joinpath(cache_dir, "wr$wr"))
            r = polish_from(wobj, theta0; nm_iterations = args["nm-iterations"],
                            multilevel = !args["no-multilevel"])
            println(@sprintf("weight_ro=%.0f: J=%.1f (J_T=%.1f J_Ro=%.1f J_reg=%.1f)",
                             wr, r.J, r.J_T, r.J_Ro, r.J_reg))
            push!(results, r)
        end
        sort!(results; by = r -> r.J)
        best = first(results)
    elseif !isempty(args["theta0-file"])
        src = JSON.parsefile(args["theta0-file"])
        theta0 = Float64.(src["best"]["theta"])
        println("POLISH from $(args["theta0-file"]): J=$(src["best"]["J"])")
        best = polish_from(obj, theta0; nm_iterations = args["nm-iterations"],
                           multilevel = !args["no-multilevel"],
                           max_f_calls = args["max-f-calls"])
        results = [best]
    else
        results = invert_ensemble(obj; ensemble = args["ensemble"], de_budget = args["de-budget"],
                                  multilevel = !args["no-multilevel"], seed0 = args["seed"],
                                  nm_iterations = args["nm-iterations"],
                                  max_f_calls = args["max-f-calls"])
        best = first(results)
    end

    # финальный прогон и критерии
    pdata = make_param_data(data, layout, best.theta)
    result = run_forward(pdata, cfg)
    rep = criteria_report(result, data, wells)
    npzwrite(joinpath(out_dir, "result.npz"), Dict(
        "x" => result.x, "z" => result.z, "ages_ma" => result.ages_ma,
        "temperature_c" => result.temperature_c,
        "final_lithology" => result.final_lithology,
        "final_porosity" => result.final_porosity,
        "final_surface_m" => result.final_surface_m,
        "final_base_m" => result.final_base_m))

    solution = OrderedDict{String,Any}(
        "status" => "complete",
        "data_dir" => abspath(args["data-dir"]),
        "wells_spec" => args["wells"],
        "grid" => OrderedDict("nx" => nx, "nz" => nz),
        "layout" => OrderedDict("names" => layout.names, "nodes_x" => layout.nodes_x,
                                "epochs" => [collect(e) for e in layout.epochs],
                                "lower" => layout.lower, "upper" => layout.upper),
        "settings" => OrderedDict(k => args[k] for k in
            ("ensemble", "workers", "de-budget", "nm-iterations", "sigma-T", "sigma-Ro",
             "seed", "reg", "weight-ro", "no-multilevel")),
        "observations" => wellset_summary(wells),
        "n_dropped_T_inactive" => obj.n_dropped_T,
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
            "per_well" => [OrderedDict(
                "well" => w.well, "n_T" => w.n_T, "n_Ro" => w.n_Ro,
                "rmse_T" => w.rmse_T, "rmse_Ro" => w.rmse_Ro,
                "frac_1s_T" => w.frac_1s_T, "frac_2s_T" => w.frac_2s_T,
                "frac_1s_Ro" => w.frac_1s_Ro, "frac_2s_Ro" => w.frac_2s_Ro)
                for w in rep.wells]),
        "elapsed_total_s" => time() - t_start,
    )
    open(joinpath(out_dir, "solution.json"), "w") do f
        JSON.print(f, nan2null(solution), 2)
    end
    # (NaN в критериях допустим только при отсутствии точек какого-то типа)

    println("ENSEMBLE J: ", join([round(r.J; digits = 1) for r in results], ", "))
    println("BEST θ̂: ", join([round(v; digits = 2) for v in best.theta], ", "))
    println(@printf("RMSE_T=%.2f RMSE_Ro=%.3f  T in 2σ: %.1f%%  Ro in 2σ: %.1f%%  E^L2=%.2f",
                    rep.rmse_T, rep.rmse_Ro, 100 * rep.frac_2s_T, 100 * rep.frac_2s_Ro, rep.l2))
    println("SAVED ", out_dir)
end

try
    main()
catch err
    showerror(stderr, err)
    println(stderr)
    exit(1)
end

