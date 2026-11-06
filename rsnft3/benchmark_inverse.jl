# 4.1: точечный бенчмарк инверсии — время forward-оценок J, cache IO, DE-оверхед.
#
# Запуск:
#   julia --project=. benchmark_inverse.jl --data-dir "../Данные для обратной задачи" \
#       --wells "T.xlsx;V.xlsx" --n 50
# Таблица в stdout: forward/eval по сеткам 20x40 и 30x60, cache hit/miss, save.

using ArgParse
using Printf
using Random

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))
include(joinpath(@__DIR__, "observations.jl"))
include(joinpath(@__DIR__, "objective.jl"))
include(joinpath(@__DIR__, "invert.jl"))

function parse_cli(argv = ARGS)
    s = ArgParseSettings(description = "Бенчмарк оценки J инверсии (4.1)")
    @add_arg_table! s begin
        "--data-dir"; arg_type = String; required = true
        "--wells";    arg_type = String; required = true
        "--n";        arg_type = Int; default = 50; help = "свежих оценок на сетку"
        "--grid";     arg_type = String; default = "20,40,30,60"; help = "сетки nx,nz через запятую"
    end
    return parse_args(argv, s)
end

function bench_grid(data, wells, nx, nz, n)
    layout = make_q_layout(data; n_nodes = 3, mid_x = 23400.0)
    cfg = SolverConfig(nx = nx, nz = nz)
    mkpath("results/_bench_cache")
    obj = make_objective(data, wells, cfg, layout; reg = 1.0, weight_ro = 1.0,
                         cache_dir = "results/_bench_cache")
    d = length(layout)
    θ = collect(range(mean(layout.lower), mean(layout.upper); length = d))

    # cold (miss): уникальные θ
    t_miss = 0.0
    for i in 1:n
        θi = θ .+ 0.001 .* i
        t0 = time()
        J, info = obj(θi)
        t_miss += time() - t0
    end
    # warm (hit): те же θ повторно
    t_hit = 0.0
    for i in 1:n
        θi = θ .+ 0.001 .* i
        t0 = time()
        J, info = obj(θi)
        t_hit += time() - t0
    end
    t_save = 0.0
    t0 = time()
    cache_save(obj; force = true)
    t_save = time() - t0
    return (nx = nx, nz = nz, d = d,
            miss_ms = 1e3 * t_miss / n, hit_ms = 1e3 * t_hit / n, save_ms = 1e3 * t_save)
end

function main()
    args = parse_cli()
    validate_inputs(args["data-dir"], args["wells"], joinpath(@__DIR__, "..", "results", "_bench_cache"))
    data = load_dataset_derived(args["data-dir"], Workbook())
    wells = load_wells_multi(args["wells"], data; sigma_T = 2.0, sigma_Ro = 0.05)
    grids = parse.(Int, split(args["grid"], ","))
    @printf("%-8s %-8s %-10s %-14s %-12s %-10s\n", "nx", "nz", "dim", "miss ms/eval", "hit ms/eval", "save ms")
    for (nx, nz) in zip(grids[1:2:end], grids[2:2:end])
        r = bench_grid(data, wells, nx, nz, args["n"])
        @printf("%-8d %-8d %-10d %-14.1f %-12.3f %-10.1f\n",
                r.nx, r.nz, r.d, r.miss_ms, r.hit_ms, r.save_ms)
    end
    println("cached J-файл: results/_bench_cache (можно удалить)")
end

try
    main()
catch err
    println(stderr, "ОШИБКА: ", sprint(showerror, err))
    exit(1)
end
