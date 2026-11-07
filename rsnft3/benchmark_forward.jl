# Бенчмарк и профиль прямого решателя: baseline-замеры и сравнение результатов.
# Использование:
#   julia --project=. benchmark_forward.jl [--data-dir D] [--nx 60] [--nz 120]
#       [--profile]  — вывести Profile-разбиение по функциям
#       [--vs RUN_ID] — сравнить результат с сохранённым прогоном (max |ΔT|)
using Dates
using Profile

const HERE = dirname(@__FILE__)
include(joinpath(HERE, "basin_fvm.jl"))

function parse_cli(argv)
    opts = Dict{String,Any}("data-dir" => normpath(joinpath(HERE, "..", "Данные_для_прямой_задачи", "data1")),
                            "cache-dir" => normpath(joinpath(HERE, "..", "results")),
                            "nx" => 60, "nz" => 120, "max-dt-ma" => 2.5,
                            "picard-iterations" => 3, "profile" => false, "vs" => nothing)
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--profile"
            opts["profile"] = true
        elseif a == "--vs"
            opts["vs"] = argv[i + 1]; i += 1
        elseif startswith(a, "--")
            opts[replace(a[3:end], "-" => "-")] = argv[i + 1]; i += 1
        end
        i += 1
    end
    return opts
end

function main()
    opts = parse_cli(ARGS)
    config = SolverConfig(nx = opts["nx"], nz = opts["nz"],
                          max_dt_ma = opts["max-dt-ma"],
                          picard_iterations = opts["picard-iterations"])
    data = load_dataset(opts["data-dir"], opts["cache-dir"])

    # Прогрев (JIT) — не учитываем.
    run_forward(data, config)
    # Замеренный прогон.
    result = run_forward(data, config)
    diag = result.diagnostics
    n = length(diag)
    println("BENCH steps=$n elapsed=$(round(result.elapsed_seconds, digits=3))s ",
            "per_step=$(round(result.elapsed_seconds / n * 1000, digits=2))ms ",
            "avg_picard=$(round(sum(d -> d["picard_iterations"], diag) / n, digits=2)) ",
            "sum_linear_iters=$(sum(d -> d["linear_iterations"], diag))")

    if opts["profile"]
        Profile.clear()
        @profile run_forward(data, config)
        println("=== Profile (top) ===")
        Profile.print(format = :flat, sortedby = :count, mincount = 50)
    end

    if opts["vs"] !== nothing
        other = joinpath(opts["cache-dir"], opts["vs"], "result.npz")
        isfile(other) || error("not found: $other")
        ref = NPZ.npzread(other)["temperature_c"]
        cur = result.temperature_c
        size(ref) == size(cur) || error("shape mismatch: $(size(ref)) vs $(size(cur))")
        maxd = 0.0
        for k in axes(cur, 1)
            for j in axes(cur, 2), i in axes(cur, 3)
                a = cur[k, j, i]; b = ref[k, j, i]
                (isfinite(a) && isfinite(b)) && (maxd = max(maxd, abs(a - b)))
            end
        end
        println("COMPARE vs=$(opts["vs"]) max_abs_diff_T=$(maxd) C")
    end
    return result
end

main()
