# Run several adaptive meshes and create a convergence comparison report.
#
# Запуск:
#   julia run_adaptive_grid.jl --data-dir data --case 16,16,1 --case 16,16,2 --case 16,16,3
#
# Файлы adaptive_basin_fvm.jl, basin_fvm.jl, compare_runs.jl, visualize_results.jl
# и visualize_adaptive_mesh.jl должны лежать рядом.

isdefined(Main, :DEFAULT_DATA_DIR) || include(joinpath(@__DIR__, "adaptive_basin_fvm.jl"))
isdefined(Main, :interpolate_field) || include(joinpath(@__DIR__, "compare_runs.jl"))

const DEFAULT_CASES = ("16,16,1", "16,16,2", "16,16,3")

function parse_case(value::AbstractString)
    format_msg = "case must have format BASE_NX,BASE_NZ,MAX_LEVEL"
    parts = split(value, ",")
    length(parts) == 3 || throw(ArgumentError(format_msg))
    nums = try
        Tuple(parse(Int, strip(p)) for p in parts)
    catch
        throw(ArgumentError(format_msg))
    end
    base_nx, base_nz, max_level = nums
    if base_nx < 2 || base_nz < 2 || !(0 <= max_level <= 5)
        throw(ArgumentError(
            "BASE_NX and BASE_NZ must be at least 2; MAX_LEVEL must be between 0 and 5"))
    end
    return (base_nx, base_nz, max_level)
end

"Запускает Julia-скрипт в подпроцессе (в том же окружении), печатает вывод и возвращает stdout."
function run_script(script::AbstractString, args::Vector{String})
    project = dirname(Base.active_project())
    cmd = `$(Base.julia_cmd()) --project=$project $script $args`
    out, err = IOBuffer(), IOBuffer()
    try
        run(pipeline(cmd; stdout = out, stderr = err))
    catch
        print(stderr, String(take!(out)), String(take!(err)))
        rethrow()
    end
    stdout_text = String(take!(out))
    stderr_text = String(take!(err))
    isempty(stdout_text) || print(stdout_text)
    isempty(stderr_text) || print(stderr, stderr_text)
    return stdout_text
end

function run_id_from_output(output::AbstractString)
    m = match(r"(?:RUN|CACHE_HIT)\s+([0-9a-f]{16})", output)
    m === nothing && error("adaptive solver did not report a run id")
    return String(m.captures[1])
end

function comparison(run_dir::AbstractString, reference_dir::AbstractString)
    current = npzread(joinpath(run_dir, "result.npz"))
    reference = npzread(joinpath(reference_dir, "result.npz"))
    mapped = interpolate_field(
        current["x"], current["z"], current["temperature_c"][end, :, :],
        reference["x"], reference["z"],
    )
    reference_final = reference["temperature_c"][end, :, :]
    mask = isfinite.(mapped) .& isfinite.(reference_final)
    any(mask) || throw(ErrorException(
        "no overlapping finite cells for $run_dir and $reference_dir"))
    difference = mapped[mask] .- reference_final[mask]
    return OrderedDict{String,Any}(
        "compared_cells" => count(mask),
        "rms_difference_c" => sqrt(mean(difference .^ 2)),
        "relative_l2" => norm(difference) / norm(reference_final[mask]),
        "max_abs_difference_c" => maximum(abs.(difference)),
    )
end

function format_number(value, digits::Int = 6)
    value === nothing && return "—"
    value isa AbstractFloat && return Printf.format(Printf.Format("%.$(digits)g"), value)
    return string(value)
end

function csv_field(v)
    v === nothing && return ""
    s = v isa Bool ? (v ? "True" : "False") : string(v)
    return any(c -> c in (',', '"', '\n', '\r'), s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
end

function write_reports(rows::Vector{<:AbstractDict}, reference_id::AbstractString,
                       report_dir::AbstractString)
    mkpath(report_dir)
    payload = OrderedDict{String,Any}("reference_run_id" => reference_id, "runs" => rows)
    write(joinpath(report_dir, "comparison.json"), sprint(io -> JSON.print(io, payload, 2)))

    fieldnames = collect(keys(rows[1]))
    open(joinpath(report_dir, "comparison.csv"), "w") do io
        write(io, "\ufeff")                        # utf-8-sig, чтобы Excel распознал кодировку
        write(io, join(fieldnames, ","), "\r\n")
        for row in rows
            write(io, join((csv_field(row[k]) for k in fieldnames), ","), "\r\n")
        end
    end

    lines = String[
        "# Сравнение адаптивных сеток",
        "",
        "Эталон для попарных отклонений: `$reference_id` (самая мелкая ячейка).",
        "",
        "| Сетка | Уровень | Ячеек | Мин. ячейка, м | Время, с | L2 | RMS, °C | Max, °C | Сходимость |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows
        run_link = "../$(row["run_id"])"
        mesh = "$(row["base_nx"])×$(row["base_nz"])"
        min_cell = "$(format_number(row["minimum_dx_m"], 5))×$(format_number(row["minimum_dz_m"], 5))"
        push!(lines,
            "| [$mesh]($run_link) | $(row["max_level"]) | $(row["final_cells"]) | " *
            "$min_cell | $(format_number(row["elapsed_seconds"], 5)) | " *
            "$(format_number(row["relative_l2"], 5)) | " *
            "$(format_number(row["rms_difference_c"], 5)) | " *
            "$(format_number(row["max_abs_difference_c"], 5)) | " *
            "$(row["nonlinear_converged"] ? "да" : "нет") |")
    end
    append!(lines, ["", "## Изображения", ""])
    for row in rows
        append!(lines, [
            "### $(row["base_nx"])×$(row["base_nz"]), уровень $(row["max_level"])",
            "",
            "[Температура](../$(row["run_id"])/temperature_0Ma.png) · " *
            "[Адаптивная сетка](../$(row["run_id"])/adaptive_mesh_0Ma.png)",
            "",
        ])
    end
    write(joinpath(report_dir, "README.md"), join(lines, "\n"))
end

function fail(message::AbstractString)
    println(stderr, "error: ", message)
    exit(2)
end

function main_grid(argv = ARGS)
    s = ArgParseSettings(description = "Run a series of adaptive FVM meshes and compare them")
    @add_arg_table! s begin
        "--data-dir"
            arg_type = String
            default = DEFAULT_DATA_DIR
        "--cache-dir"
            arg_type = String
            default = "adaptive_results"
        "--report-dir"
            arg_type = String
            default = nothing
        "--case"
            arg_type = String
            action = :append_arg
            metavar = "NX,NZ,LEVEL"
            help = "repeatable; defaults: 16,16,1; 16,16,2; 16,16,3"
        "--max-dt-ma"
            arg_type = Float64
            default = 2.5
        "--interface-band"
            arg_type = Float64
            default = 0.55
        "--picard-iterations"
            arg_type = Int
            default = 8
        "--picard-tolerance-c"
            arg_type = Float64
            default = 1.0e-3
        "--linear-tolerance"
            arg_type = Float64
            default = 1.0e-8
        "--force"
            action = :store_true
        "--skip-images"
            action = :store_true
    end
    args = parse_args(argv, s)

    case_strings = isempty(args["case"]) ? collect(DEFAULT_CASES) : String.(args["case"])
    cases = try
        [parse_case(v) for v in case_strings]
    catch e
        e isa ArgumentError ? fail(e.msg) : rethrow()
    end
    length(unique(cases)) == length(cases) || fail("duplicate --case values are not allowed")
    args["max-dt-ma"] <= 0 && fail("--max-dt-ma must be positive")

    cache_dir = args["cache-dir"]
    solver = joinpath(@__DIR__, "adaptive_basin_fvm.jl")
    temperature_visualizer = joinpath(@__DIR__, "visualize_results.jl")
    mesh_visualizer = joinpath(@__DIR__, "visualize_adaptive_mesh.jl")
    run_dirs = String[]

    for (number, (base_nx, base_nz, max_level)) in enumerate(cases)
        println("\n[$number/$(length(cases))] base=$(base_nx)x$(base_nz), level=$max_level")
        command = String[
            "--data-dir", args["data-dir"], "--cache-dir", cache_dir,
            "--base-nx", string(base_nx), "--base-nz", string(base_nz),
            "--max-level", string(max_level),
            "--max-dt-ma", string(args["max-dt-ma"]),
            "--interface-band", string(args["interface-band"]),
            "--picard-iterations", string(args["picard-iterations"]),
            "--picard-tolerance-c", string(args["picard-tolerance-c"]),
            "--linear-tolerance", string(args["linear-tolerance"]),
        ]
        args["force"] && push!(command, "--force")
        run_id = run_id_from_output(run_script(solver, command))
        run_dir = joinpath(cache_dir, run_id)
        push!(run_dirs, run_dir)
        if !args["skip-images"]
            run_script(temperature_visualizer, String[run_dir, "--age", "0"])
            run_script(mesh_visualizer, String[run_dir])
        end
    end

    metadata = [JSON.parsefile(joinpath(d, "metadata.json"); dicttype = OrderedDict{String,Any})
                for d in run_dirs]
    reference_index = argmin(i -> (metadata[i]["minimum_dx_m"] * metadata[i]["minimum_dz_m"],
                                   -metadata[i]["final_cells"]), eachindex(metadata))
    reference_dir = run_dirs[reference_index]
    reference_id = metadata[reference_index]["run_id"]

    rows = OrderedDict{String,Any}[]
    for (run_dir, item) in zip(run_dirs, metadata)
        metrics = comparison(run_dir, reference_dir)
        sc = item["solver"]
        push!(rows, merge(OrderedDict{String,Any}(
            "run_id" => item["run_id"],
            "is_reference" => item["run_id"] == reference_id,
            "base_nx" => sc["base_nx"],
            "base_nz" => sc["base_nz"],
            "max_level" => sc["max_level"],
            "final_cells" => item["final_cells"],
            "final_active_cells" => item["final_active_cells"],
            "minimum_dx_m" => item["minimum_dx_m"],
            "minimum_dz_m" => item["minimum_dz_m"],
            "elapsed_seconds" => item["elapsed_seconds"],
            "temperature_min_c" => item["final_temperature_min_c"],
            "temperature_max_c" => item["final_temperature_max_c"],
            "nonlinear_converged" => get(item, "nonlinear_converged", false),
            "unconverged_time_steps" => get(item, "unconverged_time_steps", nothing),
        ), metrics))
    end

    report_dir = args["report-dir"] === nothing ? joinpath(cache_dir, "grid_comparison") : args["report-dir"]
    write_reports(rows, reference_id, report_dir)
    println("\nREFERENCE $reference_id")
    println("REPORT $(joinpath(report_dir, "README.md"))")
    println("CSV $(joinpath(report_dir, "comparison.csv"))")
    println("JSON $(joinpath(report_dir, "comparison.json"))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_grid()
end
