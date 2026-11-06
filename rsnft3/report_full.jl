# Единый HTML-отчёт (русский, интерактивные графики Plotly, самодостаточный):
# прямая задача (result.npz + metadata.json) и/или обратная (solution.json).
# Заменяет report_inverse.jl.
#
# Запуск:
#   julia --project=. report_full.jl --direct results/<run_id>
#   julia --project=. report_full.jl --inverse results/<inverse_run>
#   julia --project=. report_full.jl --direct ... --inverse ... --out report.html

using ArgParse
using JSON
using NPZ
using Printf
using Dates
using Base64

include(joinpath(@__DIR__, "report", "html.jl"))
include(joinpath(@__DIR__, "report", "plotly_assets.jl"))

function parse_cli(argv = ARGS)
    s = ArgParseSettings(description = "Единый HTML-отчёт: прямая и/или обратная задача")
    @add_arg_table! s begin
        "--direct"
            help = "каталог результата прямой задачи (result.npz + metadata.json)"
            arg_type = String
            default = ""
        "--inverse"
            help = "каталог результата инверсии (solution.json + result.npz)"
            arg_type = String
            default = ""
        "--out"
            help = "путь к итоговому HTML (по умолчанию <run_dir>/report.html)"
            arg_type = String
            default = ""
    end
    return parse_args(argv, s)
end

function build_report(direct_dir::String, inverse_dir::String, out_path::String)
    io = IOBuffer()
    println(io, "<!DOCTYPE html><html lang=\"ru\"><head>")
    println(io, "<meta charset=\"utf-8\">")
    println(io, "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
    println(io, "<title>Отчёт по бассейновому моделированию</title>")
    println(io, "<style>", REPORT_CSS, "</style>")
    println(io, "</head><body>")

    has_direct = !isempty(direct_dir)
    has_inverse = !isempty(inverse_dir)
    header = has_direct && has_inverse ? "Прямая и обратная задачи" :
             has_direct ? "Прямая задача" : "Обратная задача"
    println(io, "<header class=\"hero\"><div class=\"wrap\">")
    println(io, "<h1>Бассейновое моделирование — отчёт</h1>")
    println(io, "<div class=\"sub\">", esc(header), " · сгенерировано ",
            Dates.format(Dates.now(), "dd.uu.yyyy HH:MM"), "</div>")
    println(io, "</div></header>")

    println(io, "<nav class=\"toc\"><b>Содержание:</b>")
    has_direct && println(io, "<a href=\"#direct\">Прямая задача</a>")
    has_inverse && println(io, "<a href=\"#inverse\">Обратная задача</a>")
    println(io, "</nav><main>")

    offline = plotly_script_tag(io)

    if has_direct
        Base.invokelatest(include, joinpath(@__DIR__, "report", "section_direct.jl"))
        dd = Base.invokelatest(load_direct, direct_dir)
        Base.invokelatest(section_direct, io, dd)
    end
    if has_inverse
        for m in ("parameterized_data.jl", "ro_kinetics.jl", "observations.jl",
                  "objective.jl", "criteria.jl")
            Base.invokelatest(include, joinpath(@__DIR__, m))
        end
        Base.invokelatest(include, joinpath(@__DIR__, "report", "section_inverse.jl"))
        iv = Base.invokelatest(load_inverse, inverse_dir)
        Base.invokelatest(section_inverse, io, iv)
    end

    println(io, "</main><footer>Отчёт сгенерирован report_full.jl",
            offline ? " · графики встроены (работает офлайн)" : " · plotly.js с CDN",
            "</footer>")
    println(io, "</body></html>")
    return String(take!(io))
end

const CLI_ARGS = parse_cli()
const DIRECT_DIR = CLI_ARGS["direct"]
const INVERSE_DIR = CLI_ARGS["inverse"]

isempty(DIRECT_DIR) && isempty(INVERSE_DIR) &&
    error("укажите --direct и/или --inverse")

if !isempty(DIRECT_DIR)
    isfile(joinpath(DIRECT_DIR, "result.npz")) || error("нет result.npz в $DIRECT_DIR")
end
if !isempty(INVERSE_DIR)
    isfile(joinpath(INVERSE_DIR, "solution.json")) || error("нет solution.json в $INVERSE_DIR")
end

has_direct() = !isempty(DIRECT_DIR)
has_inverse() = !isempty(INVERSE_DIR)

if has_direct()
    include(joinpath(@__DIR__, "report", "section_direct.jl"))
end
if has_inverse()
    for m in ("parameterized_data.jl", "ro_kinetics.jl", "observations.jl",
              "objective.jl", "criteria.jl")
        include(joinpath(@__DIR__, m))
    end
    include(joinpath(@__DIR__, "report", "section_inverse.jl"))
end

function main()
    out_path = CLI_ARGS["out"]
    if isempty(out_path)
        base = !isempty(DIRECT_DIR) ? DIRECT_DIR : INVERSE_DIR
        out_path = joinpath(base, "report.html")
    end
    isempty(DIRECT_DIR) || isdir(DIRECT_DIR) || error("каталог прямой задачи не найден: $DIRECT_DIR")
    isempty(INVERSE_DIR) || isdir(INVERSE_DIR) || error("каталог обратной задачи не найден: $INVERSE_DIR")
    html = build_report(DIRECT_DIR, INVERSE_DIR, out_path)
    mkpath(dirname(abspath(out_path)))
    write(out_path, html)
    println("SAVED ", out_path, " (", round(length(html) / 1e6; digits = 1), " МБ)")
end

try
    main()
catch err
    showerror(stderr, err)
    println(stderr)
    exit(1)
end
