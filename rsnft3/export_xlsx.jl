# Экспорт результатов моделирования в XLSX формата входных/эталонных данных:
# один лист на слой, колонки X, Time_Ma, Z_Top [m], Z_Bot [m], Z_Aver [m],
# Top/Aver/Bot [Celsius] (Temperature.xlsx) / [%Ro]
# (Sweeney_Burnham_1990_EASY_Ro.xlsx). Прямая задача: result.npz прогона +
# данные прямой. Обратная: result.npz инверсии (контрольный прогон с найденным
# q) + данные обратной задачи.
#
# Запуск:
#   julia --project=. export_xlsx.jl --run results/<id> --data-dir <данные прямой>
#   julia --project=. export_xlsx.jl --inverse results/.../inverse --inv-data-dir <данные обратной>
#   [--out-dir DIR]  (по умолчанию <run_dir>/xlsx)

isdefined(@__MODULE__, :BasinDataset) || include(joinpath(@__DIR__, "basin_fvm.jl"))
include(joinpath(@__DIR__, "objective.jl"))   # bilinear_sample (+ ro_kinetics, parameterized_data)
using ArgParse
using JSON
using NPZ
using XLSX

struct ExportInput
    npz_path::String
    data::BasinDataset
    out_dir::String
    label::String
end

function parse_cli(argv = ARGS)
    s = ArgParseSettings(description = "Экспорт T/Ro в XLSX формата эталона")
    @add_arg_table! s begin
        "--run"
            help = "каталог результата прямой задачи (result.npz) + --data-dir"
            arg_type = String
            default = ""
        "--data-dir"
            help = "данные прямой задачи (геометрия слоёв)"
            arg_type = String
            default = ""
        "--inverse"
            help = "каталог результата инверсии (result.npz) + --inv-data-dir"
            arg_type = String
            default = ""
        "--inv-data-dir"
            help = "данные обратной задачи (геометрия слоёв)"
            arg_type = String
            default = ""
        "--out-dir"
            help = "каталог вывода (по умолчанию <run_dir>/xlsx)"
            arg_type = String
            default = ""
    end
    return parse_args(argv, s)
end

"T среза модели на возраст age: точное совпадение или линейная интерполяция
по времени между соседними сохранёнными срезами."
function temp_at_age(temp::Array{Float64,3}, ages_ma::Vector{Float64}, age::Float64)
    k = findfirst(a -> a == age, ages_ma)
    k !== nothing && return temp[k, :, :]
    i_old = findlast(a -> a >= age, ages_ma)
    i_young = findfirst(a -> a <= age, ages_ma)
    (i_old === nothing || i_young === nothing) &&
        error("age $age outside saved range $(ages_ma[1])…$(ages_ma[end])")
    i_old == i_young && return temp[i_old, :, :]
    f = (ages_ma[i_old] - age) / (ages_ma[i_old] - ages_ma[i_young])
    return (1.0 - f) .* temp[i_old, :, :] .+ f .* temp[i_young, :, :]
end

struct TruncResult
    x::Vector{Float64}
    z::Vector{Float64}
    ages_ma::Vector{Float64}
    temperature_c::Array{Float64,3}
end

"Ro частицы (слой, eta) на возраст ages_ma[k] по усечённой истории T."
function ro_at(result::TruncResult, data::BasinDataset, xq::Float64,
               layer::String, eta::Float64, k::Int)
    k <= 1 && return 0.2   # частица ещё не отложена / нет нагрева — минимальное Ro
    trunc = TruncResult(result.x, result.z, result.ages_ma[1:k], result.temperature_c[1:k, :, :])
    return particle_ro(trunc, data, xq, layer, eta)
end

function write_field_xlsx(path::String, ex::ExportInput, kind::Symbol)
    npz = npzread(ex.npz_path)
    xg = Float64.(npz["x"])
    zg = Float64.(npz["z"])
    ages_ma = Float64.(npz["ages_ma"])
    temp = Float64.(npz["temperature_c"])
    result = TruncResult(xg, zg, ages_ma, temp)
    unit = kind === :T ? "Celsius" : "%Ro"
    XLSX.openxlsx(path, mode = "w") do xf
        first_used = Ref(false)   # пустой Sheet1 переименовывается в первый слой
        for (layer, series) in ex.data.depth
            local sheet
            if first_used[]
                sheet = XLSX.addsheet!(xf, layer)
            else
                sheet = XLSX.getsheet(xf, 1)
                XLSX.rename!(sheet, layer)
                first_used[] = true
            end
            row = 1
            for (col, h) in enumerate(["X", "Time_Ma", "Z_Top [m]", "Z_Bot [m]",
                                       "Z_Aver [m]", "Top [$unit]",
                                       "Bot [$unit]", "Aver [$unit]"])
                sheet[row, col] = h
            end
            for age in sort(collect(keys(series)))
                haskey(series, age) || continue
                rec = series[age]
                k = findfirst(a -> a == age, ages_ma)
                tk = temp_at_age(temp, ages_ma, age)
                for i in eachindex(rec.x)
                    xq = rec.x[i]
                    top, bot = rec.top[i], rec.bot[i]
                    zaver = 0.5 * (top + bot)
                    t_top = bilinear_sample(xg, zg, tk, xq, top)
                    t_bot = bilinear_sample(xg, zg, tk, xq, bot)
                    t_aver = bilinear_sample(xg, zg, tk, xq, zaver)
                    row += 1
                    sheet[row, 1] = xq
                    sheet[row, 2] = age
                    sheet[row, 3] = top
                    sheet[row, 4] = bot
                    sheet[row, 5] = zaver
                    if kind === :T
                        sheet[row, 6] = t_top
                        sheet[row, 7] = t_bot
                        sheet[row, 8] = t_aver
                    else
                        sheet[row, 6] = ro_at(result, ex.data, xq, layer, 0.0, k === nothing ? 0 : k)
                        sheet[row, 7] = ro_at(result, ex.data, xq, layer, 1.0, k === nothing ? 0 : k)
                        sheet[row, 8] = ro_at(result, ex.data, xq, layer, 0.5, k === nothing ? 0 : k)
                    end
                end
            end
        end
    end
    return path
end

function main(argv = ARGS)
    args = parse_cli(argv)
    has_run = !isempty(args["run"])
    has_inv = !isempty(args["inverse"])
    (has_run || has_inv) || error("укажите --run <dir> --data-dir <dir> и/или --inverse <dir> --inv-data-dir <dir>")
    has_run && isempty(args["data-dir"]) && error("--run требует --data-dir")
    has_inv && isempty(args["inv-data-dir"]) && error("--inverse требует --inv-data-dir")
    cache = tempdir()
    inputs = ExportInput[]
    if has_run
        run_dir = rstrip(args["run"], '/')
        out_dir = isempty(args["out-dir"]) ? joinpath(run_dir, "xlsx") : args["out-dir"]
        push!(inputs, ExportInput(joinpath(run_dir, "result.npz"),
                                  load_dataset(args["data-dir"], cache), out_dir, "прямая задача"))
    end
    if has_inv
        inv_dir = rstrip(args["inverse"], '/')
        out_dir = isempty(args["out-dir"]) ? joinpath(inv_dir, "xlsx") : args["out-dir"]
        push!(inputs, ExportInput(joinpath(inv_dir, "result.npz"),
                                  load_dataset(args["inv-data-dir"], cache), out_dir, "обратная задача"))
    end
    for ex in inputs
        isfile(ex.npz_path) || error("нет файла: $(ex.npz_path)")
        mkpath(ex.out_dir)
        t_path = write_field_xlsx(joinpath(ex.out_dir, "Temperature.xlsx"), ex, :T)
        println("EXPORTED $t_path ($(ex.label))")
        ro_path = write_field_xlsx(joinpath(ex.out_dir, "Sweeney_Burnham_1990_EASY_Ro.xlsx"), ex, :Ro)
        println("EXPORTED $ro_path ($(ex.label))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
