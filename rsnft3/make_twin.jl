# C4: генерация twin-наблюдений. Прямой прогон с θ_true на РАБОЧЕЙ сетке,
# выборка T/Ro в точках реальных скважин (x и глубины из XLSX организаторов,
# точки уже провалидированы по геометрии), добавка шума N(0, σ) (σ=0 → точный
# twin). Ro берётся в тех же глубинах, что и T (формат load_wells_json).
# Выход: JSON + _field.npz с полем twin-прогона.
#
# Запуск:
#   julia --project=. make_twin.jl --data-dir "../Данные для обратной задачи" \
#         --wells "../.../TemperatureWellData.xlsx" \
#         --grid 50,60 --q 75,80,85,90 --sigma-T 0 --sigma-Ro 0 \
#         --seed 42 --out ../data/twin_exact.json

using ArgParse
using Random
using JSON
using NPZ

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))
include(joinpath(@__DIR__, "observations.jl"))
include(joinpath(@__DIR__, "objective.jl"))   # bilinear_sample

function parse_cli()
    s = ArgParseSettings(description = "Twin observations from a forward run with known theta_true")
    @add_arg_table s begin
        "--data-dir" ; required = true
        "--wells"    ; required = true; help = "XLSX defining the well sampling points"
        "--grid"     ; default = "50,60"
        "--q"        ; default = "75,80,85,90"; help = "theta_true, one q per epoch, mW/m2"
        "--sigma-T"  ; arg_type = Float64; default = 0.0
        "--sigma-Ro" ; arg_type = Float64; default = 0.0
        "--seed"     ; arg_type = Int; default = 42
        "--out"      ; required = true
    end
    return parse_args(s)
end

"T/Ro синтетики в точке скважины; NaN, если точка вне осадочной геометрии."
function point_truth(result, data, x::Float64, z::Float64)
    T = bilinear_sample(result.x, result.z, result.temperature_c[end, :, :], x, z)
    le = layer_eta_at(data, x, z)
    Ro = le === nothing ? NaN : particle_ro(result, data, x, le[1], le[2])
    return T, Ro
end

function main()
    args = parse_cli()
    nx, nz = parse.(Int, split(args["grid"], ","))
    q_epochs = parse.(Float64, split(args["q"], ","))
    data = load_dataset_derived(args["data-dir"], Workbook())
    wells = load_wells_xlsx(args["wells"], data; kind = :T, sigma_T = 1.0, sigma_Ro = 1.0)
    layout = make_q_layout(data)
    cfg = SolverConfig(nx = nx, nz = nz)
    length(q_epochs) == length(layout.epochs) ||
        error("--q must have one value per epoch ($(length(layout.epochs)))")

    θ = theta_constant(layout, q_epochs)
    pdata = make_param_data(data, layout, θ)
    result = run_forward(pdata, cfg)
    fin = filter(isfinite, vec(result.temperature_c[end, :, :]))
    println(@sprintf("twin forward done: %.1f s, T_final %.2f..%.2f C",
                     result.elapsed_seconds, minimum(fin), maximum(fin)))
    npzwrite(replace(args["out"], ".json" => "_field.npz"), Dict(
        "x" => result.x, "z" => result.z, "ages_ma" => result.ages_ma,
        "temperature_c" => result.temperature_c))

    Random.seed!(args["seed"])
    docs = []
    for wname in unique(p.well for p in wells.points)
        pts = filter(p -> p.well == wname && p.kind == :T, wells.points)
        isempty(pts) && continue
        x = first(pts).x
        zs = Float64[]; Tobs = Float64[]; Roobs = Float64[]
        for p in pts
            T, Ro = point_truth(result, data, x, p.z)
            (isfinite(T) && isfinite(Ro)) || continue   # вне осадочной геометрии
            push!(zs, p.z)
            push!(Tobs, T + args["sigma-T"] * randn())
            push!(Roobs, Ro + args["sigma-Ro"] * randn())
        end
        isempty(zs) && continue
        doc = OrderedDict{String,Any}(
            "well" => wname, "x" => x, "z" => zs,
            "T_obs" => Tobs, "Ro_obs" => Roobs,
            "sigma_T" => fill(args["sigma-T"], length(zs)),
            "sigma_Ro" => fill(args["sigma-Ro"], length(zs)))
        push!(docs, doc)
    end
    open(args["out"], "w") do f
        JSON.print(f, docs, 2)
    end
    println("SAVED ", args["out"])
end

try
    main()
catch err
    showerror(stderr, err)
    println(stderr)
    exit(1)
end
