# 2-D basin heat-flow forward solver using cell-centred finite volumes.
#
# The solver reads the tournament XLSX files directly.  Geological time is
# traversed from the oldest configuration to the present.  Between two dated
# configurations the geometry is linearly interpolated and the temperature is
# conservatively transported with the layers by a material-coordinate remap.
# Diffusion and radiogenic heating are then advanced with backward Euler and
# Picard iterations for temperature-dependent properties.
#
# Зависимости:
#   using Pkg; Pkg.add(["ArgParse", "DataFrames", "JSON", "NPZ", "OrderedCollections", "XLSX"])
#
# Запуск:
#   julia basin_fvm.jl --data-dir data --cache-dir results --nx 60 --nz 120

using ArgParse
using DataFrames
using JSON
using LinearAlgebra
using NPZ
using OrderedCollections
using Printf
using Serialization
using SHA
using Statistics
using XLSX

const SECONDS_PER_MA = 1.0e6 * 365.25 * 24.0 * 3600.0

# ----------------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------------

Base.@kwdef struct SolverConfig
    nx::Int = 60
    nz::Int = 120
    max_dt_ma::Float64 = 2.5
    picard_iterations::Int = 3
    picard_tolerance_c::Float64 = 1.0e-3
    fluid_lambda::Float64 = 0.60
    fluid_density::Float64 = 1000.0
    fluid_heat_capacity::Float64 = 4180.0
    initial_lambda::Float64 = 2.5
end

function validate(c::SolverConfig)
    (c.nx < 3 || c.nz < 3) && throw(ArgumentError("nx and nz must be at least 3"))
    c.max_dt_ma <= 0 && throw(ArgumentError("max_dt_ma must be positive"))
    c.picard_iterations < 1 && throw(ArgumentError("picard_iterations must be positive"))
    return nothing
end

# Аналог dataclasses.asdict (порядок полей сохраняется)
function asdict(c::SolverConfig)
    return OrderedDict{String,Any}(String(f) => getfield(c, f) for f in fieldnames(SolverConfig))
end

# ----------------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------------

"Линейная интерполяция как np.interp(xq, x, y, left=y[1], right=y[end]); x возрастает."
function interp1(xq::AbstractVector, x::AbstractVector, y::AbstractVector)
    out = Vector{Float64}(undef, length(xq))
    for k in eachindex(xq)
        q = xq[k]
        if isnan(q)
            out[k] = NaN
        elseif q <= x[1]
            out[k] = y[1]
        elseif q >= x[end]
            out[k] = y[end]
        else
            i = searchsortedlast(x, q)
            t = (q - x[i]) / (x[i+1] - x[i])
            out[k] = y[i] + t * (y[i+1] - y[i])
        end
    end
    return out
end

waples_heat_capacity(temperature_k, c20) =
    let tc = temperature_k - 273.15
        c20 * (0.953 + 2.29e-3 * tc - 2.835e-6 * tc^2 + 1.191e-9 * tc^3)
    end

function sekiguchi_conductivity(temperature_k, lambda20)
    t = max(temperature_k, 150.0)
    return 1.84 + 358.0 * (1.0227 * lambda20 - 1.882) * (1.0 / t - 0.00068)
end

harmonic(a, b) = 2.0 * a * b / max(a + b, 1.0e-30)

"Jacobi-preconditioned conjugate gradients for an FVM graph matrix."
function pcg_solve(diagonal::Vector{Float64}, pair_p::Vector{Int}, pair_q::Vector{Int},
                   conductance::Vector{Float64}, rhs::Vector{Float64}, initial::Vector{Float64};
                   tolerance::Float64 = 1.0e-9, max_iterations::Int = 1000)
    function matvec(v::Vector{Float64})
        result = diagonal .* v
        @inbounds for k in eachindex(pair_p)
            p = pair_p[k]; q = pair_q[k]; g = conductance[k]
            result[p] -= g * v[q]
            result[q] -= g * v[p]
        end
        return result
    end

    solution = copy(initial)
    residual = rhs - matvec(solution)
    scale = max(norm(rhs), 1.0)
    relative = norm(residual) / scale
    relative <= tolerance && return solution, 0, relative
    z = residual ./ diagonal
    direction = copy(z)
    rz_old = dot(residual, z)
    for iteration in 1:max_iterations
        applied = matvec(direction)
        denominator = dot(direction, applied)
        denominator <= 0.0 && error("PCG lost positive definiteness")
        alpha = rz_old / denominator
        solution .+= alpha .* direction
        residual .-= alpha .* applied
        relative = norm(residual) / scale
        relative <= tolerance && return solution, iteration, relative
        z = residual ./ diagonal
        rz_new = dot(residual, z)
        direction = z .+ (rz_new / rz_old) .* direction
        rz_old = rz_new
    end
    error(@sprintf("PCG did not converge in %d iterations (residual=%.3e)", max_iterations, relative))
end

# ----------------------------------------------------------------------------
# Структуры данных
# ----------------------------------------------------------------------------

struct Record
    x::Vector{Float64}
    top::Vector{Float64}
    bot::Vector{Float64}
    value::Vector{Float64}
end

const Workbook = OrderedDict{String,Dict{Float64,Record}}

struct LithProp
    name::String
    rho_s::Float64
    lambda20::Float64
    c20::Float64
    As::Float64
end

struct Layer
    top::Vector{Float64}
    bot::Vector{Float64}
    lith::Vector{Int}
    phi::Vector{Float64}
end

struct State
    age::Float64
    layers::Dict{String,Layer}
    surface::Vector{Float64}
    base::Vector{Float64}
    tsurf::Vector{Float64}
    qbasal::Vector{Float64}
end

struct Grid
    active::BitMatrix
    layer_id::Matrix{Int}
    lith::Matrix{Int}
    phi::Matrix{Float64}
    names::Vector{String}
end

# ----------------------------------------------------------------------------
# Чтение входных данных
# ----------------------------------------------------------------------------

tofloat(v) = (ismissing(v) || v === nothing) ? NaN : Float64(v)

function find_column(df::DataFrame, prefix::AbstractString)
    for col in names(df)
        startswith(String(col), prefix) && return String(col)
    end
    throw(KeyError("No column starting with \"$prefix\""))
end

function series_from_frame(df::DataFrame)
    value_col = find_column(df, "Aver")
    time = tofloat.(df[!, "Time_Ma"])
    X    = tofloat.(df[!, "X"])
    top  = tofloat.(df[!, "Z_Top [m]"])
    bot  = tofloat.(df[!, "Z_Bot [m]"])
    val  = tofloat.(df[!, value_col])

    groups = Dict{Float64,Vector{Int}}()
    for k in eachindex(time)
        isnan(time[k]) && continue
        push!(get!(groups, time[k], Int[]), k)
    end

    result = Dict{Float64,Record}()
    for (age, rows) in groups
        rows = sort(rows; by = k -> X[k], alg = MergeSort)   # sort_values("X")
        keep = Int[]                                         # drop_duplicates("X", keep="last")
        for (n, k) in enumerate(rows)
            if n == length(rows) || X[rows[n+1]] != X[k]
                push!(keep, k)
            end
        end
        result[age] = Record(X[keep], top[keep], bot[keep], val[keep])
    end
    return result
end

function read_workbook(path::AbstractString)
    xf = XLSX.readxlsx(path)
    book = Workbook()
    for sheet in XLSX.sheetnames(xf)
        df = DataFrame(XLSX.readtable(path, sheet))
        book[sheet] = series_from_frame(df)
    end
    return book
end

const REQUIRED = (
    "Depth.xlsx",
    "Lithology.xlsx",
    "Porosity.xlsx",
    "Tsurf.xlsx",
    "lithotype_properties.xlsx",
)

# Heat_Flow_Basal.xlsx опционален: в постановке обратной задачи 2026 q_basal(x,t) —
# неизвестный, он инжектируется в памяти (parameterized_data.jl), а не читается из файла.
const OPTIONAL = ("Heat_Flow_Basal.xlsx",)

struct BasinDataset
    data_dir::String
    fingerprint::String
    depth::Workbook
    lith::Workbook
    porosity::Workbook
    heat::Workbook
    tsurf_data::Workbook
    properties::Dict{Int,LithProp}
    ages::Vector{Float64}
    bounds::NTuple{4,Float64}
end

function input_fingerprint(data_dir::AbstractString)
    ctx = SHA.SHA256_CTX()
    for name in REQUIRED
        st = stat(joinpath(data_dir, name))
        mtime_ns = round(Int, st.mtime * 1.0e9)
        SHA.update!(ctx, Vector{UInt8}(codeunits("$name:$(st.size):$mtime_ns")))
    end
    return bytes2hex(SHA.digest!(ctx))[1:16]
end

function read_all(data_dir::AbstractString)
    depth    = read_workbook(joinpath(data_dir, "Depth.xlsx"))
    lith     = read_workbook(joinpath(data_dir, "Lithology.xlsx"))
    porosity = read_workbook(joinpath(data_dir, "Porosity.xlsx"))
    heat_path = joinpath(data_dir, "Heat_Flow_Basal.xlsx")
    heat     = isfile(heat_path) ? read_workbook(heat_path) : Workbook()
    tsurf    = read_workbook(joinpath(data_dir, "Tsurf.xlsx"))

    prop = DataFrame(XLSX.readtable(joinpath(data_dir, "lithotype_properties.xlsx"), 1))
    c_id, c_name = find_column(prop, "ID"), find_column(prop, "Name")
    c_rho, c_lam = find_column(prop, "ρs"), find_column(prop, "λ20")
    c_c20, c_as  = find_column(prop, "c20"), find_column(prop, "As")
    properties = Dict{Int,LithProp}()
    for r in 1:nrow(prop)
        properties[Int(round(tofloat(prop[r, c_id])))] = LithProp(
            string(prop[r, c_name]),
            tofloat(prop[r, c_rho]),
            tofloat(prop[r, c_lam]),
            tofloat(prop[r, c_c20]),
            tofloat(prop[r, c_as]) * 1.0e-6,
        )
    end

    ages = sort(collect(Set(age for layer in values(depth) for age in keys(layer))); rev = true)
    recs = [rec for layer in values(depth) for rec in values(layer)]
    bounds = (
        minimum(minimum(r.x) for r in recs),
        maximum(maximum(r.x) for r in recs),
        minimum(minimum(r.top) for r in recs),
        maximum(maximum(r.bot) for r in recs),
    )
    return (; depth, lith, porosity, heat, tsurf_data = tsurf, properties, ages, bounds)
end

function load_dataset(data_dir::AbstractString, cache_dir::AbstractString)
    data_dir = abspath(data_dir)
    missing_files = [n for n in REQUIRED if !isfile(joinpath(data_dir, n))]
    isempty(missing_files) ||
        throw(ErrorException("Missing input files: " * join(missing_files, ", ")))
    fingerprint = input_fingerprint(data_dir)
    input_cache = joinpath(cache_dir, "_input_cache")
    mkpath(input_cache)
    cache_file = joinpath(input_cache, fingerprint * ".jls")

    payload = nothing
    if isfile(cache_file)
        try
            payload = open(deserialize, cache_file)
        catch
            payload = nothing   # кэш от другой версии Julia/кода — читаем заново
        end
    end
    if payload === nothing
        payload = read_all(data_dir)
        open(io -> serialize(io, payload), cache_file, "w")
    end
    return BasinDataset(data_dir, fingerprint, payload.depth, payload.lith, payload.porosity,
                        payload.heat, payload.tsurf_data, payload.properties,
                        payload.ages, payload.bounds)
end

interp_record(rec::Record, x::AbstractVector, key::Symbol) =
    interp1(x, rec.x, getfield(rec, key))

function boundary_value(workbook::Workbook, age::Float64, x::AbstractVector)
    candidates = Vector{Vector{Float64}}()
    for layer in values(workbook)
        haskey(layer, age) && push!(candidates, interp_record(layer[age], x, :value))
    end
    isempty(candidates) && throw(ErrorException("No boundary data at age $age Ma"))
    out = Vector{Float64}(undef, length(x))
    for i in eachindex(x)                       # np.nanmedian
        vals = [c[i] for c in candidates if !isnan(c[i])]
        out[i] = isempty(vals) ? NaN : median(vals)
    end
    return out
end

function exact_state(data::BasinDataset, age::Float64, x::AbstractVector)
    layers = Dict{String,Layer}()
    for (name, depth_series) in data.depth
        haskey(depth_series, age) || continue
        d   = depth_series[age]
        lit = data.lith[name][age]
        phi = data.porosity[name][age]
        layers[name] = Layer(
            interp_record(d, x, :top),
            interp_record(d, x, :bot),
            Int.(floor.(interp_record(lit, x, :value) .+ 1.0e-8)),
            clamp.(interp_record(phi, x, :value) ./ 100.0, 0.0, 0.95),
        )
    end
    isempty(layers) && throw(ErrorException("No geometry at age $age Ma"))
    surface = fill(Inf, length(x))
    base = fill(-Inf, length(x))
    for l in values(layers)
        surface .= min.(surface, l.top)
        base .= max.(base, l.bot)
    end
    return State(
        age, layers, surface, base,
        boundary_value(data.tsurf_data, age, x),
        boundary_value(data.heat, age, x) .* 1.0e-3,
    )
end

# ----------------------------------------------------------------------------
# Геометрия и перенос
# ----------------------------------------------------------------------------

function interpolate_state(old::State, young::State, fraction::Real)
    f = clamp(Float64(fraction), 0.0, 1.0)
    layers = Dict{String,Layer}()
    for name in sort!(collect(union(keys(old.layers), keys(young.layers))))
        a = get(old.layers, name, nothing)
        b = get(young.layers, name, nothing)
        if a === nothing
            a = Layer(old.surface, old.surface, b.lith, b.phi)
        elseif b === nothing
            b = Layer(young.surface, young.surface, a.lith, a.phi)
        end
        layers[name] = Layer(
            (1.0 - f) .* a.top .+ f .* b.top,
            (1.0 - f) .* a.bot .+ f .* b.bot,
            f < 0.5 ? copy(a.lith) : copy(b.lith),
            (1.0 - f) .* a.phi .+ f .* b.phi,
        )
    end
    return State(
        (1.0 - f) * old.age + f * young.age,
        layers,
        (1.0 - f) .* old.surface .+ f .* young.surface,
        (1.0 - f) .* old.base .+ f .* young.base,
        (1.0 - f) .* old.tsurf .+ f .* young.tsurf,
        (1.0 - f) .* old.qbasal .+ f .* young.qbasal,
    )
end

function rasterize(state::State, z::AbstractVector)
    nz, nx = length(z), length(state.surface)
    layer_id = zeros(Int, nz, nx)          # 0 = неактивная ячейка
    lith = fill(-1, nz, nx)
    phi = fill(NaN, nz, nx)
    names = sort!(collect(keys(state.layers)))
    for (layer_number, name) in enumerate(names)
        item = state.layers[name]
        for i in 1:nx
            lo = min(item.top[i], item.bot[i])
            hi = max(item.top[i], item.bot[i])
            (hi - lo) > 1.0e-8 || continue
            for j in 1:nz
                if z[j] >= lo && z[j] <= hi
                    layer_id[j, i] = layer_number
                    lith[j, i] = item.lith[i]
                    phi[j, i] = item.phi[i]
                end
            end
        end
    end
    return Grid(layer_id .> 0, layer_id, lith, phi, names)
end

function material_remap(temperature::Matrix{Float64}, old_state::State,
                        new_state::State, z::AbstractVector)
    old_grid = rasterize(old_state, z)
    new_grid = rasterize(new_state, z)
    nz, nx = size(temperature)
    out = fill(NaN, nz, nx)
    names = sort!(collect(keys(new_state.layers)))   # порядок фиксирован для воспроизводимости
    for i in 1:nx
        valid_column = [old_grid.active[j, i] && isfinite(temperature[j, i]) for j in 1:nz]
        if !any(valid_column)
            out[new_grid.active[:, i], i] .= new_state.tsurf[i]
            continue
        end
        z_valid = z[valid_column]
        t_valid = temperature[valid_column, i]
        for name in names
            new_layer = new_state.layers[name]
            lo_new, hi_new = minmax(new_layer.top[i], new_layer.bot[i])
            cells = (z .>= lo_new) .& (z .<= hi_new) .& new_grid.active[:, i]
            any(cells) || continue
            old_layer = get(old_state.layers, name, nothing)
            if old_layer === nothing || abs(old_layer.bot[i] - old_layer.top[i]) < 1.0e-8
                out[cells, i] .= new_state.tsurf[i]
                continue
            end
            thickness_new = max(hi_new - lo_new, 1.0e-12)
            eta = clamp.((z[cells] .- lo_new) ./ thickness_new, 0.0, 1.0)
            lo_old, hi_old = minmax(old_layer.top[i], old_layer.bot[i])
            material_z = lo_old .+ eta .* (hi_old - lo_old)
            out[cells, i] = interp1(material_z, z_valid, t_valid)
        end
    end
    for i in 1:nx, j in 1:nz
        if new_grid.active[j, i] && !isfinite(out[j, i])
            out[j, i] = new_state.tsurf[i]
        end
    end
    return out
end

# ----------------------------------------------------------------------------
# Тепловые свойства и неявный шаг МКО
# ----------------------------------------------------------------------------

function thermal_properties(temperature_c::Matrix{Float64}, grid::Grid,
                            lith_properties::Dict{Int,LithProp}, config::SolverConfig)
    lam = fill(NaN, size(temperature_c))
    rho_c = fill(NaN, size(temperature_c))
    source = fill(NaN, size(temperature_c))
    for idx in eachindex(temperature_c)
        grid.active[idx] || continue
        code = grid.lith[idx]
        haskey(lith_properties, code) ||
            throw(KeyError("Unknown lithotype code $code; available: $(sort(collect(keys(lith_properties))))"))
        prop = lith_properties[code]
        tk = temperature_c[idx] + 273.15
        phi = grid.phi[idx]
        lam_s = sekiguchi_conductivity(tk, prop.lambda20)
        c_s = waples_heat_capacity(tk, prop.c20)
        lam[idx] = lam_s^(1.0 - phi) * config.fluid_lambda^phi
        rho_c[idx] = (1.0 - phi) * prop.rho_s * c_s +
                     phi * config.fluid_density * config.fluid_heat_capacity
        source[idx] = (1.0 - phi) * prop.As
    end
    return lam, rho_c, source
end

function implicit_fvm_step(transported_c::Matrix{Float64}, state::State, z::AbstractVector,
                           dx::Float64, dz::Float64, dt_seconds::Float64,
                           lith_properties::Dict{Int,LithProp}, config::SolverConfig)
    grid = rasterize(state, z)
    active = grid.active
    nz, nx = size(active)
    ncell = count(active)
    indices = zeros(Int, nz, nx)
    indices[active] = 1:ncell
    trial = copy(transported_c)
    max_change = Inf
    linear_iterations = 0
    linear_residual = Inf
    picard_used = 0

    for iteration in 1:config.picard_iterations
        picard_used = iteration
        lam, rho_c, source = thermal_properties(trial, grid, lith_properties, config)
        volume = dx * dz
        diagonal = rho_c[active] .* volume ./ dt_seconds
        rhs = diagonal .* transported_c[active] .+ source[active] .* volume
        pair_p = Int[]
        pair_q = Int[]
        pair_g = Float64[]

        for idx in CartesianIndices(active)
            j, i = Tuple(idx)
            active[j, i] || continue
            p = indices[j, i]

            # Each internal face is inserted once, from its west/north cell.
            for (dj, di, area, distance) in ((0, 1, dz, dx), (1, 0, dx, dz))
                jj, ii = j + dj, i + di
                if 1 <= jj <= nz && 1 <= ii <= nx && active[jj, ii]
                    conductance = harmonic(lam[j, i], lam[jj, ii]) * area / distance
                    q = indices[jj, ii]
                    diagonal[p] += conductance
                    diagonal[q] += conductance
                    push!(pair_p, p); push!(pair_q, q); push!(pair_g, conductance)
                end
            end

            # Physical top/bottom boundaries on the stair-step domain.
            for dj in (-1, 1)
                jj = j + dj
                if !(1 <= jj <= nz) || !active[jj, i]
                    face_z = z[j] + 0.5 * dj * dz
                    surface_distance = abs(face_z - state.surface[i])
                    base_distance = abs(face_z - state.base[i])
                    if surface_distance <= base_distance
                        conductance = lam[j, i] * dx / (0.5 * dz)
                        diagonal[p] += conductance
                        rhs[p] += conductance * state.tsurf[i]
                    else
                        rhs[p] += state.qbasal[i] * dx
                    end
                end
            end
        end

        solution, linear_iterations, linear_residual = pcg_solve(
            diagonal, pair_p, pair_q, pair_g, rhs, trial[active];
            tolerance = 1.0e-8, max_iterations = max(2000, 8 * config.nx * config.nz),
        )
        updated = fill(NaN, nz, nx)
        updated[active] = solution
        max_change = -Inf                      # np.nanmax(np.abs(updated - trial))
        for k in eachindex(updated)
            d = abs(updated[k] - trial[k])
            isnan(d) || (max_change = max(max_change, d))
        end
        trial = updated
        max_change <= config.picard_tolerance_c && break
    end

    return trial, OrderedDict{String,Any}(
        "picard_iterations" => picard_used,
        "picard_change_c" => max_change,
        "linear_iterations" => linear_iterations,
        "linear_relative_residual" => linear_residual,
    )
end

function initialise_temperature(state::State, z::AbstractVector, config::SolverConfig)
    grid = rasterize(state, z)
    nz, nx = length(z), length(state.surface)
    temperature = fill(NaN, nz, nx)
    for i in 1:nx, j in 1:nz
        grid.active[j, i] || continue
        depth = max(z[j] - state.surface[i], 0.0)
        temperature[j, i] = state.tsurf[i] + state.qbasal[i] * depth / config.initial_lambda
    end
    return temperature
end

# ----------------------------------------------------------------------------
# Прямая задача
# ----------------------------------------------------------------------------

function run_forward(data::BasinDataset, config::SolverConfig)
    validate(config)
    x0, x1, z0, z1 = data.bounds
    dx = (x1 - x0) / config.nx
    dz = (z1 - z0) / config.nz
    x = collect(range(x0 + 0.5 * dx, x1 - 0.5 * dx; length = config.nx))
    z = collect(range(z0 + 0.5 * dz, z1 - 0.5 * dz; length = config.nz))
    exact = Dict(age => exact_state(data, age, x) for age in data.ages)
    temperature = initialise_temperature(exact[data.ages[1]], z, config)
    saved_temperature = [copy(temperature)]
    diagnostics = OrderedDict{String,Any}[]

    start = time_ns()
    for (age_old, age_young) in zip(data.ages[1:end-1], data.ages[2:end])
        span = age_old - age_young
        n_substeps = max(1, Int(ceil(span / config.max_dt_ma)))
        previous_state = exact[age_old]
        for substep in 1:n_substeps
            fraction = substep / n_substeps
            state = interpolate_state(exact[age_old], exact[age_young], fraction)
            transported = material_remap(temperature, previous_state, state, z)
            dt_seconds = (span / n_substeps) * SECONDS_PER_MA
            temperature, diag = implicit_fvm_step(
                transported, state, z, dx, dz, dt_seconds, data.properties, config)
            push!(diagnostics, merge(OrderedDict{String,Any}("age_ma" => state.age), diag))
            previous_state = state
        end
        push!(saved_temperature, copy(temperature))
    end

    final_state = exact[data.ages[end]]
    final_grid = rasterize(final_state, z)
    elapsed = (time_ns() - start) / 1.0e9

    # (время, z, x) — та же раскладка осей, что у NumPy-версии
    nt = length(saved_temperature)
    temperature_c = Array{Float64,3}(undef, nt, config.nz, config.nx)
    for k in 1:nt
        temperature_c[k, :, :] = saved_temperature[k]
    end

    return (
        x = x, z = z, ages_ma = copy(data.ages),
        temperature_c = temperature_c,
        final_lithology = final_grid.lith,
        final_porosity = final_grid.phi,
        final_surface_m = final_state.surface,
        final_base_m = final_state.base,
        diagnostics = diagnostics,
        elapsed_seconds = elapsed,
        dx = dx, dz = dz,
    )
end

# ----------------------------------------------------------------------------
# Кэш результатов
# ----------------------------------------------------------------------------

function run_id_for(data::BasinDataset, config::SolverConfig)
    solver = OrderedDict(sort!(collect(asdict(config)); by = first))
    payload = OrderedDict{String,Any}(
        "input" => data.fingerprint,
        "solver" => solver,
        "version" => 1,
    )
    return bytes2hex(sha256(JSON.json(payload)))[1:16]
end

function save_result(result, run_dir::AbstractString, metadata)
    mkpath(run_dir)
    npzwrite(joinpath(run_dir, "result.npz"), Dict(
        "x" => result.x, "z" => result.z, "ages_ma" => result.ages_ma,
        "temperature_c" => result.temperature_c,
        "final_lithology" => result.final_lithology,
        "final_porosity" => result.final_porosity,
        "final_surface_m" => result.final_surface_m,
        "final_base_m" => result.final_base_m,
    ))
    write(joinpath(run_dir, "metadata.json"), sprint(io -> JSON.print(io, metadata, 2)))
end

function update_manifest(cache_dir::AbstractString, metadata)
    manifest_path = joinpath(cache_dir, "manifest.json")
    if isfile(manifest_path)
        manifest = JSON.parsefile(manifest_path; dicttype = OrderedDict{String,Any})
    else
        manifest = OrderedDict{String,Any}("runs" => OrderedDict{String,Any}())
    end
    manifest["runs"][metadata["run_id"]] = metadata
    write(manifest_path, sprint(io -> JSON.print(io, manifest, 2)))
end

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

function main(argv = ARGS)
    s = ArgParseSettings(description = "2-D basin finite-volume forward solver")
    @add_arg_table! s begin
        "--data-dir"
            arg_type = String
            required = true
        "--cache-dir"
            arg_type = String
            default = "results"
        "--nx"
            arg_type = Int
            default = 60
        "--nz"
            arg_type = Int
            default = 120
        "--max-dt-ma"
            arg_type = Float64
            default = 2.5
        "--picard-iterations"
            arg_type = Int
            default = 3
        "--force"
            action = :store_true
    end
    args = parse_args(argv, s)

    cache_dir = args["cache-dir"]
    mkpath(cache_dir)
    config = SolverConfig(
        nx = args["nx"], nz = args["nz"], max_dt_ma = args["max-dt-ma"],
        picard_iterations = args["picard-iterations"],
    )
    data = load_dataset(args["data-dir"], cache_dir)
    run_id = run_id_for(data, config)
    run_dir = joinpath(cache_dir, run_id)
    if isfile(joinpath(run_dir, "result.npz")) && !args["force"]
        println("CACHE_HIT $run_id $(joinpath(run_dir, "result.npz"))")
        return
    end

    println("RUN $run_id: grid=$(config.nx)x$(config.nz), dt<=$(config.max_dt_ma) Ma")
    result = run_forward(data, config)
    last_field = result.temperature_c[end, :, :]
    finite = filter(isfinite, vec(last_field))
    metadata = OrderedDict{String,Any}(
        "run_id" => run_id,
        "status" => "complete",
        "input_directory" => data.data_dir,
        "input_fingerprint" => data.fingerprint,
        "solver" => asdict(config),
        "elapsed_seconds" => result.elapsed_seconds,
        "dx_m" => result.dx, "dz_m" => result.dz,
        "final_temperature_min_c" => minimum(finite),
        "final_temperature_max_c" => maximum(finite),
        "assumptions" => [
            "Initial temperature is a linear conductive profile based on the oldest surface temperature and basal heat flow.",
            "Fluid properties are constant and configurable.",
            "Layer motion is represented by material-coordinate remapping between interpolated geometries.",
            "EASY%Ro is not computed because kinetic constants are absent from the supplied files.",
        ],
        "diagnostics" => result.diagnostics,
    )
    save_result(result, run_dir, metadata)
    update_manifest(cache_dir, metadata)
    println("SAVED $(joinpath(run_dir, "result.npz"))")
    println(@sprintf("T_FINAL %.3f %.3f C", minimum(finite), maximum(finite)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
