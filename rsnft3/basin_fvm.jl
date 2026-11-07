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
using SparseArrays
using NPZ
using OrderedCollections
using Printf
using Serialization
using SHA
using Statistics

include(joinpath(@__DIR__, "progress.jl"))
using .Progress
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
    pcg_tolerance::Float64 = 1.0e-6
    linear_solver::Symbol = :pcg
end

function validate(c::SolverConfig)
    (c.nx < 3 || c.nz < 3) && throw(ArgumentError("nx and nz must be at least 3"))
    c.max_dt_ma <= 0 && throw(ArgumentError("max_dt_ma must be positive"))
    c.picard_iterations < 1 && throw(ArgumentError("picard_iterations must be positive"))
    c.pcg_tolerance <= 0 && throw(ArgumentError("pcg_tolerance must be positive"))
    c.linear_solver in (:pcg, :direct) ||
        throw(ArgumentError("linear_solver must be :pcg or :direct"))
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

"Jacobi-preconditioned conjugate gradients for an FVM graph matrix (preallocated buffers)."
function pcg_solve!(solution::Vector{Float64}, residual::Vector{Float64},
                    z::Vector{Float64}, direction::Vector{Float64}, applied::Vector{Float64},
                    diagonal::Vector{Float64}, pair_p::Vector{Int}, pair_q::Vector{Int},
                    conductance::Vector{Float64}, rhs::Vector{Float64}, initial::Vector{Float64};
                    tolerance::Float64 = 1.0e-9, max_iterations::Int = 1000)
    n = length(diagonal)
    copyto!(solution, initial)
    @inbounds @simd for i in 1:n
        residual[i] = diagonal[i] * solution[i]
    end
    @inbounds for k in eachindex(pair_p)
        p = pair_p[k]; q = pair_q[k]; g = conductance[k]
        residual[p] -= g * solution[q]
        residual[q] -= g * solution[p]
    end
    @inbounds @simd for i in 1:n
        residual[i] = rhs[i] - residual[i]
    end
    scale = max(norm(rhs), 1.0)
    relative = norm(residual) / scale
    relative <= tolerance && return 0, relative
    @inbounds @simd for i in 1:n
        z[i] = residual[i] / diagonal[i]
    end
    copyto!(direction, z)
    rz_old = dot(residual, z)
    for iteration in 1:max_iterations
        @inbounds @simd for i in 1:n
            applied[i] = diagonal[i] * direction[i]
        end
        @inbounds for k in eachindex(pair_p)
            p = pair_p[k]; q = pair_q[k]; g = conductance[k]
            applied[p] -= g * direction[q]
            applied[q] -= g * direction[p]
        end
        denominator = dot(direction, applied)
        denominator <= 0.0 && error("PCG lost positive definiteness")
        alpha = rz_old / denominator
        @inbounds @simd for i in 1:n
            solution[i] += alpha * direction[i]
            residual[i] -= alpha * applied[i]
        end
        relative = norm(residual) / scale
        relative <= tolerance && return iteration, relative
        @inbounds @simd for i in 1:n
            z[i] = residual[i] / diagonal[i]
        end
        rz_new = dot(residual, z)
        beta = rz_new / rz_old
        @inbounds @simd for i in 1:n
            direction[i] = z[i] + beta * direction[i]
        end
        rz_old = rz_new
    end
    error(@sprintf("PCG did not converge in %d iterations (residual=%.3e)", max_iterations, relative))
end

function pcg_solve(diagonal::Vector{Float64}, pair_p::Vector{Int}, pair_q::Vector{Int},
                   conductance::Vector{Float64}, rhs::Vector{Float64}, initial::Vector{Float64};
                   tolerance::Float64 = 1.0e-9, max_iterations::Int = 1000)
    n = length(diagonal)
    solution = Vector{Float64}(undef, n)
    iterations, relative = pcg_solve!(solution, Vector{Float64}(undef, n),
                                      Vector{Float64}(undef, n), Vector{Float64}(undef, n),
                                      Vector{Float64}(undef, n),
                                      diagonal, pair_p, pair_q, conductance, rhs, initial;
                                      tolerance = tolerance, max_iterations = max_iterations)
    return solution, iterations, relative
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
"Предвычисленные массивы свойств литологий (code -> scalar)."
struct LithoPropLookup
    codes::Vector{Int}
    lambda20::Vector{Float64}
    c20::Vector{Float64}
    rho_s::Vector{Float64}
    As::Vector{Float64}
    by_code::Dict{Int,Int}   # code -> index
end

function LithoPropLookup(props::Dict{Int,LithProp})
    codes = collect(keys(props))
    lambda20 = Float64[props[c].lambda20 for c in codes]
    c20      = Float64[props[c].c20      for c in codes]
    rho_s    = Float64[props[c].rho_s    for c in codes]
    As       = Float64[props[c].As       for c in codes]
    by_code = Dict{Int,Int}(c => i for (i, c) in enumerate(codes))
    return LithoPropLookup(codes, lambda20, c20, rho_s, As, by_code)
end

@inline function lookup_index(l::LithoPropLookup, code::Int)
    idx = get(l.by_code, code, 0)
    idx == 0 && throw(KeyError("Unknown lithotype code $code; available: $(sort(l.codes))"))
    return idx
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

"Построенный один раз на (grid, state) стенсил FVM: компактные индексы, пары ячеек,
геометрия граней, граничные ячеи. Не зависит от температуры и qbasal (кроме
значений tsurf, зафиксированных при построении)."
struct Stencil
    ncell::Int
    cell_lin::Vector{Int}
    cell_col::Vector{Int}
    cell_tsurf::Vector{Float64}
    pair_first::Vector{Int}
    pair_last::Vector{Int}
    pair_p::Vector{Int}
    pair_q::Vector{Int}
    pair_plin::Vector{Int}
    pair_qlin::Vector{Int}
    pair_area::Vector{Float64}
    pair_dist::Vector{Float64}
    pair_g::Vector{Float64}
    btop::Vector{Int8}
    bbot::Vector{Int8}
    east_k::Vector{Int}
    north_k::Vector{Int}
    west_k::Vector{Int}
    south_k::Vector{Int}
    idx_west::Vector{Int}
    idx_south::Vector{Int}
    idx_north::Vector{Int}
    idx_east::Vector{Int}
    idx_diag::Vector{Int}
    colptr::Vector{Int}
    rowval::Vector{Int}
    dx::Float64
    dz::Float64
end

"Кеш символических факторизаций CHOLMOD по объекту стенсила (паттерн матрицы
фиксирован для данного стенсила; численная факторизация обновляется cholesky!)."
const CHOLMOD_FACTOR_CACHE = Base.IdDict{Stencil, Any}()

"Переиспользуемое рабочее пространство шага: свойства, диагональ, правая часть,
буферы PCG и double-buffer полей температуры (NaN в неактивных ячейках)."
mutable struct FvmWorkspace
    nz::Int
    nx::Int
    ncell::Int
    lam::Matrix{Float64}
    rho_c::Matrix{Float64}
    source::Matrix{Float64}
    diag::Vector{Float64}
    rhs::Vector{Float64}
    sol::Vector{Float64}
    residual::Vector{Float64}
    zvec::Vector{Float64}
    direction::Vector{Float64}
    applied::Vector{Float64}
    field_a::Matrix{Float64}
    field_b::Matrix{Float64}
end

function FvmWorkspace(nz::Int, nx::Int)
    ncell = nz * nx
    return FvmWorkspace(nz, nx, ncell,
        fill(NaN, nz, nx), fill(NaN, nz, nx), fill(NaN, nz, nx),
        Vector{Float64}(undef, ncell), Vector{Float64}(undef, ncell),
        Vector{Float64}(undef, ncell), Vector{Float64}(undef, ncell),
        Vector{Float64}(undef, ncell), Vector{Float64}(undef, ncell),
        Vector{Float64}(undef, ncell),
        fill(NaN, nz, nx), fill(NaN, nz, nx))
end

function build_stencil(grid::Grid, state::State, z::AbstractVector, dx::Float64, dz::Float64)
    active = grid.active
    nz, nx = size(active)
    ncell = count(active)
    cell_lin = Vector{Int}(undef, ncell)
    cell_col = Vector{Int}(undef, ncell)
    cell_tsurf = Vector{Float64}(undef, ncell)
    pair_first = Vector{Int}(undef, ncell)
    pair_last = Vector{Int}(undef, ncell)
    btop = Vector{Int8}(undef, ncell)
    bbot = Vector{Int8}(undef, ncell)
    index = zeros(Int, nz, nx)
    pos = 0
    for i in 1:nx, j in 1:nz
        active[j, i] || continue
        pos += 1
        index[j, i] = pos
        cell_lin[pos] = j + (i - 1) * nz
        cell_col[pos] = i
        cell_tsurf[pos] = state.tsurf[i]
    end
    pair_p = Int[]; pair_q = Int[]
    pair_plin = Int[]; pair_qlin = Int[]
    pair_area = Float64[]; pair_dist = Float64[]
    sizehint!(pair_p, 2 * ncell); sizehint!(pair_q, 2 * ncell)
    sizehint!(pair_plin, 2 * ncell); sizehint!(pair_qlin, 2 * ncell)
    sizehint!(pair_area, 2 * ncell); sizehint!(pair_dist, 2 * ncell)
    for i in 1:nx, j in 1:nz
        active[j, i] || continue
        p = index[j, i]
        lin = j + (i - 1) * nz
        pair_first[p] = length(pair_p) + 1

        for (dj, di, area, distance) in ((0, 1, dz, dx), (1, 0, dx, dz))
            jj, ii = j + dj, i + di
            if 1 <= jj <= nz && 1 <= ii <= nx && active[jj, ii]
                push!(pair_p, p); push!(pair_q, index[jj, ii])
                push!(pair_plin, lin); push!(pair_qlin, jj + (ii - 1) * nz)
                push!(pair_area, area); push!(pair_dist, distance)
            end
        end
        pair_last[p] = length(pair_p)

        btop[p] = 0; bbot[p] = 0
        for dj in (-1, 1)
            jj = j + dj
            (1 <= jj <= nz && active[jj, i]) && continue
            face_z = z[j] + 0.5 * dj * dz
            kind = abs(face_z - state.surface[i]) <= abs(face_z - state.base[i]) ? Int8(1) : Int8(2)
            dj == -1 ? (btop[p] = kind) : (bbot[p] = kind)
        end
    end
    east_k = zeros(Int, ncell); north_k = zeros(Int, ncell)
    west_k = zeros(Int, ncell); south_k = zeros(Int, ncell)
    @inbounds for k in eachindex(pair_p)
        p = pair_p[k]; q = pair_q[k]
        if q == p + 1
            north_k[p] = k; south_k[q] = k
        else
            east_k[p] = k; west_k[q] = k
        end
    end
    # CSC-паттерн: в столбце c — соседи снизу (запад, юг), диагональ, соседи сверху (север, восток).
    colptr = Vector{Int}(undef, ncell + 1)
    colptr[1] = 1
    @inbounds for c in 1:ncell
        colptr[c + 1] = colptr[c] + (west_k[c] > 0) + (south_k[c] > 0) + 1 +
                        (north_k[c] > 0) + (east_k[c] > 0)
    end
    nnz = colptr[end] - 1
    rowval = Vector{Int}(undef, nnz)
    idx_west = zeros(Int, ncell); idx_south = zeros(Int, ncell)
    idx_north = zeros(Int, ncell); idx_east = zeros(Int, ncell)
    idx_diag = zeros(Int, ncell)
    @inbounds for c in 1:ncell
        pos = colptr[c]
        if west_k[c] > 0
            idx_west[c] = pos; rowval[pos] = pair_p[west_k[c]]; pos += 1
        end
        if south_k[c] > 0
            idx_south[c] = pos; rowval[pos] = pair_p[south_k[c]]; pos += 1
        end
        idx_diag[c] = pos; rowval[pos] = c; pos += 1
        if north_k[c] > 0
            idx_north[c] = pos; rowval[pos] = pair_q[north_k[c]]; pos += 1
        end
        if east_k[c] > 0
            idx_east[c] = pos; rowval[pos] = pair_q[east_k[c]]; pos += 1
        end
    end
    return Stencil(ncell, cell_lin, cell_col, cell_tsurf,
                   pair_first, pair_last,
                   pair_p, pair_q, pair_plin, pair_qlin, pair_area, pair_dist,
                   similar(pair_area),
                   btop, bbot,
                   east_k, north_k, west_k, south_k,
                   idx_west, idx_south, idx_north, idx_east, idx_diag,
                   colptr, rowval, dx, dz)
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

function exact_state(data::BasinDataset, age::Float64, x::AbstractVector; with_heat::Bool = true)
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
    q = with_heat ? boundary_value(data.heat, age, x) .* 1.0e-3 : zeros(length(x))
    return State(age, layers, surface, base, boundary_value(data.tsurf_data, age, x), q)
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
                        new_state::State, z::AbstractVector;
                        old_grid::Union{Nothing,Grid} = nothing,
                        new_grid::Union{Nothing,Grid} = nothing)
    old_grid === nothing && (old_grid = rasterize(old_state, z))
    new_grid === nothing && (new_grid = rasterize(new_state, z))
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
    thermal_properties!(lam, rho_c, source, temperature_c, grid, lith_properties, config)
    return lam, rho_c, source
end

function thermal_properties!(lam::Matrix{Float64}, rho_c::Matrix{Float64},
                             source::Matrix{Float64}, temperature_c::Matrix{Float64},
                             grid::Grid, lookup::LithoPropLookup, config::SolverConfig)
    fill!(lam, NaN); fill!(rho_c, NaN); fill!(source, NaN)
    @inbounds for idx in eachindex(temperature_c)
        grid.active[idx] || continue
        li = lookup_index(lookup, grid.lith[idx])
        tk = temperature_c[idx] + 273.15
        phi = grid.phi[idx]
        lam_s = sekiguchi_conductivity(tk, lookup.lambda20[li])
        c_s = waples_heat_capacity(tk, lookup.c20[li])
        lam[idx] = lam_s^(1.0 - phi) * config.fluid_lambda^phi
        rho_c[idx] = (1.0 - phi) * lookup.rho_s[li] * c_s +
                     phi * config.fluid_density * config.fluid_heat_capacity
        source[idx] = (1.0 - phi) * lookup.As[li]
    end
    return lam, rho_c, source
end

function thermal_properties!(lam::Matrix{Float64}, rho_c::Matrix{Float64},
                             source::Matrix{Float64}, temperature_c::Matrix{Float64},
                             grid::Grid, lith_properties::Dict{Int,LithProp}, config::SolverConfig)
    return thermal_properties!(lam, rho_c, source, temperature_c, grid,
                               LithoPropLookup(lith_properties), config)
end

"""7.2: Picard-цикл с переиспользуемым workspace.
assemble!(trial, w, stencil) заполняет w.diag, w.rhs, stencil.pair_g и возвращает guess;
scatter!(w, sol) пишет решение в w.field_b и возвращает ссылку.
Double-buffer: trial/updated — матрицы nz×nx с NaN в неактивных ячейках."""
function picard_cycle(config, transported::Matrix{Float64}, w::FvmWorkspace, stencil::Stencil,
                      assemble!, scatter!;
                      linear_tolerance::Float64, linear_max_iterations::Int)
    trial = w.field_a
    updated = w.field_b
    copyto!(trial, transported)
    max_change = Inf
    linear_iterations = 0
    linear_residual = Inf
    picard_used = 0
    for iteration in 1:config.picard_iterations
        picard_used = iteration
        guess = assemble!(trial, w, stencil)
        linear_iterations, linear_residual = pcg_solve!(
            w.sol, w.residual, w.zvec, w.direction, w.applied,
            w.diag, stencil.pair_p, stencil.pair_q, stencil.pair_g, w.rhs, guess;
            tolerance = linear_tolerance, max_iterations = linear_max_iterations)
        updated = scatter!(updated, w, w.sol)
        max_change = 0.0
        for k in eachindex(stencil.cell_lin)
            d = abs(updated[stencil.cell_lin[k]] - trial[stencil.cell_lin[k]])
            max_change = max(max_change, d)
        end
        trial, updated = updated, trial
        max_change <= config.picard_tolerance_c && break
    end
    return trial, picard_used, max_change, linear_iterations, linear_residual
end

"""Прямой Picard-цикл: матрица шага с фиксированным CSC-паттерном перестраивается
и факторизуется (Холецкий) на каждой итерации Пикара; решение — обратная подстановка.
Численно эквивалентен PCG-пути с точностью до линейной точности (прямое решение
точнее допуска pcg_tolerance)."""
function picard_cycle_direct(config, transported::Matrix{Float64}, w::FvmWorkspace,
                             s::Stencil, lookup, grid, state::State,
                             transported_c::Matrix{Float64}, dt_seconds::Float64,
                             volume::Float64, scatter!)
    ncell = s.ncell
    nzval = Vector{Float64}(undef, length(s.rowval))
    A = SparseMatrixCSC(ncell, ncell, s.colptr, s.rowval, nzval)
    trial = w.field_a
    updated = w.field_b
    copyto!(trial, transported)
    max_change = Inf
    linear_residual = Inf
    picard_used = 0
    factorization = nothing
    for iteration in 1:config.picard_iterations
        picard_used = iteration
        thermal_properties!(w.lam, w.rho_c, w.source, trial, grid, lookup, config)
        @inbounds for k in eachindex(s.pair_p)
            s.pair_g[k] = harmonic(w.lam[s.pair_plin[k]], w.lam[s.pair_qlin[k]]) *
                          s.pair_area[k] / s.pair_dist[k]
        end
        @inbounds for c in 1:ncell
            lin = s.cell_lin[c]
            d = w.rho_c[lin] * volume / dt_seconds
            r = d * transported_c[lin] + w.source[lin] * volume
            if s.west_k[c] > 0
                g = s.pair_g[s.west_k[c]]
                d += g; nzval[s.idx_west[c]] = -g
            end
            if s.south_k[c] > 0
                g = s.pair_g[s.south_k[c]]
                d += g; nzval[s.idx_south[c]] = -g
            end
            if s.north_k[c] > 0
                g = s.pair_g[s.north_k[c]]
                d += g; nzval[s.idx_north[c]] = -g
            end
            if s.east_k[c] > 0
                g = s.pair_g[s.east_k[c]]
                d += g; nzval[s.idx_east[c]] = -g
            end
            nzval[s.idx_diag[c]] = d
            if s.btop[c] == 1
                g = w.lam[lin] * s.dx / (0.5 * s.dz)
                nzval[s.idx_diag[c]] = d + g
                r += g * s.cell_tsurf[c]
            elseif s.btop[c] == 2
                r += state.qbasal[s.cell_col[c]] * s.dx
            end
            if s.bbot[c] == 1
                g = w.lam[lin] * s.dx / (0.5 * s.dz)
                nzval[s.idx_diag[c]] += g
                r += g * s.cell_tsurf[c]
            elseif s.bbot[c] == 2
                r += state.qbasal[s.cell_col[c]] * s.dx
            end
            w.rhs[c] = r
        end
        # Символьная факторизация зависит только от паттерна (стенсила): кешируется
        # по объекту стенсила, что устраняет повторный анализ между оценками theta.
        factorization = get(CHOLMOD_FACTOR_CACHE, s, nothing)
        if factorization === nothing
            factorization = cholesky(Symmetric(A))
            CHOLMOD_FACTOR_CACHE[s] = factorization
        else
            cholesky!(factorization, Symmetric(A))
        end
        ldiv!(w.sol, factorization, w.rhs)
        scatter!(updated, w, w.sol)
        max_change = 0.0
        @inbounds for k in eachindex(s.cell_lin)
            d = abs(updated[s.cell_lin[k]] - trial[s.cell_lin[k]])
            max_change = max(max_change, d)
        end
        trial, updated = updated, trial
        max_change <= config.picard_tolerance_c && break
    end
    # Фактическая относительная невязка последнего решения.
    matvec = A * w.sol
    residual = 0.0
    rhs_norm = 0.0
    @inbounds for k in 1:ncell
        residual += (matvec[k] - w.rhs[k])^2
        rhs_norm += w.rhs[k]^2
    end
    linear_residual = sqrt(residual / max(rhs_norm, 1.0e-300))
    return trial, picard_used, max_change, 0, linear_residual
end

"""Общий Picard-цикл для legacy/адаптивного решателя (создаёт буферы сам)."""
function picard_cycle(config, transported, assemble!, scatter;
                      linear_tolerance::Float64, linear_max_iterations::Int)
    trial = copy(transported)
    updated = similar(transported)
    max_change = Inf
    linear_iterations = 0
    linear_residual = Inf
    picard_used = 0
    for iteration in 1:config.picard_iterations
        picard_used = iteration
        diagonal, pair_p, pair_q, pair_g, rhs, guess = assemble!(trial)
        solution, linear_iterations, linear_residual = pcg_solve(
            diagonal, pair_p, pair_q, pair_g, rhs, guess;
            tolerance = linear_tolerance, max_iterations = linear_max_iterations)
        updated = scatter(solution)
        max_change = -Inf
        for k in eachindex(updated)
            d = abs(updated[k] - trial[k])
            isnan(d) || (max_change = max(max_change, d))
        end
        trial, updated = updated, trial
        max_change <= config.picard_tolerance_c && break
    end
    return trial, picard_used, max_change, linear_iterations, linear_residual
end

function implicit_fvm_step(transported_c::Matrix{Float64}, state::State, z::AbstractVector,
                           dx::Float64, dz::Float64, dt_seconds::Float64,
                           lookup::LithoPropLookup, config::SolverConfig;
                           grid::Union{Nothing,Grid} = nothing,
                           stencil::Union{Nothing,Stencil} = nothing,
                           workspace::Union{Nothing,FvmWorkspace} = nothing)
    grid === nothing && (grid = rasterize(state, z))
    stencil === nothing && (stencil = build_stencil(grid, state, z, dx, dz))
    workspace === nothing && (workspace = FvmWorkspace(config.nz, config.nx))
    active = grid.active
    ncell = stencil.ncell
    workspace.ncell = ncell
    resize!(workspace.diag, ncell); resize!(workspace.rhs, ncell)
    resize!(workspace.sol, ncell); resize!(workspace.residual, ncell)
    resize!(workspace.zvec, ncell); resize!(workspace.direction, ncell)
    resize!(workspace.applied, ncell)
    volume = dx * dz

    function assemble!(trial::Matrix{Float64}, w::FvmWorkspace, s::Stencil)
        thermal_properties!(w.lam, w.rho_c, w.source, trial, grid, lookup, config)
        @inbounds for k in 1:ncell
            lin = s.cell_lin[k]
            w.diag[k] = w.rho_c[lin] * volume / dt_seconds
            w.rhs[k] = w.diag[k] * transported_c[lin] + w.source[lin] * volume
        end
        # Порядок накопления повторяет прежнюю сборку: по ячейкам в column-major,
        # пары ячейки, затем верхняя грань, затем нижняя.
        @inbounds for c in 1:ncell
            for k in s.pair_first[c]:s.pair_last[c]
                g = harmonic(w.lam[s.pair_plin[k]], w.lam[s.pair_qlin[k]]) * s.pair_area[k] / s.pair_dist[k]
                s.pair_g[k] = g
                w.diag[s.pair_p[k]] += g
                w.diag[s.pair_q[k]] += g
            end
            if s.btop[c] == 1
                g = w.lam[s.cell_lin[c]] * s.dx / (0.5 * s.dz)
                w.diag[c] += g
                w.rhs[c] += g * s.cell_tsurf[c]
            elseif s.btop[c] == 2
                w.rhs[c] += state.qbasal[s.cell_col[c]] * s.dx
            end
            if s.bbot[c] == 1
                g = w.lam[s.cell_lin[c]] * s.dx / (0.5 * s.dz)
                w.diag[c] += g
                w.rhs[c] += g * s.cell_tsurf[c]
            elseif s.bbot[c] == 2
                w.rhs[c] += state.qbasal[s.cell_col[c]] * s.dx
            end
        end
        return trial[active]
    end

    function scatter!(out::Matrix{Float64}, w::FvmWorkspace, sol::Vector{Float64})
        fill!(out, NaN)
        @inbounds for k in 1:ncell
            out[stencil.cell_lin[k]] = sol[k]
        end
        return out
    end

    if config.linear_solver == :direct
        trial, picard_used, max_change, linear_iterations, linear_residual =
            picard_cycle_direct(config, transported_c, workspace, stencil, lookup, grid,
                                state, transported_c, dt_seconds, volume, scatter!)
    else
        trial, picard_used, max_change, linear_iterations, linear_residual = picard_cycle(
            config, transported_c, workspace, stencil, assemble!, scatter!;
            linear_tolerance = config.pcg_tolerance,
            linear_max_iterations = max(2000, 8 * config.nx * config.nz))
    end

    return trial, OrderedDict{String,Any}(
        "picard_iterations" => picard_used,
        "picard_change_c" => max_change,
        "linear_iterations" => linear_iterations,
        "linear_relative_residual" => linear_residual,
    )
end

"Legacy-совместимая обёртка с Dict-lookup литологий и без workspace."
function implicit_fvm_step(transported_c::Matrix{Float64}, state::State, z::AbstractVector,
                           dx::Float64, dz::Float64, dt_seconds::Float64,
                           lith_properties::Dict{Int,LithProp}, config::SolverConfig;
                           grid::Union{Nothing,Grid} = nothing)
    grid === nothing && (grid = rasterize(state, z))
    stencil = build_stencil(grid, state, z, dx, dz)
    workspace = FvmWorkspace(config.nz, config.nx)
    return implicit_fvm_step(transported_c, state, z, dx, dz, dt_seconds,
                             LithoPropLookup(lith_properties), config;
                             grid = grid, stencil = stencil, workspace = workspace)
end

function initialise_temperature(state::State, z::AbstractVector, config::SolverConfig;
                                grid::Union{Nothing,Grid} = nothing)
    grid === nothing && (grid = rasterize(state, z))
    nz, nx = length(z), length(state.surface)
    temperature = fill(NaN, nz, nx)
    for i in 1:nx, j in 1:nz
        grid.active[j, i] || continue
        depth = max(z[j] - state.surface[i], 0.0)
        temperature[j, i] = state.tsurf[i] + state.qbasal[i] * depth / config.initial_lambda
    end
    return temperature
end

struct StepGeometry
    state::State                       # тета-независимая часть, qbasal = zeros
    grid::Grid
    stencil::Stencil                   # построенный один раз стенсил
    age_old::Float64
    age_young::Float64
    fraction::Float64
    dt_seconds::Float64
end

struct ForwardGeometry
    steps::Vector{StepGeometry}
    exact::Dict{Float64,State}
    exact_grids::Dict{Float64,Grid}
    x::Vector{Float64}
    z::Vector{Float64}
end

patched_q(s::State, q::Vector{Float64}) =
    State(s.age, s.layers, s.surface, s.base, s.tsurf, q)

"Предвычисление всей тета-независимой части прогона: геометрия по шагам, гриды."
function precompute_geometry(data::BasinDataset, config::SolverConfig)
    x0, x1, z0, z1 = data.bounds
    dx = (x1 - x0) / config.nx
    dz = (z1 - z0) / config.nz
    x = collect(range(x0 + 0.5 * dx, x1 - 0.5 * dx; length = config.nx))
    z = collect(range(z0 + 0.5 * dz, z1 - 0.5 * dz; length = config.nz))
    exact = Dict(age => exact_state(data, age, x; with_heat = false) for age in data.ages)
    exact_grids = Dict(age => rasterize(exact[age], z) for age in data.ages)
    steps = StepGeometry[]
    for (age_old, age_young) in zip(data.ages[1:end-1], data.ages[2:end])
        span = age_old - age_young
        n_substeps = max(1, Int(ceil(span / config.max_dt_ma)))
        for substep in 1:n_substeps
            fraction = substep / n_substeps
            state = interpolate_state(exact[age_old], exact[age_young], fraction)
            grid = rasterize(state, z)
            stencil = build_stencil(grid, state, z, dx, dz)
            push!(steps, StepGeometry(state, grid, stencil, age_old, age_young,
                                      fraction, (span / n_substeps) * SECONDS_PER_MA))
        end
    end
    return ForwardGeometry(steps, exact, exact_grids, x, z)
end

# ----------------------------------------------------------------------------
# Прямая задача
# ----------------------------------------------------------------------------

function run_forward_param(data::BasinDataset, config::SolverConfig,
                           geometry::ForwardGeometry,
                           qba::Dict{Float64,Vector{Float64}})
    x, z = geometry.x, geometry.z
    dx = (data.bounds[2] - data.bounds[1]) / config.nx
    dz = (data.bounds[4] - data.bounds[3]) / config.nz
    age0 = data.ages[1]
    s0 = patched_q(geometry.exact[age0], qba[age0])
    temperature = initialise_temperature(s0, z, config; grid = geometry.exact_grids[age0])
    saved_temperature = [copy(temperature)]
    diagnostics = OrderedDict{String,Any}[]
    lookup = LithoPropLookup(data.properties)
    workspace = FvmWorkspace(config.nz, config.nx)

    Progress.bar_start(length(geometry.steps), "Forward time steps")
    start = time_ns()
    previous_state = s0
    previous_grid = geometry.exact_grids[age0]
    for si in eachindex(geometry.steps)
        step = geometry.steps[si]
        qk = (1.0 - step.fraction) .* qba[step.age_old] .+
             step.fraction .* qba[step.age_young]
        state = patched_q(step.state, qk)
        transported = material_remap(temperature, previous_state, state, z;
                                     old_grid = previous_grid, new_grid = step.grid)
        temperature, diag = implicit_fvm_step(
            transported, state, z, dx, dz, step.dt_seconds, lookup, config;
            grid = step.grid, stencil = step.stencil, workspace = workspace)
        push!(diagnostics, merge(OrderedDict{String,Any}("age_ma" => state.age), diag))
        Progress.bar_add(1, @sprintf("Forward time steps · age %.1f Ma", state.age))
        previous_state = state
        previous_grid = step.grid
        # снимок на границе эпохи — как в classic (push внутри цикла по эпохам)
        if si == length(geometry.steps) ||
           geometry.steps[si + 1].age_old != step.age_old
            push!(saved_temperature, copy(temperature))
        end
    end
    Progress.bar_finish()

    final_grid = geometry.exact_grids[data.ages[end]]
    elapsed = (time_ns() - start) / 1.0e9

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
        final_surface_m = geometry.exact[data.ages[end]].surface,
        final_base_m = geometry.exact[data.ages[end]].base,
        diagnostics = diagnostics,
        elapsed_seconds = elapsed,
        dx = dx, dz = dz,
    )
end

function run_forward(data::BasinDataset, config::SolverConfig;
                     geometry::Union{Nothing,ForwardGeometry} = nothing,
                     qbasal_by_age::Union{Nothing,Dict{Float64,Vector{Float64}}} = nothing)
    validate(config)
    if geometry !== nothing
        return run_forward_param(data, config, geometry,
                                 qbasal_by_age === nothing ?
                                 Dict(age => zeros(length(geometry.x)) for age in data.ages) :
                                 qbasal_by_age)
    end
    x0, x1, z0, z1 = data.bounds
    dx = (x1 - x0) / config.nx
    dz = (z1 - z0) / config.nz
    x = collect(range(x0 + 0.5 * dx, x1 - 0.5 * dx; length = config.nx))
    z = collect(range(z0 + 0.5 * dz, z1 - 0.5 * dz; length = config.nz))
    exact = Dict(age => exact_state(data, age, x) for age in data.ages)
    temperature = initialise_temperature(exact[data.ages[1]], z, config)
    saved_temperature = [copy(temperature)]
    diagnostics = OrderedDict{String,Any}[]

    total_steps = sum(max(1, Int(ceil((data.ages[k] - data.ages[k+1]) / config.max_dt_ma)))
                      for k in 1:length(data.ages)-1)
    Progress.bar_start(total_steps, "Forward time steps")
    start = time_ns()
    lookup = LithoPropLookup(data.properties)
    workspace = FvmWorkspace(config.nz, config.nx)
    for (age_old, age_young) in zip(data.ages[1:end-1], data.ages[2:end])
        span = age_old - age_young
        n_substeps = max(1, Int(ceil(span / config.max_dt_ma)))
        previous_state = exact[age_old]
        previous_grid = rasterize(previous_state, z)
        for substep in 1:n_substeps
            fraction = substep / n_substeps
            state = interpolate_state(exact[age_old], exact[age_young], fraction)
            grid = rasterize(state, z)
            transported = material_remap(temperature, previous_state, state, z;
                                         old_grid = previous_grid, new_grid = grid)
            dt_seconds = (span / n_substeps) * SECONDS_PER_MA
            temperature, diag = implicit_fvm_step(
                transported, state, z, dx, dz, dt_seconds, lookup, config;
                grid = grid, workspace = workspace)
            push!(diagnostics, merge(OrderedDict{String,Any}("age_ma" => state.age), diag))
            Progress.bar_add(1, @sprintf("Forward time steps · age %.1f Ma", state.age))
            previous_state = state
            previous_grid = grid
        end
        push!(saved_temperature, copy(temperature))
    end
    Progress.bar_finish()


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
    Progress.init()
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
    nsteps = length(result.diagnostics)
    @printf("TIME total %s | forward %s | %d steps, %.2f s/step | 1 J-eval ~ 1 forward
",
            Progress.fmt_time_public(result.elapsed_seconds),
            Progress.fmt_time_public(result.elapsed_seconds), nsteps,
            result.elapsed_seconds / max(nsteps, 1))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
