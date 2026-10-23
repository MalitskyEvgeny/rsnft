# Adaptive quadtree FVM basin solver.
#
# Использует basin_fvm.jl (должен лежать рядом): BasinDataset, exact_state,
# interpolate_state, pcg_solve, harmonic, sekiguchi_conductivity,
# waples_heat_capacity, update_manifest, SECONDS_PER_MA.
#
# Запуск:
#   julia adaptive_basin_fvm.jl --data-dir data --cache-dir adaptive_results

include(joinpath(@__DIR__, "basin_fvm.jl"))

const DEFAULT_DATA_DIR = raw"C:\Users\solo\Desktop\МЭИ\3 мага\роснефть\data1"

# ----------------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------------

Base.@kwdef struct AdaptiveConfig
    base_nx::Int = 16
    base_nz::Int = 32
    max_level::Int = 5
    interface_band::Float64 = 0.55
    max_dt_ma::Float64 = 2.5
    picard_iterations::Int = 8
    picard_tolerance_c::Float64 = 1.0e-3
    linear_tolerance::Float64 = 1.0e-8
    fluid_lambda::Float64 = 0.60
    fluid_density::Float64 = 1000.0
    fluid_heat_capacity::Float64 = 4180.0
    initial_lambda::Float64 = 2.5
    preview_nx::Int = 160
    preview_nz::Int = 240
end

function validate(c::AdaptiveConfig)
    (c.base_nx < 2 || c.base_nz < 2) &&
        throw(ArgumentError("base_nx and base_nz must be at least 2"))
    !(0 <= c.max_level <= 5) && throw(ArgumentError("max_level must be between 0 and 5"))
    (c.max_dt_ma <= 0 || c.interface_band < 0) &&
        throw(ArgumentError("time step must be positive and interface_band non-negative"))
    c.picard_iterations < 1 && throw(ArgumentError("picard_iterations must be positive"))
    (c.picard_tolerance_c <= 0 || c.linear_tolerance <= 0) &&
        throw(ArgumentError("solver tolerances must be positive"))
    (c.preview_nx < 2 || c.preview_nz < 2) &&
        throw(ArgumentError("preview_nx and preview_nz must be at least 2"))
    return nothing
end

asdict(c::AdaptiveConfig) =
    OrderedDict{String,Any}(String(f) => getfield(c, f) for f in fieldnames(AdaptiveConfig))

# ----------------------------------------------------------------------------
# Сетка (сбалансированное квадродерево)
#
# Координаты ячеек ix0/ix1/iz0/iz1 — целые индексы на самой мелкой сетке,
# полуоткрытый интервал [ix0, ix1) и отсчёт от 0, как в Python-версии.
# Номера ячеек (индексы в массивах) — с 1.
# ----------------------------------------------------------------------------

struct Cell
    ix0::Int
    ix1::Int
    iz0::Int
    iz1::Int
    level::Int
end

struct FacePairs
    p::Vector{Int}
    q::Vector{Int}
    len::Vector{Float64}
    distance::Vector{Float64}
    face_x::Vector{Float64}
    face_z::Vector{Float64}
    orientation::Vector{Int}     # 0 — вертикальная грань (боковая), 1 — горизонтальная
end

struct AdaptiveMesh
    cells::Vector{Cell}
    bounds::NTuple{4,Float64}
    config::AdaptiveConfig
    scale::Int
    nxf::Int
    nzf::Int
    dxf::Float64
    dzf::Float64
    x0::Vector{Float64}
    x1::Vector{Float64}
    z0::Vector{Float64}
    z1::Vector{Float64}
    xc::Vector{Float64}
    zc::Vector{Float64}
    width::Vector{Float64}
    height::Vector{Float64}
    area::Vector{Float64}
    level::Vector{Int8}
    lookup::Matrix{Int}
    pairs::FacePairs
end

function AdaptiveMesh(cells::Vector{Cell}, bounds::NTuple{4,Float64}, config::AdaptiveConfig)
    scale = 2^config.max_level
    nxf = config.base_nx * scale
    nzf = config.base_nz * scale
    x0, x1, z0, z1 = bounds
    dxf = (x1 - x0) / nxf
    dzf = (z1 - z0) / nzf
    cx0 = [x0 + c.ix0 * dxf for c in cells]
    cx1 = [x0 + c.ix1 * dxf for c in cells]
    cz0 = [z0 + c.iz0 * dzf for c in cells]
    cz1 = [z0 + c.iz1 * dzf for c in cells]
    width = cx1 .- cx0
    height = cz1 .- cz0
    lookup = zeros(Int, nzf, nxf)
    for (index, cell) in enumerate(cells)
        lookup[(cell.iz0+1):cell.iz1, (cell.ix0+1):cell.ix1] .= index
    end
    return AdaptiveMesh(
        cells, bounds, config, scale, nxf, nzf, dxf, dzf,
        cx0, cx1, cz0, cz1, 0.5 .* (cx0 .+ cx1), 0.5 .* (cz0 .+ cz1),
        width, height, width .* height, Int8[c.level for c in cells],
        lookup, face_pairs(cells, dxf, dzf, bounds),
    )
end

"Номер ячейки, содержащей точку (x, z) — как AdaptiveMesh.locate, но для одной точки."
function locate_one(mesh::AdaptiveMesh, x::Float64, z::Float64)
    x_min, x_max, z_min, z_max = mesh.bounds
    fx = (x - x_min) / (x_max - x_min) * mesh.nxf
    fz = (z - z_min) / (z_max - z_min) * mesh.nzf
    ix = isnan(fx) ? 0 : clamp(trunc(Int, clamp(fx, -1.0e15, 1.0e15)), 0, mesh.nxf - 1)
    iz = isnan(fz) ? 0 : clamp(trunc(Int, clamp(fz, -1.0e15, 1.0e15)), 0, mesh.nzf - 1)
    return mesh.lookup[iz+1, ix+1]
end

function split_cell(cell::Cell)
    ixm = (cell.ix0 + cell.ix1) ÷ 2
    izm = (cell.iz0 + cell.iz1) ÷ 2
    level = cell.level + 1
    return [
        Cell(cell.ix0, ixm, cell.iz0, izm, level),
        Cell(ixm, cell.ix1, cell.iz0, izm, level),
        Cell(cell.ix0, ixm, izm, cell.iz1, level),
        Cell(ixm, cell.ix1, izm, cell.iz1, level),
    ]
end

"Возвращает (ячейка слева/сверху, ячейка справа/снизу, перекрытие lo, hi, ориентация)."
function integer_face_pairs(cells::Vector{Cell})
    left = Dict{Int,Vector{Int}}()
    right = Dict{Int,Vector{Int}}()
    top = Dict{Int,Vector{Int}}()
    bottom = Dict{Int,Vector{Int}}()
    for (i, c) in enumerate(cells)
        push!(get!(left, c.ix0, Int[]), i)
        push!(get!(right, c.ix1, Int[]), i)
        push!(get!(top, c.iz0, Int[]), i)
        push!(get!(bottom, c.iz1, Int[]), i)
    end
    result = Tuple{Int,Int,Int,Int,Int}[]
    for coordinate in sort!(collect(intersect(keys(right), keys(left))))
        for a in right[coordinate], b in left[coordinate]
            lo = max(cells[a].iz0, cells[b].iz0)
            hi = min(cells[a].iz1, cells[b].iz1)
            hi > lo && push!(result, (a, b, lo, hi, 0))
        end
    end
    for coordinate in sort!(collect(intersect(keys(bottom), keys(top))))
        for a in bottom[coordinate], b in top[coordinate]
            lo = max(cells[a].ix0, cells[b].ix0)
            hi = min(cells[a].ix1, cells[b].ix1)
            hi > lo && push!(result, (a, b, lo, hi, 1))
        end
    end
    return result
end

function balance_cells(cells::Vector{Cell}, max_level::Int)
    for _ in 1:(max_level+1)
        marked = Set{Int}()
        for (a, b, _, _, _) in integer_face_pairs(cells)
            if abs(cells[a].level - cells[b].level) > 1
                push!(marked, cells[a].level < cells[b].level ? a : b)
            end
        end
        isempty(marked) && return cells
        updated = Cell[]
        for (i, cell) in enumerate(cells)
            if i in marked && cell.level < max_level
                append!(updated, split_cell(cell))
            else
                push!(updated, cell)
            end
        end
        cells = updated
    end
    return cells
end

function face_pairs(cells::Vector{Cell}, dxf::Float64, dzf::Float64,
                    bounds::NTuple{4,Float64})
    raw = integer_face_pairs(cells)
    n = length(raw)
    p = Vector{Int}(undef, n); q = Vector{Int}(undef, n)
    len = Vector{Float64}(undef, n); distance = Vector{Float64}(undef, n)
    face_x = Vector{Float64}(undef, n); face_z = Vector{Float64}(undef, n)
    orientation = Vector{Int}(undef, n)
    x_min, _, z_min, _ = bounds
    for (k, (a, b, lo, hi, axis)) in enumerate(raw)
        ca, cb = cells[a], cells[b]
        p[k] = a; q[k] = b; orientation[k] = axis
        if axis == 0
            len[k] = (hi - lo) * dzf
            distance[k] = 0.5 * ((ca.ix1 - ca.ix0) + (cb.ix1 - cb.ix0)) * dxf
            face_x[k] = x_min + ca.ix1 * dxf
            face_z[k] = z_min + 0.5 * (lo + hi) * dzf
        else
            len[k] = (hi - lo) * dxf
            distance[k] = 0.5 * ((ca.iz1 - ca.iz0) + (cb.iz1 - cb.iz0)) * dzf
            face_x[k] = x_min + 0.5 * (lo + hi) * dxf
            face_z[k] = z_min + ca.iz1 * dzf
        end
    end
    return FacePairs(p, q, len, distance, face_x, face_z, orientation)
end

# ----------------------------------------------------------------------------
# Построение сетки по геометрии
# ----------------------------------------------------------------------------

function state_at(data::BasinDataset, older_age::Float64, younger_age::Float64,
                  fraction::Float64, x::AbstractVector)
    older = exact_state(data, older_age, x)
    older_age == younger_age && return older
    younger = exact_state(data, younger_age, x)
    return interpolate_state(older, younger, fraction)
end

function nan_extrema(v::AbstractVector)
    lo, hi = Inf, -Inf
    for a in v
        isnan(a) && continue
        lo = min(lo, a); hi = max(hi, a)
    end
    return isinf(lo) ? (NaN, NaN) : (lo, hi)    # все NaN -> NaN, сравнения дают false
end

function cell_intersects_refinement_zone(cell::Cell, probe_x::AbstractVector, state::State,
                                         bounds::NTuple{4,Float64}, config::AdaptiveConfig)
    x_min, x_max, z_min, z_max = bounds
    scale = 2^config.max_level
    dxf = (x_max - x_min) / (config.base_nx * scale)
    dzf = (z_max - z_min) / (config.base_nz * scale)
    xa, xb = x_min + cell.ix0 * dxf, x_min + cell.ix1 * dxf
    za, zb = z_min + cell.iz0 * dzf, z_min + cell.iz1 * dzf
    indices = findall(px -> px >= xa - 1.0e-10 && px <= xb + 1.0e-10, probe_x)
    if isempty(indices)
        indices = [argmin(abs.(probe_x .- 0.5 * (xa + xb)))]
    end
    band = config.interface_band * (zb - za)
    for layer in values(state.layers)
        for boundary in (layer.top[indices], layer.bot[indices])
            bmin, bmax = nan_extrema(boundary)
            (bmin <= zb + band && bmax >= za - band) && return true
        end
        lith = layer.lith[indices]
        if length(lith) > 1 && any(lith[2:end] .!= lith[1:end-1])
            lo = min.(layer.top[indices], layer.bot[indices])
            hi = max.(layer.top[indices], layer.bot[indices])
            if nan_extrema(lo)[1] <= zb + band && nan_extrema(hi)[2] >= za - band
                return true
            end
        end
    end
    return false
end

function refine!(cells::Vector{Cell}, cell::Cell, probe_x, probe_state::State,
                 data::BasinDataset, config::AdaptiveConfig)
    if cell.level < config.max_level &&
       cell_intersects_refinement_zone(cell, probe_x, probe_state, data.bounds, config)
        for child in split_cell(cell)
            refine!(cells, child, probe_x, probe_state, data, config)
        end
    else
        push!(cells, cell)
    end
    return cells
end

function build_mesh(data::BasinDataset, older_age::Float64, younger_age::Float64,
                    fraction::Float64, config::AdaptiveConfig)
    x_min, x_max, _, _ = data.bounds
    probe_count = max(257, config.base_nx * 2^config.max_level * 4 + 1)
    probe_x = collect(range(x_min, x_max; length = probe_count))
    probe_state = state_at(data, older_age, younger_age, fraction, probe_x)
    scale = 2^config.max_level
    cells = Cell[]
    for j in 0:(config.base_nz-1)          # порядок обхода как в Python: j внешний, i внутренний
        for i in 0:(config.base_nx-1)
            refine!(cells, Cell(i * scale, (i + 1) * scale, j * scale, (j + 1) * scale, 0),
                    probe_x, probe_state, data, config)
        end
    end
    cells = balance_cells(cells, config.max_level)
    return AdaptiveMesh(cells, data.bounds, config)
end

# ----------------------------------------------------------------------------
# Литология, свойства, начальное условие
# ----------------------------------------------------------------------------

struct CellGrid
    active::BitVector
    layer_index::Vector{Int}      # 0 = неактивная ячейка
    lith::Vector{Int}
    phi::Vector{Float64}
    names::Vector{String}
end

function rasterize_cells(state::State, mesh::AdaptiveMesh)
    n = length(mesh.cells)
    layer_index = zeros(Int, n)
    lith = fill(-1, n)
    phi = fill(NaN, n)
    names = sort!(collect(keys(state.layers)))
    for (number, name) in enumerate(names)
        layer = state.layers[name]
        for c in 1:n
            lo = min(layer.top[c], layer.bot[c])
            hi = max(layer.top[c], layer.bot[c])
            if mesh.zc[c] >= lo && mesh.zc[c] <= hi && (hi - lo) > 1.0e-8
                layer_index[c] = number
                lith[c] = layer.lith[c]
                phi[c] = layer.phi[c]
            end
        end
    end
    return CellGrid(layer_index .> 0, layer_index, lith, phi, names)
end

function adaptive_properties(temperature_c::Vector{Float64}, grid::CellGrid,
                             properties::Dict{Int,LithProp}, config::AdaptiveConfig)
    lam = fill(NaN, length(temperature_c))
    rho_c = fill(NaN, length(temperature_c))
    source = fill(NaN, length(temperature_c))
    for c in eachindex(temperature_c)
        grid.active[c] || continue
        code = grid.lith[c]
        haskey(properties, code) || throw(KeyError("Unknown lithotype $code"))
        prop = properties[code]
        tk = temperature_c[c] + 273.15
        phi = grid.phi[c]
        lam_s = sekiguchi_conductivity(tk, prop.lambda20)
        c_s = waples_heat_capacity(tk, prop.c20)
        lam[c] = lam_s^(1.0 - phi) * config.fluid_lambda^phi
        rho_c[c] = (1.0 - phi) * prop.rho_s * c_s +
                   phi * config.fluid_density * config.fluid_heat_capacity
        source[c] = (1.0 - phi) * prop.As
    end
    return lam, rho_c, source
end

function initialise(mesh::AdaptiveMesh, state::State, grid::CellGrid, config::AdaptiveConfig)
    result = Vector{Float64}(undef, length(mesh.cells))
    for c in eachindex(result)
        depth = max(mesh.zc[c] - state.surface[c], 0.0)
        result[c] = grid.active[c] ?
            state.tsurf[c] + state.qbasal[c] * depth / config.initial_lambda : NaN
    end
    return result
end

function adaptive_remap(old_mesh::AdaptiveMesh, old_temperature::Vector{Float64},
                        old_grid::CellGrid, old_state_on_new_x::State,
                        new_mesh::AdaptiveMesh, new_state::State, new_grid::CellGrid)
    n = length(new_mesh.cells)
    result = fill(NaN, n)
    active_old = old_grid.active .& isfinite.(old_temperature)
    for (number, name) in enumerate(new_grid.names)
        new_layer = new_state.layers[name]
        old_layer = get(old_state_on_new_x.layers, name, nothing)
        for c in 1:n
            (new_grid.active[c] && new_grid.layer_index[c] == number) || continue
            if old_layer === nothing
                result[c] = new_state.tsurf[c]
                continue
            end
            new_lo = min(new_layer.top[c], new_layer.bot[c])
            new_hi = max(new_layer.top[c], new_layer.bot[c])
            old_lo = min(old_layer.top[c], old_layer.bot[c])
            old_hi = max(old_layer.top[c], old_layer.bot[c])
            if !((old_hi - old_lo) > 1.0e-8)
                result[c] = new_state.tsurf[c]
                continue
            end
            eta = clamp((new_mesh.zc[c] - new_lo) / max(new_hi - new_lo, 1.0e-12), 0.0, 1.0)
            target_z = old_lo + eta * (old_hi - old_lo)
            oc = locate_one(old_mesh, new_mesh.xc[c], target_z)
            result[c] = active_old[oc] ? old_temperature[oc] : old_state_on_new_x.tsurf[c]
        end
    end
    for c in 1:n
        if new_grid.active[c] && !isfinite(result[c])
            result[c] = new_state.tsurf[c]
        end
    end
    return result
end

# ----------------------------------------------------------------------------
# Неявный шаг на адаптивной сетке
# ----------------------------------------------------------------------------

function adaptive_step(transported::Vector{Float64}, mesh::AdaptiveMesh, state::State,
                       grid::CellGrid, dt_seconds::Float64,
                       properties::Dict{Int,LithProp}, config::AdaptiveConfig)
    active = grid.active
    ncell = length(mesh.cells)
    local_index = zeros(Int, ncell)
    local_index[active] = 1:count(active)
    trial = copy(transported)
    pairs = mesh.pairs
    last_linear_iterations = 0
    last_linear_residual = Inf
    max_change = Inf
    picard_used = 0

    for nonlinear_iteration in 1:config.picard_iterations
        picard_used = nonlinear_iteration
        lam, rho_c, source = adaptive_properties(trial, grid, properties, config)
        diagonal = rho_c[active] .* mesh.area[active] ./ dt_seconds
        rhs = diagonal .* transported[active] .+ source[active] .* mesh.area[active]
        pp = Int[]; qq = Int[]; gg = Float64[]

        for face in eachindex(pairs.p)
            a, b = pairs.p[face], pairs.q[face]
            a_active, b_active = active[a], active[b]
            if a_active && b_active
                g = harmonic(lam[a], lam[b]) * pairs.len[face] / pairs.distance[face]
                pa, pb = local_index[a], local_index[b]
                diagonal[pa] += g; diagonal[pb] += g
                push!(pp, pa); push!(qq, pb); push!(gg, g)
            elseif a_active != b_active
                cell = a_active ? a : b
                p = local_index[cell]
                face_z = pairs.face_z[face]
                pairs.orientation[face] == 0 && continue   # боковые границы — без потока
                if abs(face_z - state.surface[cell]) <= abs(face_z - state.base[cell])
                    distance = 0.5 * mesh.height[cell]
                    g = lam[cell] * pairs.len[face] / distance
                    diagonal[p] += g
                    rhs[p] += g * state.tsurf[cell]
                else
                    rhs[p] += state.qbasal[cell] * pairs.len[face]
                end
            end
        end

        for cell in findall(active)
            p = local_index[cell]
            descriptor = mesh.cells[cell]
            if descriptor.iz0 == 0
                g = lam[cell] * mesh.width[cell] / (0.5 * mesh.height[cell])
                diagonal[p] += g
                rhs[p] += g * state.tsurf[cell]
            end
            if descriptor.iz1 == mesh.nzf
                rhs[p] += state.qbasal[cell] * mesh.width[cell]
            end
        end

        solution, last_linear_iterations, last_linear_residual = pcg_solve(
            diagonal, pp, qq, gg, rhs, trial[active];
            tolerance = config.linear_tolerance,
            max_iterations = max(800, 6 * floor(Int, sqrt(length(diagonal)))),
        )
        updated = fill(NaN, length(trial))
        updated[active] = solution
        max_change = -Inf                        # np.nanmax(np.abs(updated - trial))
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
        "nonlinear_converged" => max_change <= config.picard_tolerance_c,
        "linear_iterations" => last_linear_iterations,
        "linear_relative_residual" => last_linear_residual,
    )
end

# ----------------------------------------------------------------------------
# Проекция на регулярную сетку и основной цикл
# ----------------------------------------------------------------------------

function mesh_to_regular(mesh::AdaptiveMesh, values::AbstractVector, nx::Int, nz::Int)
    x0, x1, z0, z1 = mesh.bounds
    dx, dz = (x1 - x0) / nx, (z1 - z0) / nz
    x = collect(range(x0 + 0.5 * dx, x1 - 0.5 * dx; length = nx))
    z = collect(range(z0 + 0.5 * dz, z1 - 0.5 * dz; length = nz))
    out = Matrix{Float64}(undef, nz, nx)
    for i in 1:nx, j in 1:nz
        out[j, i] = values[locate_one(mesh, x[i], z[j])]
    end
    return x, z, out
end

function run_adaptive(data::BasinDataset, config::AdaptiveConfig)
    validate(config)
    ages = data.ages
    mesh = build_mesh(data, ages[1], ages[1], 0.0, config)
    state = state_at(data, ages[1], ages[1], 0.0, mesh.xc)
    grid = rasterize_cells(state, mesh)
    temperature = initialise(mesh, state, grid, config)
    px, pz, preview = mesh_to_regular(mesh, temperature, config.preview_nx, config.preview_nz)
    saved = [preview]
    cell_counts = [length(mesh.cells)]
    diagnostics = OrderedDict{String,Any}[]
    start = time_ns()

    for (older_age, younger_age) in zip(ages[1:end-1], ages[2:end])
        span = older_age - younger_age
        steps = max(1, Int(ceil(span / config.max_dt_ma)))
        previous_fraction = 0.0
        previous_older, previous_younger = older_age, younger_age
        for step in 1:steps
            fraction = step / steps
            new_mesh = build_mesh(data, older_age, younger_age, fraction, config)
            new_state = state_at(data, older_age, younger_age, fraction, new_mesh.xc)
            new_grid = rasterize_cells(new_state, new_mesh)
            old_state_on_new_x = state_at(
                data, previous_older, previous_younger, previous_fraction, new_mesh.xc)
            transported = adaptive_remap(
                mesh, temperature, grid, old_state_on_new_x, new_mesh, new_state, new_grid)
            temperature, diag = adaptive_step(
                transported, new_mesh, new_state, new_grid,
                span / steps * SECONDS_PER_MA, data.properties, config)
            push!(diagnostics, merge(OrderedDict{String,Any}(
                "age_ma" => new_state.age,
                "cells" => length(new_mesh.cells),
                "active_cells" => count(new_grid.active),
            ), diag))
            mesh, state, grid = new_mesh, new_state, new_grid
            previous_fraction = fraction
        end
        _, _, preview = mesh_to_regular(mesh, temperature, config.preview_nx, config.preview_nz)
        push!(saved, preview)
        push!(cell_counts, length(mesh.cells))
    end

    elapsed = (time_ns() - start) / 1.0e9
    final_lith = mesh_to_regular(mesh, Float64.(grid.lith), config.preview_nx, config.preview_nz)[3]
    final_phi = mesh_to_regular(mesh, grid.phi, config.preview_nx, config.preview_nz)[3]
    final_preview_state = exact_state(data, ages[end], px)

    nt = length(saved)
    temperature_c = Array{Float64,3}(undef, nt, config.preview_nz, config.preview_nx)
    for k in 1:nt
        temperature_c[k, :, :] = saved[k]       # (время, z, x)
    end

    return (
        x = px, z = pz, ages_ma = copy(ages),
        temperature_c = temperature_c, final_lithology = final_lith,
        final_porosity = final_phi, mesh = mesh, state = state,
        final_surface_m = final_preview_state.surface,
        final_base_m = final_preview_state.base,
        grid = grid, final_temperature_cells = temperature,
        cell_counts = cell_counts, diagnostics = diagnostics,
        elapsed_seconds = elapsed,
    )
end

# ----------------------------------------------------------------------------
# Кэш и сохранение
# ----------------------------------------------------------------------------

function adaptive_run_id(data::BasinDataset, config::AdaptiveConfig)
    adaptive = OrderedDict(sort!(collect(asdict(config)); by = first))
    payload = OrderedDict{String,Any}(            # ключи в алфавитном порядке (sort_keys=True)
        "adaptive" => adaptive,
        "input" => data.fingerprint,
        "version" => 3,
    )
    return bytes2hex(sha256(JSON.json(payload)))[1:16]
end

function save_adaptive(result, run_dir::AbstractString, metadata)
    mkpath(run_dir)
    mesh = result.mesh
    grid = result.grid
    npzwrite(joinpath(run_dir, "result.npz"), Dict(
        "x" => result.x, "z" => result.z, "ages_ma" => result.ages_ma,
        "temperature_c" => result.temperature_c,
        "final_lithology" => result.final_lithology,
        "final_porosity" => result.final_porosity,
        "final_surface_m" => result.final_surface_m,
        "final_base_m" => result.final_base_m,
    ))
    npzwrite(joinpath(run_dir, "final_adaptive_mesh.npz"), Dict(
        "x0" => mesh.x0, "x1" => mesh.x1, "z0" => mesh.z0, "z1" => mesh.z1,
        "level" => mesh.level, "temperature_c" => result.final_temperature_cells,
        "active" => collect(grid.active), "lithology" => grid.lith, "porosity" => grid.phi,
    ))
    write(joinpath(run_dir, "metadata.json"), sprint(io -> JSON.print(io, metadata, 2)))
end

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

function main_adaptive(argv = ARGS)
    d = AdaptiveConfig()
    s = ArgParseSettings(description = "Adaptive quadtree FVM basin solver")
    @add_arg_table! s begin
        "--data-dir"
            arg_type = String
            default = DEFAULT_DATA_DIR
        "--cache-dir"
            arg_type = String
            default = "adaptive_results"
        "--base-nx"
            arg_type = Int
            default = d.base_nx
        "--base-nz"
            arg_type = Int
            default = d.base_nz
        "--max-level"
            arg_type = Int
            default = d.max_level
        "--interface-band"
            arg_type = Float64
            default = d.interface_band
        "--max-dt-ma"
            arg_type = Float64
            default = d.max_dt_ma
        "--picard-iterations"
            arg_type = Int
            default = d.picard_iterations
            help = "maximum Picard iterations per time step"
        "--picard-tolerance-c"
            arg_type = Float64
            default = d.picard_tolerance_c
        "--linear-tolerance"
            arg_type = Float64
            default = d.linear_tolerance
        "--force"
            action = :store_true
    end
    args = parse_args(argv, s)

    config = AdaptiveConfig(
        base_nx = args["base-nx"], base_nz = args["base-nz"], max_level = args["max-level"],
        interface_band = args["interface-band"], max_dt_ma = args["max-dt-ma"],
        picard_iterations = args["picard-iterations"],
        picard_tolerance_c = args["picard-tolerance-c"],
        linear_tolerance = args["linear-tolerance"],
    )
    validate(config)
    cache_dir = args["cache-dir"]
    mkpath(cache_dir)
    data = load_dataset(args["data-dir"], cache_dir)
    run_id = adaptive_run_id(data, config)
    run_dir = joinpath(cache_dir, run_id)
    if isfile(joinpath(run_dir, "result.npz")) && !args["force"]
        println("CACHE_HIT $run_id $(joinpath(run_dir, "result.npz"))")
        return
    end

    println("RUN $run_id: adaptive base=$(config.base_nx)x$(config.base_nz), level=$(config.max_level)")
    result = run_adaptive(data, config)
    final = result.final_temperature_cells
    finite = filter(isfinite, final)
    mesh = result.mesh
    unconverged_steps = count(item -> !item["nonlinear_converged"], result.diagnostics)
    final_active = count(result.grid.active)
    metadata = OrderedDict{String,Any}(
        "run_id" => run_id, "status" => "complete", "mesh_type" => "balanced_quadtree",
        "input_directory" => data.data_dir, "input_fingerprint" => data.fingerprint,
        "solver" => asdict(config), "elapsed_seconds" => result.elapsed_seconds,
        "cell_counts_at_saved_ages" => result.cell_counts,
        "final_cells" => length(mesh.cells), "final_active_cells" => final_active,
        "minimum_dx_m" => minimum(mesh.width), "minimum_dz_m" => minimum(mesh.height),
        "final_temperature_min_c" => minimum(finite),
        "final_temperature_max_c" => maximum(finite),
        "nonlinear_converged" => unconverged_steps == 0,
        "unconverged_time_steps" => unconverged_steps,
        "refinement" => "layer top/bottom interfaces and lateral lithology transitions (fault/contact zones)",
        "diagnostics" => result.diagnostics,
    )
    save_adaptive(result, run_dir, metadata)
    update_manifest(cache_dir, metadata)
    println("SAVED $(joinpath(run_dir, "result.npz"))")
    println(@sprintf("CELLS %d ACTIVE %d MIN_CELL %.3fx%.3f m",
                     length(mesh.cells), final_active, metadata["minimum_dx_m"], metadata["minimum_dz_m"]))
    println(@sprintf("T_FINAL %.3f %.3f C", minimum(finite), maximum(finite)))
    unconverged_steps > 0 && println("WARNING NONLINEAR_NOT_CONVERGED $unconverged_steps time steps")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_adaptive()
end
