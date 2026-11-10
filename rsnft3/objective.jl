# B3: функционал J(θ) для инверсии q. Прямая задача + выборка наблюдений:
# T(t=0) билинейно по (x, z), Ro — по частицам скважин (ro_kinetics).
# J = Σ((T−T_obs)/σ_T)² + weight_ro·Σ((Ro−Ro_obs)/σ_Ro)² + reg·Σ(Δθ)².
# Кэш θ→J: SHA-256(round(θ,10)), память + опциональный JSON-дамп.

using SHA

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))
include(joinpath(@__DIR__, "observations.jl"))
# Адаптивный бэкенд (включает basin_fvm.jl — уже загружен, guard внутри)
isdefined(@__MODULE__, :AdaptiveConfig) || include(joinpath(@__DIR__, "adaptive_basin_fvm.jl"))

struct TTarget
    x::Float64
    z::Float64
    value::Float64
    sigma::Float64
end

struct RoTarget
    x::Float64
    layer::String
    eta::Float64
    value::Float64
    sigma::Float64
end

struct Objective
    data::BasinDataset
    cfg::SolverConfig
    layout::ThetaLayout
    reg::Float64
    weight_ro::Float64
    t_targets::Vector{TTarget}
    ro_targets::Vector{RoTarget}
    n_dropped_T::Int                       # точки в неактивных ячейках сетки
    n_artifact_T::Base.RefValue{Int}       # кумулятивно: артефактные выборки T
    n_artifact_Ro::Base.RefValue{Int}      # кумулятивно: артефактные выборки Ro
    cache::Dict{String,Vector{Float64}}    # key -> [J, J_T, J_Ro, J_reg]
    cache_path::String                     # "" — только в памяти
    ctx_hash::String                       # контекст задачи в ключе кэша (F4)
    _dirty::Base.RefValue{Int}             # свежие оценки с последней записи кэша
    evals::Base.RefValue{Int}
    geometry::Any                          # ForwardGeometry | AdaptiveGeometry
    backend::Symbol                        # :regular | :adaptive
    adaptive_cfg::Union{Nothing,AdaptiveConfig}
    theta_map::Function                    # δ → θ (монотонный скалярный режим) | identity
end

"Маска активных ячеек (nz, nx): ячейка накрыта хотя бы одним слоем на age 0."
sediment_mask(data::BasinDataset, cfg::SolverConfig) = sediment_mask_xy(data, cfg.nx, cfg.nz)

function sediment_mask_xy(data::BasinDataset, nx::Int, nz::Int)
    b = data.bounds
    dx = (b[2] - b[1]) / nx; dz = (b[4] - b[3]) / nz
    xg = [b[1] + (i - 0.5) * dx for i in 1:nx]
    zg = [b[3] + (j - 0.5) * dz for j in 1:nz]
    active = falses(nz, nx)
    for (_, ser) in data.depth
        haskey(ser, 0.0) || continue
        rec = ser[0.0]
        for i in 1:nx
            (rec.x[1] > xg[i] || rec.x[end] < xg[i]) && continue
            j = clamp(searchsortedlast(rec.x, xg[i]), 1, length(rec.x) - 1)
            t = (xg[i] - rec.x[j]) / max(rec.x[j+1] - rec.x[j], 1e-9)
            top = rec.top[j] + t * (rec.top[j+1] - rec.top[j])
            bot = rec.bot[j] + t * (rec.bot[j+1] - rec.bot[j])
            for k in 1:nz
                (zg[k] >= top && zg[k] <= bot) && (active[k, i] = true)
            end
        end
    end
    return active, xg, zg
end

"""
    make_objective(data, wells, cfg, layout; reg, weight_ro, cache_dir)
        -> objective(θ) -> (J, info)

data — датасет без heat (инжекция внутри objective); reg — λ_r регуляризации
гладкости (соседние X-узлы и эпохи); cache_dir — каталог JSON-кэша θ→J
(пусто — только память; у каждого воркера свой). T-точки, чей билинейный
стенсил задевает неактивные ячейки сетки (тонкий клин осадков — артефакт
разрешения, не ошибка данных), исключаются из J и считаются в n_dropped_T.
"""
function make_objective(data::BasinDataset, wells::WellSet, cfg::SolverConfig, layout::ThetaLayout;
                        reg::Real = 1.0, weight_ro::Real = 1.0, cache_dir::AbstractString = "",
                        backend::Symbol = :regular, adaptive_cfg = nothing)
    if backend === :adaptive && adaptive_cfg !== nothing && !(wells === nothing)
        spots = Tuple{Float64,Float64}[(p.x, p.z) for p in wells.points]
        adaptive_cfg = AdaptiveConfig(base_nx = adaptive_cfg.base_nx, base_nz = adaptive_cfg.base_nz,
                                      max_level = adaptive_cfg.max_level,
                                      interface_band = adaptive_cfg.interface_band,
                                      max_dt_ma = adaptive_cfg.max_dt_ma,
                                      picard_iterations = adaptive_cfg.picard_iterations,
                                      picard_tolerance_c = adaptive_cfg.picard_tolerance_c,
                                      linear_tolerance = adaptive_cfg.linear_tolerance,
                                      preview_nx = adaptive_cfg.preview_nx,
                                      preview_nz = adaptive_cfg.preview_nz,
                                      well_spots = spots,
                                      well_refine_radius = adaptive_cfg.well_refine_radius)
    end
    adaptive_geo = backend === :adaptive ? precompute_adaptive_geometry(data, adaptive_cfg) : nothing
    if backend === :adaptive
        # маска = проекция активных ячеек финального меша (состояние age 0) на preview
        last = adaptive_geo.steps[end]
        _, _, act = mesh_to_regular(last.mesh, Float64.(last.grid.active),
                                    adaptive_cfg.preview_nx, adaptive_cfg.preview_nz)
        b = data.bounds
        pdx = (b[2] - b[1]) / adaptive_cfg.preview_nx
        pdz = (b[4] - b[3]) / adaptive_cfg.preview_nz
        xg = collect(range(b[1] + 0.5 * pdx, b[2] - 0.5 * pdx; length = adaptive_cfg.preview_nx))
        zg = collect(range(b[3] + 0.5 * pdz, b[4] - 0.5 * pdz; length = adaptive_cfg.preview_nz))
        active = act .> 0.0
    else
        active, xg, zg = sediment_mask(data, cfg)
    end
    t_targets = TTarget[]
    n_dropped = 0
    gnx, gnz = size(active, 2), size(active, 1)
    # Дополнительно к активному стенсилу отсекаем вырожденные: если активные
    # углы билинейного стенсила несут < BILINEAR_DEN_RATIO полной суммы весов,
    # точка у тонкого клина не представима — исключаем на этапе маски (DE не
    # видит «геометрию попаданий»). Веса считаются в конвенции bilinear_sample.
    function stencil_ok(x, z)
        nx, nz = gnx, gnz
        ux = (x - xg[1]) / (xg[2] - xg[1])
        uz = (z - zg[1]) / (zg[2] - zg[1])
        i0 = clamp(floor(Int, ux) + 1, 1, nx); i1 = clamp(i0 + 1, 1, nx)
        j0 = clamp(floor(Int, uz) + 1, 1, nz); j1 = clamp(j0 + 1, 1, nz)
        tx = clamp(ux - (i0 - 1), 0.0, 1.0); tz = clamp(uz - (j0 - 1), 0.0, 1.0)
        (active[j0, i0] && active[j0, i1] && active[j1, i0] && active[j1, i1]) || return false
        w00 = (1-tx)*(1-tz); w10 = tx*(1-tz); w01 = (1-tx)*tz; w11 = tx*tz
        den = (active[j0, i0] ? w00 : 0.0) + (active[j0, i1] ? w10 : 0.0) +
              (active[j1, i0] ? w01 : 0.0) + (active[j1, i1] ? w11 : 0.0)
        den >= BILINEAR_DEN_RATIO
    end
    for p in wells.points
        if p.kind === :T
            stencil_ok(p.x, p.z) ? push!(t_targets, TTarget(p.x, p.z, p.value, p.sigma)) : (n_dropped += 1)
        end
    end
    ro_targets = RoTarget[]
    for p in wells.points
        p.kind === :Ro || continue
        le = layer_eta_at(data, p.x, p.z)
        le === nothing && continue
        push!(ro_targets, RoTarget(p.x, le[1], le[2], p.value, p.sigma))
    end
    cache_path = isempty(cache_dir) ? "" : joinpath(cache_dir, "j_cache_v2.json")
    if isempty(cache_dir) == false && isfile(joinpath(cache_dir, "j_cache.json"))
        @warn "Найден старый j_cache.json — формат v1 без контекстного ключа, игнорируется (новый: j_cache_v2.json)"
    end
    cache = Dict{String,Vector{Float64}}()
    if !isempty(cache_path) && isfile(cache_path)
        for (k, v) in JSON.parsefile(cache_path)
            v isa AbstractVector ?
                cache[k] = Float64[x === nothing ? NaN : Float64(x) for x in v] :
                cache[k] = [Float64(v), NaN, NaN, NaN]   # старый формат (скаляр J)
        end
    end
    geometry = backend === :adaptive ? adaptive_geo : precompute_geometry(data, cfg)
    # ctx_hash должен покрывать всё, что влияет на значение J (иначе кэш
    # молча возвращает устаревшие J при смене sigma/допусков/таргетов):
    targets_ctx = Any[
        ["T", p.x, p.z, p.value, p.sigma] for p in t_targets] ∪
        Any[["Ro", p.x, p.layer, p.eta, p.value, p.sigma] for p in ro_targets]
    ctx = OrderedDict(
        "version" => J_CACHE_VERSION, "input" => data.fingerprint,
        "nx" => cfg.nx, "nz" => cfg.nz,
        "max_dt_ma" => cfg.max_dt_ma,
        "picard_iterations" => cfg.picard_iterations,
        "picard_tolerance_c" => cfg.picard_tolerance_c,
        "pcg_tolerance" => cfg.pcg_tolerance,
        "epochs" => layout.epochs, "nodes_x" => layout.nodes_x,
        "lower" => layout.lower, "upper" => layout.upper,
        "reg" => reg, "weight_ro" => weight_ro,
        "backend" => String(backend),
        "adaptive_cfg" => backend === :adaptive ? adaptive_cfg : nothing,
        "targets" => sort(targets_ctx; by = x -> (x[1], x[2], x[3])),
        "n_T" => length(t_targets), "n_Ro" => length(ro_targets))
    return Objective(data, cfg, layout, Float64(reg), Float64(weight_ro),
                     t_targets, ro_targets, n_dropped, Ref(0), Ref(0),
                     cache, cache_path, bytes2hex(sha256(JSON.json(ctx)))[1:16], Ref(0),
                     Ref(0), geometry, backend, backend === :adaptive ? adaptive_cfg : nothing,
                     layout.monotone ? (ue -> theta_expand_mono(layout, ue)) : identity)
end

"θ → qbasal (Вт/м²) на x-сетке прогона для каждого возраста; эквивалент
инъекции make_param_data + boundary_value (те же interp1 по nodes_x, ×1e-3)."
function qbasal_by_age(layout::ThetaLayout, θ::AbstractVector{<:Real},
                       x::AbstractVector, ages::Vector{Float64})
    nn = length(layout.nodes_x)
    out = Dict{Float64,Vector{Float64}}()
    for age in ages
        ie = epoch_index_of(layout, age)
        q = [Float64(θ[(ie-1)*nn + ix]) for ix in 1:nn]
        out[age] = interp1(x, layout.nodes_x, q) .* 1.0e-3
    end
    return out
end

n_targets(obj::Objective, kind::Symbol) =
    kind === :T ? length(obj.t_targets) : length(obj.ro_targets)

"Вызов objective(θ) -> (J, info); сам Objective хранит кэш, счётчики и мишени."
(obj::Objective)(θ::AbstractVector{<:Real}) = evaluate(obj, θ)

const J_CACHE_VERSION = 3

"Ключ кэша: θ ⊕ контекст (данные, сетка, layout, reg, weight_ro, backend, версия) — F4."
cache_key(θ::AbstractVector{<:Real}, ctx_hash::AbstractString = "") =
    bytes2hex(sha256(join(string.(round.(Float64.(θ); digits = 10)), ",") * "|" * ctx_hash))[1:16]

json_safe(v::Vector{Float64}) = Any[isfinite(x) ? x : nothing for x in v]

const CACHE_SAVE_EVERY = 64

function cache_save(obj::Objective; force::Bool = false)
    isempty(obj.cache_path) && return
    # 2.2: пишем раз в CACHE_SAVE_EVERY свежих оценок (или force) — не на каждой
    force = force || obj._dirty[] >= CACHE_SAVE_EVERY
    force || return
    mkpath(dirname(obj.cache_path))
    tmp = obj.cache_path * ".tmp"
    open(tmp, "w") do f
        JSON.print(f, OrderedDict(k => json_safe(v) for (k, v) in obj.cache))
    end
    # Windows: антивирус может кратковременно держать целевой файл — EACCES на
    # перезаписи. Кэш — только оптимизация: тихо пропускаем запись, переживём.
    try
        mv(tmp, obj.cache_path; force = true)  # атомарно на практике
        obj._dirty[] = 0
    catch
        rm(tmp; force = true)
    end
end

"Оценка J: прогон прямой задачи, выборка T/Ro, регуляризация. Возвращает (J, info).
Монотонный скалярный режим: θ здесь — δ; он сначала разворачивается
в реальные q (theta_map), кэш-ключ — по δ."
function evaluate(obj::Objective, θ::AbstractVector{<:Real})
    θ = obj.theta_map(θ)
    key = cache_key(θ, obj.ctx_hash)
    if haskey(obj.cache, key)
        v = obj.cache[key]
        return v[1], (J_T = v[2], J_Ro = v[3], J_reg = v[4], cached = true,
                      n_artifact_T = 0, n_artifact_Ro = 0,
                      evals = obj.evals[], t_seconds = 0.0)
    end
    t0 = time()
    obj.evals[] += 1
    if obj.backend === :adaptive
        ac = obj.adaptive_cfg
        ax0, ax1 = obj.data.bounds[1], obj.data.bounds[2]
        adx = (ax1 - ax0) / ac.preview_nx
        axg = collect(range(ax0 + 0.5 * adx, ax1 - 0.5 * adx; length = ac.preview_nx))
        result = run_adaptive(obj.data, ac;
                              qbasal_by_age = qbasal_by_age(obj.layout, θ, axg, obj.data.ages),
                              qbasal_nodes_x = obj.layout.nodes_x,
                              geometry = obj.geometry)
    else
        result = run_forward(obj.data, obj.cfg; geometry = obj.geometry,
                             qbasal_by_age = qbasal_by_age(obj.layout, θ, obj.geometry.x, obj.data.ages))
    end
    xg, zg = result.x, result.z
    temp0 = result.temperature_c[end, :, :]          # (nz, nx), t = 0

    J_T = 0.0; n_art_T = 0
    for t in obj.t_targets
        v, degenerate = bilinear_sample_flagged(xg, zg, temp0, t.x, t.z)
        w = t.sigma > 0 ? 1.0 / t.sigma^2 : 1.0
        if isfinite(v) && !degenerate
            J_T += w * (v - t.value)^2
        else
            n_art_T += 1                    # артефакт: вклад 0, счётчик в info
        end
    end

    J_Ro = 0.0; n_art_Ro = 0
    for t in obj.ro_targets
        rc = particle_ro(result, obj.data, t.x, t.layer, t.eta)
        w = t.sigma > 0 ? 1.0 / t.sigma^2 : 1.0
        if isfinite(rc)
            J_Ro += w * (rc - t.value)^2
        else
            n_art_Ro += 1
        end
    end
    obj.n_artifact_T[] += n_art_T
    obj.n_artifact_Ro[] += n_art_Ro

    d_node, d_epoch = theta_diffs(obj.layout, θ)
    J_reg = obj.reg * (sum(abs2, d_node) + sum(abs2, d_epoch))

    J = J_T + obj.weight_ro * J_Ro + J_reg
    obj.cache[key] = [J, J_T, J_Ro, J_reg]
    obj._dirty[] += 1
    cache_save(obj)
    return J, (J_T = J_T, J_Ro = J_Ro, J_reg = J_reg, cached = false,
               n_artifact_T = n_art_T, n_artifact_Ro = n_art_Ro,
               evals = obj.evals[], t_seconds = time() - t0)
end
