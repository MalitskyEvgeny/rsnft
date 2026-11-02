# C1/C2: инверсия θ. DE (BlackBoxOptim, боксы, pop≈4·dim) → NM-полировка
# (Optim Fminbox+NelderMead) с многоуровневостью по эпохам: уровень 1 —
# молодая эпоха (10–0), уровень 2 — +35–10, уровень 3 — все. DE-оценки
# параллелятся по воркерам (ParallelEvaluation/pmap; каждый воркер держит
# свой Objective со своим датасетом); NM — последовательный. Ансамбль —
# N стартов с разными seed.

using Distributed
using BlackBoxOptim
using Optim

include(joinpath(@__DIR__, "objective.jl"))

struct InversionResult
    theta::Vector{Float64}
    J::Float64
    J_T::Float64
    J_Ro::Float64
    J_reg::Float64
    de_evals::Int
    nm_evals::Int
    elapsed_s::Float64
    seed::Int
end

# --- воркеры (C2): Objective строится локально на каждом процессе ------------

const _WORKER_OBJ = Ref{Any}(nothing)

"Инициализация на воркере: свой Objective (свой датасет и кэш). Вызывать
remotecall_fetch(worker_setup, w, ...) после addprocs."
function worker_setup(data, wells, cfg, layout; reg = 1.0, weight_ro = 1.0, cache_dir = "")
    _WORKER_OBJ[] = make_objective(data, wells, cfg, layout;
                                   reg = reg, weight_ro = weight_ro, cache_dir = cache_dir)
    return myid()
end

worker_J(θ::AbstractVector{<:Real}) = _WORKER_OBJ[](θ)[1]

"Синхронизация кэшей воркеров в главный JSON (слияние)."
function merge_worker_caches!(obj::Objective)
    for w in workers()
        wc = remotecall_fetch(() -> _WORKER_OBJ[].cache, w)
        merge!(obj.cache, wc)
    end
    cache_save(obj)
    return obj
end

# --- многоуровневость ---------------------------------------------------------

"Индексы активных параметров уровня: k=1 — только молодая эпоха."
function level_indices(layout::ThetaLayout, level::Int)
    nn = length(layout.nodes_x)
    ne = length(layout.epochs)
    1 <= level <= ne || error("level must be 1..$ne")
    idx = Int[]
    for e in (ne-level+1):ne          # от молодой к старой
        append!(idx, ((e-1)*nn+1):(e*nn))
    end
    return idx
end

"DE-поиск по полному θ в боксы; f_eval(θ) -> J."
function de_search(f_eval, layout; de_budget::Int, seed::Int, trace::Bool = false)
    res = bboptimize(f_eval;
                     SearchRange = collect(zip(layout.lower, layout.upper)),
                     NumDimensions = length(layout),
                     MaxFuncEvals = de_budget,
                     TraceMode = trace ? :verbose : :silent,
                     ParallelEvaluation = nprocs() > 1,
                     RandomSeed = seed)
    return best_candidate(res), best_fitness(res), BlackBoxOptim.f_calls(res)
end

"Одна инверсия: DE по всему θ → NM-полировка по уровням эпох. f_eval(θ) -> J."
function invert_single(f_eval, layout; de_budget::Int = 4000, multilevel::Bool = true,
                       seed::Int = 1, nm_iterations::Int = 200, trace::Bool = false)
    t0 = time()
    θde, Jde, de_evals = de_search(f_eval, layout; de_budget = de_budget, seed = seed, trace = trace)
    θ = collect(Float64.(θde))
    nm_evals = 0
    levels = multilevel ? collect(1:length(layout.epochs)) : [length(layout.epochs)]
    for level in levels
        idx = level_indices(layout, level)
        θstart = copy(θ)
        active = fill(false, length(θ)); active[idx] .= true
        θstart[.!active] .= layout.x0[.!active]     # неактивные — приоры
        fsub(θsub) = begin
            θfull = copy(θstart)
            θfull[idx] .= θsub
            f_eval(θfull)
        end
        res = Optim.optimize(fsub, layout.lower[idx], layout.upper[idx], θstart[idx],
                             Fminbox(NelderMead()),
                             Optim.Options(iterations = nm_iterations,
                                           x_abstol = 1e-6, f_abstol = 1e-6))
        θ[idx] .= Optim.minimizer(res)
        nm_evals += Optim.f_calls(res)
    end
    return θ, de_evals, nm_evals, time() - t0
end

"Инверсия с объективом в текущем процессе (один процесс или главный)."
function invert_local(obj::Objective; kwargs...)
    θ, de_evals, nm_evals, elapsed = invert_single(θ -> obj(θ)[1], obj.layout; kwargs...)
    J, info = obj(θ)
    return InversionResult(θ, J, info.J_T, info.J_Ro, info.J_reg,
                           de_evals, nm_evals, elapsed, get(kwargs, :seed, 1))
end

"Инверсия в воркер-режиме: DE через worker_J на всех процессах; финальная
оценка и NM-полировка — на главном (кэш воркеров сливается в главный)."
function invert_workers(obj::Objective; de_budget::Int = 4000, multilevel::Bool = true,
                        seed::Int = 1, nm_iterations::Int = 200, trace::Bool = false)
    nprocs() > 1 || return invert_local(obj; de_budget = de_budget, multilevel = multilevel,
                                        seed = seed, nm_iterations = nm_iterations, trace = trace)
    t0 = time()
    θde, Jde, de_evals = de_search(worker_J, obj.layout; de_budget = de_budget, seed = seed, trace = trace)
    θ = collect(Float64.(θde))
    nm_evals = 0
    levels = multilevel ? collect(1:length(obj.layout.epochs)) : [length(obj.layout.epochs)]
    for level in levels
        idx = level_indices(obj.layout, level)
        θstart = copy(θ)
        active = fill(false, length(θ)); active[idx] .= true
        θstart[.!active] .= obj.layout.x0[.!active]
        fsub(θsub) = begin
            θfull = copy(θstart)
            θfull[idx] .= θsub
            obj(θfull)[1]                       # NM — последовательный, на главном
        end
        res = Optim.optimize(fsub, obj.layout.lower[idx], obj.layout.upper[idx], θstart[idx],
                             Fminbox(NelderMead()),
                             Optim.Options(iterations = nm_iterations,
                                           x_abstol = 1e-6, f_abstol = 1e-6))
        θ[idx] .= Optim.minimizer(res)
        nm_evals += Optim.f_calls(res)
    end
    merge_worker_caches!(obj)
    J, info = obj(θ)
    return InversionResult(θ, J, info.J_T, info.J_Ro, info.J_reg,
                           de_evals, nm_evals, time() - t0, seed)
end

"Ансамбль стартов; сортировка по J. Старты — последовательно, DE внутри — по воркерам."
function invert_ensemble(obj::Objective; ensemble::Int = 4, de_budget::Int = 4000,
                         multilevel::Bool = true, seed0::Int = 1, nm_iterations::Int = 200,
                         trace::Bool = false)
    mode = nprocs() > 1 ? invert_workers : invert_local
    results = InversionResult[]
    for s in 1:ensemble
        push!(results, mode(obj; de_budget = de_budget, multilevel = multilevel,
                            seed = seed0 + s - 1, nm_iterations = nm_iterations, trace = trace))
    end
    return sort!(results; by = r -> r.J)
end

"Загрузка данных и параметров на воркере (вызывается remotecall_fetch)."
function worker_load!(data_dir::AbstractString, wells::WellSet, epochs::Vector{Tuple{Float64,Float64}},
                      n_nodes::Int, mid_x::Float64, nx::Int, nz::Int)
    global INV_DATA, INV_WELLS, INV_CFG, INV_LAYOUT
    INV_DATA = load_dataset_derived(data_dir, Workbook())
    INV_LAYOUT = make_q_layout(INV_DATA; epochs = epochs, n_nodes = n_nodes, mid_x = mid_x)
    INV_WELLS = wells
    INV_CFG = SolverConfig(nx = nx, nz = nz)
    return myid()
end

"Полная инициализация воркера: данные + Objective (один remotecall)."
function worker_init!(data_dir::AbstractString, wells::WellSet, epochs::Vector{Tuple{Float64,Float64}},
                      n_nodes::Int, mid_x::Float64, nx::Int, nz::Int;
                      reg::Real = 1.0, weight_ro::Real = 1.0, cache_dir::AbstractString = "")
    global INV_DATA, INV_WELLS, INV_CFG, INV_LAYOUT
    INV_DATA = load_dataset_derived(data_dir, Workbook())
    INV_LAYOUT = make_q_layout(INV_DATA; epochs = epochs, n_nodes = n_nodes, mid_x = mid_x)
    INV_WELLS = wells
    INV_CFG = SolverConfig(nx = nx, nz = nz)
    worker_setup(INV_DATA, INV_WELLS, INV_CFG, INV_LAYOUT;
                 reg = reg, weight_ro = weight_ro, cache_dir = cache_dir)
    return myid()
end
