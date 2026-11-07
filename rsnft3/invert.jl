# C1/C2: инверсия θ. DE (BlackBoxOptim, боксы, pop≈4·dim) → NM-полировка
# (Optim Fminbox+NelderMead) с многоуровневостью по эпохам: уровень 1 —
# молодая эпоха (10–0), уровень 2 — +35–10, уровень 3 — все. DE-оценки
# параллелятся по воркерам (ParallelEvaluation/pmap; каждый воркер держит
# свой Objective со своим датасетом); NM — последовательный. Ансамбль —
# N стартов с разными seed.

using Distributed
using Random
using BlackBoxOptim
using Optim

include(joinpath(@__DIR__, "objective.jl"))

include(joinpath(@__DIR__, "progress.jl"))
using .Progress

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

# 7.3: почему Ref, а не явная передача контекста.
# Objective нельзя пересылать между процессами: он держит BasinDataset
# (десятки МБ XLSX-данных), Ref'ы (n_artifact/_dirty/evals), Dict-кэш с
# bytes2hex-ключами и geometry-структуры; Distributed сериализует замыкания
# по значению, поэтому каждый воркер строит СВОЙ Objective через
# remotecall_fetch(worker_init!/worker_init_c2f!, ...) (см. run_inverse.jl),
# а _WORKER_OBJ[] — локальная ссылка на этот экземпляр. Функции типа
# worker_J(θ) и () -> _WORKER_OBJ[].cache читают Ref на СВОЕМ процессе.
# Явная передача контекста (Objective как аргумент remotecall) сделала бы
# сериализацию данных на каждый вызов — в разы медленнее и с риском
# рассинхронизации кэшей; протокол «инициализация через remotecall_fetch +
# локальный Ref» остаётся осознанным решением.

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
        wctx = remotecall_fetch(() -> _WORKER_OBJ[].ctx_hash, w)
        # v2-ключи контекстны: воркеры обязаны иметь тот же контекст, что и главный
        wctx == obj.ctx_hash || error("cache context mismatch: worker $w ($(wctx[1:8])) != main ($(obj.ctx_hash[1:8]))")
        wc = remotecall_fetch(() -> _WORKER_OBJ[].cache, w)
        merge!(obj.cache, wc)
    end
    cache_save(obj; force = true)
    return obj
end

# --- воркеры c2f (ансамбль в c2f-режиме): воркер держит два Objective ---------

const _WORKER_OBJ_COARSE = Ref{Any}(nothing)

worker_J_coarse(θ::AbstractVector{<:Real}) = _WORKER_OBJ_COARSE[](θ)[1]

"Инициализация воркера для c2f-ансамбля: два Objective (coarse + fine) с
раздельными кэшами, за один remotecall. Параметры: именованные coarse_* и
fine_* (SolverConfig-поля, reg, weight_ro, cache_dir, nodes_x)."
function worker_init_c2f!(data_dir::AbstractString, wells::WellSet,
                          epochs::Vector{Tuple{Float64,Float64}}, n_nodes::Int,
                          mid_x::Float64;
                          coarse_nx::Int, coarse_nz::Int, coarse_cache_dir::AbstractString,
                          fine_nx::Int, fine_nz::Int, fine_cache_dir::AbstractString,
                          nodes_x::Union{Nothing,Vector{Float64}} = nothing,
                          reg::Real = 1.0, weight_ro::Real = 1.0,
                          max_dt_ma::Real = 2.5, picard_iterations::Int = 3,
                          picard_tolerance_c::Real = 1.0e-3,
                          pcg_tolerance::Real = 1.0e-6,
                          linear_solver::Symbol = :pcg)
    data = load_dataset_derived(data_dir, Workbook())
    layout = make_q_layout(data; epochs = epochs, n_nodes = n_nodes, mid_x = mid_x,
                           nodes_x = nodes_x)
    common = (max_dt_ma = Float64(max_dt_ma), picard_iterations = picard_iterations,
              picard_tolerance_c = Float64(picard_tolerance_c),
              pcg_tolerance = Float64(pcg_tolerance),
              linear_solver = linear_solver)
    ccfg = SolverConfig(nx = coarse_nx, nz = coarse_nz; common...)
    fcfg = SolverConfig(nx = fine_nx, nz = fine_nz; common...)
    _WORKER_OBJ_COARSE[] = make_objective(data, wells, ccfg, layout;
                                          reg = reg, weight_ro = weight_ro,
                                          cache_dir = coarse_cache_dir)
    _WORKER_OBJ[] = make_objective(data, wells, fcfg, layout;
                                   reg = reg, weight_ro = weight_ro,
                                   cache_dir = fine_cache_dir)
    return myid()
end

"Слияние кэшей воркеров c2f-ансамбля: coarse-кэш в obj_coarse, fine-кэш в
obj_fine; оба главных кэша сохраняются."
function merge_worker_caches_c2f!(obj_coarse::Objective, obj_fine::Objective)
    for w in workers()
        wctx = remotecall_fetch(() -> _WORKER_OBJ_COARSE[].ctx_hash, w)
        wctx == obj_coarse.ctx_hash ||
            error("cache context mismatch (coarse): worker $w != main")
        wctxF = remotecall_fetch(() -> _WORKER_OBJ[].ctx_hash, w)
        wctxF == obj_fine.ctx_hash ||
            error("cache context mismatch (fine): worker $w != main")
        wc = remotecall_fetch(() -> _WORKER_OBJ_COARSE[].cache, w)
        merge!(obj_coarse.cache, wc)
        wf = remotecall_fetch(() -> _WORKER_OBJ[].cache, w)
        merge!(obj_fine.cache, wf)
    end
    cache_save(obj_coarse; force = true)
    cache_save(obj_fine; force = true)
    return nothing
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

"DE-поиск по полному θ в боксы; f_eval(θ) -> J. Seed сеем сами: RandomSeed
BBO 0.6.12 obsolete и молча игнорируется — без Random.seed! прогон
невоспроизводим даже при том же seed."
function de_search(f_eval, layout; de_budget::Int, seed::Int, trace::Bool = false,
                   warm_start = nothing, progress_chan = nothing)
    Random.seed!(seed)
    de_cb(ctrl) = begin
        fe = BlackBoxOptim.num_func_evals(ctrl)
        Jb = best_fitness(ctrl)
        if progress_chan === nothing
            Progress.bar_add(0, @sprintf("DE search · eval %d/%d · best J=%.1f",
                                         fe, de_budget, Jb))
        else
            put!(progress_chan, (myid(), fe, Jb))
        end
    end
    Progress.bar_start(de_budget, "DE search")
    res = bboptimize(f_eval, warm_start;
                     SearchRange = collect(zip(layout.lower, layout.upper)),
                     NumDimensions = length(layout),
                     MaxFuncEvals = de_budget,
                     TraceMode = trace ? :verbose : :silent,
                     CallbackFunction = de_cb,
                     CallbackInterval = 1.0,
                     ParallelEvaluation = (myid() == 1 && nprocs() > 1))
    Progress.bar_finish()
    return best_candidate(res), best_fitness(res), BlackBoxOptim.f_calls(res)
end

"Общая NM-полировка по уровням эпох (дедуп invert_single/invert_workers/polish_from):
модифицирует θ на месте, возвращает число NM-оценок. f_eval(θfull) -> J."
function polish_nm!(f_eval, layout::ThetaLayout, θ::Vector{Float64};
                    multilevel::Bool = true, nm_iterations::Int = 200, max_f_calls::Int = 0,
                    progress_chan = nothing, fe_offset::Int = 0)
    nm_evals = 0
    levels = multilevel ? collect(1:length(layout.epochs)) : [length(layout.epochs)]
    nm_t0 = time()
    for (li, level) in enumerate(levels)
        Progress.step_note(@sprintf("NM polish · level %d/%d · elapsed %s",
                                    li, length(levels), Progress.fmt_time_public(time() - nm_t0)))
        idx = level_indices(layout, level)
        θstart = copy(θ)
        active = fill(false, length(θ)); active[idx] .= true
        θstart[.!active] .= layout.x0[.!active]     # неактивные — приоры
        fsub(θsub) = begin
            θfull = copy(θstart)
            θfull[idx] .= θsub
            f_eval(θfull)
        end
        fe = Ref(0)
        fsub_raw = (θsub) -> begin
            θfull = copy(θstart)
            θfull[idx] .= θsub
            f_eval(θfull)
        end
        fsub = (θsub) -> begin
            J = fsub_raw(θsub)
            fe[] += 1
            progress_chan === nothing || put!(progress_chan, (myid(), fe_offset + fe[], J))
            J
        end
        res = Optim.optimize(fsub, layout.lower[idx], layout.upper[idx], θstart[idx],
                             Fminbox(NelderMead()),
                             Optim.Options(iterations = nm_iterations,
                                           f_calls_limit = max_f_calls > 0 ? max_f_calls : typemax(Int),
                                           x_abstol = 1e-6, f_abstol = 1e-6))
        θ[idx] .= Optim.minimizer(res)
        nm_evals += Optim.f_calls(res)
    end
    return nm_evals
end

"Одна инверсия: DE по всему θ → NM-полировка по уровням эпох. f_eval(θ) -> J."
function invert_single(f_eval, layout; de_budget::Int = 4000, multilevel::Bool = true,
                       seed::Int = 1, nm_iterations::Int = 200, trace::Bool = false,
                       max_f_calls::Int = 0, warm_start = nothing, progress_chan = nothing)
    t0 = time()
    θde, Jde, de_evals = de_search(f_eval, layout; de_budget = de_budget, seed = seed,
                                   trace = trace, warm_start = warm_start,
                                   progress_chan = progress_chan)
    θ = collect(Float64.(θde))
    nm_evals = polish_nm!(f_eval, layout, θ; multilevel = multilevel,
                          nm_iterations = nm_iterations, max_f_calls = max_f_calls,
                          progress_chan = progress_chan, fe_offset = de_evals)
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
                        seed::Int = 1, nm_iterations::Int = 200, trace::Bool = false,
                        max_f_calls::Int = 0, warm_start = nothing)
    nprocs() > 1 || return invert_local(obj; de_budget = de_budget, multilevel = multilevel,
                                        seed = seed, nm_iterations = nm_iterations,
                                        trace = trace, warm_start = warm_start)
    t0 = time()
    θde, Jde, de_evals = de_search(worker_J, obj.layout; de_budget = de_budget, seed = seed,
                                   trace = trace, warm_start = warm_start)
    θ = collect(Float64.(θde))
    nm_evals = polish_nm!(θ -> obj(θ)[1], obj.layout, θ; multilevel = multilevel,
                          nm_iterations = nm_iterations, max_f_calls = max_f_calls)
    merge_worker_caches!(obj)
    J, info = obj(θ)
    return InversionResult(θ, J, info.J_T, info.J_Ro, info.J_reg,
                           de_evals, nm_evals, time() - t0, seed)
end

"Старт ансамбля целиком на воркере: DE серийно + NM серийно на локальном
Objective. Вызывается remotecall'ом из invert_ensemble_parallel; параметры —
именованный кортеж (позиционный аргумент: без kwargs-замыкания через
Distributed нет world-age конфликта с @everywhere-include)."
function invert_start_on_worker(kw::NamedTuple)
    _WORKER_OBJ[] === nothing && error("воркер не инициализирован (worker_init!)")
    return invert_local(_WORKER_OBJ[]; kw...)
end

"Ансамбль стартов параллельно: один старт целиком (DE серийно + NM серийно) на
одном воркере, старты отправляются динамически по свободным воркерам — DE
старта s+1 перекрывается с NM старта s. Детерминизм: seed = seed0+s-1, сбор в
порядке стартов, сортировка по J; слияние кэшей воркеров в главный.

F11: воркеры инициализируются явным remotecall(worker_init!/worker_init_c2f!)
в run_inverse.jl ДО вызовов invert_ensemble_parallel — Objective не
пересылается по сети (нельзя: BasinDataset/Ref'ы не сериализуемы надёжно,
отсюда warning «Cannot transfer global variable _WORKER_OBJ», если пытаться).
Воркеры уже инициализированы с собственным _WORKER_OBJ[] (см. _WORKER_OBJ выше).
Многопоточность (Threads): Random.seed! в de_search сеет глобальный RNG
задачи; при будущей threads-параллелизации сеять per-task RNG
(Random.default_rng()/Xoshiro(seed)), иначе синхронизация семян сломается."
function invert_ensemble_parallel(obj::Objective; ensemble::Int = 4, de_budget::Int = 4000,
                                  multilevel::Bool = true, seed0::Int = 1,
                                  nm_iterations::Int = 200, trace::Bool = false,
                                  max_f_calls::Int = 0, warm_start = nothing)
    ws = collect(workers())
    isempty(ws) && error("нет воркеров")
    futures = Vector{Future}(undef, ensemble)
    idle = ws[:]
    busy = Dict{Int,Vector{Future}}()          # воркер -> отправленные туда старты
    next_start = 1
    # живой DE-прогресс с воркеров: колбэк de_search кладёт (pid, f_calls, J),
    # читатель на главном двигает полосу; сумма по всем стартам = ensemble·budget
    chan = RemoteChannel(() -> Channel{Tuple{Int,Int,Float64}}(64))
    nm_soft_per = max_f_calls > 0 ? max_f_calls : 2 * nm_iterations * 4
    nm_soft = ensemble * nm_soft_per
    Progress.bar_start(ensemble * de_budget + nm_soft, "DE + NM polish (ensemble)")
    last_fe = Dict{Int,Int}()
    reader = @async begin
        while true
            msg = try take!(chan) catch; break end
            delta = msg[2] - get(last_fe, msg[1], 0)
            last_fe[msg[1]] = msg[2]
            phase = msg[2] <= de_budget ? "DE search" : "NM polish"
            tot_fe = sum(values(last_fe))
            Progress.bar_add(delta, @sprintf("eval %d/%d · best J=%.1f",
                                             tot_fe, ensemble * de_budget + nm_soft, msg[3]))
            Progress.worker_bar_update(msg[1], min(msg[2], de_budget + nm_soft_per),
                                       de_budget + nm_soft_per,
                                       @sprintf("%s · eval %d/%d · best J=%.1f",
                                                phase, min(msg[2], de_budget + nm_soft_per),
                                                de_budget + nm_soft_per, msg[3]))
        end
        Progress.bar_finish()
    end
    while next_start <= ensemble || !isempty(busy)
        while next_start <= ensemble && !isempty(idle)
            w = popfirst!(idle)
            s = next_start
            f = remotecall(invert_start_on_worker, w,
                           (de_budget = de_budget, multilevel = multilevel,
                            seed = seed0 + s - 1, nm_iterations = nm_iterations,
                            trace = trace, max_f_calls = max_f_calls,
                            warm_start = (s == 1 ? warm_start : nothing),
                            progress_chan = chan))
            futures[s] = f
            push!(get!(busy, w, Future[]), f)
            next_start += 1
        end
        done_ws = Int[w for w in keys(busy) if all(isready, busy[w])]
        if isempty(done_ws)
            wait(first(first(busy)[2]))         # ждём ближайший готовый старт
            continue
        end
        done_count = ensemble - next_start + 1 + length(done_ws)
        for w in done_ws
            push!(idle, w)
            delete!(busy, w)
            Progress.worker_bar_done(w)
        end
        Progress.step_note(@sprintf("ensemble starts: %d/%d done", done_count, ensemble))
    end
    results = [fetch(f) for f in futures]
    close(chan)
    wait(reader)
    merge_worker_caches!(obj)
    return sort!(results; by = r -> r.J)
end

"Ансамбль стартов; сортировка по J. Диспетч по заполнению ядер: стартов
больше или равно воркерам — старт целиком на воркере, параллельно (все ядра
заняты волнами, NM перекрыт); стартов меньше — legacy: DE каждого старта
pmap'ит по всем воркерам, последовательно."
function invert_ensemble(obj::Objective; ensemble::Int = 4, de_budget::Int = 4000,
                         multilevel::Bool = true, seed0::Int = 1, nm_iterations::Int = 200,
                         trace::Bool = false, max_f_calls::Int = 0, warm_start = nothing)
    if nprocs() > 1 && ensemble >= nworkers()
        return invert_ensemble_parallel(obj; ensemble = ensemble, de_budget = de_budget,
                                        multilevel = multilevel, seed0 = seed0,
                                        nm_iterations = nm_iterations, trace = trace,
                                        max_f_calls = max_f_calls, warm_start = warm_start)
    end
    mode = nprocs() > 1 ? invert_workers : invert_local
    results = InversionResult[]
    for s in 1:ensemble
        Progress.step_note(@sprintf("ensemble start %d/%d (seed %d)", s, ensemble, seed0 + s - 1))
        push!(results, mode(obj; de_budget = de_budget, multilevel = multilevel,
                            seed = seed0 + s - 1, nm_iterations = nm_iterations,
                            trace = trace, max_f_calls = max_f_calls,
                            warm_start = (s == 1 ? warm_start : nothing)))
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
                      reg::Real = 1.0, weight_ro::Real = 1.0, cache_dir::AbstractString = "",
                      max_dt_ma::Real = 2.5, picard_iterations::Int = 3,
                      picard_tolerance_c::Real = 1.0e-3, pcg_tolerance::Real = 1.0e-6,
                      linear_solver::Symbol = :pcg,
                      nodes_x::Union{Nothing,Vector{Float64}} = nothing)
    global INV_DATA, INV_WELLS, INV_CFG, INV_LAYOUT
    INV_DATA = load_dataset_derived(data_dir, Workbook())
    INV_LAYOUT = make_q_layout(INV_DATA; epochs = epochs, n_nodes = n_nodes, mid_x = mid_x,
                               nodes_x = nodes_x)
    INV_WELLS = wells
    INV_CFG = SolverConfig(nx = nx, nz = nz, max_dt_ma = Float64(max_dt_ma),
                           picard_iterations = picard_iterations,
                           picard_tolerance_c = Float64(picard_tolerance_c),
                           pcg_tolerance = Float64(pcg_tolerance),
                           linear_solver = linear_solver)
    worker_setup(INV_DATA, INV_WELLS, INV_CFG, INV_LAYOUT;
                 reg = reg, weight_ro = weight_ro, cache_dir = cache_dir)
    return myid()
end

"Полировка без DE: NM по уровням эпох из заданного старта (иерархия сеток:
DE на грубой, полировка на рабочей)."
function polish_from(obj::Objective, θ0::Vector{Float64}; nm_iterations::Int = 200,
                     multilevel::Bool = true, max_f_calls::Int = 0, progress_chan = nothing)
    t0 = time()
    θ = collect(Float64.(θ0))
    length(θ) == length(obj.layout) || error("theta0 length mismatch")
    nm_evals = polish_nm!(θ -> obj(θ)[1], obj.layout, θ; multilevel = multilevel,
                          nm_iterations = nm_iterations, max_f_calls = max_f_calls,
                          progress_chan = progress_chan)
    J, info = obj(θ)
    return InversionResult(θ, J, info.J_T, info.J_Ro, info.J_reg,
                           0, nm_evals, time() - t0, 0)
end

# --- coarse-to-fine (иерархия сеток) ------------------------------------------

"Результат coarse-стадии c2f: θ, J и времена на грубой сетке."
struct C2fStage
    theta::Vector{Float64}
    J::Float64
    de_evals::Int
    elapsed_s::Float64
    seed::Int
end

"DE-поиск на грубом Objective (без NM): θ, J, число оценок DE.
Воркер-режим требует, чтобы воркеры И главный имели проинициализированный
_WORKER_OBJ (смотри c2f_stage_init! в run_inverse.jl)."
function de_stage(obj::Objective; de_budget::Int, seed::Int, trace::Bool = false)
    f_eval = nprocs() > 1 && _WORKER_OBJ[] !== nothing ? worker_J : (θ -> obj(θ)[1])
    θde, Jde, de_evals = de_search(f_eval, obj.layout;
                                   de_budget = de_budget, seed = seed, trace = trace)
    return C2fStage(collect(Float64.(θde)), Float64(Jde), de_evals, 0.0, seed)
end

"""
    invert_c2f(obj_coarse, obj_fine; de_budget, nm_iterations, multilevel,
               seed, trace, max_f_calls) -> (InversionResult, C2fStage)

Двухстадийная инверсия: DE на obj_coarse (грубая сетка; воркер-режим — через
worker_J), затем NM-полировка на obj_fine от θ_coarse (polish_from, с
многоуровневостью по эпохам). Layout у обоих Objective одинаков — θ
переносится без пересчёта. Второе значение — coarse-стадия для отчёта.

Вызов (пере)инициализации воркеров и главного `_WORKER_OBJ[]` под нужную
стадию — обязанность вызывающей стороны (см. c2f_worker_reinit! в
run_inverse.jl): invert_c2f не знает параметров датасета.
"""
function invert_c2f(obj_coarse::Objective, obj_fine::Objective;
                    de_budget::Int = 4000, nm_iterations::Int = 200,
                    multilevel::Bool = true, seed::Int = 1, trace::Bool = false,
                    max_f_calls::Int = 0)
    length(obj_coarse.layout) == length(obj_fine.layout) ||
        error("c2f: coarse and fine layouts must match")
    t0 = time()
    stage = de_stage(obj_coarse; de_budget = de_budget, seed = seed, trace = trace)
    println(@sprintf("C2F stage A (coarse): J=%.1f in %.1fs (DE evals %d)",
                     stage.J, time() - t0, stage.de_evals))
    polished = polish_from(obj_fine, stage.theta; nm_iterations = nm_iterations,
                           multilevel = multilevel, max_f_calls = max_f_calls)
    return polished, stage
end

"Старт c2f-ансамбля целиком на воркере: серийный DE на coarse-Objective (через
worker_J_coarse) -> NM-полировка на fine-Objective. Вызывается remotecall'ом;
параметры — именованный кортеж (позиционный аргумент, без kwargs-замыкания)."
function invert_start_c2f_on_worker(kw::NamedTuple)
    de_budget::Int = kw.de_budget
    nm_iterations::Int = kw.nm_iterations
    multilevel::Bool = kw.multilevel
    seed::Int = kw.seed
    trace::Bool = kw.trace
    max_f_calls::Int = kw.max_f_calls
    (_WORKER_OBJ_COARSE[] === nothing || _WORKER_OBJ[] === nothing) &&
        error("воркер не инициализирован для c2f (worker_init_c2f!)")
    t0 = time()
    θde, Jde, de_evals = de_search(worker_J_coarse, _WORKER_OBJ_COARSE[].layout;
                                   de_budget = de_budget, seed = seed, trace = trace,
                                   progress_chan = get(kw, :progress_chan, nothing))
    stage = C2fStage(collect(Float64.(θde)), Float64(Jde), de_evals, 0.0, seed)
    polished = polish_from(_WORKER_OBJ[], stage.theta; nm_iterations = nm_iterations,
                           multilevel = multilevel, max_f_calls = max_f_calls,
                           progress_chan = get(kw, :progress_chan, nothing))
    result = InversionResult(polished.theta, polished.J, polished.J_T, polished.J_Ro,
                             polished.J_reg, polished.de_evals, polished.nm_evals,
                             polished.elapsed_s, seed)
    return result, stage
end

"Ансамбль c2f-стартов параллельно: воркер держит два Objective (coarse + fine) и
выполняет старт целиком, без переинициализаций между стадиями и стартами.
Детерминизм: seed = seed0+s-1; сбор в порядке стартов, сортировка по J; слияние
кэшей воркеров (coarse и fine) в главные Objective."
function invert_ensemble_c2f(obj_coarse::Objective, obj_fine::Objective;
                             ensemble::Int = 4, de_budget::Int = 4000,
                             nm_iterations::Int = 200, multilevel::Bool = true,
                             seed0::Int = 1, trace::Bool = false, max_f_calls::Int = 0)
    length(obj_coarse.layout) == length(obj_fine.layout) ||
        error("c2f: coarse and fine layouts must match")
    ws = collect(workers())
    isempty(ws) && error("нет воркеров")
    futures = Vector{Future}(undef, ensemble)
    idle = ws[:]
    busy = Dict{Int,Vector{Future}}()
    next_start = 1
    # # живой прогресс стадии A (coarse DE) с воркеров
    chan = RemoteChannel(() -> Channel{Tuple{Int,Int,Float64}}(64))
    nm_soft_per = max_f_calls > 0 ? max_f_calls : 2 * nm_iterations * 4
    nm_soft = ensemble * nm_soft_per
    Progress.bar_start(ensemble * de_budget + nm_soft, "DE + NM polish (c2f coarse ensemble)")
    last_fe = Dict{Int,Int}()
    reader = @async begin
        while true
            msg = try take!(chan) catch; break end
            delta = msg[2] - get(last_fe, msg[1], 0)
            last_fe[msg[1]] = msg[2]
            phase = msg[2] <= de_budget ? "DE search (coarse)" : "NM polish (coarse)"
            tot_fe = sum(values(last_fe))
            Progress.bar_add(delta, @sprintf("eval %d/%d · best J=%.1f",
                                             tot_fe, ensemble * de_budget + nm_soft, msg[3]))
            Progress.worker_bar_update(msg[1], min(msg[2], de_budget + nm_soft_per),
                                       de_budget + nm_soft_per,
                                       @sprintf("%s · eval %d/%d · best J=%.1f",
                                                phase, min(msg[2], de_budget + nm_soft_per),
                                                de_budget + nm_soft_per, msg[3]))
        end
        Progress.bar_finish()
    end
    while next_start <= ensemble || !isempty(busy)
        while next_start <= ensemble && !isempty(idle)
            w = popfirst!(idle)
            s = next_start
            f = remotecall(invert_start_c2f_on_worker, w,
                           (de_budget = de_budget, nm_iterations = nm_iterations,
                            multilevel = multilevel, seed = seed0 + s - 1,
                            trace = trace, max_f_calls = max_f_calls,
                            progress_chan = chan))
            futures[s] = f
            push!(get!(busy, w, Future[]), f)
            next_start += 1
        end
        done_ws = Int[w for w in keys(busy) if all(isready, busy[w])]
        if isempty(done_ws)
            wait(first(first(busy)[2]))
            continue
        end
        for w in done_ws
            push!(idle, w)
            delete!(busy, w)
            Progress.worker_bar_done(w)
        end
    end
    pairs = [fetch(f) for f in futures]
    close(chan)
    wait(reader)
    results = InversionResult[p[1] for p in pairs]
    stages = C2fStage[p[2] for p in pairs]
    merge_worker_caches_c2f!(obj_coarse, obj_fine)
    order = sortperm(results; by = r -> r.J)
    return results[order], stages[order]
end

"Переинициализация воркеров под стадию c2f: для каждого воркера —
удалённый вызов пере-инициализатора f под SolverConfig данной стадии."
function c2f_stage_init!(f::Function, args...; kwargs...)
    for w in workers()
        remotecall_fetch(f, w, args...; kwargs...)
    end
    return nothing
end
