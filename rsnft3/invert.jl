# C1/C2: инверсия θ. DE (BlackBoxOptim, боксы, pop≈4·dim) → NM-полировка
# (Optim Fminbox+NelderMead) с многоуровневостью по эпохам: уровень 1 —
# молодая эпоха (10–0), уровень 2 — +35–10, уровень 3 — все. DE-оценки
# параллелятся по воркерам (ParallelEvaluation/pmap, у каждого свой датасет);
# NM — последовательный. Ансамбль — N стартов с разными seed.

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

"Индексы активных параметров уровня многоуровневости: k=1 — только молодая эпоха."
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

"DE-поиск по полному θ в боксы."
function de_search(J, layout; de_budget::Int, seed::Int, trace::Bool = false)
    res = bboptimize(θ -> J(θ)[1];
                     SearchRange = collect(zip(layout.lower, layout.upper)),
                     NumDimensions = length(layout),
                     MaxFuncEvals = de_budget,
                     TraceMode = trace ? :verbose : :silent,
                     ParallelEvaluation = nprocs() > 1,
                     RandomSeed = seed)
    return best_candidate(res), best_fitness(res), BlackBoxOptim.f_calls(res)
end

"Одна инверсия: DE по всему θ → NM-полировка по уровням эпох."
function invert_single(J, layout; de_budget::Int = 4000, multilevel::Bool = true,
                       seed::Int = 1, nm_iterations::Int = 200, trace::Bool = false)
    t0 = time()
    θde, Jde, de_evals = de_search(J, layout; de_budget = de_budget, seed = seed, trace = trace)
    θ = collect(Float64.(θde))
    nm_evals = 0
    levels = multilevel ? collect(1:length(layout.epochs)) : [length(layout.epochs)]
    for level in levels
        idx = level_indices(layout, level)
        θstart = copy(θ)
        # неактивные координаты — на приорах (x0), активные — с текущей точки
        active = fill(false, length(θ)); active[idx] .= true
        θstart[.!active] .= layout.x0[.!active]
        fsub(θsub) = begin
            θfull = copy(θstart)
            θfull[idx] .= θsub
            J(θfull)[1]
        end
        res = Optim.optimize(fsub, layout.lower[idx], layout.upper[idx], θstart[idx],
                             Fminbox(NelderMead()),
                             Optim.Options(iterations = nm_iterations,
                                           x_abstol = 1e-6, f_abstol = 1e-6))
        θ[idx] .= Optim.minimizer(res)
        nm_evals += Optim.f_calls(res)
    end
    Jfin, info = J(θ)
    return InversionResult(θ, Jfin, info.J_T, info.J_Ro, info.J_reg,
                           de_evals, nm_evals, time() - t0, seed)
end

"Ансамбль стартов; сортировка по J. Старты — последовательно, DE внутри — по воркерам."
function invert_ensemble(J, layout; ensemble::Int = 4, de_budget::Int = 4000,
                         multilevel::Bool = true, seed0::Int = 1, nm_iterations::Int = 200,
                         trace::Bool = false)
    results = InversionResult[]
    for s in 1:ensemble
        push!(results, invert_single(J, layout; de_budget = de_budget, multilevel = multilevel,
                                     seed = seed0 + s - 1, nm_iterations = nm_iterations, trace = trace))
    end
    return sort!(results; by = r -> r.J)
end
