# Parameterized data layer for the inverse problem (R3).
#
# Обёртка над BasinDataset: синтезирует Workbook heat из вектора q (мВт/м²),
# геометрия/Tsurf/свойства — из исходных данных без изменений. Форвард-модель
# не правится. Части плана: constant_heat_workbook (шаг 0.0), ThetaLayout,
# make_param_data (B1/B2).
#
# Запуск автономно не предполагается — include из objective.jl / make_twin.jl /
# run_inverse.jl.

isdefined(@__MODULE__, :BasinDataset) || include(joinpath(@__DIR__, "basin_fvm.jl"))

# ----------------------------------------------------------------------------
# Синтетический Workbook heat
# ----------------------------------------------------------------------------

"Workbook heat из константы q [мВт/м²] на все возрасты (одна запись `Constant`)."
function constant_heat_workbook(data, q::Real; n_nodes::Int = 33)
    ages = ages_of(data)
    bounds = bounds_of(data)
    xs = collect(range(bounds[1], bounds[2]; length = n_nodes))
    d = Dict{Float64,Record}()
    for age in ages
        d[age] = Record(xs, fill(NaN, n_nodes), fill(NaN, n_nodes), fill(Float64(q), n_nodes))
    end
    return Workbook("Constant" => d)
end

"Workbook heat из ступенчатого профиля q [мВт/м²] по эпохам (константа внутри эпохи)."
function piecewise_heat_workbook(data, epoch_q::Vector{Pair{Tuple{Float64,Float64},Float64}}; n_nodes::Int = 33)
    ages = ages_of(data)
    bounds = bounds_of(data)
    xs = collect(range(bounds[1], bounds[2]; length = n_nodes))
    d = Dict{Float64,Record}()
    for age in ages
        q = q_for_age(epoch_q, age)
        d[age] = Record(xs, fill(NaN, n_nodes), fill(NaN, n_nodes), fill(q, n_nodes))
    end
    return Workbook("Constant" => d)
end

function q_for_age(epoch_q::Vector{Pair{Tuple{Float64,Float64},Float64}}, age::Float64)
    for ((a_old, a_young), q) in epoch_q
        age <= a_old && age >= a_young && return q
    end
    dist(p) = abs((p.first[1] + p.first[2]) / 2 - age)
    return epoch_q[argmin(dist.(epoch_q))].second
end

ages_of(data) = data isa BasinDataset ? data.ages : data.ages
bounds_of(data) = data isa BasinDataset ? data.bounds : data.bounds

# ----------------------------------------------------------------------------
# Загрузка набора с подменённым heat (файл Heat_Flow_Basal.xlsx может отсутствовать)
# ----------------------------------------------------------------------------

function load_dataset_derived(data_dir::AbstractString, heat::Workbook)
    data_dir = abspath(data_dir)
    missing_files = [n for n in REQUIRED if !isfile(joinpath(data_dir, n))]
    isempty(missing_files) || throw(ErrorException("Missing input files: " * join(missing_files, ", ")))
    payload = read_all(data_dir)
    isempty(payload.heat) || @warn "Heat_Flow_Basal.xlsx присутствует и будет заменён инжектированным"
    return BasinDataset(data_dir, "derived-" * string(hash(string(heat)))[1:8],
                        payload.depth, payload.lith, payload.porosity, heat,
                        payload.tsurf_data, payload.properties, payload.ages, payload.bounds)
end

"Датасет с константным q [мВт/м²] (шаг 0, контрольные прогоны)."
load_dataset_constant(data_dir::AbstractString, q::Real) =
    load_dataset_derived(data_dir, constant_heat_workbook(read_all(abspath(data_dir)), q))

# ----------------------------------------------------------------------------
# ThetaLayout (B1): декларативная схема неизвестных — только q (решение 2e)
# ----------------------------------------------------------------------------

struct ThetaLayout
    names::Vector{String}
    lower::Vector{Float64}
    upper::Vector{Float64}
    x0::Vector{Float64}
    epochs::Vector{Tuple{Float64,Float64}}   # (старая Ma, молодая Ma)
    nodes_x::Vector{Float64}                 # X-узлы q, включая границы профиля
    monotone::Bool                           # скалярный режим: оптимизируется δ, не q
end
ThetaLayout(names, lower, upper, x0, epochs, nodes_x) =
    ThetaLayout(names, lower, upper, x0, epochs, nodes_x, false)

Base.length(l::ThetaLayout) = length(l.names)

"Имя параметра: q |<эпоха>Ma |x<N>м."
function q_param_name(epoch::Tuple{Float64,Float64}, x::Float64)
    e = epoch[1] == round(epoch[1]) ? string(Int(epoch[1])) : string(epoch[1])
    y = epoch[2] == round(epoch[2]) ? string(Int(epoch[2])) : string(epoch[2])
    xn = x == round(x) ? string(Int(x)) : string(x)
    return "q_[" * e * "-" * y * "]Ma_x" * xn
end

"""
    make_q_layout(data; epochs, n_nodes, mid_x, q_box) -> ThetaLayout

Схема θ = q_basal: n_epochs × n_nodes значений мВт/м². X-узлы всегда включают
границы профиля (data.bounds) → интерполяция q без экстраполяции. При
n_nodes = 2 — только границы; при 3 добавляется mid_x (по умолчанию 23400 м —
середина профиля по фактуре данных). Индексация: (i_epoch-1)*n_nodes + i_x.
"""
function make_q_layout(data;
                       epochs::Vector{Tuple{Float64,Float64}} =
                           [(200.0, 100.0), (100.0, 35.0), (35.0, 10.0), (10.0, 0.0)],
                       n_nodes::Int = 3,
                       mid_x::Float64 = 23400.0,
                       q_box::Tuple{Float64,Float64} = (30.0, 120.0),
                       nodes_x::Union{Nothing,Vector{Float64}} = nothing)
    bounds = bounds_of(data)
    xmin, xmax = Float64(bounds[1]), Float64(bounds[2])
    nodes = if nodes_x !== nothing
        isempty(nodes_x) && error("nodes_x must not be empty")
        ns = sort(unique(clamp.(Float64.(nodes_x), xmin, xmax)))
        ns[1] == xmin || pushfirst!(ns, xmin)
        ns[end] == xmax || push!(ns, xmax)
        ns
    elseif n_nodes == 1
        [xmin]
    elseif n_nodes == 2
        [xmin, xmax]
    elseif n_nodes == 3
        [xmin, clamp(mid_x, xmin, xmax), xmax]
    else
        error("n_nodes must be 1, 2 or 3 (or pass nodes_x)")
    end
    names = String[]; lower = Float64[]; upper = Float64[]; x0 = Float64[]
    for epoch in epochs
        for x in nodes
            push!(names, q_param_name(epoch, x))
            push!(lower, q_box[1]); push!(upper, q_box[2])
            push!(x0, (q_box[1] + q_box[2]) / 2)
        end
    end
    return ThetaLayout(names, lower, upper, x0, epochs, nodes)
end

"θ из константного q на каждую эпоху (твин θ_true, отчёты)."
function theta_constant(layout::ThetaLayout, epoch_q::Vector{Float64})
    length(epoch_q) == length(layout.epochs) ||
        error("expected $(length(layout.epochs)) epoch values")
    nn = length(layout.nodes_x)
    return [epoch_q[ie] for ie in eachindex(layout.epochs) for _ in 1:nn]
end

# ----------------------------------------------------------------------------
# Монотонный скалярный q(t) (--q-scalar): жёсткое ограничение через
# репараметризацию. Эпохи от старой к молодой; оптимизируемые переменные
# δ = [q_1, δ_2, …, δ_ne], где q_1 — q самой старой эпохи, а
# q_e = q_1 − Σ_{k<e} (softplus(δ_k) − ln2) → q невозрастающий ко времени
# при любых δ. δ = 0 → константный q.
const SOFTPLUS_LN2 = log(2.0)
softplus_stable(x::Real) = x > 30 ? Float64(x) : log1p(exp(Float64(x)))
"δ (скалярный монотонный) → значения q по эпохам [мВт/м²]."
function theta_monotone_qs(ue::AbstractVector{<:Real})
    ne = length(ue)
    ne >= 1 || error("empty theta")
    qs = Vector{Float64}(undef, ne)
    qs[1] = Float64(ue[1])
    for e in 2:ne
        qs[e] = qs[e-1] - (softplus_stable(ue[e]) - SOFTPLUS_LN2)
    end
    return qs
end
"δ → θ в координатах layout (эпохи × узлы; скалярный режим: 1 узел)."
function theta_expand_mono(layout::ThetaLayout, ue::AbstractVector{<:Real})
    layout.monotone || error("theta_expand_mono: layout is not monotone")
    nn = length(layout.nodes_x)
    nn == 1 || error("monotone layout expects a single X-node")
    qs = theta_monotone_qs(ue)
    return [qs[ie] for ie in eachindex(layout.epochs) for _ in 1:nn]
end
"θ с реальными q по эпохам → δ (обратная задача для warm-start)."
function theta_to_delta(qs::AbstractVector{<:Real})
    ne = length(qs)
    ue = Vector{Float64}(undef, ne)
    ue[1] = Float64(qs[1])
    for e in 2:ne
        d = Float64(qs[e-1]) - Float64(qs[e])      # падение за эпоху ≥ 0
        ue[e] = log(expm1(d + SOFTPLUS_LN2))       # softplus⁻¹(d + ln2)
    end
    return ue
end
"""
    make_mono_scalar_layout(base) -> ThetaLayout
Скалярный монотонный режим: 1 X-узел, оптимизация в δ-координатах.
Бокс q_1 — из base (30–120 по умолчанию); бокс δ — [0, delta_max] —
падение за эпоху до delta_max мВт/м² (с запасом для наблюдаемых изменений).
"""
function make_mono_scalar_layout(base::ThetaLayout)
    xmin = base.nodes_x[1]
    qlo, qhi = base.lower[1], base.upper[1]
    delta_max = 60.0
    ne = length(base.epochs)
    names = [q_param_name(base.epochs[ie], xmin) for ie in 1:ne]
    lower = [qlo, fill(0.0, ne - 1)...]
    upper = [qhi, fill(delta_max, ne - 1)...]
    x0 = [base.x0[1], fill(0.0, ne - 1)...]      # δ=0 → константный q
    return ThetaLayout(names, lower, upper, x0, base.epochs, [xmin], true)
end

"Разности θ по соседним эпохам и соседним X-узлам (для регуляризации B3)."
# ----------------------------------------------------------------------------
# make_param_data (B2): BasinDataset с инжектированным θ-heat
# ----------------------------------------------------------------------------

"Возраст принадлежит эпохе [старая, молодая]; вне сетки эпох — ближайшая по середине."
function epoch_index_of(layout::ThetaLayout, age::Float64)
    for (ie, (a_old, a_young)) in enumerate(layout.epochs)
        age <= a_old && age >= a_young && return ie
    end
    mids = [(e[1] + e[2]) / 2 for e in layout.epochs]
    return argmin(abs.(mids .- age))
end

"""
    make_param_data(data, layout, θ) -> BasinDataset

Копия датасета с синтетическим Workbook heat: на каждый возраст — Record
с value в X-узлах layout (константа по эпохе); boundary_value решателя
линейно интерполирует по x → ступенчатая по эпохам q(x, t). Tsurf,
геометрия, свойства — из исходных данных без изменений.
"""
function make_param_data(data::BasinDataset, layout::ThetaLayout, θ::AbstractVector{<:Real})
    length(θ) == length(layout) ||
        throw(DimensionMismatch("theta has $(length(θ)) components, layout expects $(length(layout))"))
    nn = length(layout.nodes_x)
    ages = ages_of(data)
    d = Dict{Float64,Record}()
    for age in ages
        ie = epoch_index_of(layout, age)
        q = [Float64(θ[(ie-1)*nn + ix]) for ix in 1:nn]
        d[age] = Record(layout.nodes_x, fill(NaN, nn), fill(NaN, nn), q)
    end
    heat = Workbook("Param" => d)
    return BasinDataset(data.data_dir, "param-" * string(hash(round.(θ; digits = 10)))[1:8],
                        data.depth, data.lith, data.porosity, heat,
                        data.tsurf_data, data.properties, data.ages, data.bounds)
end

function theta_diffs(layout::ThetaLayout, θ::AbstractVector{<:Real})
    nn = length(layout.nodes_x)
    ne = length(layout.epochs)
    d_node = Float64[]; d_epoch = Float64[]
    for ie in 1:ne, ix in 1:nn-1
        push!(d_node, θ[(ie-1)*nn + ix + 1] - θ[(ie-1)*nn + ix])
    end
    for ie in 1:ne-1, ix in 1:nn
        push!(d_epoch, θ[ie*nn + ix] - θ[(ie-1)*nn + ix])
    end
    return d_node, d_epoch
end
