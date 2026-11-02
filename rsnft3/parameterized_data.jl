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
end

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
                       q_box::Tuple{Float64,Float64} = (30.0, 120.0))
    bounds = bounds_of(data)
    xmin, xmax = Float64(bounds[1]), Float64(bounds[2])
    nodes = n_nodes == 2 ? [xmin, xmax] :
            n_nodes == 3 ? [xmin, clamp(mid_x, xmin, xmax), xmax] :
            error("n_nodes must be 2 or 3")
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

"Разности θ по соседним эпохам и соседним X-узлам (для регуляризации B3)."
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
