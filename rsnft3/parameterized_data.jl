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
