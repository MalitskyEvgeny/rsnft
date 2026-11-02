# B4: критерии качества инверсии. J_dir по скважинам (T/Ro), доли |res| <= σ/2σ
# (условие 18 приёмки), E^L2 между прогонами (трапеции по срезам времени,
# активные ячейки).

include(joinpath(@__DIR__, "objective.jl"))

struct WellStat
    well::String
    n_T::Int
    n_Ro::Int
    rmse_T::Float64
    rmse_Ro::Float64
    frac_1s_T::Float64
    frac_2s_T::Float64
    frac_1s_Ro::Float64
    frac_2s_Ro::Float64
end

struct CriteriaReport
    wells::Vector{WellStat}
    rmse_T::Float64                     # общие RMSE
    rmse_Ro::Float64
    frac_1s_T::Float64
    frac_2s_T::Float64
    frac_1s_Ro::Float64
    frac_2s_Ro::Float64
    l2::Float64                         # E^L2 к reference-прогону (NaN если нет)
end

frac_within(res::Vector{Float64}, sigma::Float64, k::Real) =
    (isempty(res) || sigma <= 0) ? NaN : count(r -> abs(r) <= k * sigma, res) / length(res)

rmse(res::Vector{Float64}) = isempty(res) ? NaN : sqrt(sum(abs2, res) / length(res))

"Невязки T по точкам WellSet (T_calc билинейно на t=0)."
function residuals_T(result, wells::WellSet)
    temp0 = result.temperature_c[end, :, :]
    acc = Dict{String,Vector{Tuple{Float64,Float64}}}()   # well -> [(res, sigma)]
    for p in wells.points
        p.kind === :T || continue
        v = bilinear_sample(result.x, result.z, temp0, p.x, p.z)
        res = isfinite(v) ? v - p.value : NaN
        push!(get!(acc, p.well, Tuple{Float64,Float64}[]), (res, p.sigma))
    end
    return acc
end

"Невязки Ro по точкам WellSet (particle_ro)."
function residuals_Ro(result, data::BasinDataset, wells::WellSet)
    acc = Dict{String,Vector{Tuple{Float64,Float64}}}()
    for p in wells.points
        p.kind === :Ro || continue
        le = layer_eta_at(data, p.x, p.z)
        le === nothing && continue
        rc = particle_ro(result, data, p.x, le[1], le[2])
        res = isfinite(rc) ? rc - p.value : NaN
        push!(get!(acc, p.well, Tuple{Float64,Float64}[]), (res, p.sigma))
    end
    return acc
end

function _stat(name::String, t::Vector{Tuple{Float64,Float64}}, r::Vector{Tuple{Float64,Float64}})
    keepT = [x for x in t if isfinite(x[1])]
    keepR = [x for x in r if isfinite(x[1])]
    resT = first.(keepT); sigT = last.(keepT)
    resR = first.(keepR); sigR = last.(keepR)
    return WellStat(name, length(resT), length(resR),
                    rmse(resT), rmse(resR),
                    frac_within(resT, isempty(sigT) ? 1.0 : sigT[1], 1),
                    frac_within(resT, isempty(sigT) ? 1.0 : sigT[1], 2),
                    frac_within(resR, isempty(sigR) ? 1.0 : sigR[1], 1),
                    frac_within(resR, isempty(sigR) ? 1.0 : sigR[1], 2))
end

"""
    criteria_report(result_calc, data, wells; result_ref = nothing) -> CriteriaReport

Сводка по скважинам и общая: RMSE, доли |res| <= σ и <= 2σ, E^L2 к
result_ref (если задан).
"""
function criteria_report(result_calc, data::BasinDataset, wells::WellSet;
                         result_ref = nothing)
    resT = residuals_T(result_calc, wells)
    resR = residuals_Ro(result_calc, data, wells)
    stats = WellStat[]
    for wname in sort(collect(union(keys(resT), keys(resR))))
        push!(stats, _stat(wname, get(resT, wname, Tuple{Float64,Float64}[]),
                           get(resR, wname, Tuple{Float64,Float64}[])))
    end
    allT = [x for x in vcat(values(resT)...) if isfinite(x[1])]
    allR = [x for x in vcat(values(resR)...) if isfinite(x[1])]
    l2 = result_ref === nothing ? NaN : l2_error(result_calc, result_ref)
    return CriteriaReport(stats,
                          rmse(first.(allT)), rmse(first.(allR)),
                          frac_within(first.(allT), length(allT) > 0 ? allT[1][2] : 1.0, 1),
                          frac_within(first.(allT), length(allT) > 0 ? allT[1][2] : 1.0, 2),
                          frac_within(first.(allR), length(allR) > 0 ? allR[1][2] : 1.0, 1),
                          frac_within(first.(allR), length(allR) > 0 ? allR[1][2] : 1.0, 2),
                          l2)
end

"""
    l2_error(result_calc, result_ref) -> Float64

E^L2 между прогонами: на каждом временном срезе — RMS расхождения T по
активным (конечным в обоих прогонах) ячейкам; по времени — квадратурный
корень из трапеций e(t)², нормированный на длительность.
"""
function l2_error(result_calc, result_ref)
    Tc, Tr = result_calc.temperature_c, result_ref.temperature_c
    size(Tc) == size(Tr) || throw(DimensionMismatch("temperature grids differ"))
    nt = size(Tc, 1)
    ages = result_calc.ages_ma
    size(ages) == (nt,) || throw(DimensionMismatch("ages differ"))
    e2 = Float64[]
    for k in 1:nt
        m = isfinite.(Tc[k, :, :]) .& isfinite.(Tr[k, :, :])
        n = count(m)
        push!(e2, n > 0 ? sum(abs2, Tc[k, :, :][m] .- Tr[k, :, :][m]) / n : 0.0)
    end
    # трапеции по t (Ma), е2(t)^0.5 в квадрате => e2 интегрируем напрямую
    total = 0.0
    for k in 2:nt
        dt = ages[k-1] - ages[k]                       # возраст убывает
        dt > 0 || continue
        total += 0.5 * (e2[k-1] + e2[k]) * dt
    end
    span = ages[1] - ages[end]
    return span > 0 ? sqrt(total / span) : sqrt(e2[1])
end
