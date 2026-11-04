# EASY%Ro kinetics and particle tracing (R1).
#
# Вариант A констант Sweeney & Burnham (1990), тот же, что в эталонной читалке
# организаторов basin_data.py (_EASY_RO): 20 реакций, A = 1e13 c^-1,
# Ei = 34..72 ккал/моль (шаг 2), веса Σ = 0.85.
# Ro считается пост-процессингом поверх прогона basin_fvm.jl: траектория частицы
# (x, слой, η) восстанавливается из геометрии (material_remap сохраняет η),
# кумулятивный интеграл Аррениуса I_i = ∫k_i dt, F = Σ w_i (1 − exp(−I_i)),
# Ro = exp(−1.6 + 3.7·F).
#
# База: research/proto/ro_calc.jl; ловушки учтены (ro_kinetics.md §6):
#   1) геометрия слоя на промежуточных возрастах — линейная интерполяция срезов;
#   2) интеграл кумулятивный (не пошаговый (1−e^−ΔI));
#   3) NaN-зоны T вне бассейна — клэмп к ближайшему конечному по z.
#
# Запуск автономно не предполагается — include из objective.jl / run_inverse.jl.

isdefined(@__MODULE__, :BasinDataset) || include(joinpath(@__DIR__, "basin_fvm.jl"))

const R_GAS = 8.314462618                  # Дж/(моль·К)
const EA_A = 1.0e13                        # c^-1
const EA_E_KCAL = collect(34.0:2.0:72.0)   # ккал/моль
const EA_ER = EA_E_KCAL .* 1000.0 .* 4.184 ./ R_GAS   # E_i/R, К
const EA_W = (0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07,
              0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01)
const A1, A2, B1, B2 = 2.334733, 0.250621, 3.330675, 1.681534

"Скорости реакций k_i(T), c^-1 (рациональная аппроксимация Аррениуса из статьи)."
function isothermal_rate(TK::Float64)
    out = Vector{Float64}(undef, length(EA_ER))
    for (n, er) in enumerate(EA_ER)
        x = er / TK
        out[n] = EA_A * TK * exp(-x) * (1.0 - (x^2 + A1 * x + A2) / (x^2 + B1 * x + B2))
    end
    return out
end

easy_ro(F::Float64) = exp(-1.6 + 3.7 * F)

# ----------------------------------------------------------------------------
# Геометрия слоя на произвольном возрасте
# ----------------------------------------------------------------------------

_rec_interp(rec::Record, xq::Float64) =
    let j = clamp(searchsortedlast(rec.x, xq), 1, length(rec.x) - 1)
        t = (xq - rec.x[j]) / max(rec.x[j+1] - rec.x[j], 1.0e-9)
        (rec.top[j] + t * (rec.top[j+1] - rec.top[j]),
         rec.bot[j] + t * (rec.bot[j+1] - rec.bot[j]))
    end

"Состояние слоя (top, bot) в столбце x на возрасте age с линейной интерполяцией
между срезами (как interpolate_state); nothing если слоя ещё нет или он древнее."
function layer_at(data, age::Float64, name::String, xq::Float64)
    ages = data.ages
    age > ages[1] && return nothing
    ser = data.depth[name]
    haskey(ser, age) && return _rec_interp(ser[age], xq)
    i_old = findlast(a -> a >= age && haskey(ser, a), ages)
    i_young = findfirst(a -> a <= age && haskey(ser, a), ages)
    (i_old === nothing || i_young === nothing) && return nothing
    a_old, a_young = ages[i_old], ages[i_young]
    a_old == a_young && return _rec_interp(ser[a_old], xq)
    f = (a_old - age) / (a_old - a_young)
    r_old = _rec_interp(ser[a_old], xq)
    r_young = _rec_interp(ser[a_young], xq)
    (r_old[1] + f * (r_young[1] - r_old[1]), r_old[2] + f * (r_young[2] - r_old[2]))
end

# ----------------------------------------------------------------------------
# Ro частиц
# ----------------------------------------------------------------------------

"Ro для частицы (x, слой, η) по полю temperature_c прогона (nt, nz, nx)."
function particle_ro(result, data, xq::Float64, layer_name::String, eta::Float64;
                     t_min_k::Float64 = 200.0)
    temp = result.temperature_c
    nt, nz, nx = size(temp)
    xg, zg, ages = result.x, result.z, result.ages_ma
    dxg = xg[2] - xg[1]
    dzg = zg[2] - zg[1]

    I_tot = zeros(length(EA_W))
    deposited = false
    for k in 2:nt
        age_mid = 0.5 * (ages[k-1] + ages[k])
        lo = layer_at(data, age_mid, layer_name, xq)
        lo === nothing && (deposited && break; continue)
        top, bot = lo
        zp = top + eta * (bot - top)
        # билинейная выборка T с клэмпом к активной зоне
        xi = clamp(round(Int, (xq - xg[1]) / dxg) + 1, 1, nx)
        zi = clamp(round(Int, (zp - zg[1]) / dzg) + 1, 1, nz)
        v = temp[k, zi, xi]
        if isnan(v)
            for d in 1:nz
                if zi + d <= nz && isfinite(temp[k, zi+d, xi])
                    v = temp[k, zi+d, xi]; break
                elseif zi - d >= 1 && isfinite(temp[k, zi-d, xi])
                    v = temp[k, zi-d, xi]; break
                end
            end
        end
        TK = v + 273.15
        if TK < t_min_k
            deposited = true
            continue
        end
        dt = (ages[k-1] - ages[k]) * SECONDS_PER_MA
        dt <= 0.0 && continue
        rate = isothermal_rate(TK)
        for n in eachindex(I_tot)
            I_tot[n] += rate[n] * dt
        end
        deposited = true
    end
    F = 0.0
    for n in eachindex(I_tot)
        F += EA_W[n] * (1.0 - exp(-I_tot[n]))
    end
    return easy_ro(min(F, 1.0))
end

"Ro по всем частицам сетки (x-узлы прогона × η=0.1..0.9 каждого слоя).
Возвращает Dict(layer_name => Matrix(nx × 5)). Используется для верификации A1c."
function particle_ro_grid(result, data; etas = (0.1, 0.3, 0.5, 0.7, 0.9))
    out = Dict{String,Matrix{Float64}}()
    layers = sort(collect(keys(data.depth)))
    for name in layers
        m = Matrix{Float64}(undef, length(result.x), length(etas))
        for (xi, xq) in enumerate(result.x), (ei, eta) in enumerate(etas)
            m[xi, ei] = particle_ro(result, data, xq, name, eta)
        end
        out[name] = m
    end
    return out
end
