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
#   3) NaN-зоны T вне бассейна — масочная билинейная интерполяция с 1D-фолбэком
#      по ближайшей конечной строке/столбцу (bilinear_sample_flagged, basin_fvm.jl).
#
# Запуск автономно не предполагается — include из objective.jl / run_inverse.jl.

isdefined(@__MODULE__, :BasinDataset) || include(joinpath(@__DIR__, "basin_fvm.jl"))

const R_GAS = 8.314                        # Дж/(моль·К) — как в basin_data.py
const EA_A = 1.0e13                        # c^-1
const EA_E_KCAL = collect(34.0:2.0:72.0)   # ккал/моль
const EA_ER = EA_E_KCAL .* 4186.8 ./ R_GAS            # E_i/R, К (калория 4186.8 Дж —
                                                      # как в basin_data.py организаторов)
const EA_W = (0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07,
              0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01)
"Скорости реакций k_i(T), c^-1 — точный закон Аррениуса (как в читалке
# организаторов basin_data.py: k = A*exp(-E_i/(R*T))).
# Внимание: встречающаяся в сторонних реализациях формула
# k = A*T*exp(-x)*(1 - (x^2+a1*x+a2)/(x^2+b1*x+b2)) — это аппроксимация
# ИНТЕГРАЛА Аррениуса при постоянной скорости нагрева; использованная как
# изотермическая скорость, она завышает k в 4-7 раз и систематически
# завышает Ro (сверено с эталоном организаторов, 2026-11-10)."
function isothermal_rate(TK::Float64)
    out = Vector{Float64}(undef, length(EA_ER))
    for (n, er) in enumerate(EA_ER)
        out[n] = EA_A * exp(-er / TK)
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

"Ro для частицы (x, слой, η) по полю temperature_c прогона (nt, nz, nx).

Квадратура — трапеция по скорости на каждом интервале срезов [k-1, k]: константа
на горячем (молодом) конце интервала систематически завышала интеграл Аррениуса
(k(T) растёт экспоненциально, интервалы срезов 12.5–20 Ma) — Ro уползала вправо.
T на обоих концах — bilinear_sample (масочная интерполяция с честным 1D-фолбэком
по ближайшей конечной строке/столбцу; прежний фолбэк к ближайшему узлу предпочёл
бы более горячий узел ниже частицы)."
function particle_ro(result, data, xq::Float64, layer_name::String, eta::Float64;
                     t_min_k::Float64 = 200.0)
    temp = result.temperature_c
    nt = size(temp, 1)
    xg, zg, ages = result.x, result.z, result.ages_ma

    I_tot = zeros(length(EA_W))
    deposited = false
    for k in 2:nt
        # Интервал учитывается, только если частица существует на его середине:
        # до депонирования точка (zp по геометрии молодого конца) лежит в
        # NaN-зоне над древней поверхностью осадков, и любой фолбэк тянет
        # температуру из тёплого осадка ниже - ложное выдерживание, Ro завышена.
        # Скорость при температуре дна пренебрежимо мала, потеря точности нет.
        layer_at(data, 0.5 * (ages[k-1] + ages[k]), layer_name, xq) === nothing &&
            (deposited && break; continue)
        lo_old = layer_at(data, ages[k-1], layer_name, xq)
        lo_young = layer_at(data, ages[k], layer_name, xq)
        top_young, bot_young = lo_young
        zp_young = top_young + eta * (bot_young - top_young)
        v_young = bilinear_sample(xg, zg, temp[k, :, :], xq, zp_young)
        # NaN-выборка T (точка на границе активной зоны сетки) — пропуск интервала,
        # не обрыв интегрирования: частица существует, терять весь дальнейший
        # прогрев неверно (подошвенные точки Layer_7, 55-0 Ma).
        isfinite(v_young) || continue
        if lo_old === nothing
            v_old = v_young    # депонирование в первой половине интервала
        else
            top_old, bot_old = lo_old
            zp_old = top_old + eta * (bot_old - top_old)
            v_old = bilinear_sample(xg, zg, temp[k-1, :, :], xq, zp_old)
            isfinite(v_old) || continue
        end
        TK_old = v_old + 273.15
        TK_young = v_young + 273.15
        if max(TK_old, TK_young) < t_min_k
            deposited = true
            continue
        end
        dt = (ages[k-1] - ages[k]) * SECONDS_PER_MA
        dt <= 0.0 && continue
        rate_old = isothermal_rate(TK_old)
        rate_young = isothermal_rate(TK_young)
        for n in eachindex(I_tot)
            I_tot[n] += 0.5 * (rate_old[n] + rate_young[n]) * dt
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
