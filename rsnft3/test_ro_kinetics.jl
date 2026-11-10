# Юнит-тесты ro_kinetics.jl (A1b): константы против эталона basin_data.py (R8)
# + физичность изотермических точек.
# Запуск: julia --project=. test_ro_kinetics.jl

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))

using Printf

failures = 0
function check(name, ok)
    global failures
    println((ok ? "PASS " : "FAIL "), name)
    ok || (failures += 1)
end

# --- 1. Константы == _EASY_RO из basin_data.py (дословная таблица) ---
et_E = collect(34.0:2.0:72.0)
et_w = [0.03,0.03,0.04,0.04,0.05,0.05,0.06,0.04,0.04,0.07,
        0.06,0.06,0.06,0.05,0.05,0.04,0.03,0.02,0.02,0.01]
check("E_kcal == basin_data.py", EA_E_KCAL == et_E)
check("weights == basin_data.py", collect(EA_W) == et_w)
check("weights sum == 0.85", abs(sum(EA_W) - 0.85) < 1e-12)
check("A == 1e13", EA_A == 1.0e13)
# basin_data.py: E_J = E_kcal*4186.8, Rg_J = 8.314 (то же, что ER в easyro_original.py)
et_ER = et_E .* 4186.8 ./ 8.314
check("E/R (K) == basin_data.py (rel < 1e-3)", maximum(abs.(EA_ER .- et_ER) ./ et_ER) < 1.0e-3)
# инкрементные константы статьи (easyro_original.py дословно)
check("A1..B2 == 2.334733/0.250621/3.330675/1.681534",
      A1 == 2.334733 && A2 == 0.250621 && B1 == 3.330675 && B2 == 1.681534)

# --- 2. isothermal_rate: конечность и положительность в рабочем диапазоне ---
rate = isothermal_rate(373.15)
check("rate(100C): all finite", all(isfinite, rate))
check("rate(100C): all >= 0", all(>=(0), rate))
check("rate(100C): low-E reactions faster than high-E", rate[1] > rate[end])

# --- 3. Физичность изотермических точек (геологические выдерживания) ---
function ro_isothermal(TC, Myr)
    dt = Myr * SECONDS_PER_MA
    I = isothermal_rate(TC + 273.15) .* dt
    F = sum(EA_W[n] * (1 - exp(-I[n])) for n in eachindex(I))
    return easy_ro(min(F, 1.0))
end
# Общепринятое поведение EASY%Ro (Sweeney&Burnham 1990):
# 60C/100Myr — незрелая зона (Ro < 0.5, начало нефтяного окна ~0.5-0.6);
# 100C/100Myr — середина нефтяного окна (~0.7-0.8);
# 140C/100Myr — конец нефтяного окна / газ (>1.3).
check("Ro(60C, 100 Myr) < 0.55 (immature)", ro_isothermal(60.0, 100.0) < 0.55)
check("Ro(100C, 100 Myr) in [0.6, 0.95] (oil window)",
      0.6 <= ro_isothermal(100.0, 100.0) <= 0.95)
check("Ro(140C, 100 Myr) > 1.3 (overmature)", ro_isothermal(140.0, 100.0) > 1.3)
check("Ro monotone in T", ro_isothermal(80.0, 50.0) < ro_isothermal(110.0, 50.0) < ro_isothermal(140.0, 50.0))
check("Ro monotone in time", ro_isothermal(120.0, 10.0) < ro_isothermal(120.0, 100.0))
check("Ro(0C, any) == easy_ro(0) == 0.202", abs(easy_ro(0.0) - exp(-1.6)) < 1e-12)

# --- 4. Квадратура particle_ro: трапеция по скорости на синтетическом прогоне ---
# Фейки: layer_at требует data.ages и data.depth[name][age]::Record; result —
# NamedTuple с temperature_c (nt, nz, nx), x, z, ages_ma.
let
    rec_old = Record([0.0, 1000.0], [0.0, 0.0], [100.0, 100.0], [0.0, 0.0])
    rec_young = Record([0.0, 1000.0], [0.0, 0.0], [100.0, 100.0], [0.0, 0.0])
    fake_data = (ages = [10.0, 0.0],
                 depth = Dict("L" => Dict(10.0 => rec_old, 0.0 => rec_young)))
    xg = [0.0, 1000.0]
    zg = collect(range(0.0, 100.0; length = 11))
    function fake_result(T_old_c, T_young_c)
        T = zeros(2, length(zg), length(xg))
        T[1, :, :] .= T_old_c
        T[2, :, :] .= T_young_c
        (temperature_c = T, x = xg, z = zg, ages_ma = [10.0, 0.0])
    end
    xq, eta = 500.0, 0.5

    # (а) изотермический случай: трапеция == точному изотермическому значению
    ro_iso = particle_ro(fake_result(100.0, 100.0), fake_data, xq, "L", eta)
    check("particle_ro isothermal == exact", abs(ro_iso - ro_isothermal(100.0, 10.0)) < 1e-12)

    # (б) линейный разогрев 70->110C за 10 Myr: трапеция заметно ближе к эталону
    #     мелкого шага, чем старая константа на горячем конце
    ro_trap = particle_ro(fake_result(70.0, 110.0), fake_data, xq, "L", eta)
    function ro_fine_linear(T0, T1, Myr; n = 4000)
        dt = Myr * SECONDS_PER_MA / n
        I = zeros(length(EA_W))
        for i in 0:(n - 1)
            TK = (T0 + (T1 - T0) * (i + 0.5) / n) + 273.15
            I .+= isothermal_rate(TK) .* dt
        end
        F = sum(EA_W[m] * (1 - exp(-I[m])) for m in eachindex(I))
        easy_ro(min(F, 1.0))
    end
    ro_fine = ro_fine_linear(70.0, 110.0, 10.0)
    dt_tot = 10.0 * SECONDS_PER_MA
    I_hot = isothermal_rate(110.0 + 273.15) .* dt_tot
    F_hot = sum(EA_W[m] * (1 - exp(-I_hot[m])) for m in eachindex(I_hot))
    ro_hot = easy_ro(min(F_hot, 1.0))
    check("particle_ro heating: closer to fine grid than hot-end constant",
          abs(ro_trap - ro_fine) < abs(ro_hot - ro_fine))
    # k(T) сильно выпукла: трапеция на 2 точках всё же завышает, но вдвое меньше
    check("particle_ro heating: error < 3/4 of hot-end error",
          abs(ro_trap - ro_fine) < 0.75 * abs(ro_hot - ro_fine))

    # (в) регресс: новая Ro не превышает старую (смещение было в плюс)
    check("particle_ro heating: Ro <= old hot-end Ro", ro_trap <= ro_hot + 1e-12)

    # холодный интервал (оба конца ниже t_min_k=200K, т.е. < -73C): вклада нет
    ro_cold = particle_ro(fake_result(-100.0, -100.0), fake_data, xq, "L", eta)
    check("particle_ro cold interval: no contribution", abs(ro_cold - easy_ro(0.0)) < 1e-12)
end

println()
if failures == 0
    println("ALL PASS")
else
    println(@sprintf("%d FAILURES", failures))
    exit(1)
end
