# Сравнение Ro нашего прогона data1 с эталоном организаторов.
# Вход:  results/_refcomp_ro_input.json  [{layer,x,t,eta,ro_ref}...]
# Выход: results/_refcomp_ro_output.json [{layer,x,t,eta,ro_ref,ro_our,t_our}...]
using JSON
using NPZ

include(joinpath(@__DIR__, "basin_fvm.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))

struct NpzResult
    temperature_c::Array{Float64,3}
    x::Vector{Float64}
    z::Vector{Float64}
    ages_ma::Vector{Float64}
end

function particle_ro_at(result, data, xq, layer_name, eta, t_stop; t_min_k=200.0)
    temp = result.temperature_c
    nt, nz, nx = size(temp)
    xg, zg, ages = result.x, result.z, result.ages_ma
    dxg = xg[2] - xg[1]
    dzg = zg[2] - zg[1]
    I_tot = zeros(length(EA_W))
    deposited = false
    for k in 2:nt
        age_mid = 0.5 * (ages[k-1] + ages[k])
        age_mid < t_stop - 1e-9 && (deposited && break; continue)
        lo = layer_at(data, age_mid, layer_name, xq)
        lo === nothing && (deposited && break; continue)
        top, bot = lo
        zp = top + eta * (bot - top)
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

function main()
    run_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "results", "efdab5d17bbb3817")
    out_path = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "results", "_refcomp_ro_output.json")
    rows = JSON.parsefile(joinpath(@__DIR__, "..", "results", "_refcomp_ro_input.json"))
    npz = npzread(joinpath(run_dir, "result.npz"))
    result = NpzResult(Float64.(npz["temperature_c"]), Float64.(npz["x"]),
                       Float64.(npz["z"]), Float64.(npz["ages_ma"]))
    cache = mkpath(joinpath(@__DIR__, "..", "results", "_refcomp_cache"))
    data = load_dataset(joinpath(@__DIR__, "..", "Данные_для_прямой_задачи", "data1"), cache)
    out = Vector{OrderedDict{String,Any}}()
    for (i, r) in enumerate(rows)
        ro = try
            particle_ro_at(result, data, r["x"], r["layer"], r["eta"], r["t"])
        catch
            NaN
        end
        if i % 5000 == 0
            println("... $i / $(length(rows))")
        end
        push!(out, OrderedDict(
            "layer" => r["layer"], "x" => r["x"], "t" => r["t"], "eta" => r["eta"],
            "ro_ref" => r["ro_ref"], "ro_our" => ro))
    end
    open(out_path, "w") do f
        JSON.print(f, out)
    end
    println("SAVED results/_refcomp_ro_output.json")
end

main()
