"""Compare final temperature fields from two cached runs without SciPy."""

using ArgParse
using JSON
using LinearAlgebra
using NPZ
using OrderedCollections
using Statistics

"""
Линейная интерполяция, эквивалент np.interp: x должен возрастать,
за пределами диапазона берётся краевое значение.
"""
function interp1(xq::AbstractVector, x::AbstractVector, y::AbstractVector)
    out = similar(xq, Float64)
    for (k, q) in pairs(xq)
        if isnan(q)
            out[k] = NaN
        elseif q <= x[1]
            out[k] = y[1]
        elseif q >= x[end]
            out[k] = y[end]
        else
            i = searchsortedlast(x, q)
            t = (q - x[i]) / (x[i+1] - x[i])
            out[k] = y[i] + t * (y[i+1] - y[i])
        end
    end
    return out
end

function interpolate_field(x0::AbstractVector, z0::AbstractVector,
                           field::AbstractMatrix,
                           x1::AbstractVector, z1::AbstractVector)
    horizontal = fill(NaN, length(z0), length(x1))
    for j in eachindex(z0)
        valid = isfinite.(field[j, :])
        if count(valid) >= 2
            horizontal[j, :] = interp1(x1, x0[valid], field[j, valid])
        end
    end
    result = fill(NaN, length(z1), length(x1))
    for i in eachindex(x1)
        valid = isfinite.(horizontal[:, i])
        if count(valid) >= 2
            result[:, i] = interp1(z1, z0[valid], horizontal[valid, i])
        end
    end
    return result
end

function main_compare(argv = ARGS)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "coarse"
            required = true
        "fine"
            required = true
        "--output"
            default = "comparison.json"
    end
    args = parse_args(argv, s)

    coarse_path = isdir(args["coarse"]) ? joinpath(args["coarse"], "result.npz") : args["coarse"]
    fine_path   = isdir(args["fine"])   ? joinpath(args["fine"], "result.npz")   : args["fine"]

    coarse = npzread(coarse_path)
    fine   = npzread(fine_path)

    # temperature_c имеет форму (время, z, x); берём последний шаг по времени
    coarse_final = coarse["temperature_c"][end, :, :]
    fine_final   = fine["temperature_c"][end, :, :]

    mapped = interpolate_field(coarse["x"], coarse["z"], coarse_final, fine["x"], fine["z"])
    mask = isfinite.(mapped) .& isfinite.(fine_final)
    difference = mapped[mask] .- fine_final[mask]

    report = OrderedDict(
        "coarse" => coarse_path,
        "fine" => fine_path,
        "compared_cells" => count(mask),
        "rms_difference_c" => sqrt(mean(difference .^ 2)),
        "relative_l2" => norm(difference) / norm(fine_final[mask]),
        "max_abs_difference_c" => maximum(abs.(difference)),
    )

    text = sprint(io -> JSON.print(io, report, 2))
    write(args["output"], text)
    println(text)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_compare()
end
