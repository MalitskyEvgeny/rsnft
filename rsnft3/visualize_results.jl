# Quick PNG visualisations from a cached basin_fvm run (Luxor/Cairo instead of Pillow).
#
# Зависимости:
#   using Pkg; Pkg.add(["ArgParse", "JSON", "Luxor", "NPZ"])
#
# Запуск:
#   julia visualize_results.jl results/<run_id> --age 0

using ArgParse
using JSON
using Luxor
using NPZ
using Printf
using Colors

const PALETTE = Float64[
    0 0 4; 31 12 72; 85 15 109; 136 34 106;
    186 54 85; 227 89 51; 249 140 10; 249 201 50;
    252 255 164
]

"Установить шрифт (аналог font(size) в Python-версии; запасной шрифт подбирает Cairo)."
function set_font(size::Real)
    fontface("DejaVu Sans")
    fontsize(size)
end

function draw_text(s::AbstractString, x::Real, y::Real, size::Real, color = "black")
    set_font(size)
    setcolor(color)
    text(s, Point(x, y); halign = :left, valign = :top)
end

"Цвет для значения: линейная интерполяция по палитре, не-конечные значения — серый (245)."
function colour_rgb(v::Real, low::Real, high::Real)
    if !isfinite(v)
        return RGB{Float64}(245 / 255, 245 / 255, 245 / 255)
    end
    scaled = clamp((v - low) / max(high - low, 1.0e-12), 0.0, 1.0)
    position = scaled * (size(PALETTE, 1) - 1)
    left = floor(Int, position)
    right = min(left + 1, size(PALETTE, 1) - 1)
    weight = position - left
    c = ntuple(k -> clamp(floor(Int, PALETTE[left+1, k] * (1.0 - weight) +
                                    PALETTE[right+1, k] * weight), 0, 255) / 255, 3)
    return RGB{Float64}(c[1], c[2], c[3])
end

colour_at(v, low, high) = colour_rgb(v, low, high)

"Матрица цветов размера h×w, масштабирование «ближайший сосед» как в PIL (NEAREST)."
function nearest_resize(colours::Matrix{RGB{Float64}}, w::Int, h::Int)
    nz, nx = size(colours)
    img = Matrix{RGB{Float64}}(undef, h, w)
    for jd in 1:h, id in 1:w
        js = min(nz, floor(Int, (jd - 0.5) * nz / h) + 1)
        is = min(nx, floor(Int, (id - 0.5) * nx / w) + 1)
        img[jd, id] = colours[js, is]
    end
    return img
end

fmt_g(x) = @sprintf("%g", x)

function outline_box(x1, y1, x2, y2; color = "black", width = 1)
    setcolor(color)
    setline(width)
    box(Point(x1, y1), Point(x2, y2), :stroke)
end

function main_results(argv = ARGS)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "run"
            help = "Run directory or result.npz"
            required = true
        "--age"
            help = "Nearest saved geological age, Ma"
            arg_type = Float64
            default = 0.0
        "--output"
            arg_type = String
            default = nothing
    end
    args = parse_args(argv, s)

    result_path = isdir(args["run"]) ? joinpath(args["run"], "result.npz") : args["run"]
    run_dir = dirname(abspath(result_path))
    data = npzread(result_path)
    ages = data["ages_ma"]
    index = argmin(abs.(ages .- args["age"]))
    temperature = data["temperature_c"][index, :, :]       # (z, x)
    x, z = data["x"], data["z"]
    metadata = JSON.parsefile(joinpath(run_dir, "metadata.json"))
    solver = metadata["solver"]
    if get(metadata, "mesh_type", nothing) == "balanced_quadtree"
        mesh_label = "adaptive $(solver["base_nx"])x$(solver["base_nz"]) · level $(solver["max_level"])"
    else
        mesh_label = "$(solver["nx"])x$(solver["nz"])"
    end

    finite = filter(isfinite, vec(temperature))
    low, high = minimum(finite), maximum(finite)

    output = args["output"] === nothing ?
        joinpath(run_dir, "temperature_$(fmt_g(ages[index]))Ma.png") : args["output"]
    Drawing(1400, 620, output)
    background("white")

    draw_text("FVM run $(metadata["run_id"]) | $mesh_label | dt <= $(solver["max_dt_ma"]) Ma",
              40, 18, 22)

    left, top, width, height = 70, 85, 820, 450
    colours = [colour_rgb(v, low, high) for v in temperature]
    placeimage(nearest_resize(colours, width, height), Point(left, top))
    outline_box(left, top, left + width, top + height; width = 2)
    draw_text("Temperature at $(fmt_g(ages[index])) Ma", left, 56, 16)
    draw_text("x, m", left + width ÷ 2 - 20, top + height + 30, 16)
    draw_text("z, m", 12, top + height ÷ 2, 16)
    for fraction in range(0.0, 1.0; length = 5)
        px = left + trunc(Int, fraction * width)
        pz = top + trunc(Int, fraction * height)
        draw_text(@sprintf("%.0f", x[1] + fraction * (x[end] - x[1])),
                  px - 15, top + height + 7, 13)
        draw_text(@sprintf("%.0f", z[1] + fraction * (z[end] - z[1])),
                  left - 55, pz - 8, 13)
    end

    bar_x = left + width + 18
    gradient = collect(range(high, low; length = height))
    bar = Matrix{RGB{Float64}}(undef, height, 28)
    for j in 1:height
        bar[j, :] .= colour_rgb(gradient[j], low, high)
    end
    placeimage(bar, Point(bar_x, top))
    outline_box(bar_x, top, bar_x + 28, top + height)
    draw_text(@sprintf("%.1f °C", high), bar_x + 36, top - 6, 13)
    draw_text(@sprintf("%.1f °C", low), bar_x + 36, top + height - 12, 13)

    plot_left, plot_top, plot_width, plot_height = 1040, 85, 300, 450
    outline_box(plot_left, plot_top, plot_left + plot_width, plot_top + plot_height; width = 2)
    centre = length(x) ÷ 2 + 1                       # len(x)//2 с отсчётом от 0
    valid = isfinite.(temperature[:, centre])
    tv, zv = temperature[valid, centre], z[valid]
    points = Point[]
    for (temp, depth) in zip(tv, zv)
        px = plot_left + trunc(Int, (temp - low) / max(high - low, 1.0e-12) * plot_width)
        py = plot_top + trunc(Int, (depth - z[1]) / max(z[end] - z[1], 1.0e-12) * plot_height)
        push!(points, Point(px, py))
    end
    for fraction in range(0.0, 1.0; length = 5)
        px = plot_left + trunc(Int, fraction * plot_width)
        py = plot_top + trunc(Int, fraction * plot_height)
        setcolor(225 / 255, 225 / 255, 225 / 255)
        setline(1)
        line(Point(px, plot_top), Point(px, plot_top + plot_height), :stroke)
        line(Point(plot_left, py), Point(plot_left + plot_width, py), :stroke)
        draw_text(@sprintf("%.0f", low + fraction * (high - low)),
                  px - 12, plot_top + plot_height + 7, 13)
    end
    if length(points) > 1
        setcolor(205 / 255, 45 / 255, 35 / 255)
        setline(3)
        poly(points, :stroke)
    end
    draw_text(@sprintf("Profile at x=%.1f m", x[centre]), plot_left, 56, 16)
    draw_text("Temperature, °C", plot_left + 80, plot_top + plot_height + 30, 16)

    finish()
    println(output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_results()
end
