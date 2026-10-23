# Render the final adaptive mesh with temperature fill and refinement levels.
#
# Запуск:
#   julia visualize_adaptive_mesh.jl adaptive_results/<run_id>

isdefined(Main, :colour_rgb) || include(joinpath(@__DIR__, "visualize_results.jl"))

const LEVEL_COLOURS = [
    (75, 85, 99), (37, 99, 235), (220, 38, 38), (147, 51, 234), (5, 150, 105),
]

level_colour(lev::Integer) = let c = LEVEL_COLOURS[lev%length(LEVEL_COLOURS)+1]
    RGB(c[1] / 255, c[2] / 255, c[3] / 255)
end

function main_mesh(argv = ARGS)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "run"
            help = "Adaptive run directory"
            required = true
        "--output"
            arg_type = String
            default = nothing
    end
    args = parse_args(argv, s)
    run_dir = args["run"]
    mesh = npzread(joinpath(run_dir, "final_adaptive_mesh.npz"))
    meta = JSON.parsefile(joinpath(run_dir, "metadata.json"))
    x0, x1, z0, z1 = mesh["x0"], mesh["x1"], mesh["z0"], mesh["z1"]
    level = Int.(mesh["level"])
    active = Bool.(mesh["active"])
    temperature = mesh["temperature_c"]
    finite = [t for (t, a) in zip(temperature, active) if a && isfinite(t)]
    low, high = minimum(finite), maximum(finite)

    width, height = 1500, 900
    margin_left, margin_top, plot_width, plot_height = 90, 95, 1280, 700
    xmin, xmax = minimum(x0), maximum(x1)
    zmin, zmax = minimum(z0), maximum(z1)
    px(x) = margin_left + trunc(Int, (x - xmin) / (xmax - xmin) * plot_width)
    py(z) = margin_top + trunc(Int, (z - zmin) / (zmax - zmin) * plot_height)

    output = args["output"] === nothing ? joinpath(run_dir, "adaptive_mesh_0Ma.png") : args["output"]
    Drawing(width, height, output)
    background("white")

    draw_text("Adaptive FVM mesh · run $(meta["run_id"]) · $(meta["final_cells"]) cells",
              40, 22, 24)
    for i in eachindex(x0)
        p1, p2 = Point(px(x0[i]), py(z0[i])), Point(px(x1[i]), py(z1[i]))
        setcolor(active[i] ? colour_rgb(temperature[i], low, high) :
                 RGB(244 / 255, 244 / 255, 245 / 255))
        box(p1, p2, :fill)
        setcolor(level_colour(level[i]))
        setline(level[i] < 2 ? 1 : 2)
        box(p1, p2, :stroke)
    end
    outline_box(margin_left, margin_top, margin_left + plot_width, margin_top + plot_height; width = 2)
    draw_text("x, m", margin_left + plot_width ÷ 2 - 20, 835, 17)
    draw_text("z, m", 22, margin_top + plot_height ÷ 2, 17)
    for fraction in range(0.0, 1.0; length = 5)
        xx = margin_left + trunc(Int, fraction * plot_width)
        yy = margin_top + trunc(Int, fraction * plot_height)
        draw_text(@sprintf("%.0f", xmin + fraction * (xmax - xmin)), xx - 15, 805, 14)
        draw_text(@sprintf("%.0f", zmin + fraction * (zmax - zmin)), margin_left - 62, yy - 8, 14)
    end
    legend_x = 1050
    for lev in sort!(unique(level))
        x = legend_x + lev * 105
        outline_box(x, 43, x + 20, 63; color = level_colour(lev), width = 3)
        draw_text("level $lev", x + 27, 44, 14)
    end
    draw_text(@sprintf("min cell %.1f × %.1f m · temperature %.1f…%.1f °C",
                       meta["minimum_dx_m"], meta["minimum_dz_m"], low, high),
              90, 62, 14, RGB(70 / 255, 70 / 255, 70 / 255))
    finish()
    println(output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_mesh()
end
