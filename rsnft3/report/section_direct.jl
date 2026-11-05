const REPORT_RUN_DIR_REF = Ref("")

# Секция «Прямая задача»: температурное поле по эпохам (слайдер),
# геометрия бассейна, стратиграфия, пористость, сводка прогона.

struct DirectData
    x::Vector{Float64}
    z::Vector{Float64}
    ages::Vector{Float64}
    temperature::Array{Float64,3}   # (nt, nz, nx)
    porosity::Matrix{Float64}       # (nz, nx)
    lithology::Matrix{Int}          # (nz, nx)
    surface::Vector{Float64}
    base::Vector{Float64}
    meta::Dict{String,Any}
    run_name::String
end

function load_direct(run_dir::AbstractString)
    REPORT_RUN_DIR_REF[] = run_dir
    npz = npzread(joinpath(run_dir, "result.npz"))
    meta_path = joinpath(run_dir, "metadata.json")
    meta = isfile(meta_path) ? JSON.parsefile(meta_path) : Dict{String,Any}()
    meta = Dict{String,Any}(string(k) => v for (k, v) in meta)
    return DirectData(
        Float64.(npz["x"]), Float64.(npz["z"]), Float64.(npz["ages_ma"]),
        Float64.(npz["temperature_c"]),
        Float64.(npz["final_porosity"]), Int.(npz["final_lithology"]),
        Float64.(npz["final_surface_m"]), Float64.(npz["final_base_m"]),
        meta, basename(rstrip(run_dir, '/')))
end

function _temp_frames(dd::DirectData; max_frames::Int = 10)
    nt = length(dd.ages)
    idx = nt <= max_frames ? collect(1:nt) :
          round.(Int, range(1, nt; length = max_frames))
    unique!(idx)
    frames = [Dict("name" => @sprintf("%.0f", dd.ages[i]),
                   "data" => [Dict("z" => dd.temperature[i, :, :],
                                   "zmin" => 0.0, "zmax" => maximum(filter(isfinite, dd.temperature)))])
              for i in idx]
    steps = [Dict("method" => "animate",
                  "label" => @sprintf("%.0f", dd.ages[i]),
                  "args" => [[@sprintf("%.0f", dd.ages[i])],
                             Dict("mode" => "immediate",
                                  "frame" => Dict("duration" => 0, "redraw" => true),
                                  "transition" => Dict("duration" => 0))])
             for i in idx]
    return frames, steps
end

function section_direct(io::IOBuffer, dd::DirectData)
    xkm = dd.x ./ 1000
    zkm = dd.z ./ 1000
    finite_T = filter(isfinite, dd.temperature)
    tmin, tmax = extrema(finite_T)

    println(io, "<section class=\"card\" id=\"direct\">")
    println(io, "<h2 class=\"sec\">Прямая задача — «", esc(dd.run_name), "»</h2>")

    # --- сводка прогона
    solver = get(dd.meta, "solver", Dict{String,Any}())
    println(io, "<h3>Сводка прогона</h3><div class=\"cards\">")
    _metric(io, "Сетка", "$(get(solver, "nx", "—")) × $(get(solver, "nz", "—"))")
    _metric(io, "Шаги, м", "x≈$(fmt(get(dd.meta, "dx_m", NaN), d = 0)), z≈$(fmt(get(dd.meta, "dz_m", NaN), d = 0))")
    _metric(io, "Расчёт, с", fmt(get(dd.meta, "elapsed_seconds", NaN), d = 1))
    _metric(io, "T финальная", "$(fmt(get(dd.meta, "final_temperature_min_c", NaN), d = 1)) … $(fmt(get(dd.meta, "final_temperature_max_c", NaN), d = 1)) °C")
    _metric(io, "Δt ≤, Ма", fmt(get(solver, "max_dt_ma", NaN), d = 1))
    println(io, "</div>")

    _grid_section(io, dd)
    # --- температурное поле со слайдером по эпохам
    println(io, "<h3>Температурное поле T(x, z) по эпохам</h3>")
    println(io, "<p class=\"small\">Поле температуры, °C. Слайдер — время (Ма, назад от 0).</p>")
    frames, steps = _temp_frames(dd)
    data = Dict("type" => "heatmap",
                "x" => xkm, "y" => zkm,
                "z" => dd.temperature[end, :, :],
                "zmin" => 0.0, "zmax" => tmax,
                "colorscale" => "Inferno", "reversescale" => false,
                "colorbar" => Dict("title" => Dict("text" => "T, °C", "side" => "right")))
    layout = Dict("xaxis" => Dict("title" => "X, км"),
                  "yaxis" => Dict("title" => "Глубина z, км", "autorange" => "reversed"),
                  "sliders" => [Dict("active" => length(steps) - 1,
                                     "currentvalue" => Dict("prefix" => "Эпоха: ", "suffix" => " Ма"),
                                     "pad" => Dict("t" => 30), "steps" => steps)],
                  "transition" => Dict("duration" => 0))
    plot_div(io, "direct-temp", data, layout; frames = frames, height = 520, width = 1140)

    # --- температура по глубине в нескольких колонках
    println(io, "<h3>Термические профили по колонкам</h3>")
    println(io, "<div class=\"grid2\">")
    _depth_profiles(io, dd, xkm, zkm)
    println(io, "</div>")

    # --- геометрия бассейна
    println(io, "<h3>Геометрия бассейна (текущий момент)</h3>")
    println(io, "<div class=\"grid2\">")
    _geometry(io, dd, xkm)
    _stratigraphy(io, dd, xkm, zkm)
    println(io, "</div>")

    # --- пористость
    println(io, "<h3>Пористость (текущий момент)</h3>")
    _porosity(io, dd, xkm, zkm)

    println(io, "</section>")
end

function _metric(io::IOBuffer, label::String, value::String)
    println(io, "<div class=\"metric\"><div class=\"v\">", value,
            "</div><div class=\"l\">", esc(label), "</div></div>")
end

function _depth_profiles(io::IOBuffer, dd::DirectData, xkm, zkm)
    nx = length(xkm)
    cols = unique(round.(Int, range(1, nx; length = 4)))
    pal = ["#2c5f8a", "#c0392b", "#1e8449", "#8e44ad"]
    for (j, i) in enumerate(cols)
        traces = Any[]
        sel = round.(Int, range(1, length(dd.ages); length = 5)) |> unique
        for (k, t) in enumerate(sel)
            push!(traces, Dict("type" => "scatter", "mode" => "lines",
                "name" => @sprintf("%.0f Ма", dd.ages[t]),
                "x" => dd.temperature[t, :, i], "y" => zkm,
                "line" => Dict("color" => pal[mod1(k, 4)], "width" => 1.8)))
        end
        layout = Dict("title" => @sprintf("x = %.1f км", xkm[i]),
                      "xaxis" => Dict("title" => "T, °C"),
                      "yaxis" => Dict("title" => "Глубина, км", "autorange" => "reversed"),
                      "legend" => Dict("orientation" => "h", "y" => -0.2))
        plot_div(io, "direct-col-$i", traces, layout; height = 380, width = 555)
    end
end

function _geometry(io::IOBuffer, dd::DirectData, xkm)
    traces = Any[
        Dict("type" => "scatter", "mode" => "lines", "name" => "Дневная поверхность",
             "fill" => "tonexty", "fillcolor" => "rgba(120,170,220,0.35)",
             "x" => xkm, "y" => dd.base ./ 1000,
             "line" => Dict("color" => "#5b6b7b", "width" => 1.5)),
        Dict("type" => "scatter", "mode" => "lines", "name" => "Кровля фундамента",
             "x" => xkm, "y" => dd.surface ./ 1000,
             "line" => Dict("color" => "#2c5f8a", "width" => 2.5))]
    layout = Dict("title" => "Осадочная толща",
                  "xaxis" => Dict("title" => "X, км"),
                  "yaxis" => Dict("title" => "Глубина, км", "autorange" => "reversed"),
                  "legend" => Dict("orientation" => "h", "y" => -0.2))
    plot_div(io, "direct-geometry", traces, layout; height = 380, width = 555)
end

function _stratigraphy(io::IOBuffer, dd::DirectData, xkm, zkm)
    codes = sort(unique(dd.lithology))
    n = length(codes)
    positions = [(c - minimum(codes)) / max(maximum(codes) - minimum(codes), 1) for c in codes]
    colorscale = Any[]
    for (p, c) in zip(positions, codes)
        # дискретные полосы, позиции зажаты в [0, 1]: Plotly отвергает цвета вне диапазона
        lo = clamp(p - 0.5 / n, 0.0, 1.0)
        hi = clamp(p + 0.5 / n, 0.0, 1.0)
        push!(colorscale, [lo, litho_color(c)])
        push!(colorscale, [hi, litho_color(c)])
    end
    sort!(colorscale, by = first)
    trace = Dict("type" => "heatmap", "x" => xkm, "y" => zkm,
                 "z" => dd.lithology, "coloraxis" => "c1",
                 "hovertemplate" => "x: %{x:.1f} км<br>z: %{y:.2f} км<br>код: %{z}<extra></extra>")
    layout = Dict("title" => "Литология",
                  "xaxis" => Dict("title" => "X, км"),
                  "yaxis" => Dict("title" => "Глубина, км", "autorange" => "reversed"),
                  "coloraxis" => Dict("colorscale" => colorscale,
                                      "cmin" => minimum(codes), "cmax" => maximum(codes),
                                      "colorbar" => Dict(
                                          "tickvals" => codes,
                                          "ticktext" => [get(LITHO_NAMES, c, "код $c") for c in codes],
                                          "len" => 0.9)))
    plot_div(io, "direct-litho", trace, layout; height = 380, width = 555)
end

function _porosity(io::IOBuffer, dd::DirectData, xkm, zkm)
    finite = filter(isfinite, dd.porosity)
    trace = Dict("type" => "heatmap", "x" => xkm, "y" => zkm,
                 "z" => dd.porosity, "colorscale" => "Blues",
                 "zmin" => 0.0, "zmax" => maximum(finite),
                 "colorbar" => Dict("title" => Dict("text" => "φ", "side" => "right")),
                 "hovertemplate" => "x: %{x:.1f} км<br>z: %{y:.2f} км<br>φ: %{z:.3f}<extra></extra>")
    layout = Dict("xaxis" => Dict("title" => "X, км"),
                  "yaxis" => Dict("title" => "Глубина, км", "autorange" => "reversed"))
    plot_div(io, "direct-poro", trace, layout; height = 400, width = 1140)
end

# --- Рабочая сетка -------------------------------------------------------------

"Схема рабочей сетки (линии субдискретизированы до ≤40 на ось) и adaptive-меш,
если прогон его сохранил (mesh_adaptive.png)."
function _grid_section(io::IOBuffer, dd::DirectData)
    println(io, "<h3>Рабочая сетка расчёта</h3>")
    println(io, "<div class=\"grid2\">")
    nx, nz = length(dd.x), length(dd.z)
    # границы ячеек: середины между центрами + внешние края
    xedges = vcat(dd.x[1] - (dd.x[2] - dd.x[1]) / 2,
                  (dd.x[1:end-1] .+ dd.x[2:end]) ./ 2,
                  dd.x[end] + (dd.x[end] - dd.x[end-1]) / 2)
    zedges = vcat(dd.z[1] - (dd.z[2] - dd.z[1]) / 2,
                  (dd.z[1:end-1] .+ dd.z[2:end]) ./ 2,
                  dd.z[end] + (dd.z[end] - dd.z[end-1]) / 2)
    step_line(n) = max(1, ceil(Int, n / 40))
    sx, sz = step_line(nx), step_line(nz)
    shapes = Any[]
    for i in 1:sx:nx+1
        push!(shapes, Dict("type" => "line", "x0" => xedges[i] / 1000,
                           "x1" => xedges[i] / 1000, "y0" => zedges[1] / 1000,
                           "y1" => zedges[end] / 1000,
                           "line" => Dict("color" => "#c3cedb", "width" => 1)))
    end
    for j in 1:sz:nz+1
        push!(shapes, Dict("type" => "line", "x0" => xedges[1] / 1000,
                           "x1" => xedges[end] / 1000, "y0" => zedges[j] / 1000,
                           "y1" => zedges[j] / 1000,
                           "line" => Dict("color" => "#c3cedb", "width" => 1)))
    end
    trace = Dict("type" => "heatmap", "x" => dd.x ./ 1000, "y" => dd.z ./ 1000,
                 "z" => dd.temperature[end, :, :], "colorscale" => "Inferno",
                 "opacity" => 0.55, "showscale" => false,
                 "hoverinfo" => "skip")
    layout = Dict("title" => "Рабочая сетка $(nx) × $(nz)" *
                                 (sx > 1 || sz > 1 ? " (показана субдискретизация)" : ""),
                  "xaxis" => Dict("title" => "X, км"),
                  "yaxis" => Dict("title" => "Глубина, км", "autorange" => "reversed"),
                  "shapes" => shapes)
    plot_div(io, "direct-grid", trace, layout; height = 380, width = 555)

    mesh_png = joinpath(REPORT_RUN_DIR_REF[], "mesh_adaptive.png")
    if isfile(mesh_png)
        b64 = base64encode(read(mesh_png))
        println(io, "<div><p class=\"small\">Адаптивная сетка финального шага ",
                "(mesh_adaptive.png):</p>",
                "<img src=\"data:image/png;base64,", b64,
                "\" style=\"max-width:100%;height:auto;border:1px solid var(--border);",
                "border-radius:8px\" alt=\"adaptive mesh\"></div>")
    else
        println(io, "<div><p class=\"small\">Регулярный прогон: адаптивная сетка ",
                "не использовалась, расчёт ведётся на равномерной сетке слева ",
                "($(nx) × $(nz) ячеек, шаг ≈ $(fmt((dd.x[2] - dd.x[1]); d = 0)) × ",
                "$(fmt((dd.z[2] - dd.z[1]); d = 0)) м).</p></div>")
    end
    println(io, "</div>")
end
