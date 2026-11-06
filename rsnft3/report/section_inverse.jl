# Секция «Обратная задача»: θ̂ (q по эпохам и x-узлам), критерии приёмки,
# профили скважин, ансамбль и настройки. Вход — каталог прогона run_inverse.jl.

struct InverseData
    sol::Dict{String,Any}
    data::Any                      # BasinDataset
    wells::Any                     # WellSet из observations.jl
    npz::Dict{String,Any}
    run_name::String
end

const REPORT_REPO_ROOT = dirname(dirname(@__DIR__))

"Путь из solution.json может быть относительным к корню репозитория."
function resolve_path(spec::AbstractString)
    parts = split(spec, ';')
    out = String[]
    for p in parts
        isempty(strip(p)) && continue
        isfile(p) ? push!(out, String(p)) :
            push!(out, joinpath(REPORT_REPO_ROOT, p))
    end
    return join(out, ';')
end

function load_inverse(run_dir::AbstractString)
    sol = JSON.parsefile(joinpath(run_dir, "solution.json"))
    npz = npzread(joinpath(run_dir, "result.npz"))
    data = load_dataset_derived(sol["data_dir"], Workbook())
    wells = load_wells_multi(resolve_path(sol["wells_spec"]), data;
                             sigma_T = sol["settings"]["sigma-T"],
                             sigma_Ro = sol["settings"]["sigma-Ro"])
    sol = Dict{String,Any}(string(k) => v for (k, v) in sol)
    return InverseData(sol, data, wells, npz, basename(rstrip(run_dir, '/')))
end

function section_inverse(io::IOBuffer, iv::InverseData)
    sol = iv.sol
    c = sol["criteria"]
    st = sol["settings"]
    println(io, "<section class=\"card\" id=\"inverse\">")
    println(io, "<h2 class=\"sec\">Обратная задача — «", esc(iv.run_name), "»</h2>")

    pass2s = _pass(c)
    n_art = get(c, "n_artifact", nothing)
    n_kept = get(c, "n_kept", nothing)
    println(io, "<h3>Приёмка (условие 18, строгая)</h3>")
    println(io, "<p><span class=\"badge ", pass2s ? "ok" : "bad", "\">",
            pass2s ? "ПРОЙДЕНО" : "НЕ ВЫПОЛНЕНО", "</span> ",
            "<span class=\"small\">RMSE ≤ 2σ и ≥ 95% точек в пределах 2σ и 0 артефактных точек</span></p>")
    println(io, "<div class=\"cards\">")
    if n_art !== nothing && n_art > 0
        println(io, "<p class=\"small\"><b>Артефактные точки: ", n_art,
                "</b> — модель в этих точках не определена (вырожденная/NaN-выборка); ",
                "приёмка не может считаться пройденной.</p>")
    end
    if n_art !== nothing && n_kept !== nothing
        tot = n_art + n_kept
        tot > 0 && println(io, "<p class=\"small\">Доля артефактных точек: ",
                @sprintf("%.1f%%", 100.0 * n_art / tot), " (kept ", n_kept, " / dropped ", n_art, ")</p>")
    end
    okT = c["frac_2s_T"] === nothing || c["frac_2s_T"] >= 0.95
    okR = c["frac_2s_Ro"] === nothing || c["frac_2s_Ro"] >= 0.95
    defT = c["frac_2s_T"] === nothing ? "" :
           @sprintf(" · дефицит %.1f п.п. до порога 95%%", 100 * max(0.0, 0.95 - c["frac_2s_T"]))
    defR = c["frac_2s_Ro"] === nothing ? "" :
           @sprintf(" · дефицит %.1f п.п. до порога 95%%", 100 * max(0.0, 0.95 - c["frac_2s_Ro"]))
    _metric(io, "RMSE T", fmt(c["rmse_T"]) * " °C")
    _metric(io, "RMSE Ro", fmt(c["rmse_Ro"]; d = 3))
    _metric(io, "T в 1σ / 2σ", fmtP(c["frac_1s_T"]) * " / " * fmtP(c["frac_2s_T"]) * defT)
    _metric(io, "Ro в 1σ / 2σ", fmtP(c["frac_1s_Ro"]) * " / " * fmtP(c["frac_2s_Ro"]) * defR)
    _metric(io, "E^L2", fmt(c["l2"]; d = 3))
    _metric(io, "Точек T / Ro", string(sol["observations"]["n_T_total"]) * " / " *
                                 string(sol["observations"]["n_Ro_total"]))
    println(io, "</div>")

    _theta_section(io, iv)
    _ensemble_q_section(io, iv)
    _ensemble_scatter(io, iv)
    _ensemble_table(io, iv)
    _well_profiles(io, iv)
    _residual_diagnostics(io, iv)
    _dropped_points_block(io, iv)
    _settings_block(io, iv)
    _repro_block(io, iv)
    println(io, "</section>")
end

function _pass(c::AbstractDict)
    haskey(c, "passed") && return c["passed"] === true
    f2T, f2R = c["frac_2s_T"], c["frac_2s_Ro"]
    okT = f2T === nothing || f2T >= 0.95
    okR = f2R === nothing || f2R >= 0.95
    n_art = get(c, "n_artifact", 0)
    return okT && okR && n_art == 0
end

function _metric(io::IOBuffer, label::String, value::String)
    println(io, "<div class=\"metric\"><div class=\"v\">", value,
            "</div><div class=\"l\">", esc(label), "</div></div>")
end

function _theta_section(io::IOBuffer, iv::InverseData)
    sol = iv.sol
    layout = sol["layout"]
    nodes_x = Vector{Float64}(layout["nodes_x"])
    epochs = layout["epochs"]
    epoch_labels = [@sprintf("%.0f–%.0f", e[1], e[2]) for e in epochs]
    theta = Vector{Float64}(sol["best"]["theta"])
    nq = length(layout["names"])
    nxq = length(nodes_x)
    Q = permutedims(reshape(theta, nxq, :))     # (n_epochs, nxq)

    println(io, "<h3>Восстановленный поток q(x, t)</h3>")
    println(io, "<div class=\"grid2\">")
    trace = Dict("type" => "heatmap", "z" => Q, "x" => nodes_x ./ 1000,
                 "y" => epoch_labels, "colorscale" => "Portland",
                 "colorbar" => Dict("title" => Dict("text" => "мВт/м²", "side" => "right")),
                 "hovertemplate" => "x: %{x:.1f} км<br>эпоха: %{y}<br>q: %{z:.1f} мВт/м²<extra></extra>")
    plot_div(io, "inv-theta", trace,
             Dict("xaxis" => Dict("title" => "X, км"),
                  "yaxis" => Dict("title" => "Эпоха")); height = 340, width = 555)

    rows = join(("<tr><td>" * esc(n) * "</td><td>" * fmt(theta[i]; d = 1) *
                 "</td><td>" * fmt(layout["lower"][i]; d = 0) *
                 "</td><td>" * fmt(layout["upper"][i]; d = 0) * "</td></tr>")
                for (i, n) in enumerate(layout["names"]))
    println(io, "<div><table><tr><th>Параметр</th><th>θ̂, мВт/м²</th><th>мин</th><th>макс</th></tr>",
            rows, "</table></div>")
    println(io, "</div>")
end

function _ensemble_scatter(io::IOBuffer, iv::InverseData)
    ens = iv.sol["ensemble"]
    js = [e["J"] for e in ens]
    jt = [e["J_T"] for e in ens]
    jr = [e["J_Ro"] for e in ens]
    traces = Any[
        Dict("type" => "bar", "name" => "J_T", "x" => string.(1:length(js)), "y" => jt,
             "marker" => Dict("color" => "#4684b8")),
        Dict("type" => "bar", "name" => "J_Ro", "x" => string.(1:length(js)), "y" => jr,
             "marker" => Dict("color" => "#c0653a")),
        Dict("type" => "scatter", "mode" => "markers", "name" => "J (итог)",
             "x" => string.(1:length(js)), "y" => js,
             "marker" => Dict("color" => "#1c2733", "size" => 9))]
    plot_div(io, "inv-ens", traces,
             Dict("barmode" => "group",
                  "title" => "Вклад в функционал по стартам",
                  "xaxis" => Dict("title" => "Старт ансамбля"),
                  "yaxis" => Dict("title" => "Значение", "type" => "log"),
                  "legend" => Dict("orientation" => "h", "y" => -0.25)); height = 340, width = 1140)
end

function _ensemble_table(io::IOBuffer, iv::InverseData)
    ens = iv.sol["ensemble"]
    names = iv.sol["layout"]["names"]
    th = hcat([Vector{Float64}(e["theta"]) for e in ens]...)
    rows = join(("<tr><td>" * esc(n) * "</td><td>" * fmt(mean(th[i, :]); d = 1) *
                 "</td><td>" * fmt(std(th[i, :]); d = 1) *
                 "</td><td>" * fmt(minimum(th[i, :]); d = 1) *
                 "</td><td>" * fmt(maximum(th[i, :]); d = 1) * "</td></tr>")
                for (i, n) in enumerate(names))
    println(io, "<div><p class=\"small\">Разброс ансамбля (невзаимозаменяемость решения)</p>",
            "<table><tr><th>Параметр</th><th>среднее</th><th>std</th><th>мин</th><th>макс</th></tr>",
            rows, "</table></div>")
end

"Расчётные значения T и Ro в точках скважин (прогон уже в result.npz)."
function _well_values(iv::InverseData)
    npz = iv.npz
    xg, zg, temp = Float64.(npz["x"]), Float64.(npz["z"]), Float64.(npz["temperature_c"])
    Tfield = temp[end, :, :]
    data = iv.data
    calc_T = Dict{String,Vector{Tuple{Float64,Float64}}}()
    obs_T = Dict{String,Vector{Tuple{Float64,Float64}}}()
    calc_Ro = Dict{String,Vector{Tuple{Float64,Float64}}}()
    obs_Ro = Dict{String,Vector{Tuple{Float64,Float64}}}()
    sig_T = Dict{String,Vector{Float64}}()
    sig_Ro = Dict{String,Vector{Float64}}()
    for p in iv.wells.points
        if p.kind == :T
            v = bilinear_sample(xg, zg, Tfield, p.x, p.z)
            isfinite(v) || continue
            push!(get!(calc_T, p.well, Tuple{Float64,Float64}[]), (v, p.z))
            push!(get!(obs_T, p.well, Tuple{Float64,Float64}[]), (p.value, p.z))
            push!(get!(sig_T, p.well, Float64[]), p.sigma)
        else
            le = layer_eta_at(data, p.x, p.z)
            le === nothing && continue
            v = particle_ro((x = xg, z = zg, ages_ma = npz["ages_ma"],
                             temperature_c = temp,
                             final_lithology = npz["final_lithology"],
                             final_porosity = npz["final_porosity"],
                             final_surface_m = npz["final_surface_m"],
                             final_base_m = npz["final_base_m"],
                             elapsed_seconds = 0.0, dx = xg[2] - xg[1],
                             dz = zg[2] - zg[1], diagnostics = nothing),
                            data, p.x, le[1], le[2])
            isfinite(v) || continue
            push!(get!(calc_Ro, p.well, Tuple{Float64,Float64}[]), (v, p.z))
            push!(get!(obs_Ro, p.well, Tuple{Float64,Float64}[]), (p.value, p.z))
            push!(get!(sig_Ro, p.well, Float64[]), p.sigma)
        end
    end
    return calc_T, obs_T, calc_Ro, obs_Ro, sig_T, sig_Ro
end

function _well_profiles(io::IOBuffer, iv::InverseData)
    calc_T, obs_T, calc_Ro, obs_Ro, sig_T, sig_Ro = _well_values(iv)
    well_x = Dict{String,Float64}(string(k) => w["x_m"] for (k, w) in iv.sol["observations"]["wells"])
    wells = sort(collect(keys(obs_T)); by = w -> get(well_x, w, 0.0))
    println(io, "<h3>Профили по скважинам: модель (линия) против наблюдений (точки, ±σ)</h3>")
    println(io, "<div class=\"grid2\">")
    for w in wells
        _profile_plot(io, "T", w, obs_T[w], calc_T[w], get(sig_T, w, Float64[]), "T, °C")
        haskey(obs_Ro, w) && _profile_plot(io, "Ro", w, obs_Ro[w], calc_Ro[w],
                                           get(sig_Ro, w, Float64[]), "Ro, %")
    end
    println(io, "</div>")
end

function _profile_plot(io::IOBuffer, kind::String, w::String,
                       obs::Vector{Tuple{Float64,Float64}},
                       calc::Vector{Tuple{Float64,Float64}},
                       sigma::Vector{Float64}, xlabel::String)
    zobs = [p[2] ./ 1000 for p in obs]
    zcalc = [p[2] ./ 1000 for p in calc]
    vals = vcat(first.(obs), first.(calc))
    pad = 0.05 * max(maximum(vals) - minimum(vals), 1e-9)
    traces = Any[
        Dict("type" => "scatter", "mode" => "lines", "name" => "модель",
             "x" => first.(calc), "y" => zcalc,
             "line" => Dict("color" => "#4684b8", "width" => 2.2)),
        Dict("type" => "scatter", "mode" => "markers", "name" => "наблюдения",
             "x" => first.(obs), "y" => zobs,
             "error_x" => Dict("type" => "data", "array" => sigma, "visible" => true,
                               "color" => "#c0653a", "thickness" => 1),
             "marker" => Dict("color" => "#c0392b", "size" => 6))]
    plot_div(io, "inv-well-$(kind)-$(w)", traces,
             Dict("title" => "$w ($kind)",
                  "xaxis" => Dict("title" => xlabel,
                                  "range" => [minimum(vals) - pad, maximum(vals) + pad]),
                  "yaxis" => Dict("title" => "Глубина, км", "autorange" => "reversed"),
                  "legend" => Dict("orientation" => "h", "y" => -0.2),
                  "showlegend" => false); height = 330, width = 555)
end

function _settings_block(io::IOBuffer, iv::InverseData)
    st = iv.sol["settings"]
    println(io, "<h3>Настройки прогона</h3>")
    println(io, "<table><tr><th>Параметр</th><th>Значение</th></tr>")
    for k in ("ensemble", "workers", "de-budget", "nm-iterations", "sigma-T",
              "sigma-Ro", "seed", "reg", "weight-ro")
        println(io, "<tr><td>", esc(k), "</td><td>", esc(string(st[k])), "</td></tr>")
    end
    println(io, "<tr><td>полное время</td><td>", fmt(iv.sol["elapsed_total_s"]; d = 0),
            " с</td></tr>")
    g = iv.sol["grid"]
    println(io, "<tr><td>рабочая сетка</td><td>", g["nx"], " × ", g["nz"], "</td></tr>")
    println(io, "</table>")
end

# --- Диагностика невязок ------------------------------------------------------

const RESID_PALETTE = ["#4684b8", "#c0653a", "#1e7d3e", "#7d5ba6", "#b3261e",
                       "#a05a00", "#2c8a8a"]

"Невязки всех валидных точек: (well, kind, x_m, z_m, res, sigma) + отброшенные."
function _residuals(iv::InverseData)
    npz = iv.npz
    xg, zg, temp = Float64.(npz["x"]), Float64.(npz["z"]), Float64.(npz["temperature_c"])
    Tfield = temp[end, :, :]
    out = Tuple{String,Symbol,Float64,Float64,Float64,Float64}[]
    dropped = Tuple{String,Symbol,Float64,Float64}[]
    for p in iv.wells.points
        if p.kind == :T
            v = bilinear_sample(xg, zg, Tfield, p.x, p.z)
            if isfinite(v)
                push!(out, (p.well, :T, p.x, p.z, v - p.value, p.sigma))
            else
                push!(dropped, (p.well, :T, p.x, p.z))
            end
        else
            le = layer_eta_at(iv.data, p.x, p.z)
            if le === nothing
                push!(dropped, (p.well, :Ro, p.x, p.z))
                continue
            end
            rc = particle_ro((x = xg, z = zg, ages_ma = npz["ages_ma"],
                              temperature_c = temp,
                              final_lithology = npz["final_lithology"],
                              final_porosity = npz["final_porosity"],
                              final_surface_m = npz["final_surface_m"],
                              final_base_m = npz["final_base_m"],
                              elapsed_seconds = 0.0, dx = xg[2] - xg[1],
                              dz = zg[2] - zg[1], diagnostics = nothing),
                             iv.data, p.x, le[1], le[2])
            if isfinite(rc)
                push!(out, (p.well, :Ro, p.x, p.z, rc - p.value, p.sigma))
            else
                push!(dropped, (p.well, :Ro, p.x, p.z))
            end
        end
    end
    return out, dropped
end

"Гистограммы невязок T/Ro с линиями ±1σ/±2σ и scatter по глубине/X."
function _residual_diagnostics(io::IOBuffer, iv::InverseData)
    res, _ = _residuals(iv)
    rT = [r for r in res if r[2] == :T]
    rR = [r for r in res if r[2] == :Ro]
    println(io, "<h3>Диагностика невязок</h3>")
    isempty(res) && (println(io, "<p class=\"small\">Нет валидных точек</p>"); return)

    sol = iv.sol
    println(io, "<div class=\"grid2\">")
    rows = join(("<tr><td>" * esc(String(w["well"])) * "</td><td>" * string(w["n_T"]) *
                 "</td><td>" * fmt(w["rmse_T"]) * "</td><td>" * fmtP(w["frac_2s_T"]) *
                 "</td><td>" * string(w["n_Ro"]) * "</td><td>" * fmt(w["rmse_Ro"]; d = 3) *
                 "</td><td>" * fmtP(w["frac_2s_Ro"]) * "</td></tr>")
                for w in sol["criteria"]["per_well"])
    println(io, "<div><p class=\"small\">Разбивка по скважинам</p>",
            "<table><tr><th>Скважина</th><th>n T</th><th>RMSE T</th><th>T 2σ</th>",
            "<th>n Ro</th><th>RMSE Ro</th><th>Ro 2σ</th></tr>", rows, "</table></div>")
    println(io, "<div><p class=\"small\">Гистограммы построены по валидным точкам; отброшенные ",
            "перечислены ниже. Для чистого шума в коридор ±2σ попадает ~95% точек, в ±1σ — ~68%; ",
            "систематический перекос означает смещение модели, а не случайный шум.</p></div>")
    println(io, "</div>")

    function hist_div(id, pts, color, label, sig)
        vals = [r[5] for r in pts]
        sigm = max(sum([r[6] for r in pts]) / length(pts), 1e-9)
        layout = Dict("title" => label * ", n=" * string(length(vals)),
             "xaxis" => Dict("title" => label, "range" => [-3 * sigm, 3 * sigm]),
             "yaxis" => Dict("title" => "число точек"),
             "shapes" => Any[
                 Dict("type" => "line", "x0" => -sigm, "x1" => sigm, "y0" => 0, "y1" => 1,
                      "yref" => "paper", "line" => Dict("color" => "#1e7d3e", "width" => 1.5)),
                 Dict("type" => "line", "x0" => -2 * sigm, "x1" => 2 * sigm, "y0" => 0, "y1" => 1,
                      "yref" => "paper",
                      "line" => Dict("color" => "#c0653a", "width" => 1.5, "dash" => "dot"))],
             "annotations" => Any[
                 Dict("text" => "±1σ", "x" => sigm, "y" => 1.03, "yref" => "paper",
                      "showarrow" => false,
                      "font" => Dict("size" => 10, "color" => "#1e7d3e")),
                 Dict("text" => "±2σ", "x" => 2 * sigm, "y" => 1.03, "yref" => "paper",
                      "showarrow" => false,
                      "font" => Dict("size" => 10, "color" => "#c0653a"))])
        trace = Dict("type" => "histogram", "x" => vals, "nbinsx" => 30,
                     "marker" => Dict("color" => color))
        plot_div(io, id, trace, layout; height = 340, width = 555)
    end

    println(io, "<div class=\"grid2\">")
    isempty(rT) || hist_div("inv-res-hist-T", rT, "#4684b8", "Невязка T, °C", 2.0)
    isempty(rR) || hist_div("inv-res-hist-Ro", rR, "#c0653a", "Невязка Ro, %", 0.05)
    println(io, "</div>")

    well_x = Dict{String,Float64}(string(k) => w["x_m"] for (k, w) in sol["observations"]["wells"])
    wells = sort(unique(r[1] for r in res); by = w -> get(well_x, w, 0.0))
    function scat_traces(kind, coord)
        traces = Any[]
        for (iw, w) in enumerate(wells)
            pts = [r for r in res if r[2] == kind && r[1] == w]
            isempty(pts) && continue
            push!(traces, Dict("type" => "scatter", "mode" => "markers", "name" => w,
                 "x" => [r[coord] / 1000 for r in pts], "y" => [r[5] for r in pts],
                 "marker" => Dict("color" => RESID_PALETTE[mod1(iw, length(RESID_PALETTE))],
                                  "size" => 5, "opacity" => 0.75),
                 "hovertemplate" => "%{x:.2f}<br>res: %{y:.2f}<extra>" * w * "</extra>"))
        end
        return traces
    end
    println(io, "<div class=\"grid2\">")
    for (kind, suffix, unit) in ((:T, "T", "°C"), (:Ro, "Ro", "%"))
        for (coord, axname) in ((4, "Глубина, км"), (3, "X, км"))
            traces = scat_traces(kind, coord)
            isempty(traces) && continue
            plot_div(io, "inv-res-$(suffix)-$(coord)", traces,
                     Dict("title" => "Невязка $suffix vs $axname",
                          "xaxis" => Dict("title" => axname),
                          "yaxis" => Dict("title" => "Невязка $suffix, $unit",
                                          "zeroline" => true),
                          "legend" => Dict("orientation" => "h", "y" => -0.25));
                     height = 340, width = 555)
        end
    end
    println(io, "</div>")
end

# --- Согласованность ансамбля --------------------------------------------------

"Линии q(x) всех стартов ансамбля по эпохам (разброс = мера идентифицируемости)."

"Линии q(x) всех стартов ансамбля по эпохам (разброс = мера идентифицируемости)."
function _ensemble_q_section(io::IOBuffer, iv::InverseData)
    sol = iv.sol
    ens = sol["ensemble"]
    length(ens) < 2 && return
    layout = sol["layout"]
    nodes_km = Vector{Float64}(layout["nodes_x"]) ./ 1000
    epochs = layout["epochs"]
    nxq = length(nodes_km)
    println(io, "<h3>Согласованность ансамбля: q(x) по эпохам</h3>")
    println(io, "<div class=\"grid2\">")
    for (ie, e) in enumerate(epochs)
        traces = Any[]
        for (is, m) in enumerate(ens)
            th = Vector{Float64}(m["theta"])
            push!(traces, Dict("type" => "scatter", "mode" => "lines",
                 "name" => "старт " * string(is), "x" => nodes_km,
                 "y" => th[((ie - 1) * nxq + 1):(ie * nxq)],
                 "line" => Dict("color" => RESID_PALETTE[mod1(is, length(RESID_PALETTE))],
                                "width" => 1.4, "dash" => "dot"), "opacity" => 0.7))
        end
        thb = Vector{Float64}(sol["best"]["theta"])
        push!(traces, Dict("type" => "scatter", "mode" => "lines+markers", "name" => "лучший",
             "x" => nodes_km, "y" => thb[((ie - 1) * nxq + 1):(ie * nxq)],
             "line" => Dict("color" => "#1c2733", "width" => 2.4),
             "marker" => Dict("size" => 7)))
        plot_div(io, "inv-ens-q-$ie", traces,
                 Dict("title" => @sprintf("Эпоха %.0f–%.0f Ma", e[1], e[2]),
                      "xaxis" => Dict("title" => "X, км"),
                      "yaxis" => Dict("title" => "q, мВт/м²"),
                      "legend" => Dict("orientation" => "h", "y" => -0.25));
                 height = 330, width = 555)
    end
    println(io, "</div>")
end

"Список точек, исключённых из критериев (неактивная сетка / NaN-выборка)."
function _dropped_points_block(io::IOBuffer, iv::InverseData)
    _, dropped = _residuals(iv)
    n_inactive = get(iv.sol, "n_dropped_T_inactive", 0)
    println(io, "<h3>Отброшенные точки</h3>")
    println(io, "<p class=\"small\">Точки вне активной области сетки исключаются из функционала ",
            "и критериев. На этапе подготовки отброшено ", n_inactive,
            " T-точек (неактивная сетка). При финальном прогоне NaN-выборка дала ещё ",
            length(dropped), " точек.</p>")
    isempty(dropped) && (println(io, "<p class=\"small\">Все оставшиеся точки валидны.</p>"); return)
    rows = join(("<tr><td>" * esc(d[1]) * "</td><td>" * string(d[2]) * "</td><td>" *
                 fmt(d[3] / 1000; d = 2) * "</td><td>" * fmt(d[4] / 1000; d = 2) * "</td></tr>")
                for d in dropped)
    println(io, "<table><tr><th>Скважина</th><th>Тип</th><th>X, км</th><th>Глубина, км</th></tr>",
            rows, "</table>")
end

"Версии, git-коммит и входные данные прогона."
function _repro_block(io::IOBuffer, iv::InverseData)
    sol = iv.sol
    println(io, "<h3>Воспроизводимость</h3>")
    println(io, "<table><tr><th>Параметр</th><th>Значение</th></tr>")
    println(io, "<tr><td>версия Julia</td><td>", string(VERSION), "</td></tr>")
    git_hash = try
        readchomp(`git -C $REPORT_REPO_ROOT rev-parse --short HEAD`)
    catch
        "n/a"
    end
    println(io, "<tr><td>git commit</td><td>", esc(git_hash), "</td></tr>")
    println(io, "<tr><td>каталог данных</td><td>",
            esc(basename(get(sol, "data_dir", ""))), "</td></tr>")
    for f in split(get(sol, "wells_spec", ""), ';')
        isempty(strip(f)) && continue
        sz = isfile(strip(f)) ? @sprintf(" · %.0f КБ · %s", filesize(strip(f)) / 1024,
              Dates.format(Dates.unix2datetime(mtime(strip(f))), "yyyy-mm-dd")) : ""
        println(io, "<tr><td>файл наблюдений</td><td>", esc(basename(strip(f))), sz, "</td></tr>")
    end
    println(io, "<tr><td>seed ансамбля</td><td>", iv.sol["settings"]["seed"], "</td></tr>")
    println(io, "<tr><td>узлы q по X</td><td>",
            join([fmt(x / 1000; d = 1) for x in sol["layout"]["nodes_x"]], ", "), " км</td></tr>")
    println(io, "</table>")
end
