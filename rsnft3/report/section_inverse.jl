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
    println(io, "<h3>Приёмка (условие 18)</h3>")
    println(io, "<p><span class=\"badge ", pass2s ? "ok" : "bad", "\">",
            pass2s ? "ПРОЙДЕНО" : "НЕ ВЫПОЛНЕНО", "</span> ",
            "<span class=\"small\">RMSE ≤ 2σ и ≥ 95% точек в пределах 2σ</span></p>")
    println(io, "<div class=\"cards\">")
    _metric(io, "RMSE T", fmt(c["rmse_T"]) * " °C")
    _metric(io, "RMSE Ro", fmt(c["rmse_Ro"]; d = 3))
    _metric(io, "T в 1σ / 2σ", fmtP(c["frac_1s_T"]) * " / " * fmtP(c["frac_2s_T"]))
    _metric(io, "Ro в 1σ / 2σ", fmtP(c["frac_1s_Ro"]) * " / " * fmtP(c["frac_2s_Ro"]))
    _metric(io, "E^L2", fmt(c["l2"]; d = 3))
    _metric(io, "Точек T / Ro", string(sol["observations"]["n_T_total"]) * " / " *
                                 string(sol["observations"]["n_Ro_total"]))
    println(io, "</div>")

    _theta_section(io, iv)
    _ensemble_scatter(io, iv)
    _ensemble_table(io, iv)
    _well_profiles(io, iv)
    _settings_block(io, iv)
    println(io, "</section>")
end

function _pass(c::AbstractDict)
    f2T, f2R = c["frac_2s_T"], c["frac_2s_Ro"]
    okT = f2T === nothing || f2T >= 0.95
    okR = f2R === nothing || f2R >= 0.95
    return okT && okR
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
                  "yaxis" => Dict("title" => "Эпоха")); height = 340)

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
                  "legend" => Dict("orientation" => "h", "y" => -0.25)); height = 340)
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
                  "showlegend" => false); height = 330)
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
