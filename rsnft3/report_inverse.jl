# D1 (R7): самодостаточный HTML-отчёт по инверсии. Вход: каталог прогона
# run_inverse.jl (solution.json + result.npz). Таблица θ̂ ансамбля
# (mean/std/min/max), критерии (условие 18), per-well профили T и Ro
# (PNG Luxor, встроены base64), сходимость DE/NM.
#
# Запуск:
#   julia --project=. report_inverse.jl ../results/inverse_run1

using ArgParse
using JSON
using NPZ
using Base64
using Luxor

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "ro_kinetics.jl"))
include(joinpath(@__DIR__, "observations.jl"))
include(joinpath(@__DIR__, "objective.jl"))   # bilinear_sample
include(joinpath(@__DIR__, "criteria.jl"))

const STYLE = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 2em auto; max-width: 1100px;
       color: #222; background: #fafafa; }
h1 { border-bottom: 2px solid #468; padding-bottom: .3em; }
h2 { color: #345; margin-top: 1.6em; }
table { border-collapse: collapse; margin: .8em 0; }
td, th { border: 1px solid #bbb; padding: 4px 10px; text-align: right; }
th { background: #e8ecf4; }
.ok { color: #070; font-weight: bold; } .bad { color: #c00; font-weight: bold; }
img { max-width: 48%; margin: 6px; border: 1px solid #ccc; background: #fff; }
code { background: #eee; padding: 1px 5px; }
.small { color: #666; font-size: .9em; }
"""

esc(s::AbstractString) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")
fmt(x::Real; d::Int = 2) = isnan(x) ? "—" : string(round(x; digits = d))
fmt(x::Nothing; d::Int = 2) = "null"
fmtP(x::Real) = isnan(x) ? "—" : string(round(100 * x; digits = 1)) * "%"

"PNG профиля: X — значение (T °C или Ro), Y — глубина (м, вниз)."
function profile_png(wname::String, obs::Vector{Tuple{Float64,Float64}},   # (value, z)
                     calc::Vector{Tuple{Float64,Float64}};                  # (value, z)
                     xlabel::String, fname::String, sigma::Vector{Float64} = Float64[])
    isempty(obs) && return nothing
    vals = vcat(first.(obs), first.(calc))
    vmin, vmax = minimum(vals), maximum(vals)
    pad = 0.05 * max(vmax - vmin, 1e-9)
    zmin, zmax = extrema(vcat(last.(obs), last.(calc)))
    W, H = 420, 380
    Drawing(W, H, fname)
    origin()
    background("white")
    left, right, top, bot = 60.0, W - 25.0, 30.0, H - 45.0
    X(v) = left + (v - (vmin - pad)) / (vmax - vmin + 2pad) * (right - left)
    Y(z) = top + (z - zmin) / max(zmax - zmin, 1e-9) * (bot - top)
    setline(0.5); setcolor("grey80")
    for g in 0:4
        line(Point(left, top + g * (bot - top) / 4), Point(right, top + g * (bot - top) / 4), :stroke)
    end
    # расчётная линия
    setcolor("steelblue"); setline(1.6)
    for i in 2:length(calc)
        line(Point(X(calc[i-1][1]), Y(calc[i-1][2])), Point(X(calc[i][1]), Y(calc[i][2])), :stroke)
    end
    # наблюдения с σ-планками
    setcolor("firebrick")
    for (i, (v, z)) in enumerate(obs)
        circle(Point(X(v), Y(z)), 3.2, :fill)
        if get(sigma, i, 0.0) > 0
            setline(0.8)
            line(Point(X(v - sigma[i]), Y(z)), Point(X(v + sigma[i]), Y(z)), :stroke)
        end
    end
    setcolor("black"); fontsize(11)
    text(wname, Point(left, 16), halign = :left)
    text(xlabel, Point((left + right) / 2, H - 8), halign = :center)
    text(fmt(vmin; d = 1), Point(left, bot + 12), halign = :center)
    text(fmt(vmax; d = 1), Point(right, bot + 12), halign = :center)
    text(@sprintf("%.0f м", zmax), Point(left - 6, bot), halign = :right)
    text(@sprintf("%.0f м", zmin), Point(left - 6, top), halign = :right)
    finish()
    return fname
end

function main()
    isempty(ARGS) && error("usage: julia report_inverse.jl <run_dir>")
    run_dir = abspath(ARGS[1])
    isfile(joinpath(run_dir, "solution.json")) || error("no solution.json in $run_dir")
    sol = JSON.parsefile(joinpath(run_dir, "solution.json"))
    npz = npzread(joinpath(run_dir, "result.npz"))

    data = load_dataset_derived(sol["data_dir"], Workbook())
    wells = load_wells_multi(sol["wells_spec"], data;
                             sigma_T = sol["settings"]["sigma-T"],
                             sigma_Ro = sol["settings"]["sigma-Ro"])
    xg, zg, temp = npz["x"], npz["z"], npz["temperature_c"]
    Tfield = temp[end, :, :]

    # расчётные значения в точках скважин
    calc_T = Dict{String,Vector{Tuple{Float64,Float64}}}()
    obs_T = Dict{String,Vector{Tuple{Float64,Float64}}}()
    calc_Ro = Dict{String,Vector{Tuple{Float64,Float64}}}()
    obs_Ro = Dict{String,Vector{Tuple{Float64,Float64}}}()
    sig_T = Dict{String,Vector{Float64}}(); sig_Ro = Dict{String,Vector{Float64}}()
    for p in wells.points
        d = p.kind == :T ? calc_T : calc_Ro
        o = p.kind == :T ? obs_T : obs_Ro
        s = p.kind == :T ? sig_T : sig_Ro
        v = p.kind == :T ? bilinear_sample(xg, zg, Tfield, p.x, p.z) :
            begin
                le = layer_eta_at(data, p.x, p.z)
                le === nothing ? NaN : particle_ro((x = xg, z = zg,
                    ages_ma = npz["ages_ma"], temperature_c = temp,
                    final_lithology = npz["final_lithology"],
                    final_porosity = npz["final_porosity"],
                    final_surface_m = npz["final_surface_m"],
                    final_base_m = npz["final_base_m"],
                    elapsed_seconds = 0.0, dx = xg[2] - xg[1], dz = zg[2] - zg[1],
                    diagnostics = nothing), data, p.x, le[1], le[2])
            end
        isfinite(v) || continue
        push!(get!(d, p.well, Tuple{Float64,Float64}[]), (v, p.z))
        push!(get!(o, p.well, Tuple{Float64,Float64}[]), (p.value, p.z))
        push!(get!(s, p.well, Float64[]), p.sigma)
    end

    # PNG по скважинам
    tmp = mktempdir()
    imgs = String[]
    for w in sort(collect(keys(obs_T)); by = w -> get(sol["observations"]["wells"][w], "x_m", 0.0))
        for (kind, obs, calc, sg, xl) in (("T", obs_T, calc_T, sig_T, "T, °C"),
                                          ("Ro", obs_Ro, calc_Ro, sig_Ro, "Ro, %"))
            haskey(obs, w) || continue
            f = joinpath(tmp, "$(w)_$(kind).png")
            profile_png("$w ($kind)", obs[w], calc[w]; xlabel = xl, fname = f,
                        sigma = get(sg, w, Float64[])) === nothing || push!(imgs, f)
        end
    end
    img_html = join(["<img src=\"data:image/png;base64,$(base64encode(read(f)))\"/>" for f in imgs], "\n")

    # таблица ансамбля
    ens = sol["ensemble"]
    th = hcat([Vector{Float64}(e["theta"]) for e in ens]...)
    names = sol["layout"]["names"]
    stat_rows = join(["<tr><td>$(esc(n))</td><td>$(fmt(mean(th[i,:]), d=1))</td><td>$(fmt(std(th[i,:]), d=1))</td>" *
                      "<td>$(fmt(minimum(th[i,:]), d=1))</td><td>$(fmt(maximum(th[i,:]), d=1))</td></tr>"
                      for (i, n) in enumerate(names)])
    ens_rows = join(["<tr><td>$i</td><td>$(fmt(e["J"]; d=1))</td><td>$(fmt(e["J_T"]; d=1))</td>" *
                     "<td>$(fmt(e["J_Ro"]; d=1))</td><td>$(fmt(e["J_reg"]; d=2))</td>" *
                     "<td>$(e["de_evals"])</td><td>$(e["nm_evals"])</td><td>$(fmt(e["elapsed_s"], d=0))</td></tr>"
                     for (i, e) in enumerate(ens)])
    c = sol["criteria"]
    crit_rows = join(["<tr><td>RMSE T, °C</td><td>$(fmt(c["rmse_T"]))</td></tr>",
                      "<tr><td>RMSE Ro</td><td>$(fmt(c["rmse_Ro"], d=3))</td></tr>",
                      "<tr><td>T в 1σ</td><td>$(fmtP(c["frac_1s_T"]))</td></tr>",
                      "<tr><td>T в 2σ</td><td>$(fmtP(c["frac_2s_T"]))</td></tr>",
                      "<tr><td>Ro в 1σ</td><td>$(fmtP(c["frac_1s_Ro"]))</td></tr>",
                      "<tr><td>Ro в 2σ</td><td>$(fmtP(c["frac_2s_Ro"]))</td></tr>",
                      "<tr><td>E^L2</td><td>$(fmt(c["l2"], d=3))</td></tr>"])
    per_well = join(["<tr><td>$(esc(w["well"]))</td><td>$(w["n_T"])</td><td>$(w["n_Ro"])</td>" *
                     "<td>$(fmt(w["rmse_T"]))</td><td>$(fmt(w["rmse_Ro"], d=3))</td>" *
                     "<td>$(fmtP(w["frac_2s_T"]))</td><td>$(fmtP(w["frac_2s_Ro"]))</td></tr>"
                     for w in c["per_well"]])

    pass2s = (c["frac_2s_T"] >= 0.95 || isnan(c["frac_2s_T"])) &&
             (c["frac_2s_Ro"] >= 0.95 || isnan(c["frac_2s_Ro"]))
    html = """
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Inverse report</title>
<style>$STYLE</style></head><body>
<h1>Inverse problem report — $(esc(basename(run_dir)))</h1>
<p class="small">grid $(sol["grid"]["nx"])x$(sol["grid"]["nz"]),
   ensemble $(sol["settings"]["ensemble"]), DE budget $(sol["settings"]["de-budget"]),
   σ_T=$(sol["settings"]["sigma-T"]) °C, σ_Ro=$(sol["settings"]["sigma-Ro"]),
   reg=$(sol["settings"]["reg"]), weight_ro=$(sol["settings"]["weight-ro"]),
   total $(fmt(sol["elapsed_total_s"], d=0)) s</p>
<h2>Acceptance (criterion 18)</h2>
<p class="$(pass2s ? "ok" : "bad")">$(pass2s ? "PASS" : "NOT MET"): RMSE <= 2sigma and >=95% of points within 2sigma</p>
<table><tr><th>Criterion</th><th>Value</th></tr>$crit_rows</table>
<h2>Ensemble thetas (non-uniqueness)</h2>
<table><tr><th>Parameter</th><th>mean</th><th>std</th><th>min</th><th>max</th></tr>$stat_rows</table>
<h2>Ensemble starts</h2>
<table><tr><th>#</th><th>J</th><th>J_T</th><th>J_Ro</th><th>J_reg</th><th>DE evals</th><th>NM evals</th><th>time, s</th></tr>$ens_rows</table>
<h2>Per well</h2>
<table><tr><th>Well</th><th>n_T</th><th>n_Ro</th><th>RMSE_T</th><th>RMSE_Ro</th><th>T in 2sigma</th><th>Ro in 2sigma</th></tr>$per_well</table>
<h2>Profiles: model (line) vs observations (points, +-sigma)</h2>
$img_html
</body></html>"""
    out = joinpath(run_dir, "report.html")
    write(out, html)
    println("SAVED ", out)
end

try
    main()
catch err
    showerror(stderr, err)
    println(stderr)
    exit(1)
end
