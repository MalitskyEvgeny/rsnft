# Общие хелперы отчёта: экранирование, форматирование, каркас страницы.

const REPORT_CSS = """
:root {
  --bg: #f5f7fa; --card: #ffffff; --ink: #1c2733; --muted: #5b6b7b;
  --accent: #2c5f8a; --accent2: #4684b8; --border: #dde4ec;
  --ok: #1e7d3e; --bad: #b3261e; --warn: #a05a00;
}
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', 'PT Sans', Arial, sans-serif; margin: 0;
       background: var(--bg); color: var(--ink); line-height: 1.5; }
header.hero { background: linear-gradient(135deg, #1f3a56, #2c5f8a 60%, #3a7ab8);
       color: #fff; padding: 2.2em 2em 1.6em; }
header.hero h1 { margin: 0 0 .2em; font-size: 1.7em; font-weight: 600; }
header.hero .sub { color: #cfe0f0; font-size: .95em; }
.wrap { max-width: 1200px; margin: 0 auto; padding: 0 1.2em; }
nav.toc { background: var(--card); border: 1px solid var(--border);
       border-radius: 10px; margin: 1.2em auto; max-width: 1200px;
       padding: .8em 1.4em; }
nav.toc b { color: var(--muted); text-transform: uppercase; font-size: .78em;
       letter-spacing: .06em; }
nav.toc a { color: var(--accent); text-decoration: none; margin-right: 1.2em;
       white-space: nowrap; }
nav.toc a:hover { text-decoration: underline; }
main { max-width: 1200px; margin: 0 auto; padding: 0 1.2em 3em; }
section.card { background: var(--card); border: 1px solid var(--border);
       border-radius: 10px; padding: 1.4em 1.6em; margin: 1.4em 0;
       box-shadow: 0 1px 3px rgba(30,50,80,.06); }
h2.sec { margin: 0 0 .6em; font-size: 1.35em; color: var(--accent);
       border-bottom: 2px solid var(--border); padding-bottom: .35em; }
h3 { color: #34495e; margin: 1.4em 0 .5em; }
table { border-collapse: collapse; margin: .7em 0; font-size: .93em; }
td, th { border: 1px solid var(--border); padding: 5px 12px; text-align: right; }
th { background: #edf2f8; color: #33475b; }
td:first-child, th:first-child { text-align: left; }
.badge { display: inline-block; padding: 3px 14px; border-radius: 999px;
       font-weight: 700; font-size: .95em; color: #fff; }
.badge.ok { background: var(--ok); } .badge.bad { background: var(--bad); }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: .8em 0; }
.metric { background: #f2f6fb; border: 1px solid var(--border);
       border-radius: 8px; padding: 10px 16px; min-width: 150px; }
.metric .v { font-size: 1.45em; font-weight: 700; color: var(--accent); }
.metric .l { font-size: .82em; color: var(--muted); }
.plot { width: 100%; margin: 10px 0; }
.plot-wide { width: 100%; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
       gap: 10px; }
.grid2 > * { min-width: 0; }
.small { color: var(--muted); font-size: .88em; }
footer { text-align: center; color: var(--muted); font-size: .85em;
       padding: 1.5em 0 2.5em; }
@media print {
  nav.toc, .noprint { display: none; }
  body { background: #fff; }
  section.card { box-shadow: none; border: 1px solid #ccc; page-break-inside: avoid; }
}
"""

esc(s::AbstractString) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

fmt(x::Real; d::Int = 2) = (x === nothing || isnan(x)) ? "—" : string(round(x; digits = d))
fmt(x::Nothing; d::Int = 2) = "—"
fmtP(x::Real) = (x === nothing || isnan(x)) ? "—" : string(round(100 * x; digits = 1)) * "%"
fmtP(x::Nothing) = "—"

"JSON-безопасное значение: NaN/Inf -> null, ключи и матрицы -> строки-массивы."
function json_safe(x::Real)
    isnan(x) && return nothing
    isinf(x) && return nothing
    return x
end
json_safe(x::Union{Nothing,AbstractString,Bool}) = x
json_safe(x::AbstractVector) = Any[json_safe(v) for v in x]
json_safe(x::AbstractMatrix) = Any[json_safe(col) for col in eachrow(x)]
json_safe(d::AbstractDict) =
    Dict{String,Any}(string(k) => json_safe(v) for (k, v) in d)

"Div с графиком Plotly: данные/layout — Dict, сериализуются через JSON."
function plot_div(io::IOBuffer, id::String, data::Union{AbstractDict,AbstractVector},
                  layout::AbstractDict; frames::Union{Nothing,AbstractVector} = nothing,
                  height::Int = 460)
    l = Dict{String,Any}(layout)
    l["margin"] = get(layout, "margin", Dict("l" => 60, "r" => 25, "t" => 40, "b" => 45))
    l["height"] = height
    l["autosize"] = true
    l["paper_bgcolor"] = "rgba(0,0,0,0)"
    l["plot_bgcolor"] = "rgba(0,0,0,0)"
    l["font"] = merge(Dict("family" => "'Segoe UI', Arial, sans-serif", "size" => 13,
                           "color" => "#1c2733"), get(layout, "font", Dict()))
    println(io, "<div id=\"", id, "\" class=\"plot\"></div>")
    print(io, "<script>Plotly.newPlot('", id, "', ")
    # трейсы всегда массивом: одиночный объект Plotly молча не рисует
    if data isa AbstractDict
        JSON.print(io, [json_safe(data)])
    else
        JSON.print(io, json_safe(data))
    end
    print(io, ", ")
    JSON.print(io, json_safe(l))
    if frames !== nothing
        print(io, ").then(function(gd){Plotly.addFrames(gd,")
        JSON.print(io, json_safe(frames))
        print(io, ");})")
    else
        print(io, ")")
    end
    println(io, ";</script>")
end

const LITHO_NAMES = Dict{Int,String}(
    1 => "Глина", 2 => "Аргиллит", 3 => "Алевролит", 4 => "Песчаник",
    5 => "Известняк", 6 => "Доломит", 7 => "Ангидрит", 8 => "Каменная соль",
    9 => "Уголь", 10 => "Базальт")

"Цвета литотипов (случай по коду, приглушённая палитра)."
function litho_color(code::Int)
    palette = ["#8a9a5b", "#b5a642", "#c2b280", "#d9b382", "#7fb2c8",
               "#9fb8c8", "#cfd8dc", "#e57373", "#333333", "#5d4037"]
    return palette[mod1(code, length(palette))]
end
