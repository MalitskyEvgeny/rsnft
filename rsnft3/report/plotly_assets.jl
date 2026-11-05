# Встроенный plotly.js: отчёт открывается офлайн.

const PLOTLY_LOCAL = joinpath(@__DIR__, "plotly.min.js")
const PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

function plotly_script_tag(io::IOBuffer)
    if isfile(PLOTLY_LOCAL)
        js = read(PLOTLY_LOCAL, String)
        print(io, "<script>")
        print(io, js)
        println(io, "</script>")
        return true
    end
    println(io, "<script src=\"$PLOTLY_CDN\"></script>")
    return false
end
