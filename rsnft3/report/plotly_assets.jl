# Plotly.js: по умолчанию встроен (отчёт самодостаточен офлайн);
# SLIM-режим — ссылка на общий файл results/assets/plotly.min.js (6.3).

const PLOTLY_LOCAL = joinpath(@__DIR__, "plotly.min.js")
const PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
const PLOTLY_SHARED = joinpath(@__DIR__, "..", "..", "results", "assets", "plotly.min.js")
const REPORT_SLIM = Ref(false)
const REPORT_OUT_PATH = Ref("")

function plotly_script_tag(io::IOBuffer)
    if REPORT_SLIM[]
        if !isfile(PLOTLY_LOCAL)
            @warn "slim: локальный plotly.min.js не найден, подключаю CDN" file = PLOTLY_LOCAL
        else
            # P1 Review 6: копируем plotly рядом с итоговым HTML, ссылка всегда валидна
            out_dir = dirname(abspath(REPORT_OUT_PATH[]))
            shared = joinpath(out_dir, "assets", "plotly.min.js")
            mkpath(dirname(shared))
            isfile(shared) || cp(PLOTLY_LOCAL, shared)
            println(io, "<script src=\"assets/plotly.min.js\"></script>")
            return false
        end
    end
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
