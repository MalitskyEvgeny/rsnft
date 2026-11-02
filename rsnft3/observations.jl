# A2: наблюдения скважин — плоский список точек + загрузчики XLSX (формат
# организаторов, лист Wells_Interp) и JSON (twin). Валидация геометрии:
# точка вне осадков на age 0 → исключается; скважина целиком вне → исключается
# с предупреждением (счётчики — в WellSet для solution.json).

using XLSX
import JSON

struct WellPoint
    well::String
    x::Float64
    z::Float64          # глубина, м, положительна вниз (как Z геометрии)
    kind::Symbol        # :T или :Ro
    value::Float64
    sigma::Float64
end

struct WellSet
    points::Vector{WellPoint}          # вошедшие в J
    excluded::Vector{WellPoint}        # вне геометрии (поточечно)
    excluded_wells::Vector{String}     # скважины целиком вне геометрии
end

n_points(ws::WellSet, kind::Symbol) = count(p -> p.kind === kind, ws.points)
n_wells(ws::WellSet) = length(unique(p.well for p in ws.points))

"""
    layer_eta_at(data, x, z) -> (layer_name, eta) | nothing

Слой, содержащий точку (x, z) на age 0; eta ∈ [0, 1] — относительная
глубина внутри слоя (top → bot).
"""
function layer_eta_at(data::BasinDataset, x::Float64, z::Float64)
    for (lname, ser) in data.depth
        haskey(ser, 0.0) || continue
        rec = ser[0.0]
        (rec.x[1] > x || rec.x[end] < x) && continue
        j = clamp(searchsortedlast(rec.x, x), 1, length(rec.x) - 1)
        t = (x - rec.x[j]) / max(rec.x[j+1] - rec.x[j], 1e-9)
        top = rec.top[j] + t * (rec.top[j+1] - rec.top[j])
        bot = rec.bot[j] + t * (rec.bot[j+1] - rec.bot[j])
        if z >= top && z <= bot
            return lname, (z - top) / max(bot - top, 1e-9)
        end
    end
    return nothing
end

# --- внутренняя сборка WellSet с валидацией -----------------------------------

"""Точки по скважинам + фильтр геометрии; скважина без единой точки внутри —
исключается целиком (excluded_wells)."""
function build_wellset(raw::Vector{WellPoint}, data::BasinDataset)
    points = WellPoint[]
    excluded = WellPoint[]
    by_well = Dict{String,Vector{WellPoint}}()
    for p in raw
        isfinite(p.x) && isfinite(p.z) && isfinite(p.value) || continue
        p.sigma > 0 || error("sigma must be > 0 for well $(p.well) ($(p.kind))")
        push!(get!(by_well, p.well, WellPoint[]), p)
    end
    excluded_wells = String[]
    for wname in sort(collect(keys(by_well)))
        inside = WellPoint[]
        for p in by_well[wname]
            if layer_eta_at(data, p.x, p.z) === nothing
                push!(excluded, p)
            else
                push!(inside, p)
            end
        end
        if isempty(inside)
            push!(excluded_wells, wname)
            @warn "Well \"$wname\" is entirely outside the sediment geometry — excluded from J"
        else
            append!(points, inside)
        end
    end
    return WellSet(points, excluded, excluded_wells)
end

# --- XLSX организаторов (TemperatureWellData / VitriniteWellData) -------------

"""
    load_wells_xlsx(path, data; kind, sigma_T, sigma_Ro) -> WellSet

Формат организаторов: лист `Wells_Interp` (или первый), пары колонок
`"<W> (NNN m) Depth [m]"` / `"<W> (NNN m) Value"`. kind — :T или :Ro,
определяет sigma по умолчанию (файл организаторов sigma не содержит).
"""
function load_wells_xlsx(path::AbstractString, data::BasinDataset;
                         kind::Symbol, sigma_T::Float64 = 2.0, sigma_Ro::Float64 = 0.05)
    kind in (:T, :Ro) || error("kind must be :T or :Ro")
    sigma = kind === :T ? sigma_T : sigma_Ro
    xf = XLSX.readxlsx(path)
    sheet_name = "Wells_Interp" in XLSX.sheetnames(xf) ? "Wells_Interp" : first(XLSX.sheetnames(xf))
    sheet = xf[sheet_name]
    header = [string(c) for c in vec(sheet[1, 1:end])]
    depth_cols = findall(h -> occursin("Depth", h), header)
    raw = WellPoint[]
    for dc in depth_cols
        head = header[dc]                       # "A (400 m) Depth [m]"
        wname = strip(split(head, " (")[1])
        x = parse(Float64, strip(split(split(head, " (")[2], " m")[1]))
        vc = findall(h -> h != head && occursin("Value", h) &&
                        startswith(h, split(head, " (")[1]), header)
        isempty(vc) && error("No Value column for \"$head\" in $path")
        for r in 2:size(XLSX.getdata(sheet), 1)
            depth = sheet[r, dc]; value = sheet[r, vc[1]]
            (depth isa Real && value isa Real) || continue
            z = Float64(depth)                  # глубина положительна вниз
            push!(raw, WellPoint(wname, x, z, kind, Float64(value), sigma))
        end
    end
    return build_wellset(raw, data)
end

# --- JSON (twin) ---------------------------------------------------------------

"""
    load_wells_json(path, data) -> WellSet

Формат make_twin.jl: список скважин `[{"x":…, "z":[…], "T_obs":[…], "Ro_obs":[…],
"sigma_T":[…]|Float, "sigma_Ro":[…]|Float}, …]` (Ro_obs может отсутствовать).
"""
function load_wells_json(path::AbstractString, data::BasinDataset)
    doc = JSON.parsefile(path)
    doc isa AbstractVector || (doc = doc["wells"])
    raw = WellPoint[]
    for (iw, w) in enumerate(doc)
        x = Float64(w["x"])
        zs = Float64.(w["z"])
        wname = get(w, "well", "W$iw")
        if haskey(w, "T_obs")
            st = w["sigma_T"]
            st isa AbstractVector || (st = fill(Float64(st), length(zs)))
            for (k, z) in enumerate(zs)
                v = Float64(w["T_obs"][k])
                isfinite(v) || continue
                push!(raw, WellPoint(wname, x, z, :T, v, Float64(st[k])))
            end
        end
        if haskey(w, "Ro_obs")
            sr = w["sigma_Ro"]
            sr isa AbstractVector || (sr = fill(Float64(sr), length(zs)))
            for (k, z) in enumerate(zs)
                v = Float64(w["Ro_obs"][k])
                isfinite(v) || continue
                push!(raw, WellPoint(wname, x, z, :Ro, v, Float64(sr[k])))
            end
        end
    end
    return build_wellset(raw, data)
end

"""Сводка WellSet для solution.json / отчёта."""
function wellset_summary(ws::WellSet)
    per = Dict{String,Dict{String,Any}}()
    for wname in sort(unique(p.well for p in ws.points))
        pts = filter(p -> p.well == wname, ws.points)
        per[wname] = Dict(
            "n_T" => count(p -> p.kind === :T, pts),
            "n_Ro" => count(p -> p.kind === :Ro, pts),
            "x_m" => first(pts).x)
    end
    drop = Dict{String,Int}()
    for p in ws.excluded
        drop[p.well] = get(drop, p.well, 0) + 1
    end
    return Dict("wells" => per,
                "excluded_points_by_well" => drop,
                "excluded_wells" => ws.excluded_wells,
                "n_T_total" => n_points(ws, :T),
                "n_Ro_total" => n_points(ws, :Ro))
end
