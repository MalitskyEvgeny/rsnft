# Юнит-тесты observations.jl (A2): формат Wells_Interp организаторов,
# исключение внегеометричных точек, JSON-загрузчик, layer_eta_at.
# Запуск: julia --project=. test_observations.jl

include(joinpath(@__DIR__, "parameterized_data.jl"))
include(joinpath(@__DIR__, "observations.jl"))

using Printf

failures = 0
function check(name, ok)
    global failures
    println((ok ? "PASS " : "FAIL "), name)
    ok || (failures += 1)
end

const DATA = normpath(joinpath(@__DIR__, "..", "Данные для обратной задачи"))
d = load_dataset_constant(DATA, 75.0)

# --- 1. XLSX TemperatureWellData ---
wsT = load_wells_xlsx(joinpath(DATA, "TemperatureWellData.xlsx"), d; kind = :T)
check("T: 5 wells", n_wells(wsT) == 5)
check("T: ~499 points (100 per well, 1 excluded)", n_points(wsT, :T) == 499)
check("T: 1 point excluded (well A)", length(wsT.excluded) == 1 && wsT.excluded[1].well == "A")
check("T: no fully-excluded wells", isempty(wsT.excluded_wells))
check("T: all sigma > 0", all(p -> p.sigma > 0, wsT.points))
check("T: well A has 99 points inside", count(p -> p.well == "A", wsT.points) == 99)

# --- 2. XLSX VitriniteWellData ---
wsR = load_wells_xlsx(joinpath(DATA, "VitriniteWellData.xlsx"), d; kind = :Ro)
check("Ro: 5 wells", n_wells(wsR) == 5)
check("Ro: 49 points (10 per well, 1 excluded)", n_points(wsR, :Ro) == 49)
check("Ro: default sigma_Ro = 0.05", all(p -> p.sigma == 0.05, wsR.points))
check("T: default sigma_T = 2.0", all(p -> p.sigma == 2.0, wsT.points))

# --- 3. layer_eta_at: согласованность с геометрией ---
ok_layers = true
for p in first(wsT.points, 20)
    le = layer_eta_at(d, p.x, p.z)
    le === nothing && (ok_layers = false; break)
    ser = d.depth[le[1]]
    rec = ser[0.0]
    j = clamp(searchsortedlast(rec.x, p.x), 1, length(rec.x) - 1)
    t = (p.x - rec.x[j]) / max(rec.x[j+1] - rec.x[j], 1e-9)
    top = rec.top[j] + t * (rec.top[j+1] - rec.top[j])
    bot = rec.bot[j] + t * (rec.bot[j+1] - rec.bot[j])
    (p.z < top || p.z > bot || !(0.0 <= le[2] <= 1.0)) && (ok_layers = false; break)
end
check("layer_eta_at: point inside its layer, eta in [0,1]", ok_layers)
check("layer_eta_at: outside point -> nothing", layer_eta_at(d, -1000.0, 2000.0) === nothing)

# --- 4. Валидация sigma ---
raw_bad = [WellPoint("W", 10000.0, 2000.0, :T, 50.0, 0.0)]
try
    build_wellset(raw_bad, d)
    check("sigma=0 raises", false)
catch
    check("sigma=0 raises", true)
end

# --- 5. JSON-загрузчик (формат make_twin) ---
json_path = joinpath(tempdir(), "a2_test_wells.json")
open(json_path, "w") do f
    JSON.print(f, [
        Dict("well" => "WT", "x" => 10000.0, "z" => [1500.0, 2500.0, 3500.0],
             "T_obs" => [40.0, 80.0, 130.0], "sigma_T" => 2.0,
             "Ro_obs" => [0.3, 0.5, 0.7], "sigma_Ro" => 0.05),
        Dict("well" => "WOFF", "x" => -5000.0, "z" => [2000.0, 3000.0],
             "T_obs" => [50.0, 90.0], "sigma_T" => 2.0),
    ], 2)
end
wsJ = load_wells_json(json_path, d)
check("JSON: T points loaded", n_points(wsJ, :T) == 3)
check("JSON: Ro points loaded", n_points(wsJ, :Ro) == 3)
check("JSON: off-geometry well excluded entirely", wsJ.excluded_wells == ["WOFF"])
check("JSON: scalar sigma broadcast", all(p -> p.sigma == 2.0, filter(p -> p.kind === :T, wsJ.points)))

println()
if failures == 0
    println("ALL PASS")
else
    println(@printf("%d FAILURES", failures))
    exit(1)
end
