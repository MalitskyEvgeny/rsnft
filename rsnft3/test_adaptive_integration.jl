# Интеграционные тесты адаптивного бэкенда обратной задачи:
# 1) level-0 адаптивная сетка (base = регулярной) vs run_forward — сходимость
#    дискретизаций (не бит-идентичность: face_pairs против регулярного стенсила);
# 2) smoke make_objective(backend=:adaptive): конечный J, повторный вызов из кэша.
#
# Запуск: julia --project=. test_adaptive_integration.jl --data-dir <dir>

using Test
using Printf

include(joinpath(@__DIR__, "objective.jl"))   # тянет basin_fvm + adaptive_basin_fvm

data_dir = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "Данные для обратной задачи")

data = load_dataset(data_dir, joinpath(@__DIR__, "..", "results", "_test_ageprobe"))
wells_T = load_wells_xlsx(joinpath(data_dir, "TemperatureWellData.xlsx"), data; kind = :T, sigma_T = 2.0)
wells_Ro = load_wells_xlsx(joinpath(data_dir, "VitriniteWellData.xlsx"), data; kind = :Ro, sigma_Ro = 0.05)
wells = WellSet(vcat(wells_T.points, wells_Ro.points), vcat(wells_T.excluded, wells_Ro.excluded),
                union(wells_T.excluded_wells, wells_Ro.excluded_wells))

epochs = [(200.0, 100.0), (100.0, 35.0), (35.0, 10.0), (10.0, 0.0)]
nx, nz = 20, 40
layout = make_q_layout(data; epochs = epochs, n_nodes = 2, mid_x = 20000.0)
θ = Float64[]
for ie in 1:length(epochs), _ in 1:2
    push!(θ, 60.0 + 10.0 * ie)
end

dt_ma = 5.0
cfg = SolverConfig(nx = nx, nz = nz, max_dt_ma = dt_ma, picard_iterations = 3,
                   picard_tolerance_c = 1.0e-3, pcg_tolerance = 1.0e-6)
acfg = AdaptiveConfig(base_nx = nx, base_nz = nz, max_level = 0, max_dt_ma = dt_ma,
                      picard_iterations = 3, picard_tolerance_c = 1.0e-3,
                      linear_tolerance = 1.0e-6, preview_nx = nx, preview_nz = nz)

@testset "AdaptiveIntegration" begin
    @testset "level-0 vs regular grid convergence" begin
        geo = precompute_geometry(data, cfg)
        qba = qbasal_by_age(layout, θ, geo.x, data.ages)
        rreg = run_forward(data, cfg; geometry = geo, qbasal_by_age = qba)

        ageo = precompute_adaptive_geometry(data, acfg)
        ax0, ax1 = data.bounds[1], data.bounds[2]
        adx = (ax1 - ax0) / acfg.preview_nx
        axg = collect(range(ax0 + 0.5 * adx, ax1 - 0.5 * adx; length = acfg.preview_nx))
        aqba = qbasal_by_age(layout, θ, axg, data.ages)
        radp = run_adaptive(data, acfg; qbasal_by_age = aqba,
                            qbasal_nodes_x = layout.nodes_x, geometry = ageo)

        Treg = rreg.temperature_c[end, :, :]
        Tadp = radp.temperature_c[end, :, :]
        @test size(Treg) == size(Tadp)
        mask = isfinite.(Treg) .& isfinite.(Tadp)
        @test count(mask) > 0
        diff = Treg[mask] .- Tadp[mask]
        rms = sqrt(sum(abs2, diff) / length(diff))
        scale = maximum(abs.(Treg[mask]))
        @printf("level-0 vs regular: RMS diff = %.4f C (scale %.1f C, rel %.2e), max %.4f C\n",
                rms, scale, rms / scale, maximum(abs.(diff)))
        # не бит-идентичность: разные дискретизации граней; требуем < 1% от масштаба
        @test rms / scale < 0.01
    end

    @testset "objective adaptive backend smoke" begin
        acfg2 = AdaptiveConfig(base_nx = 8, base_nz = 16, max_level = 1, max_dt_ma = dt_ma,
                               picard_iterations = 3, picard_tolerance_c = 1.0e-3,
                               linear_tolerance = 1.0e-6, preview_nx = 60, preview_nz = 80)
        obj = make_objective(data, wells, cfg, layout; reg = 0.0,
                             backend = :adaptive, adaptive_cfg = acfg2)
        θt = repeat([75.0, 85.0], length(epochs))
        J1, i1 = obj(θt)
        @test isfinite(J1) && J1 > 0
        J2, i2 = obj(θt)
        @test i2.cached && J2 == J1
        @printf("adaptive J=%.2f (J_T=%.2f J_Ro=%.2f), cache OK\n", J1, i1.J_T, i1.J_Ro)
    end
end
println("ALL PASS")
