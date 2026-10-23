using Test

isdefined(Main, :AdaptiveConfig) || include(joinpath(@__DIR__, "adaptive_basin_fvm.jl"))

isdefined(Main, :parse_case) || include(joinpath(@__DIR__, "run_adaptive_grid.jl"))

@testset "AdaptiveMeshTests" begin
    @testset "coarse/fine face lengths are conservative" begin
        cells = vcat([Cell(0, 4, 0, 4, 0)], [Cell(4, 5, j, j + 1, 2) for j in 0:3])
        pairs = face_pairs(cells, 1.0, 1.0, (0.0, 8.0, 0.0, 4.0))
        touching = pairs.p .== 1            # ячейка 0 в Python -> 1 в Julia
        @test isapprox(sum(pairs.len[touching]), 4.0; atol = 1.0e-7)
    end

    @testset "balance limits neighbour level jump" begin
        cells = vcat([Cell(0, 4, 0, 4, 0)], [Cell(4, 5, j, j + 1, 2) for j in 0:3])
        balanced = balance_cells(cells, 2)
        for (a, b, _, _, _) in integer_face_pairs(balanced)
            @test abs(balanced[a].level - balanced[b].level) <= 1
        end
    end

    @testset "constant temperature reports Picard convergence" begin
        config = AdaptiveConfig(base_nx = 2, base_nz = 2, max_level = 0)
        cells = [Cell(i, i + 1, j, j + 1, 0) for j in 0:1 for i in 0:1]
        mesh = AdaptiveMesh(cells, (0.0, 2.0, 0.0, 2.0), config)
        temperature = fill(20.0, 4)
        grid = CellGrid(trues(4), ones(Int, 4), zeros(Int, 4), zeros(4), ["layer"])
        state = State(0.0, Dict{String,Layer}(), zeros(4), fill(2.0, 4),
                      fill(20.0, 4), zeros(4))
        properties = Dict{Int,LithProp}(0 => LithProp("test", 2500.0, 2.5, 900.0, 0.0))

        result, diagnostics = adaptive_step(
            temperature, mesh, state, grid, 1.0e6, properties, config)

        @test isapprox(result, temperature; rtol = 1.0e-7)
        @test diagnostics["nonlinear_converged"]
        @test diagnostics["picard_iterations"] == 1
    end

    @testset "invalid solver controls are rejected" begin
        @test_throws ArgumentError validate(AdaptiveConfig(picard_iterations = 0))
        @test_throws ArgumentError validate(AdaptiveConfig(picard_tolerance_c = 0.0))
        @test_throws ArgumentError validate(AdaptiveConfig(linear_tolerance = 0.0))
    end

    if isdefined(Main, :parse_case)
        @testset "adaptive grid case parser" begin
            @test parse_case("8, 16, 3") == (8, 16, 3)
            @test_throws ArgumentError parse_case("8,16,6")
        end
    else
        @info "parse_case не определена (нужен перевод run_adaptive_grid.py) — тест пропущен"
    end
end
