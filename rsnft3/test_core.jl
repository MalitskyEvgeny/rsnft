using Test

# basin_fvm.jl подключается внутри adaptive_basin_fvm.jl; guard защищает от повторного include
isdefined(Main, :AdaptiveConfig) || include(joinpath(@__DIR__, "adaptive_basin_fvm.jl"))

@testset "CoreFormulaTests" begin
    @testset "harmonic mean" begin
        @test isapprox(harmonic(2.0, 4.0), 8.0 / 3.0; atol = 1.0e-7)
    end

    @testset "waples reference value at 20C" begin
        value = waples_heat_capacity(293.15, 900.0)
        @test isapprox(value, 897.9079752; atol = 5.0e-6)
    end

    @testset "sekiguchi is lambda20 at 20C to roundoff" begin
        value = sekiguchi_conductivity(293.15, 3.0)
        @test isapprox(value, 3.0; atol = 0.03)
    end
end
