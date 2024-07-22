using ArgParse
using LinearAlgebra

using SCAMP

const dω = 0.01
const Ω = 30.0

struct CorrelatorProgram
    β::Float64
    τ::Vector{Float64}
    C::Vector{Float64}
    M::Matrix{Float64}
    t::Float64
    σ::Float64
end

struct PrimalCorrelatorProgram
    cp::CorrelatorProgram
end

function CorrelatorProgram(β::Float64, τ::Vector{Float64}, C::Vector{Float64}, Σ::Matrix{Float64}, t::Float64, σ::Float64; p::Float64=0.01)
    M = inv(Σ)
    # TODO
end

function primal(p::CorrelatorProgram)::PrimalCorrelatorProgram
    return PrimalCorrelatorProgram(p)
end

function initial(p::PrimalCorrelatorProgram)::Vector{Float64}
    ωs = 0:dω:Ω
    return rand(length(ωs))
end

function objective!(g::Vector{Float64}, p::PrimalCorrelatorProgram, ρ::Vector{Float64})::Vector{Float64}
    ωs = 0:dω:Ω
    # TODO
end

function badness!(g::Vector{Float64}, p::PrimalCorrelatorProgram, ρ::Vector{Float64})::Float64
    ωs = 0:dω:Ω
    r = 0.
    # Positivity
    for (ω, ρω) in zip(ωs,ρ)
        # TODO
    end
    # Measurements
    # TODO
    return r
end

function barrier!(g::Vector{Float64}, p::PrimalCorrelatorProgram, ρ::Vector{Float64})::Float64
    ωs = 0:dω:Ω
    for (ω, ρω) in zip(ωs,ρ)
        # TODO
    end
    # Measurements
    # TODO
    return r
end

function initial(p::CorrelatorProgram)::Vector{Float64}
    return rand(length(p.τ)+1)
end

function objective!(g::Vector{Float64}, p::CorrelatorProgram, y::Vector{Float64})::Float64
    # Unpack
    μ, ℓ = y[1], y[2:end]
    # TODO
end

function λ!(gℓ::Vector{Float64}, p::CorrelatorProgram, ℓ::Vector{Float64}, ω::Float64)::Float64
    # TODO
    return 0.0
end

function badness!(g::Vector{Float64}, p::CorrelatorProgram, y::Vector{Float64})::Float64
    # Unpack
    μ, ℓ = y[1], y[2:end]
    g .= 0
    r::Float64 = 0.
    if μ < 0
        r -= μ
        g[1] = -1.
    end
    # Integrate the negative bits.
    for ω in 0:dω:Ω
        # TODO
    end
    return r
end

function barrier!(g::Vector{Float64}, p::CorrelatorProgram, y::Vector{Float64})::Float64
    # Unpack
    μ, ℓ = y[1], y[2:end]
    # Integrate the logarithm.
    if μ ≤ 0
        return Inf
    end
    g .= 0
    r::Float64 = -log(μ)
    g[1] = 0. # TODO
    for ω in 0:dω:Ω
        # TODO
    end
    return r
end

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "correlator"
                required = true
                arg_type = String
        end
        parse_args(s)
    end
    cor, cov = let
        lines = readlines(args["correlator"])
    end
    
    # TODO construct the program

    solve(p)
end

main()
