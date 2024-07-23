using ArgParse
using LinearAlgebra

using SCAMP
import SCAMP: initial, badness!, barrier!, objective!

const dω = 0.01
const Ω = 30.0

struct CorrelatorProgram <: ConvexProgram
    β::Float64
    τ::Vector{Float64}
    C::Vector{Float64}
    M::Matrix{Float64}
    t::Float64
    σ::Float64

    function CorrelatorProgram(β::Float64, τ::Vector{Float64}, C::Vector{Float64}, Σ::Matrix{Float64}, t::Float64, σ::Float64; p::Float64=0.01)::CorrelatorProgram
        M = inv(Σ)
        # TODO regularize and use `p`
        new(β, τ, C, M, t, σ)
    end
end

struct PrimalCorrelatorProgram <: ConvexProgram
    cp::CorrelatorProgram
end

function primal(p::CorrelatorProgram)::PrimalCorrelatorProgram
    return PrimalCorrelatorProgram(p)
end

function initial(p::PrimalCorrelatorProgram)::Vector{Float64}
    ωs = 0:dω:Ω
    return rand(length(ωs))
end

function objective!(g::Vector{Float64}, p::PrimalCorrelatorProgram, ρ::Vector{Float64})::Float64
    ωs = 0:dω:Ω
    r = 0.
    for (i,ω) in enumerate(ωs)
        g[i] = -2 * sin(ω*p.cp.t) * sinh(p.cp.β*ω/2) * exp(-ω^2 * p.cp.σ^2 / 2)
        r += ρ[i] * g[i]
    end
    return r
end

function badness!(g::Vector{Float64}, p::PrimalCorrelatorProgram, ρ::Vector{Float64})::Float64
    ωs = 0:dω:Ω
    r = 0.
    g .= 0
    # Positivity
    for (k, (ω, ρω)) in enumerate(zip(ωs,ρ))
        if ρω < 0
            g[k] += -1.
            r += -ρω
        end
    end
    # Measurements
    for (i,(τ,C)) in enumerate(zip(p.cp.τ,p.cp.C))
        cor = 0.
        dcor = zero(g)
        for (k, (ω, ρω)) in enumerate(zip(ωs,ρ))
            dcor[k] = cosh(ω * (p.cp.β/2 - τ))
            cor += dcor[k] * ρω
        end
        # TODO
    end
    return r
end

function barrier!(g::Vector{Float64}, p::PrimalCorrelatorProgram, ρ::Vector{Float64})::Float64
    ωs = 0:dω:Ω
    g .= 0
    r = 0.
    # Positivity
    for (k, (ω, ρω)) in enumerate(zip(ωs,ρ))
        if ρω < 0
            return Inf
        end
        g[k] += -1/ρω
        r += -log(ρω)
    end
    # Measurements
    for (i,(τ,C)) in enumerate(zip(p.cp.τ,p.cp.C))
        cor = 0.
        dcor = zero(g)
        for (k, (ω, ρω)) in enumerate(zip(ωs,ρ))
            dcor[k] = cosh(ω * (p.cp.β/2 - τ))
            cor += dcor[k] * ρω
        end
        # TODO
    end
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
        map(s -> Meta.parse(s) |> eval, readlines(args["correlator"]))
    end
    β = length(cor)
    τ = collect(1.0:(β÷2))
    C = cor[1:(β÷2)]
    Σ = cov[1:(β÷2),1:(β÷2)]
   
    t = 0.0
    σ = 1.0
    p = CorrelatorProgram(Float64(β), τ, C, Σ, t, σ)
    solve(primal(p))
end

main()
