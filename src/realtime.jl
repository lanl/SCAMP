using ArgParse
using LinearAlgebra
using Statistics: mean

using SCAMP
import SCAMP: initial, badness!, barrier!, objective!

const dω = 0.01
const Ω = 30.0

function resample(f, x; K=1000)::Vector{Float64}
    fs = zeros(K)
    for k in 1:K
        fs[k] = f(rand(x,K))
    end
    return fs
end

struct CorrelatorProgram <: ConvexProgram
    β::Float64
    τ::Vector{Float64}
    C::Vector{Float64}
    M::Matrix{Float64}
    t::Float64
    σ::Float64

    function CorrelatorProgram(Cs, t::Float64, σ::Float64; p::Float64=0.01)::CorrelatorProgram
        C = mean(Cs)
        β = length(C)
        τ = collect(1:Float64(β))
        K = 1000
        M = zeros((β,β))
        for n in 1:β
            M[n,n] = 1.0
        end
        xs = resample(Cs) do Cs
            C′ = mean(Cs)
            v = C′ - C
            return v' * M * v
        end
        sort!(xs)
        x = xs[round(Int,(1-p)*K)]
        M ./= x
        new(Float64(β), τ, C, M, t, σ)
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
    cors = let
        open(args["correlator"]) do f
            cors = Vector{Vector{Float64}}()
            for l in readlines(f)
                v = eval(Meta.parse(l))
                if !isnothing(v)
                    push!(cors, v)
                end
            end
            cors
        end
    end
   
    t = 0.0
    σ = 1.0
    p = CorrelatorProgram(cors, t, σ)
    println(solve(primal(p)))
end

main()
