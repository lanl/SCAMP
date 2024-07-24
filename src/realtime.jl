using ArgParse
using LinearAlgebra
using Random: rand, randn, rand!, randn!
using Statistics: mean

using SCAMP
import SCAMP: initial, badness!, barrier!, objective!

#const dω = 0.02
#const Ω = 20.0
const dω = 0.1
const Ω = 8.0
const ωs = dω:dω:Ω

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
    return rand(length(ωs))
end

function objective!(g, p::PrimalCorrelatorProgram, ρ::Vector{Float64})::Float64
    r = 0.
    for (i,ω) in enumerate(ωs)
        coef = -2 * sin(ω*p.cp.t) * exp(-ω^2 * p.cp.σ^2 / 2) * dω
        if !isnothing(g)
            g[i] = coef
        end
        r += ρ[i] * coef
    end
    return r
end

function badness!(g, p::PrimalCorrelatorProgram, ρ::Vector{Float64})::Float64
    β = length(p.cp.C)
    r = 0.
    if !isnothing(g)
        g .= 0
    end
    # Positivity
    for (k, (ω, ρω)) in enumerate(zip(ωs,ρ))
        if ρω < 0
            if !isnothing(g)
                g[k] += -1.
            end
            r += -ρω
        end
    end
    # Measurements
    cor = zeros(β)
    dcor = zeros((β,length(ωs)))
    for (i,τ) in enumerate(p.cp.τ)
        for (k, (ω, ρω)) in enumerate(zip(ωs,ρ))
            #dcor[i,k] = cosh(ω * (p.cp.β/2 - τ)) / sinh(p.cp.β*ω/2)
            dcor[i,k] = (exp(-ω*τ) + exp(-ω * (p.cp.β -τ))) / (1 - exp(-ω*p.cp.β)) * dω
            cor[i] += dcor[i,k] * ρω
        end
    end
    v = cor - p.cp.C
    err = v' * p.cp.M * v
    if err > 1
        r += err-1
        if !isnothing(g)
            for (k, ω) in enumerate(ωs)
                for i in 1:β, j in 1:β
                    g[k] += 2 * v'[i] * p.cp.M[i,j] * dcor[j,k]
                end
            end
        end
    end
    return r
end

function barrier!(g, p::PrimalCorrelatorProgram, ρ::Vector{Float64})::Float64
    β = length(p.cp.C)
    if !isnothing(g)
        g .= 0
    end
    r = 0.
    # Positivity
    for (k, (ω, ρω)) in enumerate(zip(ωs,ρ))
        if ρω < 0
            return Inf
        end
        if !isnothing(g)
            g[k] += -1/ρω
        end
        r += -log(ρω)
    end
    # Measurements
    cor = zeros(β)
    dcor = zeros((β,length(ωs)))
    for (i,(τ,C)) in enumerate(zip(p.cp.τ,p.cp.C))
        for (k, (ω, ρω)) in enumerate(zip(ωs,ρ))
            #dcor[i,k] = cosh(ω * (p.cp.β/2 - τ)) / sinh(p.cp.β*ω/2)
            dcor[i,k] = (exp(-ω*τ) + exp(-ω * (p.cp.β -τ))) / (1 - exp(-ω*p.cp.β)) * dω
            cor[i] += dcor[i,k] * ρω
        end
    end
    v = cor - p.cp.C
    err = v' * p.cp.M * v
    if err > 1
        return Inf
    end
    r += -log(1-err)
    if !isnothing(g)
        for (k, ω) in enumerate(ωs)
            g[k] += -(2 * v' * p.cp.M * dcor[:,k])/(1-err)
        end
    end
    return r
end

function initial(p::CorrelatorProgram)::Vector{Float64}
    return rand(length(p.τ)+1)
end

function objective!(g, p::CorrelatorProgram, y::Vector{Float64})::Float64
    # Unpack
    μ, ℓ = y[1], y[2:end]
    # TODO
end

function λ!(gℓ::Vector{Float64}, p::CorrelatorProgram, ℓ::Vector{Float64}, ω::Float64)::Float64
    # TODO
    return 0.0
end

function badness!(g, p::CorrelatorProgram, y::Vector{Float64})::Float64
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

function barrier!(g, p::CorrelatorProgram, y::Vector{Float64})::Float64
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
   
    if false
        # Check derivatives of badness!
        ϵ = 1e-5
        t = 0.0
        σ = 1.0
        p = primal(CorrelatorProgram(cors, t, σ))

        ρ = initial(p)
        randn!(ρ)
        g = zero(ρ)
        g′ = zero(ρ)
        bad = badness!(g, p, ρ)
        for n in 1:length(ρ)
            ρ′ = copy(ρ)
            ρ′[n] += ϵ
            bad′ = badness!(g′, p, ρ′)
            println((bad′ - bad)/ϵ - g[n])
        end
        return
    end

    # Solve
    σ = 1.0
    for t in 0:.1:10
        p = CorrelatorProgram(cors, t, σ)
        lo = solve(primal(p); verbose=true)[1]
        println("$t  $lo")
    end
end

main()
