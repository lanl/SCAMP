# Extract a smeared spectral function.

using ArgParse
using LinearAlgebra
using Profile
using Random: rand, randn, rand!, randn!
using SpecialFunctions: erf
using Statistics: mean, cov

import Base: size

using SCAMP
using SCAMP.Utilities: check_gradients
import SCAMP: initial, constraints!, objective!

#const dω = 0.0001
const dω = 0.001
const Ω = 1.5

function resample(f, x; K=1000)::Vector{Float64}
    N = length(x)
    fs = zeros(K)
    for k in 1:K
        fs[k] = f(rand(x,N))
    end
    return fs
end

struct SpectralProgram <: ConvexProgram
    β::Float64
    τ::Vector{Float64}
    C::Vector{Float64}
    M::Matrix{Float64}
    Minv::Matrix{Float64}
    ω::Float64
    σ::Float64
    sgn::Float64

    function SpectralProgram(β::Float64, τ::Vector{Float64}, Cs, ω::Float64, σ::Float64, sgn::Float64, p::Float64=0.01)::SpectralProgram
        N = length(Cs)
        C = mean(Cs)
        B = length(τ)
        @assert B == length(C)

        # First extract the covariance matrix.
        Σ = let
            K = 1000
            cors = zeros((K,B))
            for k in 1:K
                cors[k,:] = mean(rand(Cs,N))
            end
            Σ = cov(cors)
            # Regulate
            maxeig = maximum(eigvals(Σ))
            Σ += 1e-6 * I
            Σ
        end

        # Define (1-p)% confidence region.
        M, Minv = let
            K = 1000
            M = inv(Σ)
            xs = resample(Cs; K=K) do Cs
                C′ = mean(Cs)
                v = C′ - C
                return v' * M * v
            end
            sort!(xs)
            x = xs[round(Int,(1-p)*K)]
            M ./= x
            Minv = inv(M)
            M, Minv
        end

        new(Float64(β), τ, C, M, Minv, ω, σ, sgn)
    end

    function SpectralProgram(p::SpectralProgram; ω::Float64, sgn::Float64)::SpectralProgram
        new(p.β, p.τ, p.C, p.M, p.Minv, ω, p.σ, sgn)
    end
end

function initial(p::SpectralProgram)::Vector{Float64}
    return rand(length(p.τ)+1)
end

function size(p::SpectralProgram)::Int
    return length(p.τ)+1
end

function objective!(g, h, p::SpectralProgram, y::Vector{Float64})::Float64
    # Unpack
    μ, ℓ = y[1], @view(y[2:end])
    if !isnothing(g)
        g .= 0.0
    end
    if !isnothing(h)
        h .= 0.0
    end
    r = 0.0

    # Constant (in ℓ) piece.
    if !isnothing(g)
        g[1] = -1.0
    end
    r += -μ

    # Inversion piece, and gradients
    ℓMinvℓ = (ℓ' * p.Minv * ℓ)
    r += -1 * ℓMinvℓ / (4 * μ)
    gℓ = -2 * p.Minv * ℓ / (4*μ)
    if !isnothing(g)
        g[1] += ℓMinvℓ / (4 * μ^2)
        g[2:end] .= gℓ
    end
    # Hessian of inversion piece
    if !isnothing(h)
        h[2:end,2:end] += -2*p.Minv / (4*μ)
        h[1,1] += -2 * ℓMinvℓ / (4 * μ^3)
        h[1,2:end] += 2 * p.Minv * ℓ / (4*μ^2)
        h[2:end,1] .= h[1,2:end]
    end

    # Inner product, and gradients (no Hessian)
    if !isnothing(g)
        g[2:end] .+= p.C
    end
    r += p.C' * ℓ

    # We're maximizing, not minimizing.
    r *= -1
    if !isnothing(g)
        g .*= -1
    end
    if !isnothing(h)
        h .*= -1
    end
    return r
end

function constraints!(cb, p::SpectralProgram, y::Vector{Float64})
    dλ = zero(y)
    for ω in dω:dω:Ω
        λ = λ!(dλ, p, y, ω)
        dλ .*= dω
        cb(dω*λ, dλ, 0)
    end
    dμ = zero(y)
    dμ[1] = 1.0
    μ = y[1]
    cb(μ, dμ, 0)
end

function Φ(x::Float64)
    return (1 + erf(x/sqrt(2)))/2
end

function λ!(g::Vector{Float64}, p::SpectralProgram, y::Vector{Float64}, ω::Float64)::Float64
    g .= 0.0
    μ = y[1]

    # (\mathcal K) term. No gradient.
    #r = p.sgn * -1 * sin(ω * p.t) * exp(-(p.σ^2 * ω^2)/2)
    r = p.sgn * exp(-(ω - p.ω)^2 / (2 * p.σ^2)) / (Φ(p.ω/p.σ) * sqrt(2*π)*p.σ)

    # -K^T ℓ term, with gradient
    for (i,τ) in enumerate(p.τ)
        ℓ = y[1+i]
        #gℓ = cosh(ω * (p.β/2 - τ)) / sinh(p.β * ω / 2)
        gℓ = -(exp(-ω*τ) + exp(-ω * (p.β -τ))) / (1 - exp(-ω*p.β))
        g[1+i] = gℓ
        r += ℓ * gℓ
    end
    return r
end

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "--omega"
                required = true
                arg_type = Float64
            "--sigma"
                required = true
                default = 1.0
                arg_type = Float64
            "--beta"
                required = false
                arg_type = Int
            "--skip"
                required = false
                default = 1
                arg_type = Int
            "--format"
                required = false
                default = :lists
                arg_type = Symbol
                help = "Input format"
            "--scale"
                required = false
                default = 1.0
                arg_type = Float64
            "--min-tau"
                required = false
                default = 0
                arg_type = Float64
            "--profile"
                action = :store_true
            "correlator"
                required = true
                arg_type = String
        end
        parse_args(s)
    end
    τs, cors = if args["format"] == :lists
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
        # Skip
        cors = cors[1:args["skip"]:end]
        τ = collect(1:length(cors[1]))
        τ, cors
    elseif args["format"] == :long
        open(args["correlator"]) do f
            Nstr, Kstr = split(readline(f))
            N, K = parse(Int, Nstr), parse(Int, Kstr)
            τ = zeros(Float64, K)
            cors = Vector{Vector{Float64}}()
            for n in 1:N
                c = zeros(Float64, K)
                for k in 1:K
                    l = readline(f)
                    τstr, rstr, istr = split(l)
                    τ[k] = parse(Int, τstr)
                    cor_r = parse(Float64, rstr)
                    cor_i = parse(Float64, istr)
                    c[k] = cor_r * args["scale"]
                end
                push!(cors, c)
            end
            τ, cors
        end
    end

    N = length(cors)
    # Apply minimum value of τ
    K, τs, cors = let
        K′::Int = sum(τs .>= args["min-tau"])
        τs′ = zeros(Float64, K′)
        cors′ = Vector{Vector{Float64}}()

        k′ = 1
        for k in 1:K
            if τs[k] ≥ args["min-tau"]
                τs′[k′] = τs[k]
                k′ += 1
            end
        end

        for n in 1:N
            k′ = 1
            c = zeros(Float64, K′)
            for k in 1:K
                if τs[k] ≥ args["min-tau"]
                    c[k′] = cors[n][k]
                    k′ += 1
                end
            end
            push!(cors′, c)
        end
        K′, τs′, cors′
    end

    σ = args["sigma"]
    if !isnothing(args["beta"])
        β = Float64(args["beta"])
    else
        β = Float64(length(τs))
    end

    if false
        p = SpectralProgram(β, τs, cors, 0.3, σ, 1.0)
        # Check gradients and Hessians
        y = SCAMP.IPM.feasible_initial(p)
        g = zero(y)
        h = zeros(Float64, (length(y),length(y)))
        # Objective gradients.
        SCAMP.IPM.objective!(g, h, p, y)
        @assert check_gradients(y, g, h; verbose=true) do y
            SCAMP.IPM.objective!(nothing, nothing, p, y)
        end
        # Barrier gradients.
        SCAMP.IPM.barrier!(g, h, p, y)
        @assert check_gradients(y, g, h; verbose=true) do y
            SCAMP.IPM.barrier!(nothing, nothing, p, y)
        end
        return
    end

    if args["profile"]
        plo = SpectralProgram(β, τs, cors, 0.3, σ, 1.0)
        @profile solve(plo; verbose=false)
        open("prof-flat", "w") do f
            Profile.print(f, format=:flat, sortedby=:count)
        end
        open("prof-tree", "w") do f
            Profile.print(f, noisefloor=2.0)
        end
        return
    end

    p = SpectralProgram(β, τs, cors, 0.0, σ, 1.0)
    for ω in LinRange(0,args["omega"],101)
        plo = SpectralProgram(p, ω=ω, sgn=1.0)
        phi = SpectralProgram(p, ω=ω, sgn=-1.0)
        lo, ylo = solve(plo; verbose=false)
        hi, yhi = solve(phi; verbose=false)
        println("$ω $(-lo) $hi")
    end
    return
end

main()

