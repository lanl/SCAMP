# Extract a susceptibility

using ArgParse
using LinearAlgebra
using Random: rand, randn, rand!, randn!
using Statistics: mean, cov

using SCAMP
import SCAMP: initial, constraints!, objective!

const dω = 0.001
const Ω = 2.0
const ωs = dω:dω:Ω

function resample(f, x; K=1000)::Vector{Float64}
    N = length(x)
    fs = zeros(K)
    for k in 1:K
        fs[k] = f(rand(x,N))
    end
    return fs
end

struct SusceptibilityProgram <: ConvexProgram
    β::Float64
    τ::Vector{Float64}
    C::Vector{Float64}
    M::Matrix{Float64}
    Minv::Matrix{Float64}
    sgn::Float64

    function SusceptibilityProgram(Cs, sgn::Float64; p::Float64=0.01)::SusceptibilityProgram
        N = length(Cs)
        C = mean(Cs)
        β = length(C)

        # First extract the covariance matrix.
        K = 1000
        cors = zeros((K,β))
        for k in 1:K
            cors[k,:] = mean(rand(Cs,N))
        end
        Σ = cov(cors)
        # Regulate.
        maxeig = maximum(eigvals(Σ))
        for i in 1:β
            Σ[i,i] += 1e-3 * maxeig
        end

        τ = collect(1:Float64(β))
        M = inv(Σ)
        xs = resample(Cs; K=K) do Cs
            C′ = mean(Cs)
            # TODO reconstruct M from this resampling (think about this)
            v = C′ - C
            return v' * M * v
        end
        sort!(xs)
        x = xs[round(Int,(1-p)*K)]
        M ./= x
        Minv = inv(M)
        new(Float64(β), τ, C, M, Minv, sgn)
    end
end

function initial(p::SusceptibilityProgram)::Vector{Float64}
    return rand(1+length(p.τ))
end

function objective!(g, p::SusceptibilityProgram, y::Vector{Float64})::Float64
    # Unpack
    μ, ℓ = y[1], @view(y[2:end])
    g .= 0.0
    r = 0.0

    # Constant (in ℓ) piece.
    g[1] = -1.0
    r += μ * g[1]

    # Inversion piece, and gradients
    ℓMinvℓ = (ℓ' * p.Minv * ℓ)
    r += -1 * ℓMinvℓ / (4 * μ)
    g[1] += ℓMinvℓ / (4 * μ^2)
    gℓ = -2 * p.Minv * ℓ / (4*μ)
    g[2:end] .= gℓ

    # Inner product, and gradients
    g[2:end] .+= p.C
    r += p.C' * ℓ

    # We're maximizing, not minimizing.
    r *= -1
    g .*= -1
    return r
end

function constraints!(cb, p::SusceptibilityProgram, y::Vector{Float64})
    dλ = zero(y)
    for ω in dω:dω:Ω
        λ = λ!(dλ, p, y, ω)
        dλ .*= dω
        cb(dω*λ, dλ)
    end
    dμ = zero(y)
    dμ[1] = 1.0
    μ = y[1]
    cb(μ, dμ)
end

function λ!(g::Vector{Float64}, p::SusceptibilityProgram, y::Vector{Float64}, ω::Float64)::Float64
    g .= 0.0
    μ = y[1]

    # (\mathcal K) term. No gradient.
    #r = p.sgn / ω * exp(-1/ω)
    r = p.sgn * exp(-ω^2)

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

    plo = SusceptibilityProgram(cors, 1.0)
    phi = SusceptibilityProgram(cors, -1.0)
    lo, ylo = solve(plo; verbose=true)
    hi, yhi = solve(phi; verbose=true)
    println("$(-lo) $hi")
    println(ylo)
end

main()

