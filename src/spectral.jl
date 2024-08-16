# Extract a smeared spectral function.

using ArgParse
using LinearAlgebra
using Random: rand, randn, rand!, randn!
using Statistics: mean, cov

using SCAMP
import SCAMP: initial, constraints!, objective!

const dω = 0.001
const Ω = 1.0
const ωs = dω:dω:Ω

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

    function SpectralProgram(Cs, ω::Float64, σ::Float64, sgn::Float64; p::Float64=0.01)::SpectralProgram
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
            Σ[i,i] += 1e-6 * maxeig
        end

        τ = collect(1:Float64(β))
        K = 1000
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
        new(Float64(β), τ, C, M, Minv, ω, σ, sgn)
    end

    function SpectralProgram(p::SpectralProgram; ω::Float64, sgn::Float64)::SpectralProgram
        new(p.β, p.τ, p.C, p.M, p.Minv, ω, p.σ, sgn)
    end
end

function initial(p::SpectralProgram)::Vector{Float64}
    return rand(length(p.τ)+1)
end

function objective!(g, p::SpectralProgram, y::Vector{Float64})::Float64
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

function constraints!(cb, p::SpectralProgram, y::Vector{Float64})
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

function λ!(g::Vector{Float64}, p::SpectralProgram, y::Vector{Float64}, ω::Float64)::Float64
    g .= 0.0
    μ = y[1]

    # (\mathcal K) term. No gradient.
    #r = p.sgn * -1 * sin(ω * p.t) * exp(-(p.σ^2 * ω^2)/2)
    r = p.sgn * exp(-(ω - p.ω)^2 / (2 * p.σ^2))

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
            "-s","--sigma"
                required = true
                default = 1.0
                arg_type = Float64
            "--skip"
                required = false
                default = 1
                arg_type = Int
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
    # Skip
    cors = cors[1:args["skip"]:end]

    σ = args["sigma"]
    ω = args["omega"]
    p = SpectralProgram(cors, 0.0, σ, 1.0)
    plo = SpectralProgram(p, ω=ω, sgn=1.0)
    phi = SpectralProgram(p, ω=ω, sgn=-1.0)
    lo, ylo = solve(plo; verbose=false)
    hi, yhi = solve(phi; verbose=false)
    println("$(-lo) $hi")
    return
end

main()

