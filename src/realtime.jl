using ArgParse
using DoubleFloats
using LinearAlgebra
using Random: rand, randn, rand!, randn!
using Statistics: mean, cov

using SCAMP
import SCAMP: initial, constraints!, objective!

RealT = Float64

#const dω = 0.02
#const Ω = 20.0
const dω = RealT(0.001)
const Ω = RealT(1.0)
const ωs = dω:dω:Ω

function resample(f, x; K=1000)::Vector{RealT}
    N = length(x)
    fs = zeros(RealT, K)
    for k in 1:K
        fs[k] = f(rand(x,N))
    end
    return fs
end

struct CorrelatorProgram <: ConvexProgram
    β::RealT
    τ::Vector{RealT}
    C::Vector{RealT}
    M::Matrix{RealT}
    Minv::Matrix{RealT}
    t::RealT
    σ::RealT
    sgn::Float64

    function CorrelatorProgram(Cs, t::RealT, σ::RealT, sgn::Float64; p::RealT=RealT(0.01))::CorrelatorProgram
        N = length(Cs)
        C = mean(Cs)
        β = length(C)

        # First extract the covariance matrix.
        K = 1000
        cors = zeros(RealT, (K,β))
        for k in 1:K
            cors[k,:] = mean(rand(Cs,N))
        end
        Σ = cov(cors)
        # Regulate.
        maxeig = maximum(real(eigvals(Σ)))
        for i in 1:β
            Σ[i,i] += 1e-6 * maxeig
        end

        τ = collect(1:RealT(β))
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
        new(RealT(β), τ, C, M, Minv, t, σ, sgn)
    end

    function CorrelatorProgram(p::CorrelatorProgram; t::RealT, sgn::Float64)::CorrelatorProgram
        new(p.β, p.τ, p.C, p.M, p.Minv, t, p.σ, sgn)
    end
end

function initial(p::CorrelatorProgram)::Vector{RealT}
    return rand(RealT, length(p.τ)+1)
end

function objective!(g, p::CorrelatorProgram, y::Vector{RealT})::RealT
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

function constraints!(cb, p::CorrelatorProgram, y::Vector{RealT})
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

function λ!(g::Vector{RealT}, p::CorrelatorProgram, y::Vector{RealT}, ω::RealT)::RealT
    g .= 0.0
    μ = y[1]

    # (\mathcal K) term. No gradient
    r = p.sgn * -1 * sin(ω * p.t) * exp(-(p.σ^2 * ω^2)/2)

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
            "-T","--time"
                required = true
                arg_type = RealT
            "-s","--sigma"
                required = false
                default = 1.0
                arg_type = RealT
            "--skip"
                required = false
                default = 1
                arg_type = Int
            "--scale"
                help = "Scaling hack for numerical stability"
                default = 1.0
                arg_type = Float64
            "-V","--verbose"
                action = :store_true
            "correlator"
                required = true
                arg_type = String
        end
        parse_args(s)
    end
    scale = args["scale"]
    verbose = args["verbose"]

    cors = let
        open(args["correlator"]) do f
            cors = Vector{Vector{RealT}}()
            for l in readlines(f)
                v = eval(Meta.parse(l))
                if !isnothing(v)
                    push!(cors, scale * v)
                end
            end
            cors
        end
    end
    # Skip
    cors = cors[1:args["skip"]:end]
   
    σ = args["sigma"]
    T = args["time"]
    p = CorrelatorProgram(cors, RealT(0.0), σ, 1.0)
    plo = CorrelatorProgram(p, t=T, sgn=1.0)
    phi = CorrelatorProgram(p, t=T, sgn=-1.0)
    lo, ylo = solve(plo; verbose=verbose)
    hi, yhi = solve(phi; verbose=verbose)
    println("$(-lo/scale) $(hi/scale)")
end

main()

