using ArgParse
using LinearAlgebra
using Random: rand, randn, rand!, randn!
using Statistics: mean, cov

using SCAMP
import SCAMP: initial, constraints!, objective!

#const dω = 0.02
#const Ω = 20.0
const dω = 0.02
const Ω = 0.5
const ωs = dω:dω:Ω

function resample(f, x; K=1000)::Vector{Float64}
    N = length(x)
    fs = zeros(K)
    for k in 1:K
        fs[k] = f(rand(x,N))
    end
    return fs
end

struct CorrelatorProgram <: ConvexProgram
    β::Float64
    τ::Vector{Float64}
    C::Vector{Float64}
    M::Matrix{Float64}
    Minv::Matrix{Float64}
    t::Float64
    σ::Float64
    sgn::Float64

    function CorrelatorProgram(Cs, t::Float64, σ::Float64, sgn::Float64; p::Float64=0.01)::CorrelatorProgram
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
            Σ[i,i] += 1e-5 * maxeig
        end

        τ = collect(1:Float64(β))
        K = 1000
        if false
            M = zeros((β,β))
            for n in 1:β
                M[n,n] = 1.0
            end
        else
            M = inv(Σ)
        end
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
        new(Float64(β), τ, C, M, Minv, t, σ, sgn)
    end

    function CorrelatorProgram(p::CorrelatorProgram; t::Float64, sgn::Float64)::CorrelatorProgram
        new(p.β, p.τ, p.C, p.M, p.Minv, t, p.σ, sgn)
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
        g[i] = coef * p.cp.sgn
        r += ρ[i] * coef * p.cp.sgn
    end
    return r
end

function constraints!(cb, p::PrimalCorrelatorProgram, ρ::Vector{Float64})
    β = length(p.cp.C)
    g = zero(ρ)
    # Positivity
    for (k,ρω) in enumerate(ρ)
        g[k] = dω
        cb(ρω*dω, g)
        g[k] = 0.
    end

    # Correlator error
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
    for (k, ω) in enumerate(ωs)
        for i in 1:β, j in 1:β
            g[k] -= 2 * v'[i] * p.cp.M[i,j] * dcor[j,k]
        end
    end
    cb(1-err, g)
end

function initial(p::CorrelatorProgram)::Vector{Float64}
    return rand(length(p.τ)+1)
end

function objective!(g, p::CorrelatorProgram, y::Vector{Float64})::Float64
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

function constraints!(cb, p::CorrelatorProgram, y::Vector{Float64})
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

function λ!(g::Vector{Float64}, p::CorrelatorProgram, y::Vector{Float64}, ω::Float64)::Float64
    g .= 0.0
    μ = y[1]

    # (\mathcal K) term. No gradient
    r = p.sgn * -2 * sin(ω * p.t) * exp(-(p.σ^2 * ω^2)/2)

    # -K^T ℓ term, with gradient
    for (i,τ) in enumerate(p.τ)
        if false # TODO
            if i < length(p.τ)
                # There should be a feasible point...
                continue
            end
        end
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
            "-P","--primal"
                action = :store_true
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
   
    if false
        # Check derivatives of barrier! for primal Phase1
        p = SCAMP.IPM.Phase1(primal(CorrelatorProgram(cors, 0.5, 1.0, 1.0)))
        ρ = initial(p)
        g = zero(ρ)
        g′ = zero(ρ)
        bar = SCAMP.IPM.barrier!(g, p, ρ)
        for n in 1:length(ρ)
            ϵ = 1e-5
            ρ′ = copy(ρ)
            ρ′[n] += ϵ
            bar′ = SCAMP.IPM.barrier!(g′, p, ρ′)
            println((bar′ - bar)/ϵ - g[n], "   ::   ", bar, " ", bar′, " ", g[n], " ", (bar′-bar)/ϵ)
        end
        return
    end

    if false
        # Check derivatives of barrier! for dual Phase1
        p = SCAMP.IPM.Phase1(CorrelatorProgram(cors, 0.5, 1.0, 1.0))
        ρ = initial(p)
        g = zero(ρ)
        g′ = zero(ρ)
        bar = SCAMP.IPM.barrier!(g, p, ρ)
        for n in 1:length(ρ)
            ϵ = 1e-6
            ρ′ = copy(ρ)
            ρ′[n] += ϵ
            bar′ = SCAMP.IPM.barrier!(g′, p, ρ′)
            println((bar′ - bar)/ϵ - g[n], "   ::   ", bar, " ", bar′, " ", g[n], " ", (bar′-bar)/ϵ)
        end
        return
    end

    if false
        # Check derivatives of barrier!
        p = primal(CorrelatorProgram(cors, 0.5, 1.0, 1.0))
        ρ = solve(p; verbose=false)[2]
        g = zero(ρ)
        g′ = zero(ρ)
        bar = barrier!(g, p, ρ)
        for n in 1:length(ρ)
            ϵ = ρ[n] * 1e-4
            ρ′ = copy(ρ)
            ρ′[n] += ϵ
            bar′ = barrier!(g′, p, ρ′)
            println((bar′ - bar)/ϵ - g[n], "   ::   ", bar, " ", bar′, " ", g[n], " ", log(ρ[n]))
        end
        return
    end

    if false
        # Check derivatives of (dual) objective!
        p = CorrelatorProgram(cors, 0.5, 1.0, 1.0)
        y = initial(p)
        g = zero(y)
        g′ = zero(y)
        for n in 1:length(y)
            ϵ = 1e-5
            r = objective!(g, p, y)
            y′ = copy(y)
            y′[n] += ϵ
            r′ = objective!(g′, p, y′)
            println((r′-r)/ϵ - g[n], "            ", (r′-r)/ϵ, "     ", g[n])
        end
        return
    end

    if false
        # Check derivatives of dual barrier!
        for sgn in [1.0,-1.0]
            p = CorrelatorProgram(cors, 0.5, 1.0, 1.0)
            y = initial(p)
            if !SCAMP.IPM.feasible(p, y)
                phase1 = SCAMP.IPM.Phase1(p)
                y′ = initial(phase1)
                function check(y)
                    return SCAMP.IPM.feasible(p, y[2:end])
                end
                solve(SCAMP.IPM.Phase1(p), y′; verbose=true, gd=SCAMP.UnconstrainedOptimization.GradientDescent, early=check)
                y = y′[2:end]
            end
            g = zero(y)
            g′ = zero(y)
            bar = SCAMP.IPM.barrier!(g, p, y)
            for n in 1:length(y)
                ϵ = y[n] * 1e-4
                y′ = copy(y)
                y′[n] += ϵ
                bar′ = SCAMP.IPM.barrier!(g′, p, y′)
                println((bar′ - bar)/ϵ - g[n], "   ::   ", bar, " ", bar′, " ", g[n])
            end
        end
        return
    end

    if false
        # Brute force, to demonstrate that such a feasible point exists
        t = 50.0
        σ = 5.0
        p = CorrelatorProgram(cors, t, σ, 1.)
        y = initial(p)
        y[2:end-1] .= 0.0

        for yend in 1.5:.01:2.9
            feas = true
            last = 0.0
            for y1 in 0:.00001:.02
                #y[1] = 1.1
                y[1] = y1
                #y[end] = 100.0
                y[end] = yend
                #if !SCAMP.IPM.feasible(p,y)
                #    println("NOT FEASIBLE")
                #end
                g = zero(y)
                obj = objective!(g, p, y)
                #println("Objective: $obj")
                #println()
                cs = Float64[]
                constraints!(p,y) do f, g
                    push!(cs, f)
                end
                #println("      (worst constraint:) ", minimum(cs))
                #println(y1, ",", yend, "     ", minimum(cs), "   ", obj)
                if !SCAMP.IPM.feasible(p,y) && feas
                    println(yend, "   ", obj, "     ", yend, "  ", y1)
                    feas = false
                end
                last = obj
            end
        end
        return
    end

    σ = 5.0
    if args["primal"]
        # Solve primal
        p = CorrelatorProgram(cors, 0.0, σ, 1.0)
        ρ0 = solve(primal(p); verbose=true)[2]
        for t in 2.5:2.5:100.
            plo = CorrelatorProgram(p, t=t, sgn=1.0)
            phi = CorrelatorProgram(p, t=t, sgn=-1.0)
            ρ = copy(ρ0)
            lo, ρlo = solve(primal(plo), ρ; verbose=true)
            ρ = copy(ρ0)
            hi, ρhi = solve(primal(phi), ρ; verbose=true)
            println("$t  $lo $(-hi)")
            flush(stdout)
        end
    else
        # Solve dual
        for t in 5.0:5.0:30.0
            plo = CorrelatorProgram(cors, t, σ, 1.)
            phi = CorrelatorProgram(cors, t, σ, -1.)
            lo, ylo = solve(plo; verbose=false)
            hi, yhi = solve(phi; verbose=false)
            println("$t  $(-lo) $hi")
            flush(stdout)
            #println("    ", ylo)
        end
    end
end

#=

Debugging ideas:

Why is the dual phase1 unbounded below?

It seems we are running into numerical precision issues in the dual problem.
Use quadmath? Arbitrary precision? Precondition?

Phase1 solver is still not reliable. This appears when solving the primal, and
when excluding large numbers of degrees of freedom from the dual.

=#

main()

