module IPM

using ..Programs
using ..UnconstrainedOptimization

import ..Programs: initial, constraints!, objective!

export solve

function feasible(p, y)::Bool
    ok = true
    constraints!(p, y) do f,g
        if f < 0
            ok = false
        end
    end
    return ok
end

function barrier!(g, p, y::Vector{Float64})::Float64
    N = length(g)
    r::Float64 = 0.
    g .= 0.
    constraints!(p, y) do f,g′
        if f ≤ 0
            r = Inf
        end
        if r < Inf
            r += -log(f)
            for n in 1:N
                g[n] -= g′[n] / f
            end
        end
    end
    return r
end

struct Phase1 <: ConvexProgram
    cp::ConvexProgram
end

function initial(p::Phase1)::Vector{Float64}
    y′ = initial(p.cp)
    y = zeros(Float64, 1+length(y′))
    y[2:end] .= y′
    constraints!(p.cp, y′) do f,g
        if y[1] + f < 0
            y[1] = -f + 1.0
        end
    end
    return y
end

function objective!(g, p::Phase1, y::Vector{Float64})::Float64
    g[1] = 1.
    g[2:end] .= 0.
    return y[1]
end

function constraints!(cb, p::Phase1, y::Vector{Float64})
    s = y[1]
    constraints!(p.cp, y[2:end]) do f,g
        g′ = zeros(Float64, length(g)+1)
        g′[1] = 1.
        g′[2:end] .= g
        cb(s+f, g′)
    end
end

function feasible_initial(prog::ConvexProgram; verbose::Bool)::Vector{Float64}
    println(stderr, "Finding feasible initial point...")

    y = initial(prog)
    g = zero(y)
end

function solve(prog::ConvexProgram, y; verbose::Bool=false, gd=BFGS, early=nothing)::Tuple{Float64, Vector{Float64}}
    if !feasible(prog, y)
        error("Initial point was not feasible")
    end

    N = length(y)
    g = zero(y)

    μ = 1.5
    ϵ = 1e-10
    t₀ = 1.0e-3

    t = t₀
    while t < 1/ϵ
        # Center.
        v = minimize!(gd, y) do g, y
            gobj, gbar = zero(g), zero(g)
            obj = objective!(gobj, prog, y)
            bar = barrier!(gbar, prog, y)
            r = obj + bar/t
            for n in 1:N
                g[n] = gobj[n] + gbar[n]/t
            end
            return r
        end
        obj = objective!(g, prog, y)
        if verbose
            println(stderr, t, " ", v, "   ", obj)
        end
        if !isnothing(early)
            if early(y)
                break
            end
        end
        t = μ*t
    end

    return objective!(g, prog, y), y
end

function solve(prog::ConvexProgram; verbose::Bool=false)::Tuple{Float64, Vector{Float64}}
    y = initial(prog)
    if verbose
        println(stderr, "Solving SDP: $(length(y)) degrees of freedom")
    end

    if !feasible(prog, y)
        # Perform phase-1 optimization.
        if verbose
            println(stderr, "Initial point infeasible; performing phase-1 optimization...")
        end
        phase1 = Phase1(prog)
        y′ = initial(phase1)
        function check(y)
            return feasible(prog, y[2:end])
        end
        solve(Phase1(prog), y′; verbose=verbose, gd=BFGS, early=check)
        y = y′[2:end]
    end
    # Check feasibility
    if !feasible(prog, y)
        error("Phase1 solver found infeasible point!")
    end

    println(y)
    error("TODO")

    if verbose
        println(stderr, "Performing phase-2 optimization...")
    end
    return solve(prog, y; verbose=verbose)
end

end
