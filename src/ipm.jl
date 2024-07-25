module IPM

using ..Programs
using ..UnconstrainedOptimization

export solve

function solve(prog::ConvexProgram, y; verbose::Bool=false)::Tuple{Float64, Vector{Float64}}
    N = length(y)
    g = zero(y)

    μ = 1.5
    ϵ = 1e-6
    t₀ = 1.

    t = t₀
    while t < 1/ϵ
        # Center.
        v = minimize!(LineSearch, y) do g, y
            if isnothing(g)
                gobj, gbar = nothing, nothing
            else
                gobj, gbar = zero(g), zero(g)
            end
            obj = Programs.objective!(gobj, prog, y)
            bar = Programs.barrier!(gbar, prog, y)
            r = obj + bar/t
            if !isnothing(g)
                for n in 1:N
                    g[n] = gobj[n] + gbar[n]/t
                end
            end
            return r
        end
        if verbose
            println(t, " ", v)
        end
        t = μ*t
    end

    return Programs.objective!(g, prog, y), y
end

function solve(prog::ConvexProgram; verbose::Bool=false)::Tuple{Float64, Vector{Float64}}
    @label restart
    y = Programs.initial(prog)
    N = length(y)
    g = zero(y)
    if verbose
        println("Solving SDP: $(N) degrees of freedom")
    end

    # Phase 1 (TODO should use IPM here too)
    badness = minimize!(LineSearch, y) do g, y
        return Programs.badness!(g, prog, y)
    end
    if badness > 0
        #error("No (strictly) feasible point found; restarting")
        println("No (strictly) feasible point found; restarting")
        @goto restart
    end
    if verbose
        println("Feasible point found.")
    end

    return solve(prog, y; verbose=verbose)
end

end
