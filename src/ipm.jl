module IPM

using ..Programs
using ..UnconstrainedOptimization

export solve

function solve(prog::ConvexProgram; verbose::Bool=false)::Tuple{Float64, Vector{Float64}}
    y = Programs.initial(prog)
    N = length(y)
    g = zero(y)
    if verbose
        println("Solving SDP: $(N) degrees of freedom")
    end

    # Phase 1
    if false
        feasible = false
        for step in 1:10000
            badness = Programs.badness!(g, prog, y)
            println(step, "     ", badness)
            if badness ≤ 0
                feasible = true
                break
            end
            for n in 1:N
                y[n] -= 1e-2 * g[n]
            end
        end
    end
    badness = minimize!(GradientDescent, y) do g, y
        return Programs.badness!(g, prog, y)
    end
    if badness > 0
        error("No (strictly) feasible point found")
    end
    if verbose
        println("Feasible point found.")
    end

    # Phase 2
    μ = 1.5
    ϵ = 1e-6
    t₀ = 1.

    t = t₀
    while t < 1/ϵ
        # Center.
        for step in 1:100
        end
        if verbose
        end
        t = μ*t
    end

    return Programs.objective!(g, prog, y), y
end

end
