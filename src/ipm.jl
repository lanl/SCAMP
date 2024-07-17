module IPM

using ..Programs

export solve

function solve(prog::ConvexProgram; verbose::Bool=false)::Tuple{Float64, Vector{Float64}}
    y = Programs.initial(prog)
    N = length(y)
    g = zero(y)
    if verbose
        println("Solving SDP: $(N) degrees of freedom")
    end

    # Phase 1
    feasible = false
    for step in 1:100
        badness = Programs.badness!(g, prog, y)
        if badness ≤ 0
            feasible = true
            break
        end
        for n in 1:N
            y[n] += 1e-2 * g[n]
        end
    end
    if !feasible
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
