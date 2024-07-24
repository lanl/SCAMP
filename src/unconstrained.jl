# Unconstrained optimization routines for convex functions.

module UnconstrainedOptimization

export GradientDescent, LineSearch
export minimize!

using LinearAlgebra

# Naive gradient descent
struct GradientDescent
end

function (gd::GradientDescent)(f!, y::Vector{Float64})::Float64
    N = length(y)
    δ = 0.01
    ∇::Vector{Float64} = zero(y)
    y′::Vector{Float64} = zero(y)
    for step in 1:100
        r₀ = f!(nothing, y)
        for k in 1:100
            f!(∇, y)
            for n in 1:N
                y[n] -= δ * ∇[n]
            end
        end
        r = f!(nothing, y)
        if r ≥ r₀
            return r
        end
    end
    return f!(∇, y)[1]
end

# Gradient descent with line search
struct LineSearch
    α₀::Float64
    ϵ::Float64
end

function LineSearch()
    return LineSearch(1, 1e-6)
end

function (gd::LineSearch)(f!, y::Vector{Float64})::Float64
    # TODO when to terminate gradient descent?
    N = length(y)
    ∇::Vector{Float64} = zero(y)
    y′::Vector{Float64} = zero(y)
    for step in 1:100
        r₀ = f!(∇, y)
        function at!(α::Float64)::Float64
            for n in 1:N
                y′[n] = y[n] - α * ∇[n]
            end
            return f!(nothing, y′)
        end
        α = 1.
        r = at!(α)
        while r > r₀
            α /= 2
            r = at!(α)
        end

        while true
            r′ = at!(α/2)
            if r′ < r
                r = r′
                α /= 2
            else
                break
            end
        end
        y′ = y - α * ∇
        y .= y′
    end
    return f!(∇, y)[1]
end

struct BFGS
end

struct LBFGS
end

function minimize!(f!, alg, y::Vector{Float64})::Float64
    return alg()(f!, y)
end

end
