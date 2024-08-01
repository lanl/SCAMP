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
    δ = 1e-1
    ∇::Vector{Float64} = zero(y)
    ∇′::Vector{Float64} = zero(y)
    y′::Vector{Float64} = zero(y)
    for step in 1:1000
        r₀ = f!(∇′, y)
        for k in 1:1000
            f!(∇, y)
            for n in 1:N
                y[n] -= δ * ∇[n]
            end
        end
        r = f!(∇′, y)
        if r ≥ r₀
            if δ < 1e-4
                return r
            else
                δ /= 2.
            end
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
    N = length(y)
    ∇::Vector{Float64} = zero(y)
    ∇′::Vector{Float64} = zero(y)
    y′::Vector{Float64} = zero(y)
    for step in 1:10000
        r₀ = f!(∇, y)
        function at!(α::Float64)::Float64
            for n in 1:N
                y′[n] = y[n] - α * ∇[n]
            end
            return f!(∇′, y′)
        end
        α = 1.
        r = at!(α)
        while r > r₀
            α /= 2
            #println("    shrinking α to $α, since $r > $r₀")
            r = at!(α)
        end

        while true
            r′ = at!(α/2)
            if r′ < r
                r = r′
                α /= 2
                #println("     shrinking α to $α, since $r′ < $r")
            else
                break
            end
        end
        #println("step=$step;   ", norm(∇), "   ", α)
        if α < 1e-8
            break
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
