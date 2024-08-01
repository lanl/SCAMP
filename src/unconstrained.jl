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
    for step in 1:100
        r₀ = f!(∇′, y)
        for k in 1:100
            r = f!(∇, y)
            y′ .= y
            for n in 1:N
                y′[n] -= δ * ∇[n]
            end
            r′ = f!(∇′, y′)
            if r′ < r
                y .= y′
            end
        end
        r = f!(∇′, y)
        if r ≥ r₀
            if δ < 1e-6
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
    α = 1.
    αmin = 1e-10
    δmin = 1e-10
    for step in 1:1000000
        r₀ = f!(∇, y)
        if step%200000 == 0
            println("   Taking long! step=$step   α=$α    r₀=$r₀")
            println(y)
        end
        function at!(α::Float64)::Float64
            for n in 1:N
                y′[n] = y[n] - α * ∇[n]
            end
            return f!(∇′, y′)
        end
        r = at!(α)
        while r > r₀ && α > αmin
            α /= 2
            r = at!(α)
        end

        while α > αmin
            r′ = at!(α/2)
            if r′ < r
                r = r′
                α /= 2
            else
                break
            end
        end
        δ = r₀ - r
        if α < αmin || δ < δmin
            break
        end
        y′ = y - α * ∇
        y .= y′
        α *= 10
    end
    return f!(∇, y)[1]
end

struct BFGS
end

function BFGS()
    return BFGS()
end

function (bfgs::BFGS)(f!, y::Vector{Float64})::Float64
end

struct LBFGS
end

function minimize!(f!, alg, y::Vector{Float64})::Float64
    return alg()(f!, y)
end

end
