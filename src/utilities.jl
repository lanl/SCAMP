module Utilities

using Printf

function check_gradients(f, y, g, h; verbose=false)::Bool
    ϵ = 1e-8
    N = length(y)
    @assert size(g) == (N,)
    for n in 1:N
        y₀ = y[n]
        y[n] = y₀ - ϵ
        x₋ = f(y)
        y[n] = y₀ + ϵ
        x₊ = f(y)
        y[n] = y₀
        gn = (x₊ - x₋) / (2 * ϵ)
        if abs(gn - g[n]) / abs(gn + g[n] + 1e-10) > sqrt(ϵ)
            if verbose
                println(stderr, "Gradient $n mismatch: $gn   vs   $(g[n])")
            end
            return false
        end
    end
    if verbose
        printstyled("Gradients match!\n", color=:green, bold=true)
    end
    if isnothing(h)
        return true
    end
    @assert size(h) == (N,N)
    bad = false
    for i in 1:N, j in 1:N
        yᵢ₀ = y[i]
        yⱼ₀ = y[j]
        y[i], y[j] = yᵢ₀, yⱼ₀
        y[i] += ϵ
        y[j] += ϵ
        x₊₊ = f(y)
        y[i], y[j] = yᵢ₀, yⱼ₀
        y[i] += ϵ
        y[j] -= ϵ
        x₊₋ = f(y)
        y[i], y[j] = yᵢ₀, yⱼ₀
        y[i] -= ϵ
        y[j] += ϵ
        x₋₊ = f(y)
        y[i], y[j] = yᵢ₀, yⱼ₀
        y[i] -= ϵ
        y[j] -= ϵ
        x₋₋ = f(y)
        g₊ = (x₊₊ - x₊₋) / (2*ϵ)
        g₋ = (x₋₊ - x₋₋) / (2*ϵ)
        hij = (g₊ - g₋)/(2*ϵ)
        println("$hij   $(h[i,j])")
        if abs(hij - h[i,j]) / (maximum(abs.(h)) + abs(hij) + sqrt(ϵ)) > ϵ^(1/3)
            if verbose
                println(stderr, "Hessian ($i,$j) mismatch:   $hij   vs   $(h[i,j])")
            end
            bad = true
        end
    end
    if bad
        return false
    end
    if verbose
        printstyled("Hessian matches!\n", color=:green, bold=true)
    end
    return true
end

end
