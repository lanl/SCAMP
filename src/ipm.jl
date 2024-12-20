module IPM

using LinearAlgebra
using Quadmath

import Base: size

using ..Programs
using ..UnconstrainedOptimization

import ..Programs: initial, constraints!, objective!

export solve

function feasible(p, y)::Bool
    ok = true
    constraints!(p, y) do f,g,h
        if f isa Matrix
            if minimum(eigvals(Hermitian(f))) < 0
                ok = false
            end
        elseif f isa Real
            if f < 0
                ok = false
            end
        else
            throw(ArgumentError("Expected Real or Matrix"))
        end
    end
    return ok
end

function barrier!(g, h, p, y::Vector{Float128})::Float128
    N = length(y)
    r::Float128 = 0.
    if !isnothing(g)
        g .= 0.
    end
    if !isnothing(h)
        h .= 0.
    end

    function cb(M::Matrix, D, H)
        if !isnothing(H) && H != 0
            error("Hessian is not 0")
        end
        F = eigen(Hermitian(M))
        if minimum(F.values) ≤ 0
            r = Inf
        end
        if r < Inf
            # We use the trace of the logarithm.
            r += -sum(log.(F.values))
            Minv = inv(F)
            if !isnothing(g)
                for n in 1:N
                    g[n] -= real(tr(Minv * D[:,:,n]))
                end
            end
            if !isnothing(h)
                K = size(Minv)[1]
                if true
                    mat′ = similar(Minv)
                    mat = similar(Minv)
                    for n in 1:N
                        mul!(mat′, (@view D[:,:,n]), Minv)
                        mul!(mat, Minv, mat′)
                        # Here mat == Minv * D[:,:,n] * Minv
                        for m in 1:N
                            for i in 1:K, j in 1:K
                                @inbounds h[n,m] += real(mat[i,j] * D[j,i,m])
                            end
                        end
                    end
                end
                if false
                    matn = similar(Minv)
                    matm = similar(Minv)
                    al = @allocations for n in 1:N, m in 1:N
                        mul!(matn, Minv, D[:,:,n])
                        mul!(matm, Minv, D[:,:,m])
                        for i in 1:K
                            for j in 1:K
                                h[n,m] += real(matn[i,j] * matm[j,i])
                            end
                        end
                        #h[n,m] += real(tr(Minv * D[:,:,n] * Minv * D[:,:,m]))
                    end
                    println("$al allocations")
                end
                if false
                    for n in 1:N, m in 1:N
                        h[n,m] += real(tr(Minv * D[:,:,n] * Minv * D[:,:,m]))
                    end
                end
            end
        end
    end

    function cb(f::Real, d, H)
        if !isnothing(H) && H != 0
            error("Hessian is not 0")
        end
        if f ≤ 0
            r = Inf
        end
        if r < Inf
            r += -log(f)
            finv = f^(-1)
            d .*= finv
            if !isnothing(g)
                g .-= d
            end
            if !isnothing(h)
                for n in 1:N, m in 1:N
                    h[n,m] += d[n] * d[m]
                end
            end
        end
    end

    constraints!(cb, p, y)
    return r
end

struct Phase1 <: ConvexProgram
    cp::ConvexProgram
end

function size(p::Phase1)
    return 1 + size(p.cp)
end

function initial(p::Phase1)::Vector{Float128}
    y′ = initial(p.cp)
    y = zeros(Float128, 1+length(y′))
    y[2:end] .= y′
    constraints!(p.cp, y′) do f,g
        if f isa Real
            if y[1] + f < 0
                y[1] = -f + 1.0
            end
        elseif f isa Matrix
            f = minimum(eigvals(Hermitian(f)))
            if y[1] + f < 0
                y[1] = -f + 1.0
            end
        else
            throw(ArgumentError("Expected Real or Matrix"))
        end
    end
    return y
end

function objective!(g, h, p::Phase1, y::Vector{Float128})::Float128
    if !isnothing(g)
        g[1] = 1.
        g[2:end] .= 0.
    end
    return y[1]
end

function constraints!(cb, p::Phase1, y::Vector{Float128})
    N = length(y)-1
    s = y[1]
    function fn(M::Matrix, D)
        F = eigen(Hermitian(M))
        f = F.values[1]
        v = F.vectors[:,1]
        g′ = zeros(Float128, length(D)+1)
        g′[1] = 1.
        for n in 1:N
            g′[1+n] = real(v' * D[:,:,n] * v)
        end
        cb(s+f, g′)
    end

    function fn(f::Real, d)
        g = zeros(Float128, length(d)+1)
        g[1] = 1.
        for n in 1:N
            g[1+n] = d[n]
        end
        cb(s+f, g)
    end

    constraints!(fn, p.cp, y[2:end])
end

function feasible_initial(prog::ConvexProgram; verbose::Bool=false)::Vector{Float128}
    if verbose
        println(stderr, "Finding feasible initial point...")
    end

    N = size(prog)
    y = initial(prog)
    g = zero(y)

    minimize!(BFGS, y) do g, y
        r::Float128 = 0.
        if !isnothing(g)
            g .= 0.0
        end

        function cb(f::Real,D,H)
            if !isfinite(f)
                r = Inf
                return
            end
            if f ≤ 0
                r -= f
                if !isnothing(g)
                    for n in 1:N
                        g[n] -= D[n]
                    end
                end
            end
        end

        function cb(M::Matrix,D,H)
            if any(isinf.(M)) || any(isnan.(M))
                r = Inf
                return
            end
            F = eigen(Hermitian(M))
            f = F.values[1]
            v = F.vectors[:,1]
            if f ≤ 0
                r -= f
                if !isnothing(g)
                    for n in 1:N
                        g[n] -= real(v' * D[:,:,n] * v)
                    end
                end
            end
        end

        constraints!(cb, prog, y)
        return r
    end

    if !feasible(prog, y)
        error("No (strictly) feasible point found.")
    end

    return y
end

function solve_bfgs(prog::ConvexProgram, y; verbose::Bool=false, early=nothing)::Tuple{Float128, Vector{Float128}}
    if !feasible(prog, y)
        error("Initial point was not (strictly) feasible")
    end

    N = length(y)
    g = zero(y)

    μ = 2
    ϵ = 1e-10
    t₀ = 1.0e-6

    t = t₀
    H = zeros(Float128, (N,N))
    H += I
    while t < 1/ϵ
        # Center.
        v = minimize!(BFGS, y; H0=H) do g, y
            if any(isnan.(y)) || any(isinf.(y))
                return Inf
            end
            if isnothing(g)
                gobj, gbar = nothing, nothing
            else
                gobj, gbar = zero(g), zero(g)
            end
            obj = objective!(gobj, nothing, prog, y)
            bar = barrier!(gbar, nothing, prog, y)
            r = obj + bar/t
            if !isnothing(g)
                for n in 1:N
                    g[n] = gobj[n] + gbar[n]/t
                end
            end
            return r
        end
        obj = objective!(g, nothing, prog, y)
        if verbose
            println(stderr, t, " ", v, "   ", obj)
        end
        if !isnothing(early)
            if early(y)
                break
            end
        end
        t = μ*t
        H += 1e-6 * maximum(eigvals(Hermitian(H))) * I
        #H .*= μ  # This does not help.
    end

    return objective!(g, nothing, prog, y), y
end

function solve(prog::ConvexProgram, y; verbose::Bool=false)::Tuple{Float128, Vector{Float128}}
    if !feasible(prog, y)
        error("Initial point was not (strictly) feasible")
    end

    N = length(y)
    g = zero(y)
    hobj = zeros(Float128, (N,N))
    hbar = zeros(Float128, (N,N))

    μ = 2
    ϵ::Float128 = 1e-10
    t₀::Float128 = 1.0e-2

    t = t₀
    while t < 1/ϵ
        # Center.
        v = minimize!(Newton, y) do g, h, y
            if any(isnan.(y)) || any(isinf.(y))
                return Inf
            end
            gobj, gbar = zero(g), zero(g)
            obj = objective!(gobj, hobj, prog, y)
            bar = barrier!(gbar, hbar, prog, y)
            r = obj + bar/t
            @. g = gobj + gbar/t
            @. h = hobj + hbar/t
            return r
        end
        obj = objective!(nothing, nothing, prog, y)
        if verbose
            println(stderr, t, " ", v, "   ", obj)
        end
        t = μ*t
    end

    return objective!(nothing, nothing, prog, y), y
end

function solve(prog::ConvexProgram; verbose::Bool=false)::Tuple{Float128, Vector{Float128}}
    if verbose
        println(stderr, "Solving $(typeof(prog))")
    end
    y = feasible_initial(prog; verbose=verbose)

    if verbose
        println(stderr, "Performing phase-2 optimization...")
    end
    return solve(prog, y; verbose=verbose)
end

end
