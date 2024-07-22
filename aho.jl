#!/usr/bin/env julia

using LinearAlgebra

function main()
    ω², λ = 0.01, 0.001
    ω = √ω²
    N = 100
    a = zeros(ComplexF64, (N,N))
    for i in 1:(N-1)
        a[i,i+1] = sqrt(i)
    end
    x = 1/sqrt(2*ω) * (a + a')
    p = 1im * sqrt(ω/2) * (a' - a)
    H = 0.5*p^2 + 0.5 * ω² * x^2 + 0.25 * λ * x^4
    F = eigen(Hermitian(H))

    Ω = F.vectors[:,1]

    function C(t::Float64)::ComplexF64
        U = (F.vectors')*diagm(exp.(-1im * t * F.values))*F.vectors
        return dot(x*U*Ω, U*x*Ω)
    end

    function G(τ::Float64)::ComplexF64
        V = (F.vectors')*diagm(exp.(-τ * F.values))*F.vectors
        return dot(x*V*Ω, V*x*Ω) / dot(V*Ω, V*Ω)
    end

    # Print mass gap
    println("# mass gap: ", F.values[2] - F.values[1])

    # Compute real-time correlator.
    dt = .01
    ts = 0:dt:100

    if false
        # Comparing two exponential integrals.
        κ = 2
        int1::ComplexF64 = 0.
        int2::ComplexF64 = 0.

        dt = .01
        for t in 0:dt:6
            int1 += dt * C(t) * exp(-κ*t)
        end

        dτ = .01
        for τ in 0:dτ:6
            int2 += dτ * G(τ) * exp(1im * κ * τ)
        end
        int2 *= -1im

        println("$int1    $int2")
    end
end

main()

