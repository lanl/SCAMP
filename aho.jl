#!/usr/bin/env julia

using LinearAlgebra

function main()
    ω, λ = 1., 0.3
    N = 100
    a = zeros(ComplexF64, (N,N))
    for i in 1:(N-1)
        a[i,i+1] = sqrt(ω * i)
    end
    x = 1/sqrt(2*ω) * (a + a')
    p = 1im * sqrt(ω/2) * (a' - a)
    H = 0.5*p^2 + 0.5 * ω^2 * x^2 + 0.25 * λ * x^4
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

main()

