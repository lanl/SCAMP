#!/usr/bin/env julia

using ArgParse
using LinearAlgebra

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "--correlator"
                arg_type = String
                required = false
            "--spectral"
                arg_type = String
                required = false
            "--sigma"
                arg_type = Float64
                required = true
        end
        parse_args(s)
    end
    σ = args["sigma"]

    #ω², λ = 0.0001, 0.000001
    #ω², λ = 1e-4, 1e-6
    ω², λ = 1e-4, 1e-5
    #ω², λ = parse(Float64, ARGS[1]), parse(Float64, ARGS[2])
    ω = √ω²
    N = 200
    a = zeros(ComplexF64, (N,N))
    for i in 1:(N-1)
        a[i,i+1] = sqrt(i)
    end
    x = 1/sqrt(2*ω) * (a + a')
    p = 1im * sqrt(ω/2) * (a' - a)
    H = 0.5*p^2 + 0.5 * ω² * x^2 + 0.25 * λ * x^4
    F = eigen(Hermitian(H))
    β = 30

    Ω = F.vectors[:,1]
    ρ = (F.vectors) * diagm(exp.(-β * F.values)) * F.vectors'

    function C(t::Float64)::ComplexF64
        U = (F.vectors)*diagm(exp.(-1im * t * F.values))*F.vectors'
        #return dot(x*U*Ω, U*x*Ω)
        return tr(ρ * U' * x * U * x)/ tr(ρ)
    end

    function G(τ::Float64)::ComplexF64
        V = (F.vectors)*diagm(exp.(-τ * F.values))*F.vectors'
        return dot(x*V*Ω, V*x*Ω) / dot(V*Ω, V*Ω)
    end

    # Print mass gap
    println("# mass gap: ", F.values[2] - F.values[1])

    if !isnothing(args["correlator"])
        open(args["correlator"], "w") do f
            # Compute real-time correlator.
            dt = 5.0
            ts = -100:1.0:700
            Ts = 0:dt:600

            cor = zero(ts)
            for (k,t) in enumerate(ts)
                cor[k] = imag(C(t))
            end

            for T in Ts
                exact = imag(C(T))
                smeared = 0.
                den = 0.
                for (t,c) in zip(ts, cor)
                    smeared += exp(-(T-t)^2 / (2. * σ^2)) * c
                    den += exp(-(T-t)^2 / (2. * σ^2))
                end
                smeared /= den
                println(f, "$T $smeared $exact")
            end
        end
    end

    if !isnothing(args["spectral"])
        open(args["spectral"], "w") do f
            # Compute smeared spectral function
            Z = tr(ρ)

            ωs = 0:.001:1
            for ω in ωs
                spec = 0.0
                for n in 1:N
                    for m in 1:N
                        En = F.values[n]
                        Em = F.values[m]
                        ω′ = Em-En
                        spec += 2/Z * sinh(β*ω/2) * exp(-β * (En+Em)/2) * exp(-0)
                    end
                end
                # TODO
                println(f, "$ω $spec")
            end
        end
    end

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

