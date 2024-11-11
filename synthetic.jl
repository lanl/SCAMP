#!/usr/bin/env julia

function main()
    if false
        # The full thermal thing is provided.
        N = 1000
        K = 40
        β = Float64(K)
        ω = 0.3
        println("$N $K")
        for n in 1:N, k in 1:K
            τ = k-1
            m = cosh(ω * (τ - β/2))
            c = m + randn()*10
            println("$τ $c 0.0")
        end
    else
        # Just the first part of the correlator is provided
        N = 1000
        K = 40
        β = 120
        ω = 0.3
        println("$N $K")
        for n in 1:N, k in 1:K
            τ = k-1
            m = cosh(ω * (τ - β/2))
            c = m + randn()*10
            println("$τ $c 0.0")
        end
    end
end

main()
