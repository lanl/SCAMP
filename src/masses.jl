using ArgParse
using LinearAlgebra
using Profile
using Random: rand, randn, rand!, randn!
using SpecialFunctions: erf
using Statistics: mean, cov

import Base: size

using SCAMP
using SCAMP.Utilities: check_gradients
import SCAMP: initial, constraints!, objective!

#const dω = 0.0001
const dω = 0.001
const Ω = 1.5

function resample(f, x; K=1000)::Vector{Float64}
    N = length(x)
    fs = zeros(K)
    for k in 1:K
        fs[k] = f(rand(x,N))
    end
    return fs
end

function main()
    args = let
    end
end

main()
