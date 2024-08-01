using ArgParse
using LinearAlgebra
using Random: rand, randn, rand!, randn!
using Statistics: mean, cov

using SCAMP
import SCAMP: initial, constraints!, objective!

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "correlator"
                required = true
                arg_type = String
        end
        parse_args(s)
    end
end

main()

