using ArgParse

using SCAMP

struct CorrelatorProgram
    β::Float64
    τ::Vector{Float64}
    Σ::Matrix{Float64}
end

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
    cor, cov = let
        lines = readlines(args["correlator"])
    end
end

main()
