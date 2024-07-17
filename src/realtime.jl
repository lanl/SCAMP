using ArgParse

using SCAMP

struct CorrelatorProgram
    β::Float64
    τ::Vector{Float64}
    Σ::Matrix{Float64}
    t::Float64
    σ::Float64
end

function initial(p::CorrelatorProgram)::Vector{Float64}
    # TODO
end

function badness!(g::Vector{Float64}, p::CorrelatorProgram, y::Vector{Float64})::Float64
    # TODO
end

function barrier!(g::Vector{Float64}, p::CorrelatorProgram, y::Vector{Float64})::Float64
    # TODO
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
