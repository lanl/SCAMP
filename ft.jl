#!/usr/bin/env julia

function main()
    ωs = Float64[]
    ρs = Float64[]
    open(ARGS[1]) do f
        for l in readlines(f)
            ωstr, ρstr = split(l)
            push!(ωs, Meta.parse(ωstr))
            push!(ρs, Meta.parse(ρstr))
        end
    end
    dω = ωs[2] - ωs[1]
    for t in 0:1:200
        c = 0.0
        for (ω,ρ) in zip(ωs,ρs)
            c -= sin(ω*t) * ρ * dω
        end
        println("$t $c")
    end
end

main()

