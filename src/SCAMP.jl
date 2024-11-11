module SCAMP

export solve
export initial, constraints!, objective!

export ConvexProgram

include("unconstrained.jl")
include("programs.jl")
include("simplex.jl")
include("ipm.jl")
include("utilities.jl")

using .Programs
using .IPM
using .UnconstrainedOptimization

end
