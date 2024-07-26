module SCAMP

export solve
export initial, constraints!, objective!

export ConvexProgram

include("unconstrained.jl")
include("programs.jl")
include("simplex.jl")
include("ipm.jl")

using .Programs
using .IPM
using .UnconstrainedOptimization

end
