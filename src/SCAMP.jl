module SCAMP

export solve
export initial, badness!, barrier!, objective!

export ConvexProgram

include("programs.jl")
include("simplex.jl")
include("ipm.jl")
include("unconstrained.jl")

using .Programs
using .IPM

end
