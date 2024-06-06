module Programs

export ConvexProgram
export LinearProgram

abstract type ConvexProgram end

struct LinearProgram <: ConvexProgram
end

end
