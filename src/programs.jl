module Programs

export ConvexProgram, LinearProgram
export initial, constraints!, objective!

abstract type ConvexProgram end

abstract type LinearProgram end

struct DenseLinearProgram
end

function initial end
function constraints! end
function objective! end

end
