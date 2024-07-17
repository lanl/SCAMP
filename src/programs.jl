module Programs

export ConvexProgram, LinearProgram
export initial, badness!, barrier!

abstract type ConvexProgram end

abstract type LinearProgram end

struct DenseLinearProgram
end

function initial end
function badness! end
function barrier! end

end
