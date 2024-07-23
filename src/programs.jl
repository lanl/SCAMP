module Programs

export ConvexProgram, LinearProgram
export initial, badness!, barrier!, objective!

abstract type ConvexProgram end

abstract type LinearProgram end

struct DenseLinearProgram
end

function initial end
function badness! end
function barrier! end
function objective! end

end
