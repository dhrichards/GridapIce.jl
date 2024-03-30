

function Π(f::CellField)
    return Π_op∘f
end

function Π_op(x::VectorValue{3})
    return SymTensorValue(x.data)
end


function Π_op(x::VectorValue{6})
    return SymTensorValue(x.data)
end

function Π_op(x::VectorValue{9})
    return SymFourthOrderTensorValue(x.data)
end

function Π_op(x::VectorValue{36})
    return SymFourthOrderTensorValue(x.data)
end


function invΠ(f::CellField)
    return invΠ_op∘f
end

function invΠ_op(x::Union{SymTensorValue,SymFourthOrderTensorValue})
    return VectorValue(x.data)
end


function invΠ_op(x::TensorValue{2,2,Float64,4})
    return VectorValue(x[1,1],0.5*(x[1,2]+x[2,1]),x[2,2])
end

function invΠ_op(x::TensorValue{3,3,Float64,9})
    return VectorValue(x[1,1],0.5*(x[1,2]+x[2,1]),0.5*(x[1,3]+x[3,1]),x[2,2],0.5*(x[2,3]+x[3,2]),x[3,3])
end


# Convert 2D tensor to 3D TensorValue
function T₂toT₃(T)
    return T₂toT₃_op∘T
end

function T₂toT₃_op(T::SymTensorValue{2, Float64, 3})
    return SymTensorValue(T[1,1],T[1,2],0.0,T[2,2],0.0,0.0)
end

function T₂toT₃_op(T::TensorValue{2,2})
    return TensorValue(T[1,1],T[1,2],0.0,
                T[2,1],T[2,2],0.0,
                0.0,0.0,0.0)
end

function T₂toT₃_op(T::SymFourthOrderTensorValue{2, Float64, 9})
    return SymFourthOrderTensorValue(T[1,1,1,1],T[1,1,1,2],0.0,T[1,1,2,2],0.0,0.0,
                              T[1,2,1,1],T[1,2,1,2],0.0,T[1,2,2,2],0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,
                              T[2,2,1,1],T[2,2,1,2],0.0,T[2,2,2,2],0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0)
end

# Convert 3D tensor to 2D
function T₃toT₂(T)
    return T₃toT₂_op∘T
end



function AntiCommuter(A::SymTensorValue{3,Float64,6})
    return SymFourthOrderTensorValue(2*A[1,1], A[1,2], A[1,3], 0, 0, 0,
                            A[1,2], 0.5*A[1,1] + 0.5*A[2,2], 0.5*A[2,3], A[1,2], 0.5*A[1,3], 0, 
                            A[1,3], 0.5*A[2,3], 0.5*A[1,1] + 0.5*A[3,3], 0, 0.5*A[1,2], A[1,3],
                            0, A[1,2], 0, 2*A[2,2], A[2,3], 0,
                            0, 0.5*A[1,3], 0.5*A[1,2], A[2,3], 0.5*A[2,2] + 0.5*A[3,3],
                            A[2,3], 0, 0, A[1,3], 0, A[2,3], 2*A[3,3])
end

function AntiCommuter(A::SymTensorValue{2,Float64,3})
    return SymFourthOrderTensorValue(2*A[1,1], A[1,2], 0,
                                A[1,2], 0.5*A[1,1] + 0.5*A[2,2], A[1,2],
                                0, A[1,2], 2*A[2,2])
end



function T₃toT₂_op(T::SymTensorValue{3, Float64, 6})
    return SymTensorValue(T[1,1],T[1,2],T[2,2])
end

function T₃toT₂_op(T::TensorValue{3,3})
    return TensorValue(T[1,1],T[1,2],
                T[2,1],T[2,2])
end

function T₃toT₂_op(T::SymFourthOrderTensorValue{3, Float64, 36})
    return SymFourthOrderTensorValue(T[1,1,1,1],T[1,1,1,2],T[1,1,2,2],
                              T[1,2,1,1],T[1,2,1,2],T[1,2,2,2],
                              T[2,2,1,1],T[2,2,1,2],T[2,2,2,2])
end

function inv4(T::SymFourthOrderTensorValue)
    P = Tensor2Mandel(T)

    # V = inv(P'⋅P + 1e-6*one(P))⋅P'
    V = inv(P)
    return Mandel2Tensor(V)
    
end


