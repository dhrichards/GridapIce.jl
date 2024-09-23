# using PyCall
# specfab = pyimport("specfabpy")
# sf = specfab.specfab


function makeM(W,C,λSR,f0)
    return makeM_op∘(W,C,λSR,f0)
end

function makeM_op(W::TensorValue,C::SymTensorValue{2,Float64,3},λSR::Float64,f0::VectorValue)
    C = T₂toT₃_op(C)
    W = T₂toT₃_op(W)
    return makeM_op(W,C,λSR,f0)
end

function makeM_op(W::TensorValue,C::SymTensorValue{3,Float64,6},λSR::Float64,f0::VectorValue)

    f0 = f0.data
    # put C and W in arrays
    C = [C[1,1] C[1,2] C[1,3];
            C[2,1] C[2,2] C[2,3];
            C[3,1] C[3,2] C[3,3]]

    W = [W[1,1] W[1,2] W[1,3];
            W[2,1] W[2,2] W[2,3];
            W[3,1] W[3,2] W[3,3]]

    
    M_LROT = sf.M_LROT(f0,C,W,1,0)
    M_REG = sf.M_REG(f0,C)
    M_RRX = sf.M_CDRX(f0)

    M = TensorValue(M_LROT  + M_REG + λSR*M_RRX)
    return M
end


function a2calc(f::Union{MultiFieldFEFunction,GridapDistributed.DistributedMultiFieldFEFunction})
    fr,fi = f
    return a2calc_op∘(fr,fi)
end

function a4calc(f::Union{MultiFieldFEFunction,GridapDistributed.DistributedMultiFieldFEFunction})
    fr,fi = f
    return a4calc_op∘(fr,fi)
end


function a2calc(f0::CellField)
    return a2calc_op∘(f0)
end

function a4calc(f0::CellField)
    return a4calc_op∘(f0)
end

function a2calc_op(fr::VectorValue,fi::VectorValue)
    f = fr + im*fi
    return a2calc_op(f)
end

function a4calc_op(fr::VectorValue,fi::VectorValue)
    f = fr + im*fi
    return a4calc_op(f)
end

function a2calc_op(f0::VectorValue)
    a2 = sf.a2(f0.data)
    return SymTensorValue(a2[1,1],a2[1,2],a2[1,3],
                                  a2[2,2],a2[2,3],
                                          a2[3,3])
end

function a4calc_op(f0::VectorValue)
    a4 = sf.a4(f0.data)
    return SymFourthOrderTensorValue(a4[1,1,1,1],a4[1,1,1,2],a4[1,1,1,3],a4[1,1,2,2],a4[1,1,2,3],a4[1,1,3,3],
                                        a4[1,2,1,1],a4[1,2,1,2],a4[1,2,1,3],a4[1,2,2,2],a4[1,2,2,3],a4[1,2,3,3],
                                        a4[1,3,1,1],a4[1,3,1,2],a4[1,3,1,3],a4[1,3,2,2],a4[1,3,2,3],a4[1,3,3,3],
                                        a4[2,2,1,1],a4[2,2,1,2],a4[2,2,1,3],a4[2,2,2,2],a4[2,2,2,3],a4[2,2,3,3],
                                        a4[2,3,1,1],a4[2,3,1,2],a4[2,3,1,3],a4[2,3,2,2],a4[2,3,2,3],a4[2,3,3,3],
                                        a4[3,3,1,1],a4[3,3,1,2],a4[3,3,1,3],a4[3,3,2,2],a4[3,3,2,3],a4[3,3,3,3])
end

function a2calc2d(f::Union{MultiFieldFEFunction,GridapDistributed.DistributedMultiFieldFEFunction})
    fr,fi = f
    return a2calc2d_op∘(fr,fi)
end

function a4calc2d(f::Union{MultiFieldFEFunction,GridapDistributed.DistributedMultiFieldFEFunction})
    fr,fi = f
    return a4calc2d_op∘(fr,fi)
end

function a2calc2d(f::CellField)
    return a2calc2d_op∘(f)
end
function a4calc2d(f::CellField)
    return a4calc2d_op∘(f)
end

function a2calc2d_op(fr::VectorValue,fi::VectorValue)
    f = fr + im*fi
    return a2calc2d_op(f)
end

function a4calc2d_op(fr::VectorValue,fi::VectorValue)
    f = fr + im*fi
    return a4calc2d_op(f)
end

function a2calc2d_op(f0::VectorValue)
    a2 = sf.a2(f0.data)
    return SymTensorValue(a2[1,1],a2[1,2],
                                  a2[2,2])
end

function a4calc2d_op(f0::VectorValue)
    a4 = sf.a4(f0.data)
    return SymFourthOrderTensorValue(a4[1,1,1,1],a4[1,1,1,2],a4[1,1,2,2],
                                        a4[1,2,1,1],a4[1,2,1,2],a4[1,2,2,2],
                                        a4[2,2,1,1],a4[2,2,1,2],a4[2,2,2,2])
end
                                     
