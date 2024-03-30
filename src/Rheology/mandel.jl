function Tensor2Mandel(T::SymFourthOrderTensorValue{2,Float64,9})
    return TensorValue(T[1,1,1,1],T[1,1,2,2],√2*T[1,1,1,2],
                    T[2,2,1,1],T[2,2,2,2],√2*T[2,2,1,2],
                    √2*T[1,2,1,1],√2*T[1,2,2,2],2*T[1,2,1,2])
end


function Tensor2Mandel(T::SymFourthOrderTensorValue{3,Float64,36})
   return TensorValue(  T[1,1,1,1],T[1,1,2,2],T[1,1,3,3],√2*T[1,1,2,3],√2*T[1,1,1,3],√2*T[1,1,1,2],
                            T[2,2,1,1],T[2,2,2,2],T[2,2,3,3],√2*T[2,2,2,3],√2*T[2,2,1,3],√2*T[2,2,1,2],
                            T[3,3,1,1],T[3,3,2,2],T[3,3,3,3],√2*T[3,3,2,3],√2*T[3,3,1,3],√2*T[3,3,1,2],
                            √2*T[2,3,1,1],√2*T[2,3,2,2],√2*T[2,3,3,3],2*T[2,3,2,3],2*T[2,3,1,3],2*T[2,3,1,2],
                            √2*T[1,3,1,1],√2*T[1,3,2,2],√2*T[1,3,3,3],2*T[1,3,2,3],2*T[1,3,1,3],2*T[1,3,1,2],
                            √2*T[1,2,1,1],√2*T[1,2,2,2],√2*T[1,2,3,3],2*T[1,2,2,3],2*T[1,2,1,3],2*T[1,2,1,2])
end

function Tensor2Mandel(T::SymTensorValue{2,Float64,3})
    return VectorValue(T[1,1],T[2,2],√2*T[1,2])
end

function Tensor2Mandel(T::SymTensorValue{3,Float64,6})
    return VectorValue(T[1,1],T[2,2],T[3,3],√2*T[2,3],√2*T[1,3],√2*T[1,2])
end


function Mandel2Tensor(V::TensorValue{3,3,Float64,9})
    V = V'
    return SymFourthOrderTensorValue(V[1,1],V[1,3]/√2,V[1,2],
                                        V[3,1]/√2,V[3,3]/2,V[3,2]/√2,
                                        V[2,1],V[2,3]/√2,V[2,2])
end

function Mandel2Tensor(V::TensorValue{6,6,Float64,36})
    V = V' # weird gridap indexing T[1,2] = T[2,1]
    return SymFourthOrderTensorValue(   V[1,1]   ,V[1,6]/√2 ,V[1,5]/√2 ,V[1,2]    ,V[1,4]/√2, V[1,3],
                                        V[6,1]/√2,V[6,6]/2  ,V[6,5]/2  ,V[6,2]/√2 ,V[6,4]/2,  V[6,3]/√2,
                                        V[5,1]/√2,V[5,6]/2  ,V[5,5]/2  ,V[5,2]/√2 ,V[5,4]/2,  V[5,3]/√2,
                                        V[2,1]   ,V[2,6]/√2 ,V[2,5]/√2 ,V[2,2]    ,V[2,4]/√2, V[2,3],
                                        V[4,1]/√2,V[4,6]/2  ,V[4,5]/2  ,V[4,2]/√2 ,V[4,4]/2,  V[4,3]/√2,
                                        V[3,1]   ,V[3,6]/√2 ,V[3,5]/√2 ,V[3,2]    ,V[3,4]/√2, V[3,3])
end

function Mandel2Tensor(V::VectorValue{3,Float64})
    return SymTensorValue(V[1],V[3]/√2,V[2])
end

function Mandel2Tensor(V::VectorValue{6,Float64})
    return SymTensorValue(V[1],V[6]/√2,V[5]/√2,V[2],V[4]/√2,V[3])
end
