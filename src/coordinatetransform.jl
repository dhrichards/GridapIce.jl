

function transform_gradient(z,L)
    # invJt = inv∇(z)
    invJt = inv∘∇vec(z,L)
    ∇x(u) = invJt⋅∇(u)
    return ∇x
end

function transform_gradient(z)
    invJt = inv∘∇vec(z)
    ∇x(u) = invJt⋅∇(u)
    return ∇x
end

function transform_integral(z)
    ∫_Ωx(a) = ∫(a*(det∘∇vec(z)))
    return ∫_Ωx
end
# function transform_gradient(z,L)
#     # invJt = inv∇(z)
#     ∇z = Lstretch(∇vec(z),L)
#     invJt = inv∘∇z
#     ∇x(u) = invJt⋅∇(u)
#     return ∇x
# end



function transform_integral(z,L)
    ∫_Ωx(a) = ∫(a*(det∘∇vec(z,L)))
    return ∫_Ωx
end


∇vec(f,L) = Operation(∇vec_op)(∇(f),L)

∇vec(f) = Operation(∇vec_op)(∇(f))

# function evaluate!(cache,::Broadcasting{typeof(∇vec)},f,L)
#     Broadcasting(Operation(∇vec_op))(Broadcasting(∇)(f),Broadcasting(L))
# end

function ∇vec_op(∇z::VectorValue{2},L)
    return TensorValue(L[1],0.0,
                        ∇z[1],∇z[2])
end

function ∇vec_op(∇z::VectorValue{3},L)
    return TensorValue(L[1],0.0,0.0,
                       0.0,L[2],0.0,
                          ∇z[1],∇z[2],∇z[3])
end


function ∇vec_op(∇z::VectorValue{2})
    return TensorValue(1.0,0.0,
                        ∇z[1],∇z[2])
end

function ∇vec_op(∇z::VectorValue{3})
    return TensorValue(1.0,0.0,0.0,
                       0.0,1.0,0.0,
                          ∇z[1],∇z[2],∇z[3])
end


# function ∇vec(z,L::Float64)
#     return SymTensorValue(L,0.0,0.0,1.0)⋅∇vec(z)
# end

# function ∇vec(z,L::Tuple)
#     return SymTensorValue(L[1],0.0,0.0,
#                                L[2],0.0,
#                                     1.0)⋅∇vec(z)
# end

# ∇vec(f,L) = Operation(∇vec_op)(∇(f),L)

# function evaluate!(cache,::Broadcasting{typeof(∇vec)},f,L)
#     Broadcasting(Operation(∇vec_op))(Broadcasting(∇)(f),L)
# end

# function ∇vec_op(∇z::VectorValue{2},L)
#     return TensorValue(L[1],0.0,∇z[1],∇z[2])
# end

# function ∇vec_op(∇z::VectorValue{3},L)
#     return TensorValue(L[1],0.0,0.0,
#                        0.0,L[2],0.0,
#                           ∇z[1],∇z[2],∇z[3])
# end

normal_vector(s) = Operation(normal_vector_op)(∇(s))

function evaluate!(cache,::Broadcasting{typeof(normal_vector)},s)
    Broadcasting(Operation(normal_vector_op))(Broadcasting(∇)(s))
end

function normal_vector_op(∇s::VectorValue{2,Float64})
    ∂s∂x = ∇s[1]
    return VectorValue(-∂s∂x/√(1+∂s∂x^2), 1/√(1+∂s∂x^2))
end

function normal_vector_op(∇s::VectorValue{3,Float64})
    ∂s∂x = ∇s[1]
    ∂s∂y = ∇s[2]
    return VectorValue(-∂s∂x/√(1+∂s∂x^2+∂s∂y^2), -∂s∂y/√(1+∂s∂x^2+∂s∂y^2), 1/√(1+∂s∂x^2+∂s∂y^2))
end