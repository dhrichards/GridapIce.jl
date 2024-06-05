

function transform_gradient(z)
    # invJt = inv∇(z)
    invJt = inv∘∇vec(z)
    ∇x(u) = invJt⋅∇(u)
    return ∇x
end

function transform_integral(z)
    ∫_Ωx(a) = ∫(a*(det∘∇vec(z)))
    return ∫_Ωx
end


∇vec(f) = Operation(∇vec_op)(∇(f))

function evaluate!(cache,::Broadcasting{typeof(∇vec)},f)
    Broadcasting(Operation(∇vec_op))(Broadcasting(∇)(f))
end

function ∇vec_op(∇z::VectorValue{2})
    return TensorValue(1.0,0.0,∇z[1],∇z[2])
end

function ∇vec_op(∇z::VectorValue{3})
    return TensorValue(1.0,0.0,0.0,
                       0.0,1.0,0.0,
                          ∇z[1],∇z[2],∇z[3])
end