# ∂u_i/∂x_j = M_jk ∂u_i/∂x*_k

function Transform_∇(M)
    M∇(u) = M'⋅∇(u)
    return M∇
end

Mε(u) = 0.5*(M∇(u) + M∇(u)')
Mdiv(u) = tr(M∇(u))


function Transform_op(ζ,∇b::VectorValue{2,Float64},∇s::VectorValue{2,Float64},h::Float64)
    ∂ζ∂x = (-1/h)*((1-ζ)*∇b[1] +  ζ*∇s[1])
    return TensorValue(1.0, ∂ζ∂x, 0, 1/h)
end

function Transform_op(ζ,∇b::VectorValue{3,Float64},∇s::VectorValue{3,Float64},h::Float64)
    ∂ζ∂x = (-1/h)*((1-ζ)*∇b[1] +  ζ*∇s[1])
    ∂ζ∂y = (-1/h)*((1-ζ)*∇b[2] +  ζ*∇s[2])
    return TensorValue( 1.0, 0.0, ∂ζ∂x,
                        0.0, 1.0, ∂ζ∂y,
                        0.0, 0.0,  1/h)
end

function Transform_op(ζ,∇b::VectorValue{2,Float64},∇s::VectorValue{2,Float64},h::Float64,L::Float64)
    ∂ζ∂x = (-1/(h*L))*((1-ζ)*∇b[1] +  ζ*∇s[1])
    return TensorValue(1/L, ∂ζ∂x, 0, 1/h)
end


function Transform(ζ,b,s)
    h = s - b
    return Transform_op∘(ζ,∇(b),∇(s),h)
end

function Transform(ζ,b,s,L)
    h = s - b
    return Transform_op∘(ζ,∇(b),∇(s),h,L)
end

function normal_vector(s)
    return normal_vector_op(∇(s))
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


function get_sb_fields(problem,model)
    
    F = FESpace(Triangulation(model),ReferenceFE(lagrangian,Float64,2))    
    
    s = interpolate_everywhere(problem.s,F)
    b = interpolate_everywhere(problem.b,F)

    ζ = interpolate_everywhere(x->x[2],F)
    L = interpolate_everywhere(problem.L,F)

    return s,b,ζ,L
end