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

function Transform_op(∇z::VectorValue{2,Float64},h)
    ∂ζ∂x = (1/h)*∇z[1]
    return TensorValue(1.0, ∂ζ∂x, 0, 1/h)
end

function Transform_op(∇z::VectorValue{2,Float64},h::Float64,L::Float64)
    ∂ζ∂x = (1/(h*L))*∇z[1]
    return TensorValue(1.0, ∂ζ∂x, 0, 1/h)
end

function Transform(z,h)
    return Transform_op∘(∇(z),h)
end

function Transform(z,h,L)
    return Transform_op∘(∇(z),h,L)
end
# function Transform(ζ,b,s)
#     h = s - b
#     return Transform_op∘(ζ,∇(b),∇(s),h
# end

# function Transform(ζ,b,s,L)
#     h = s - b
#     return Transform_op∘(ζ,∇(b),∇(s),h,L)
# end


function normal_vector(s)
    return normal_vector_op∘(∇(s))
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
    
    F = FESpace(Triangulation(model),ReferenceFE(lagrangian,Float64,2),conformity=:L2)

    
    s = interpolate_everywhere(problem.s,F)
    b = interpolate_everywhere(problem.b,F)

    ζ = interpolate_everywhere(x->x[2],F)

    return s,b,ζ
end

function get_sb_fields_unit(problem,model)
    
    F = FESpace(Triangulation(model),ReferenceFE(lagrangian,Float64,2))    
    
    s = interpolate_everywhere(x->problem.s(x*problem.L),F)
    b = interpolate_everywhere(x->problem.b(x*problem.L),F)

    ζ = interpolate_everywhere(x->x[2],F)
    L = interpolate_everywhere(problem.L,F)

    return s,b,ζ,L
end



# function transform_gradient(ϕ)
#     invJt = inv∘∇(ϕ)
#     ∇x(u) = invJt⋅∇(u)
#     # ∇ꜝ(ϕ,u) = ϕ'⋅∇(u)

#     return ∇x
# end

function transform_gradient(z)
    invJt = inv∇(z)
    ∇x(u) = invJt⋅∇(u)
    return ∇x
end

inv∇(f) = Operation(inv_op)(∇(f))

function evaluate!(cache,::Broadcasting{typeof(inv∇)},f)
    Broadcasting(Operation(inv_op))(Broadcasting(∇)(f))
end

function inv_op(∇z::VectorValue{2})
    a = 1.0
    b = 0.0
    c = ∇z[1]
    d = ∇z[2]
    det = a*d - b*c
    return TensorValue(d/det, -b/det, -c/det, a/det)
end


