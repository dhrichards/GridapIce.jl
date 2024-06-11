function removeZ(x::VectorValue{2,Float64})
    return VectorValue(x[1])
end

function removeZ(x::VectorValue{3,Float64})
    return VectorValue(x[1],x[2])
end

function Zonly(x::VectorValue)
    return Float64(x[end])
end

function init_z(model,b,s,order=2)
    reffeh = ReferenceFE(lagrangian,Float64,order)

    Ω = Triangulation(model)

    V = TestFESpace(Ω,reffeh,dirichlet_tags="bottom")
    Z = TrialFESpace(V,b)

    z0 = x -> x[end]*(s(x)-b(x))+b(x) 

    zh = interpolate_everywhere(z0,Z)

    return zh
end


function solve_z!(zh,model,b,dt,uh,acc=0.0,solver=LUSolver(),order=2,κ=1e-4)

    reffeh = ReferenceFE(lagrangian,Float64,order)

    Ω = Triangulation(model)
    Γ =  BoundaryTriangulation(model,tags="top")
    dΩ = Measure(Ω,2*order)
    dΓ = Measure(Γ,2*order)

    V = TestFESpace(Ω,reffeh,dirichlet_tags="bottom")
    Z = TrialFESpace(V,b)

    ∇x = transform_gradient(zh)
    ∇s(u) = removeZ∘∇x(u)
    ∇z(u) = Zonly∘∇x(u)
    ∫_Ωx = transform_integral(zh)

    us = removeZ∘uh
    w = Zonly∘uh
    
    # Solve for new surface
    aΓ(z,v) = ∫_Ωx( z*v + dt*us⋅∇s(z)*v )dΓ
    aΩ(z,v) = ∫_Ωx( κ*∇x(z)⊙∇x(v) )dΩ

    lΓ(v) = ∫_Ωx( (zh + dt*(w + acc) )*v )dΓ  #

    a(z,v) = aΓ(z,v) + aΩ(z,v)
    l(v) = lΓ(v)
    

    op = AffineFEOperator(a,l,Z,V)
    
    solve!(zh,solver,op)
    # zh = solve(solver,op)
end
