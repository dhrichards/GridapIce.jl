function removeZ(x::VectorValue{2,Float64})
    return VectorValue(x[1])
end

function removeZ(x::VectorValue{3,Float64})
    return VectorValue(x[1],x[2])
end

function Zonly(x::VectorValue)
    return Float64(x[end])
end


function solve_surface(model,sh,dt,uh,M,solver,❄️=0,order=2,κ=1e-4)

    reffeh = ReferenceFE(lagrangian,Float64,order)

    Γ =  BoundaryTriangulation(model,tags="top")
    dΓ = Measure(Γ,2*order)

    V = TestFESpace(Γ,reffeh)
    S = TrialFESpace(V)

    M∇ = Transform_∇(M)

    us = removeZ∘uh
    w = Zonly∘uh
    
    # Solve for new surface
    # a(s,v) = ∫( s*v + dt*us⋅(removeZ∘(M∇(s)))*v + dt*κ*(M∇(s)⋅M∇(v)) )dΓ #
    b(v) = ∫( (sh + dt*(w + ❄️) )*v )dΓ  #
    a(s,v) = ∫( s*v )dΓ
    # b(v) = ∫(sh*v )dΓ

    op = AffineFEOperator(a,b,S,V)
    sh = solve(solver,op)


    return sh
end


function solve_surface3d(model,sh,dt,uh,M,h,solver,❄️=0,order=2,κ=1e-4)

    reffeh = ReferenceFE(lagrangian,Float64,order)

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)

    V = TestFESpace(Ω,reffeh)
    S = TrialFESpace(V)

    M∇ = Transform_∇(M)

    us = removeZ∘uh
    w = Zonly∘uh
    
    # Solve for new surface
    a(s,v) = ∫( h*(s*v + dt*us⋅(removeZ∘(M∇(s)))*v + dt*κ*(M∇(s)⋅M∇(v))) )dΩ
    b(v) = ∫( h*(sh + dt*(w + ❄️) )*v )dΩ  #
    
    op = AffineFEOperator(a,b,S,V)
    sh = solve(solver,op)


    return sh
end



function impose_surface(s,model,hx,solver)
    # Impose surface on boundary into the mesh
    
    order = 2
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,
                reffe,
                conformity=:L2)

    U = TrialFESpace(V)

    Ω = Triangulation(model)
    Γs = BoundaryTriangulation(model,tags="top")
    Λ = SkeletonTriangulation(model)
    degree = 2*order

    dΩ = Measure(Ω,degree)
    dΓs = Measure(Γs,degree)
    dΛ = Measure(Λ,degree)

    n_Γs = get_normal_vector(Γs)
    n_Λ = get_normal_vector(Λ)

    a_Ω(u,v) = ∫( ∇(v)⊙∇(u) )dΩ
    l_Ω(v) = ∫( v*0 )dΩ

    γ = order*(order+1)
    a_Γs(u,v) = ∫( - v*(∇(u)⋅n_Γs) - (∇(v)⋅n_Γs)*u + (γ/hx)*v*u )dΓs
    l_Γs(v)   = ∫(                - (∇(v)⋅n_Γs)*s + (γ/hx)*v*s )dΓs

    a_Λ(u,v) = ∫( - jump(v*n_Λ)⊙mean(∇(u))
              - mean(∇(v))⊙jump(u*n_Λ)
              + (γ/hx)*jump(v*n_Λ)⊙jump(u*n_Λ) )dΛ

    a(u,v) = a_Ω(u,v) + a_Γs(u,v)  + a_Λ(u,v)
    l(v) = l_Ω(v) + l_Γs(v) 

    op = AffineFEOperator(a, l, U, V)
    uh = solve(solver,op)

    return uh
end






