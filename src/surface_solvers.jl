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


function solve_surface3d(model,sh,dt,uh,ϕ,h,solver,❄️=0,order=2,κ=1e-4)

    reffeh = ReferenceFE(lagrangian,Float64,order)

    
    invJt = inv∘∇(ϕ)
    ∇ꜝ(u) = invJt⋅∇(u)
    # ∇ꜝ(ϕ,u) = ϕ'⋅∇(u)
    εꜝ(u) = symmetric_part(∇ꜝ(u))
    divꜝ(u) = tr(∇ꜝ(u))

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)

    V = TestFESpace(Ω,reffeh)
    S = TrialFESpace(V)

    us = removeZ∘uh
    w = Zonly∘uh
    
    # Solve for new surface
    a(s,v) = ∫( h*(s*v + dt*us⋅(removeZ∘(∇ꜝ(s)))*v + dt*κ*(∇ꜝ(s)⋅∇ꜝ(v))) )dΩ
    b(v) = ∫( h*( dt*(w + ❄️) )*v )dΩ  #
    
    op = AffineFEOperator(a,b,S,V)
    Δsh = solve(solver,op)


    return Δsh
end


function solve_mesh(b,s,model,hx,solver)
    
    order = 2
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,
                reffe,
                conformity=:L2)

    U = TrialFESpace(V)

    Ω = Triangulation(model)
    Γs = BoundaryTriangulation(model,tags="top")
    Γb = BoundaryTriangulation(model,tags="bottom")
    Λ = SkeletonTriangulation(model)
    degree = 2*order

    dΩ = Measure(Ω,degree)
    dΓs = Measure(Γs,degree)
    dΓb = Measure(Γb,degree)
    dΛ = Measure(Λ,degree)

    n_Γs = get_normal_vector(Γs)
    n_Γb = get_normal_vector(Γb)
    n_Λ = get_normal_vector(Λ)

    a_Ω(u,v) = ∫( ∇(v)⊙∇(u) )dΩ
    l_Ω(v) = ∫( v*0 )dΩ

    γ = order*(order+1)
    a_Γs(u,v) = ∫( - v*(∇(u)⋅n_Γs) - (∇(v)⋅n_Γs)*u + (γ/hx)*v*u )dΓs
    l_Γs(v)   = ∫(                - (∇(v)⋅n_Γs)*s + (γ/hx)*v*s )dΓs

    a_Γb(u,v) = ∫( - v*(∇(u)⋅n_Γb) - (∇(v)⋅n_Γb)*u + (γ/hx)*v*u )dΓb
    l_Γb(v)   = ∫(                - (∇(v)⋅n_Γb)*b + (γ/hx)*v*b )dΓb





    a_Λ(u,v) = ∫( - jump(v*n_Λ)⊙mean(∇(u))
              - mean(∇(v))⊙jump(u*n_Λ)
              + (γ/hx)*jump(v*n_Λ)⊙jump(u*n_Λ) )dΛ


    a(u,v) = a_Ω(u,v) + a_Γs(u,v) + a_Γb(u,v) + a_Λ(u,v)
    l(v) = l_Ω(v) + l_Γs(v) + l_Γb(v)

    op = AffineFEOperator(a, l, U, V)
    uh = solve(solver,op)

    return uh
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





function impose(sh,model,solver)
    # Impose surface on boundary into the mesh
    
    order = 2
    reffe = ReferenceFE(lagrangian,Float64,order)

    V = TestFESpace(model,reffe)
    U = TrialFESpace(V)

    Ω = Triangulation(model)
    Γs = BoundaryTriangulation(model,tags="top")
    Γb = BoundaryTriangulation(model,tags="bottom")
    degree = 2*order

    dΩ = Measure(Ω,degree)
    dΓs = Measure(Γs,degree)
    dΓb = Measure(Γb,degree)

    ∇z(f) = ∇(f)⋅VectorValue(0,1.0)

    a(s,v) = ∫( s*∇z(v) )dΩ + ∫( s*v )dΓb
    l(v) = ∫( sh*v )dΓs

    op = AffineFEOperator(a, l, U, V)
    sh = solve(solver,op)

    return sh
end


function solve_surface_combined(model,zh,b,dt,uh,solver,❄️=0,order=2,κ=1e-4)

    reffeh = ReferenceFE(lagrangian,Float64,order)

    Ω = Triangulation(model)
    Γ =  BoundaryTriangulation(model,tags="top")
    dΩ = Measure(Ω,2*order)
    dΓ = Measure(Γ,2*order)

    V = TestFESpace(Ω,reffeh,dirichlet_tags="bottom")
    Z = TrialFESpace(V,b)

    ∇x = transform_gradient(zh)
    ∇s(u) = removeZ∘∇x(u)
    h = Zonly∘(∇(zh))

    us = removeZ∘uh
    w = Zonly∘uh
    
    # Solve for new surface
    aΓ(z,v) = ∫( (z*v + dt*us⋅∇s(z)*v)*h )dΓ
    aΩ(z,v) = ∫( ∇s(z)⊙∇s(v) )dΩ

    lΓ(v) = ∫( (zh + dt*(w + ❄️) )*v*h )dΓ  #

    a(z,v) = aΓ(z,v) + aΩ(z,v)
    l(v) = lΓ(v)
    

    op = AffineFEOperator(a,l,Z,V)
    zh = solve(solver,op)

    return zh

end
