struct SSA
    B::Float64
    n::Float64
    ρ::Float64
    ρsw::Float64
    g::Float64
    m::Float64
    α::Float64 # Angle of bed in x, degrees
    b # base function
    β # Basal traction
    reffeᵤ # Reference FE for velocity
    V # Test FESpace
    U # Trial FESpace
    Ω # Triangulation
    dΩ # Measure
    Γ # Boundary triangulation
    dΓ # Measure
    nΓ # Normal vector
    solver


    function SSA(B,n,β,m,dtags,dmasks,ubcs,solver,labels,order=2,degree=4)
        reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

        V = TestFESpace(model,reffeᵤ,labels=labels,dirichlet_tags=dtags,dirichlet_masks=dmasks,conformity=:H1)
        U = TrialFESpace(V,ubcs)
        Ω = Triangulation(model)
        dΩ = Measure(Ω,degree)

        Γ =  BoundaryTriangulation(model,tags="calving_front")
        dΓ = Measure(Γ,degree)
        nΓ = get_normal_vector(Γ)

        ρ = 9.138e-19; g = 9.7692e15; # Elmer units




        return new(B,n,ρ,g,β,m,reffeᵤ,V,U,Ω,dΩ,solver)
    end

    # function SSA(B,n,β,m,solver,order=2,degree=4)
    #     reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

    #     V = TestFESpace(model,reffeᵤ,conformity=:H1)
    #     U = TrialFESpace(V)
    #     Ω = Triangulation(model)
    #     dΩ = Measure(Ω,degree)

    #     ρ = 9.138e-19; g = 9.7692e15; # Elmer units

    #     return new(B,n,ρ,g,β,m,reffeᵤ,V,U,Ω,dΩ,solver)
    # end


end



function solve_ssa(μ,uh,s,h,ssa::SSA)
    # Define integration measure
    dΩ = ssa.dΩ

    B = ssa.B
    n = ssa.n
    p = 1 + 1/n
    δ = one(SymTensorValue(0.0,0.0,0.0))
    η(ε) = (0.5*ε⊙ε + 0.5*tr(ε)^2 + 1e-9)^((p-2)/2)


    b = ssa.b


    μ_2D = T₃toT₂(μ)
    μ_ijzz = extract_zz∘(μ)
    N(ε) = (η∘ε)*(μ_2D⊙ε - μ_ijzz*tr(ε))

    # N(ε) = (η∘ε)*ε

    T(N) = N + tr(N)*δ

    β = ssa.β
    f = -ssa.ρ*ssa.g*h*(∇(s) + VectorValue(tand(ssa.α),0.0))
    F = 0.5*(1-ssa.ρ/ssa.ρsw)*ssa.ρ*ssa.g*h*h*ssa.nΓ

    q = 1/ssa.m - 1 
    τb(u) = β*normc(u)^q*u
    dτb(du,u) = β*(1+q)*normc(u)^q*du

    a(u,v) = ∫( B*h*(T(N(ε(u)))⊙ε(v)) + StreamShelf∘(b,h)*(τb∘u)⋅v)dΩ 
    l((v)) = ∫( f⋅v )dΩ + ∫( F⋅v )dΓ

    res(u,v) = a(u,v) - l(v)
    jac(u,du,v) = ∫( B*h*(T(dτ(ε(du),ε(u)))⊙ε(v)) + StreamShelf∘(b,h)*(dτb∘(du,u))⋅v )dΩ
    
 
    op = FEOperator(res,jac,ssa.U,ssa.V)
    solve!(uh,solver,op)

    return uh
end


function solve_thickness(model,hn,uh,dt,acc=0,order=1,κ=1e-4,solver=LUSolver())

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)

    reffeh = ReferenceFE(lagrangian,Float64,order)
    G = TestFESpace(model,reffeh)
    H = TrialFESpace(G)

    a(h,g) = ∫( h*g + dt*(∇⋅(h*uh))*g + dt*κ*(∇(h)⋅∇(g)) )dΩ
    b(g) = ∫( (hn + dt*acc )*g )dΩ  #

    op = AffineFEOperator(a,b,H,G)
    solve!(hn,solver,op)
    return hn
end

function extract_zz(μ::SymFourthOrderTensorValue{3,Float64,36})
    return SymTensorValue(μ[1,1,3,3],μ[1,2,3,3],
                                       μ[2,2,3,3])

end


function normc(u::VectorValue{2,Float64})
    return sqrt(u[1]^2 + u[2]^2 + 1e-9)
end

function StreamShelf(b,h)
    # Returns 1 if stream, 0 if shelf
    if b + h > (1-ρ/ρsw)*h
        return 1
    else
        return 0
    end
end

