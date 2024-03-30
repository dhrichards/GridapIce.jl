struct Stokes
    D::Int # Dimension
    Td::Int # Number of tensor components
    order:: Int # Finite element order
    degree::Int # Quadrature degree
    X :: Union{GridapDistributed.DistributedMultiFieldFESpace,MultiFieldFESpace}
    Y :: Union{GridapDistributed.DistributedMultiFieldFESpace,MultiFieldFESpace}
    Ω::Union{Triangulation,GridapDistributed.DistributedTriangulation}
    Γw::Union{BoundaryTriangulation,GridapDistributed.DistributedTriangulation}
    Γs::Union{BoundaryTriangulation,GridapDistributed.DistributedTriangulation}
    β # Basal traction
    f::VectorValue
    solver

    function Stokes(model,problem,solver,order=2,degree=4)
        D = problem.D
        Td = (D==2 ? 3 : 6)
        
        ρ = 9.138e-19; g = 9.7692e15
        if D == 2
            f = VectorValue(ρ*g*sind(problem.α),-ρ*g*cosd(problem.α))
        else
            f = VectorValue(ρ*g*sind(problem.α),0.0,-ρ*g*cosd(problem.α))
        end

        reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
        reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)

        # Define test FESpaces
        V = TestFESpace(model,reffeᵤ,dirichlet_tags=problem.dtags,dirichlet_masks=problem.dmasks,conformity=:H1)
        Q = TestFESpace(model,reffeₚ,conformity=:L2,constraint=:zeromean)
        Y = MultiFieldFESpace([V,Q])

        # Define trial FESpaces from Dirichlet values
        U = TrialFESpace(V,problem.ubcs)
        P = TrialFESpace(Q)
        X = MultiFieldFESpace([U,P])

        # Define triangulation and integration measure
        Ω = Triangulation(model)
        Γw =  BoundaryTriangulation(model,tags="bottom")
        Γs =  BoundaryTriangulation(model,tags="top")

        return new(D,Td,order,degree,X,Y,Ω,Γw,Γs,problem.β,f,solver)
    end


end


function solve_up(sol,τ,M,h,stk::Stokes)
    M∇ = Transform_∇(M)
    Mε(u) = 0.5*(M∇(u) + M∇(u)')
    Mdiv(u) = tr(M∇(u))

    dΩ = Measure(stk.Ω,stk.degree)
    dΓw = Measure(stk.Γw,stk.degree)

    f = stk.f; β = stk.β

    res((u,p),(v,q)) = ∫(  h*(τ(Mε(u))⊙Mε(v) - Mdiv(u)*q - Mdiv(v)*p - v⋅f) )dΩ + ∫( β*u⋅v )dΓw # maybe add minus sign to div u for
    op = FEOperator(res,stk.X,stk.Y)

    sol, = solve!(sol,stk.solver,op)

    return sol, res
end

function solve_up_linear(sol,τ,M,h,stk::Stokes)

    M∇ = Transform_∇(M)
    Mε(u) = 0.5*(M∇(u) + M∇(u)')
    Mdiv(u) = tr(M∇(u))

    dΩ = Measure(stk.Ω,stk.degree)
    dΓw = Measure(stk.Γw,stk.degree)

    f = stk.f; β = stk.β

    res((u,p),(v,q)) = ∫(  h*(τ(Mε(u))⊙Mε(v) - Mdiv(u)*q - Mdiv(v)*p - v⋅f) )dΩ + ∫( β*u⋅v )dΓw # maybe add minus sign to div u for
    a((u,p),(v,q)) = ∫(  h*(τ(Mε(u))⊙Mε(v) - Mdiv(u)*q - Mdiv(v)*p) )dΩ + ∫( β*u⋅v )dΓw 
    b((v,q)) = ∫( v⋅f )dΩ

    op = AffineFEOperator(a,b,stk.X,stk.Y)
    sol, = solve!(sol,stk.solver,op)

    return sol, res
end



function FSSAsolve(sol⁺,res1,dt,s,stk)

    
    dΓs = Measure(stk.Γs,stk.degree)
    n_s = normal_vector(s)
    f = stk.f

    res2((u,p),(v,q)) = res1((u,p),(v,q)) - ∫( dt*(u⋅n_s)*(f⋅v))dΓs #FSSA correction


    op = FEOperator(res2,stk.X,stk.Y)
    sol, = solve!(sol⁺,stk.solver,op)

    return sol⁺
end

