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





function solve_up(sol,μ,η,z,stk::Stokes)
    

    dΩ = Measure(stk.Ω,stk.degree)
    dΓw = Measure(stk.Γw,stk.degree)

    f = stk.f; β = stk.β

    
    ∇x = transform_gradient(z)
    εx(u) = symmetric_part(∇x(u))
    divx(u) = tr(∇x(u))
    ∫_Ωx = transform_integral(z)
        
    res((u,p),(v,q)) = ∫_Ωx( η∘(εx(u))*(μ⊙εx(u))⊙εx(v) - divx(u)*q - divx(v)*p - v⋅f )dΩ #+ ∫( β*u⋅v )dΓw # maybe add minus sign to div u for
    op = FEOperator(res,stk.X,stk.Y)

    sol, = solve!(sol,stk.solver,op)

    return sol, res
end

function solve_up_linear(μ,z,stk::Stokes)

    U,P = stk.X; V,Q = stk.Y
    
    mfs = Gridap.MultiField.BlockMultiFieldStyle()
    X = MultiFieldFESpace([U,Q];style=mfs)
    Y = MultiFieldFESpace([V,Q];style=mfs)

    dΩ = Measure(stk.Ω,stk.degree)
    dΓw = Measure(stk.Γw,stk.degree)

    f = stk.f; β = stk.β

    ∇x = transform_gradient(z)
    ∫_Ωx = transform_integral(z)
    εx(u) = symmetric_part(∇x(u))
    divx(u) = tr(∇x(u))

    a((u,p),(v,q)) = ∫_Ωx(  ((μ⊙εx(u))⊙εx(v) - divx(u)*q - divx(v)*p) )dΩ #+ ∫( β*u⋅v )dΓw 
    b((v,q)) = ∫_Ωx( v⋅f )dΩ

    op = AffineFEOperator(a,b,X,Y)
    # sol, = solve!(sol,stk.solver,op)

    
    uh = block_solve(op,1.0e9,dΩ,U,Q)
    return uh
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

function block_solve(op,α,dΩ,U,Q)
    A, bb = get_matrix(op), get_vector(op);
    Auu = blocks(A)[1,1]
  
    solver_u = LUSolver()
    solver_p = LUSolver()
    # solver_p = CGSolver(RichardsonSmoother(JacobiLinearSolver(),10,0.2);maxiter=20,atol=1e-14,rtol=1.e-6,verbose=false)
  
  
  
    diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,Q,Q)]
  
    blockss = map(CartesianIndices((2,2))) do I
      (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
    end
  
    coeffs = [1.0 1.0;
              0.0 1.0]
    P = BlockTriangularSolver(blockss,[solver_u,solver_p],coeffs,:upper)
    solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6) #,verbose=i_am_main(parts))
    # solver = GMRESSolver(20,P;atol=1e-14,rtol=1.e-6) #,verbose=i_am_main(parts))
    ns = numerical_setup(symbolic_setup(solver,A),A)
  
    x = Gridap.Algebra.allocate_in_domain(A); fill!(x,0.0)
    solve!(x,ns,bb)
  
    uh = FEFunction(U,x)
    return uh
  end