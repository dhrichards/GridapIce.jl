struct Stokes
    D::Int # Dimension
    Lstretch::Union{CellField,GridapDistributed.DistributedCellField} 
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

    function Stokes(model,problem,solver,L,order=2,degree=4)
        D = problem.D
        Td = (D==2 ? 3 : 6)
        Lstretch = CellField(VectorValue(L),Triangulation(model))
        
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

        return new(D,Lstretch,Td,order,degree,X,Y,Ω,Γw,Γs,problem.β,f,solver)
    end


end





function solve_up(sol,fh,z,stk::Stokes,rheo)
    

    dΩ = Measure(stk.Ω,stk.degree)
    dΓw = Measure(stk.Γw,stk.degree)

    f = stk.f; β = stk.β

    
    ∇x = transform_gradient(z,stk.Lstretch)
    εx(u) = symmetric_part(∇x(u))
    divx(u) = tr(∇x(u)) 
    ∫_Ωx = transform_integral(z,stk.Lstretch)

    # n = 3.0; B = 100.0
    # η(ε) = B^(-1/n)*(0.5*ε⊙ε+1e-9)^((1-n)/(2*n))
    # dη(dε,ε) = B^(-1/n)*(1-n)/(2*n)*(0.5*ε⊙ε+1e-9)^((1-n)/(2*n)-1)*0.5*(dε⊙ε+ε⊙dε)
    

    # μ = SachsVisc(a2calc2d(fh),a4calc2d(fh))
    # τ(ε) = η∘(ε)*μ⊙ε
    # dτ(dε,ε) = dη∘(dε,ε)*(μ⊙ε) + η∘(ε)*(μ⊙dε)
    τ(ε) = rheo.η∘(ε)*rheo.με(fh,ε)
    dτ(dε,ε) = rheo.dη∘(dε,ε)*rheo.με(fh,ε) + rheo.η∘(ε)*rheo.d_με(fh,dε,ε)
        
    res((u,p),(v,q)) = ∫_Ωx( τ(εx(u))⊙εx(v) - divx(u)*q - divx(v)*p - v⋅f )dΩ #+ ∫_Ωx( β*u⋅v )dΓw # maybe add minus sign to div u for
    jac((u,p),(du,dp),(v,q)) = ∫_Ωx(dτ(εx(du),εx(u))⊙εx(v) - divx(v)*dp - divx(du)*q)dΩ


    op = FEOperator(res,jac,stk.X,stk.Y)

    sol, = solve!(sol,stk.solver,op)

    return sol, res
end

function solve_up_linear(μ,z,stk::Stokes,dt=0.0)

    U,P = stk.X; V,Q = stk.Y
    
    mfs = Gridap.MultiField.BlockMultiFieldStyle()
    X = MultiFieldFESpace([U,Q];style=mfs)
    Y = MultiFieldFESpace([V,Q];style=mfs)

    dΩ = Measure(stk.Ω,stk.degree)
    dΓw = Measure(stk.Γw,stk.degree)
    dΓs = Measure(stk.Γs,stk.degree)
    n_s = normal_vector(z)

    f = stk.f; β = stk.β; B = stk.B

    ∇x = transform_gradient(z,stk.Lstretch)
    ∫_Ωx = transform_integral(z,stk.Lstretch)
    εx(u) = symmetric_part(∇x(u))
    divx(u) = tr(∇x(u))

    a_stokes((u,p),(v,q)) = ∫_Ωx(  ((1/B)*(μ⊙εx(u))⊙εx(v) - divx(u)*q - divx(v)*p) )dΩ #+ ∫( β*u⋅v )dΓw 
    a_fssa((u,p),(v,q)) = ∫_Ωx(  dt*(u⋅n_s)*(f⋅v) )dΓs

    a((u,p),(v,q)) = a_stokes((u,p),(v,q)) + a_fssa((u,p),(v,q))
    
    b((v,q)) = ∫_Ωx( v⋅f )dΩ

    op = AffineFEOperator(a,b,X,Y)
    # sol, = solve!(sol,stk.solver,op)

    
    uh,ph = block_solve(op,1.0e2,dΩ,X,Q)
    return uh,ph
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

function block_solve(op,α,dΩ,X,Q)
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
  
    xh = FEFunction(X,x)
    uh,ph = xh
    return uh,ph
  end