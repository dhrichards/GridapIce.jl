
abstract type Fabric end

struct SpecFab <: Fabric
    D::Int # Dimension
    Td::Int # Number of tensor components
    order:: Int # Finite element order
    degree::Int # Quadrature degree
    solve::Function # Solver function
    F::FESpace
    G::FESpace
    Ω::Union{Triangulation,GridapDistributed.DistributedTriangulation}
    Γ::Union{BoundaryTriangulation,GridapDistributed.DistributedTriangulation}
    Λ::Union{SkeletonTriangulation,GridapDistributed.DistributedTriangulation}
    f0h::Union{CellField,GridapDistributed.DistributedCellField}
    a2::Function # Second order tensor
    a4::Function # Fourth order tensor
    λ::Float64 # 
    f0::VectorValue
    L::Int
    lm
    nlm_len::Int
    int_step::Int
    ls

    function SpecFab(model,D::Int,type,order::Int,λ::Float64,L::Int,ls=LUSolver(),int_step=1)
        Td = (D==2 ? 3 : 6)
        lm,nlm_len = sf.init(L)
        degree = 2*order
        reffe = ReferenceFE(lagrangian,VectorValue{nlm_len,ComplexF64},order)
        f0 = Complex.(zeros(nlm_len))
        f0[1] = 1/√(4π)
        f0 = VectorValue(f0)

        if type == :H1implicit
            conformity = :H1
            solve = specfab_solve
        elseif type == :H1semiexplicit
            conformity = :H1
            solve = h1_semiexplicit
        elseif type == :DGimplicit
            conformity = :L2
            solve = fabric_solve_dg
        elseif type == :DGexplicit
            conformity = :L2
            solve = dg_transient
        end

        Ω = Triangulation(model)
        Γ = BoundaryTriangulation(model)
        Λ = SkeletonTriangulation(model)

        G = TestFESpace(model,reffe,conformity=conformity,vector_type=Vector{ComplexF64})
        F = TrialFESpace(G)

        f0h = interpolate_everywhere(f0,F)

        

        if D == 2
            a2 = a2calc2d
            a4 = a4calc2d
        else
            a2 = a2calc
            a4 = a4calc
        end
        return new(D,Td,order,degree,solve,F,G,Ω,Γ,Λ,f0h,a2,a4,λ,f0,L,lm,nlm_len,int_step,ls)
    end

end


function fabric_solve_dg(model,fh,uh,C,dt,fab)


    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(model)
    Λ = SkeletonTriangulation(model)
    
    G = TestFESpace(model,fab.reffe,conformity=:L2)
    F = TrialFESpace(G)

    
    dΩ = Measure(Ω,fab.degree)
    dΓ = Measure(Γ,fab.degree)
    dΛ = Measure(Λ,fab.degree)


    n_Γ = get_normal_vector(Γ)
    n_Λ = get_normal_vector(Λ)

    SR = sqrt∘(0.5*ε(uh)⊙ε(uh))
    ∇u = uh ⊗ ∇
    W = 0.5*(∇u - (∇u)')
    λ = fab.λ
    M = makeM(W,C,λ*SR,fab.L)

    a_Ω(f,g) = ∫( f⋅g/dt - (uh⋅∇(g))⋅f - (M⋅f)⋅g  )dΩ 
     
    a_Γ(f,g) = ∫( (g⋅f)*(uh⋅n_Γ))dΓ
    
    function cf(d,w)

        c = abs(d)/2
        # note: c = abs(w)/2 gives the same result as abs(d)/2
        # note: c = -abs(d)/2 or c = -abs(w)/2 is unstable
      
        if (d+w) > 1e-10
          println("error")
          # checking in plus and minus contributions are equal and oppositie
        end
      
        return c
    end

    a_Λadv(f,g) = ∫( mean(f⊗uh)⊙jump(g⊗n_Λ) + cf∘( (uh⋅n_Λ).⁺,(uh⋅n_Λ).⁻ )*jump(f⊗n_Λ)⊙jump(g⊗n_Λ) )dΛ

    a(f,g) = a_Ω(f,g)  + a_Λadv(f,g)  + a_Γ(f,g) #+ a_Λdiff(a,b)
    l(g) = ∫( fh⋅g/dt  )dΩ 

    op = AffineFEOperator(a,l,F,G)
    fh = solve(op)


    return fh
end


function specfab_solve(fh,uh,C,T,h,dt,fab)

    κ = 1e-2

    dΩ = Measure(fab.Ω,fab.degree)

    # Switch from transformed grid to real: (also requires multiplying by h as dz = h*dζ)
    ∇ᴿ = Transform_∇(T)



    SR = sqrt∘(0.5*ε(uh)⊙ε(uh))
    ∇u = uh ⊗ ∇
    W = 0.5*(∇u - (∇u)')
    λ = fab.λ

    M = makeM(W,C,λ*SR,fab.f0h)

    a(f,g) = ∫( h*(f⋅g + dt*(uh⋅∇ᴿ(f))⋅g + dt*κ*(∇ᴿ(f)⊙∇ᴿ(g)) -dt*(M⋅f)⋅g) )dΩ # 
    b(g) = ∫( h*fh⋅g )dΩ

    op = AffineFEOperator(a,b,fab.F,fab.G)
    fh = solve(fab.ls,op)


    return fh
end


function h1_semiexplicit(model,fh,uh,C,dt,fab)

    κ = 1e-2
    Ω = Triangulation(model)
    dΩ = Measure(Ω,fab.degree)
    G = TestFESpace(model,fab.reffe)
    F = TrialFESpace(G)

    


    SR = sqrt∘(0.5*ε(uh)⊙ε(uh))
    ∇u = uh ⊗ ∇
    W = 0.5*(∇u - (∇u)')
    λ = fab.λ

    M = makeM(W,C,λ*SR,fab.L)

    a(f,g) = ∫( f⋅g -dt*(M⋅f)⋅g )dΩ # 
    b(g) = ∫( fh⋅g -dt*(uh⋅∇(fh))⋅g - dt*κ*(∇(fh)⊙∇(g)) )dΩ

    op = AffineFEOperator(a,b,F,G)
    fh = solve(op)


    return fh
end




function dg_transient(model,fh,uh,C,dt,fab)


    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(model)
    Λ = SkeletonTriangulation(model)
    
    G = TestFESpace(model,fab.reffe,conformity=:L2)
    F = TrialFESpace(G)

    
    dΩ = Measure(Ω,fab.degree)
    dΓ = Measure(Γ,fab.degree)
    dΛ = Measure(Λ,fab.degree)


    n_Γ = get_normal_vector(Γ)
    n_Λ = get_normal_vector(Λ)

    SR = sqrt∘(0.5*ε(uh)⊙ε(uh))
    ∇u = uh ⊗ ∇
    W = 0.5*(∇u - (∇u)')
    λ = fab.λ
    M = makeM(W,C,λ*SR,fab.L)

    a_Ω(f,g) = ∫( -(uh⋅∇(g))⋅f - (M⋅f)⋅g )dΩ 
     
    a_Γ(f,g) = ∫( (g⋅f)*(uh⋅n_Γ))dΓ
    
    function cf(d,w)

      
        c = abs(d)/2
        # note: c = abs(w)/2 gives the same result as abs(d)/2
        # note: c = -abs(d)/2 or c = -abs(w)/2 is unstable
      
        if (d+w) > 1e-10
          println("error")
          # checking in plus and minus contributions are equal and oppositie
        end
      
        return c
    end

    a_Λadv(f,g) = ∫( mean(f⊗uh)⊙jump(g⊗n_Λ) + cf∘( (uh⋅n_Λ).⁺,(uh⋅n_Λ).⁻ )*jump(f⊗n_Λ)⊙jump(g⊗n_Λ) )dΛ

    res(t,f,g) = a_Ω(f,g)  + a_Λadv(f,g)  + a_Γ(f,g) 
    mass(t,dft,g) = ∫( dft⋅g )dΩ

    jac(t,f,df,g) = res(t,df,g)
    jac_t(t,f,dft,g) = ∫( dft⋅g )dΩ

    opT = TransientSemilinearFEOperator(mass,res,(jac,jac_t),F,G)
   
    ls = LUSolver()

    # ode_solver = RungeKutta(ls,ls,dt/fab.int_step,:EXRK_SSP_3_3)
    ode_solver = ForwardEuler(ls,dt/fab.int_step)
    sol_t = solve(ode_solver,opT,0,dt,fh)

    return get_final_solution(sol_t,fh)
end


function get_final_solution(sol_t,fh)
    for (t,fh_new) in sol_t
        fh = fh_new
    end
    return fh
end



