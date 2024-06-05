using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField
using PartitionedArrays
using GridapDistributed
# using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver
using GridapPETSc


# options = "-ksp_type fgmres -pc_type lu -ksp_monitor"
# use mumps, 
# options = "-ksp_type fgmres -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor"
options = "-ksp_type cg -pc_type gamg -pc_factor_mat_solver_type mumps -ksp_monitor"
#use cg with jacobi preconditioner
# options = "-ksp_type cg -pc_type jacobi -ksp_monitor"


# Current state: runs but doesn't converge

function main(rank_partition,distribute)
  parts  = distribute(LinearIndices((prod(rank_partition),)))
  GridapPETSc.with(args=split(options)) do
    nc = (50,50)

    model = CartesianDiscreteModel(parts,rank_partition, (0.0,1.0,0.0,1.0), nc )
    order = 2
    qdegree = 2*(order+1)
    Dc = length(nc)

    reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"dir_top",[6,])
    add_tag_from_tags!(labels,"dir_walls",[1,2,3,4,5,7,8])

    V  = TestFESpace(model,reffe_u,dirichlet_tags=["dir_walls","dir_top"]);
    U = TrialFESpace(V,[VectorValue(0.0,0.0),VectorValue(1.0,0.0)]);

    Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)
    P = TrialFESpace(Q)


    mfs = Gridap.MultiField.BlockMultiFieldStyle()
    X = MultiFieldFESpace([U,Q];style=mfs)
    Y = MultiFieldFESpace([V,Q];style=mfs)

    α = 1e9
    # f = VectorValue(1.0,1.0)
    f = VectorValue(0.0,0.0)
    # Π_Qh = LocalProjectionMap(QUAD,lagrangian,Float64,order-1;quad_order=qdegree,space=:P)
    # graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ

    Ω = Triangulation(model)
    dΩ = Measure(Ω,qdegree)

    # ϵ = 1e-4; n = 3.0
    # η(ε) = (0.5*ε⊙ε + ϵ^2)^((1-n)/(2*n))
    # dη(dε,ε) = (1-n)/(2*n)*(0.5*ε⊙ε+ϵ^2)^((1-n)/(2*n)-1)*0.5*(dε⊙ε+ε⊙dε)

    # τ(ε) = η∘(ε)*ε
    # dτ(dε,ε) = dη∘(dε,ε)∘ε + η∘(ε)⊙dε

    # res((u,p),(v,q)) = ∫(τ(ε(u))⊙ε(v) - divergence(v)*p - divergence(u)*q - v⋅f)dΩ
    # jac((u,p),(du,dp),(v,q)) = ∫(dτ(ε(du),ε(u))⊙ε(v) - divergence(v)*dp - divergence(du)*q)dΩ

    # op = FEOperator(res,jac,X,Y)


    biform_u(u,v,dΩ) = ∫(ε(u)⊙ε(v))dΩ #+ graddiv(u,v,dΩ)
    biform((u,p),(v,q),dΩ) = biform_u(u,v,dΩ) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
    liform((v,q),dΩ) = ∫(v⋅f)dΩ

    a(u,v) = biform(u,v,dΩ)
    l(v) = liform(v,dΩ)
    op = AffineFEOperator(a,l,X,Y)

    A, b = get_matrix(op), get_vector(op);
    Auu = blocks(A)[1,1]

    solver_u = PETScLinearSolver()
    solver_p = PETScLinearSolver()
    # solver_p = CGSolver(RichardsonSmoother(JacobiLinearSolver(),10,0.2);maxiter=20,atol=1e-14,rtol=1.e-6,verbose=false)



    diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,Q,Q)]

    bblocks = map(CartesianIndices((2,2))) do I
        (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
    end

    coeffs = [1.0 1.0;
                0.0 1.0]
    P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
    solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6) #,verbose=i_am_main(parts))
    # solver = GMRESSolver(20,P;atol=1e-14,rtol=1.e-6) #,verbose=i_am_main(parts))
    ns = numerical_setup(symbolic_setup(solver,A),A)

    x = Gridap.Algebra.allocate_in_domain(A); fill!(x,0.0)
    solve!(x,ns,b)

    print(sum(x))

    uh = FEFunction(U,x) #-> gives error in mpi




    # writevtk(Ω,"stokes_p",cellfields=["uh"=>uh])
    # uh_exact = interpolate(u_exact,U)
    # eh = uh - uh_exact
    # E = sqrt(sum(∫(eh⋅eh)dΩ))
  end
end


rank_partition = (1,1)
with_mpi() do distribute
  main(rank_partition,distribute)
end