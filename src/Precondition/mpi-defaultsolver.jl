using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField
using PartitionedArrays
using GridapDistributed
# using GridapP4est
using GridapPETSc

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

function add_labels_2d!(labels)
  add_tag_from_tags!(labels,"top",[3,4,6])
  add_tag_from_tags!(labels,"bottom",[1,2,5])
  add_tag_from_tags!(labels,"walls",[7,8])
end

function add_labels_3d!(labels)
  add_tag_from_tags!(labels,"top",[5,6,7,8,11,12,15,16,22])
  add_tag_from_tags!(labels,"bottom",[1,2,3,4,9,10,13,14,21])
  add_tag_from_tags!(labels,"walls",[17,18,23,25,26])
end


function main(rank_partition,distribute)
    Dc = length(rank_partition)
    parts  = distribute(LinearIndices((prod(rank_partition),)))
    nc = (Dc ==2 ) ? (10,10) : (10,10,10)

    domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)

    model = CartesianDiscreteModel(parts,rank_partition,domain,nc,isperiodic=(true,true,false))
    add_labels! = (Dc == 2) ? add_labels_2d! : add_labels_3d!
    add_labels!(get_face_labeling(model))
  
    # FE spaces
    order = 2
    qdegree = 2*(order+1)
    reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
    reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)
  
    u_bottom = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
    u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)
  
    V = TestFESpace(model,reffe_u,dirichlet_tags=["bottom","top"]);
    U = TrialFESpace(V,[u_bottom,u_top]);
    Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 
  
    mfs = Gridap.MultiField.BlockMultiFieldStyle()
    X = MultiFieldFESpace([U,Q];style=mfs)
    Y = MultiFieldFESpace([V,Q];style=mfs)


    α = 1.e2
    f = (Dc==2) ? VectorValue(1.0,1.0) : VectorValue(1.0,1.0,1.0)


    Ω = Triangulation(model)
    dΩ = Measure(Ω,qdegree)

    biform_u(u,v,dΩ) = ∫(ε(u)⊙ε(v))dΩ #+ graddiv(u,v,dΩ)
    biform((u,p),(v,q),dΩ) = biform_u(u,v,dΩ) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
    liform((v,q),dΩ) = ∫(v⋅f)dΩ

    a(u,v) = biform(u,v,dΩ)
    l(v) = liform(v,dΩ)
    op = AffineFEOperator(a,l,X,Y)


    A, b = get_matrix(op), get_vector(op);
    Auu = blocks(A)[1,1]

    solver_u = LUSolver()
    solver_p = LUSolver()
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

    xh = FEFunction(X,x)
    uh,ph = xh

    writevtk(Ω,"stokes",cellfields=["uh"=>uh])
end

rank_partition = (1,1,4)
with_mpi() do distribute
  main(rank_partition,distribute)
end