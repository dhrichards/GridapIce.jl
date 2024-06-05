using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField
# using PartitionedArrays
# using GridapDistributed
# using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

np = 1
nc = (50,50)

model = CartesianDiscreteModel( (0.0,1.0,0.0,1.0), nc, isperiodic=(true,false) )
order = 2
qdegree = 2*(order+1)
Dc = length(nc)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"bottom",[5])
add_tag_from_tags!(labels,"top",[6])

V  = TestFESpace(model,reffe_u,dirichlet_tags=["bottom"]);
U = TrialFESpace(V,[VectorValue(0.0,0.0)]);

Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)
P = TrialFESpace(Q)


mfs = Gridap.MultiField.BlockMultiFieldStyle()
X = MultiFieldFESpace([U,Q];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

α = 1e9
# f = VectorValue(1.0,1.0)
ρ = 9.138e-19; g = 9.7692e15; angle = 0.5
f = VectorValue(ρ*g*sind(angle),-ρ*g*cosd(angle))
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


# function block_solve(op,α,dΩ,U,Q)
  A, b = get_matrix(op), get_vector(op);
  Auu = blocks(A)[1,1]

  solver_u = LUSolver()
#   solver_p = LUSolver()
  solver_p = CGSolver(RichardsonSmoother(JacobiLinearSolver(),10,0.2);maxiter=20,atol=1e-14,rtol=1.e-6,verbose=false)



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

  uh = FEFunction(U,x)
#   return uh
# end

# Postprocess
# uh = block_solve(op,α,dΩ,U,Q)



writevtk(Ω,"stokes",cellfields=["uh"=>uh])
# uh_exact = interpolate(u_exact,U)
# eh = uh - uh_exact
# E = sqrt(sum(∫(eh⋅eh)dΩ))
