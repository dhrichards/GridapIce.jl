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
using GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver, NonlinearSystemBlock, TriformBlock

np = 1
nc = (50,50) # large enough to run into problems

model = CartesianDiscreteModel( (0.0,1.0,0.0,1.0), nc )
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

f = VectorValue(0.0,0.0)

Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

α = 1.e2
poly = (Dc==2) ? QUAD : HEX
Π_Qh = LocalProjectionMap(poly,lagrangian,Float64,order-1;quad_order=qdegree,space=:P)
graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ


# Set up power law rheology
ϵ = 1e-4; n = 2.5; B = 100.0
η(ε) = B^(-1/n)*(0.5*ε⊙ε + ϵ^2)^((1-n)/(2*n))
dη(dε,ε) = B^(-1/n)*(1-n)/(2*n)*(0.5*ε⊙ε+ϵ^2)^((1-n)/(2*n)-1)*0.5*(dε⊙ε+ε⊙dε)


τ(ε) = η∘(ε)*ε
dτ(dε,ε) = dη∘(dε,ε)*ε + η∘(ε)*dε
dgraddiv(du,u,v,dΩ) = ∫(α*Π_Qh(divergence(du))⋅Π_Qh(divergence(v)))dΩ

res((u,p),(v,q)) = ∫(τ(ε(u))⊙ε(v) - divergence(v)*p - divergence(u)*q - v⋅f)dΩ #+ graddiv(u,v,dΩ)
jac((u,p),(du,dp),(v,q)) = ∫(dτ(ε(du),ε(u))⊙ε(v) - divergence(v)*dp - divergence(du)*q)dΩ #+ dgraddiv(du,u,v,dΩ)

op = FEOperator(res,jac,X,Y)

solver_u = LUSolver()
solver_p = LUSolver()

η_(ε) = η∘(ε)

Block = TriformBlock(((u,p),dp,q) -> ∫(-1.0/α*dp*q/η_(ε(u)))dΩ,X,Q,Q)
# Q = BiformBlock((p,q) -> ∫(-p*q)dΩ,Q,Q)
bblocks  = [NonlinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock()   Block]


coeffs = [1.0 1.0;
          0.0 1.0]
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)


xh = interpolate([VectorValue(1.0,1.0),0.0],X)

A = jacobian(op,xh)
b = residual(op,xh)
ss_b = symbolic_setup(P,A,b)
ss = symbolic_setup(LUSolver(),A,b)

ns_b = numerical_setup(ss_b,A)
ns = numerical_setup(ss,A)
dx_b = similar(b)
dx = similar(b)
rmul!(b,-1)
solve!(dx_b,ns_b,b)
solve!(dx,ns,b)

