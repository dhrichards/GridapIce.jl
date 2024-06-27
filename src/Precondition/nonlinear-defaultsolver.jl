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
nc = (30,30)

model = CartesianDiscreteModel( (0.0,1.0,0.0,1.0), nc )
order = 3
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
# X = MultiFieldFESpace([U,Q])
# Y = MultiFieldFESpace([V,Q])

f = VectorValue(0.0,0.0)

Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

α = 1.e2
poly = (Dc==2) ? QUAD : HEX
Π_Qh = LocalProjectionMap(poly,lagrangian,Float64,order-1;quad_order=qdegree,space=:P)
graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ


ϵ = 1e-4; n = 3.0; B = 100.0
η(ε) = B^(-1/n)*(0.5*ε⊙ε + ϵ^2)^((1-n)/(2*n))
dη(dε,ε) = B^(-1/n)*(1-n)/(2*n)*(0.5*ε⊙ε+ϵ^2)^((1-n)/(2*n)-1)*0.5*(dε⊙ε+ε⊙dε)


τ(ε) = η∘(ε)*ε
dτ(dε,ε) = dη∘(dε,ε)*ε + η∘(ε)*dε
dgraddiv(du,u,v,dΩ) = ∫(α*Π_Qh(divergence(du))⋅Π_Qh(divergence(v)))dΩ



res((u,p),(v,q)) = ∫(τ(ε(u))⊙ε(v) - divergence(v)*p - divergence(u)*q - v⋅f)dΩ #+ graddiv(u,v,dΩ)
jac((u,p),(du,dp),(v,q)) = ∫(dτ(ε(du),ε(u))⊙ε(v) - divergence(v)*dp - divergence(du)*q)dΩ #+ dgraddiv(du,u,v,dΩ)

op = FEOperator(res,jac,X,Y)

## Check everything above is correct
# using LineSearches: BackTracking
# nls = NLSolver(
#   show_trace=true, method=:newton,iterations=50,xtol=1e-8,ftol=1e-8,linesearch=BackTracking())
# solver_test = FESolver(nls)
# sol = solve(solver_test,op)
# uh, ph = sol
# writevtk(Ω,"stokes",cellfields=["uh"=>uh])


solver_u = LUSolver()
solver_p = LUSolver()
# solver_p = CGSolver(JacobiLinearSolver();maxiter=30,atol=1e-14,rtol=1.e-6,verbose=false)

# solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=false)
  

# invη(ε,p,q) = -1.0*p*q*B^(1/n)*(0.5*ε⊙ε + ϵ^2)^((n+1)/(2*n))
η_(ε) = η∘(ε)

TriForm = TriformBlock(((u,p),dp,q) -> ∫(-1.0/α*dp*q/η_(ε(u)))dΩ,X,Q,Q)
bblocks  = [NonlinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock()    TriForm]


coeffs = [1.0 1.0;
          0.0 1.0]
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)


solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6)
nlsolver = NewtonSolver(solver;maxiter=50,atol=1e-14,rtol=1.e-7,verbose=true)


xh = interpolate([x->VectorValue(x[2],0.0),0.0],X)
# xh = solve(nlsolver,op)
solve!(xh,nlsolver,op)

uh, ph = xh

writevtk(Ω,"stokes",cellfields=["uh"=>uh])

