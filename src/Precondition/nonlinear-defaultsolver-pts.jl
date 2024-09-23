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
using LineSearches

np = 1
nc = (70,70)

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


η_(ε) = η∘(ε)



solver_u = LUSolver()
solver_p = LUSolver()


TriForm = TriformBlock(((u,p),dp,q) -> ∫(-1.0/α*dp*q/η_(ε(u)))dΩ,X,Q,Q)
bblocks  = [NonlinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock()    TriForm]


coeffs = [1.0 1.0;
        0.0 1.0]
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)


solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6)
# nlsolver = NLsolveNonlinearSolver(solver; show_trace=true, method=:newton, linesearch=Static(), iterations=25)
nlsolver = NewtonSolver(solver;maxiter=25,atol=1e-14,rtol=1.e-7,verbose=true)



dt = 0.1

xh = interpolate([x->VectorValue(x[2],0.0),0.0],X)
uh, ph = xh
for i=1:10

    ptsres((u,p),(v,q)) = ∫(u⋅v/dt)dΩ + res((u,p),(v,q)) - ∫(uh⋅v/dt)dΩ
    ptsjac((u,p),(du,dp),(v,q)) = ∫(du⋅v/dt)dΩ + jac((u,p),(du,dp),(v,q))
    op = FEOperator(ptsres,ptsjac,X,Y)



    solve!(xh,nlsolver,op)

    uh, ph = xh

end

writevtk(Ω,"stokes",cellfields=["uh"=>uh])
