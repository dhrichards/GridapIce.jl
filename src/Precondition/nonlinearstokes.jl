using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapDistributed

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: LinearSystemBlock, NonlinearSystemBlock, TriformBlock, BlockTriangularSolver

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

np = (1,1)
nc = (10,10)

parts = with_mpi() do distribute
  distribute(LinearIndices((prod(np),)))
end

# Geometry
Dc = length(nc)
domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)

# model = CartesianDiscreteModel(parts,np,domain,nc)
model = CartesianDiscreteModel( domain, nc )
add_labels! = (Dc == 2) ? add_labels_2d! : add_labels_3d!
labels = get_face_labeling(model)
# add_tag_from_tags!(labels,"dir_top",[6,])
# add_tag_from_tags!(labels,"dir_walls",[1,2,3,4,5,7,8])




add_labels!(get_face_labeling(model))

# FE spaces
order = 2
qdegree = 2*(order+1)
reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

u_bottom = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)


# V  = TestFESpace(model,reffe_u,dirichlet_tags=["dir_walls","dir_top"]);
V = TestFESpace(model,reffe_u,dirichlet_tags=["bottom","top"]);
U = TrialFESpace(V,[u_bottom,u_top]);
Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

mfs = Gridap.MultiField.BlockMultiFieldStyle()
X = MultiFieldFESpace([U,Q];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

# Weak formulation
Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

p = 3
_ν(∇u) = norm(∇u)^(p-2)
_dν(∇u) = (p-2)*norm(∇u)^(p-4)
_flux(∇u) = _ν(∇u)*∇u
_dflux(∇du,∇u) = _dν(∇u)*(∇u⊙∇du)*∇u + _ν(∇u)*∇du

ν(u) = _ν∘(∇(u))
flux(u) = _flux∘(∇(u))
dflux(du,u) = _dflux∘(∇(du),∇(u))

f = (Dc==2) ? VectorValue(1.0,1.0) : VectorValue(1.0,1.0,1.0)
# f = VectorValue(0.0,0.0)

# res_u(u,v) = ∫(∇(v)⊙flux(u))dΩ - ∫(v⋅f)dΩ
# res((u,p),(v,q)) = res_u(u,v) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ

# jac_u(u,du,v) = ∫(∇(v)⊙dflux(du,u))dΩ
# jac((u,p),(du,dq),(v,q)) = jac_u(u,du,v) - ∫(divergence(v)*dq)dΩ - ∫(divergence(du)*q)dΩ

ϵ = 1e-4; n = 3.0; B = 100.0
η(ε) = B^(-1/n)*(0.5*ε⊙ε + ϵ^2)^((1-n)/(2*n))
dη(dε,ε) = B^(-1/n)*(1-n)/(2*n)*(0.5*ε⊙ε+ϵ^2)^((1-n)/(2*n)-1)*0.5*(dε⊙ε+ε⊙dε)


τ(ε) = η∘(ε)*ε
dτ(dε,ε) = dη∘(dε,ε)*ε + η∘(ε)*dε
dgraddiv(du,u,v,dΩ) = ∫(α*Π_Qh(divergence(du))⋅Π_Qh(divergence(v)))dΩ



res((u,p),(v,q)) = ∫(τ(ε(u))⊙ε(v) - divergence(v)*p - divergence(u)*q - v⋅f)dΩ #+ graddiv(u,v,dΩ)
jac((u,p),(du,dp),(v,q)) = ∫(dτ(ε(du),ε(u))⊙ε(v) - divergence(v)*dp - divergence(du)*q)dΩ #+ dgraddiv(du,u,v,dΩ)




op = FEOperator(res,jac,X,Y)

# Solver
solver_u = LUSolver()
solver_p = LUSolver()
# solver_p = CGSolver(JacobiLinearSolver();maxiter=30,atol=1e-14,rtol=1.e-6,verbose=false)
# solver_p.log.depth = 2

bblocks  = [NonlinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock()    TriformBlock(((u,p),dp,q) -> ∫(-ν(u)*dp*q)dΩ,X,Q,Q)]
coeffs = [1.0 1.0;
          0.0 1.0]
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
ls_solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-8,verbose=false)
nls_solver = NewtonSolver(ls_solver;maxiter=40,atol=1e-14,rtol=1.e-6,verbose=true)

xh = interpolate([VectorValue(1.0,1.0),0.0],X);
solve!(xh,nls_solver,op)

writevtk(Ω,"results",cellfields=["uh"=>xh[1],"ph"=>xh[2]])
# xh = solve(nls_solver,op)