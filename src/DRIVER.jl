using Gridap
using Gridap.TensorValues
using Gridap.Geometry
using Gridap.Arrays
using Gridap.MultiField
using GridapDistributed
using FillArrays, BlockArrays
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver
include("cases.jl")
include("Rheology/mandel.jl")
include("tensorfunctions.jl")
include("fabric_solve.jl")
include("surface_solvers.jl")
include("Rheology/sachs.jl")
include("meshes.jl")
include("coordinatetransform.jl")
include("stokessolvers.jl")
include("specfab.jl")


problem = ISMIPHOM(:B,5e3)

D = problem.D
ncell = (32,16)

L = (5e3,1e3)
model = CartesianDiscreteModel((0,L[1],0,1),ncell,isperiodic=problem.periodicity)
labels = get_labels(model,D)


using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton,iterations=50,xtol=1e-8,ftol=1e-8,linesearch=BackTracking())
solver = FESolver(nls)


Ecc = 1.0; Eca = 25.0; λ = 0.3

    
# n=1.0; B = 2.1430373e-1
stk = Stokes(model,3.0,100.0,problem,solver)

fab = SpecFab(model,D,:H1implicit_real,1,λ,4)

CFL = 0.1
d = minimum(L./ncell)
nt = 500

z0(x) = x[end]*(problem.s(x)-problem.b(x))+problem.b(x) 


function iterate()
    S = FESpace(Triangulation(model),ReferenceFE(lagrangian,Float64,2),conformity=:H1)
    z = interpolate_everywhere(z0,S)

    fh = fab.f0h
    μ = SachsVisc(fab.a2(fh),fab.a4(fh))

    sol, res = solve_up(zero(stk.X),μ,z,stk); uh, ph =sol
    # uh = solve_up_linear(B,μ,z,stk)
    dt = CFL*d/maximum(uh.free_values)

    # sol⁺ = FSSAsolve(zero(stk.X),res,dt,s,stk)
    # uh⁺, ph⁺ = sol⁺
    uh⁺ = uh

    writevtk(get_triangulation(model),"serial0", cellfields=["uh"=>uh,"z"=>z,"a2"=>fab.a2(fh)])


    for i = 1:nt
        print("Solving for fabric at i = $i\n")
        ∇x = transform_gradient(z)
        εx(u) = symmetric_part(∇x(u))
        # fh = fab.solve(fh,uh,εx(uh),z,dt,fab)

        print("Solving for z at i = $i\n")
        z = solve_surface_combined(model,z,problem.b,dt,uh⁺,LUSolver(),problem.❄️)
        

        μ = SachsVisc(fab.a2(fh),fab.a4(fh))
        print("Solving for velocity at i = $i\n")
        sol, res = solve_up(sol,μ,z,stk); uh, ph =sol
        # uh = solve_up_linear(B,μ,z,stk)

        dt = CFL*d/maximum(uh.free_values)

        # sol⁺ = FSSAsolve(sol⁺,res,dt,s,stk)
        # uh⁺, ph⁺ = sol⁺
        uh⁺ = uh

        writevtk(get_triangulation(model),"serial$i", cellfields=["uh"=>uh,"z"=>z,"a2"=>fab.a2(fh)])
    end
end


iterate()
