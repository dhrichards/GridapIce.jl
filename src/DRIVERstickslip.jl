using Gridap
using Gridap.TensorValues
using Gridap.Geometry
using Gridap.Arrays
using Gridap.MultiField
using GridapDistributed
using GridapGmsh
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


problem = StickSlip(100.0)
L = problem.L; D = problem.D
ncell = (20,20)
domain = (D == 2) ? (0,1,0,1) : (0,1,0,1,0,1)

model = GmshDiscreteModel("meshes/stickslip.msh")
labels = get_face_labeling(model)


using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton,iterations=50,xtol=1e-8,ftol=1e-8,linesearch=BackTracking())
solver = FESolver(nls)


Ecc = 1.0; Eca = 25.0; λ = 0.3

# n=3.0 B = 100.0  
n=1.0; B = 2.1430373e-1
stk = Stokes(model,n,B,problem,solver,L)

fab = SpecFab(model,D,:H1implicit_real,1,λ,2)

CFL = 0.1
d = minimum((L...,problem.H)./ncell)
# dt = CFL*d/
nt = 500

z_end(x) = x[end]
z_bcs = [problem.b,problem.b,z_end,z_end]
z_dtags = ["bottom_left","bottom_right","left","right"]
function iterate()
   
    z = init_z(model,bcs,problem.b,problem.s,z_dtags)

    fh = fab.f0h
    μ = SachsVisc(fab.a2(fh),fab.a4(fh))

    # sol, res = solve_up(zero(stk.X),μ,z,stk); uh, ph =sol
    uh,ph = solve_up_linear(μ,z,stk)
    dt = CFL*d/maximum(uh.free_values)

    # sol⁺ = FSSAsolve(zero(stk.X),res,dt,s,stk)
    # uh⁺, ph⁺ = sol⁺
    uh⁺ = uh

    writevtk(get_triangulation(model),"results/serial0", cellfields=["uh"=>uh,"z"=>z,"ph"=>ph,"a2"=>fab.a2(fh)])


    for i = 1:nt
        print("Solving for fabric at i = $i\n")
        ∇x = transform_gradient(z)
        εx(u) = symmetric_part(∇x(u))
        # fh = fab.solve(fh,uh,εx(uh),z,dt,fab)

        print("Solving for z at i = $i\n")
        
        solve_z!(z,model,z_bcs,dt,uh⁺,problem.❄️,z_dtags)
        

        μ = SachsVisc(fab.a2(fh),fab.a4(fh))
        print("Solving for velocity at i = $i\n")
        # sol, res = solve_up(sol,μ,z,stk); uh, ph =sol
        uh,ph = solve_up_linear(μ,z,stk)

        dt = CFL*d/maximum(uh.free_values)

        # sol⁺ = FSSAsolve(sol⁺,res,dt,s,stk)
        # uh⁺, ph⁺ = sol⁺
        uh⁺ = uh

        writevtk(get_triangulation(model),"results/serial$i", cellfields=["uh"=>uh,"z"=>z,"ph"=>ph,"a2"=>fab.a2(fh)])
    end
end


iterate()
