using Gridap
using Gridap.TensorValues
using Gridap.Geometry
using Gridap.Arrays
using Gridap.MultiField
using GridapDistributed
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

model = CartesianDiscreteModel((0,5e3,0,1),ncell,isperiodic=(true,false))
labels = get_labels(model,D)


using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton,iterations=50,xtol=1e-8,ftol=1e-8,linesearch=BackTracking())
solver = FESolver(nls)
stk = Stokes(model,problem,solver)

Ecc = 1.0; Eca = 25.0; n = 3.0; B = 100.0

η(ε) = B^(-1/n)*(0.5*ε⊙ε+1e-9)^((1-n)/(2*n))
    




fab = SpecFab(model,D,:H1implicit,1,0.3,4,LUSolver(),1)

CFL = 0.1
d = problem.L[1]/ncell[1]
nt = 500

z0(x) = x[2]*(problem.s(x[1])-problem.b(x[1]))+problem.b(x[1]) 


function iterate()
    S = FESpace(Triangulation(model),ReferenceFE(lagrangian,Float64,2),conformity=:H1)
    z = interpolate_everywhere(z0,S)

    fh = fab.f0h
    μ = SachsVisc(fab.a2(fh),fab.a4(fh))

    sol, res = solve_up(zero(stk.X),μ,η,z,stk)
    uh, ph = sol
    dt = CFL*d/maximum(uh.free_values)

    # sol⁺ = FSSAsolve(zero(stk.X),res,dt,s,stk)
    # uh⁺, ph⁺ = sol⁺
    uh⁺ = uh

    writevtk(get_triangulation(model),"serial0", cellfields=["uh"=>uh,"ph"=>ph,"z"=>z,"a2"=>fab.a2(fh)])


    for i = 1:nt
        # print("Solving for fabric at i = $i\n")
        # fh = fab.solve(fh,uh,εꜝ(ϕ,uh),ϕ,h,dt,fab)

        print("Solving for z at i = $i\n")
        z = solve_surface_combined(model,z,problem.b,dt,uh,LUSolver(),problem.❄️)
        

        μ = SachsVisc(fab.a2(fh),fab.a4(fh))
        print("Solving for velocity at i = $i\n")
        sol, res = solve_up(sol,μ,η,z,stk)
        uh, ph = sol
        dt = CFL*d/maximum(uh.free_values)

        # sol⁺ = FSSAsolve(sol⁺,res,dt,s,stk)
        # uh⁺, ph⁺ = sol⁺
        uh⁺ = uh

        writevtk(get_triangulation(model),"serial$i", cellfields=["uh"=>uh,"ph"=>ph,"z"=>z,"a2"=>fab.a2(fh)])
    end
end


iterate()
