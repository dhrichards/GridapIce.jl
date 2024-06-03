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

function ϕcalc(x::Float64,z::Float64)
    return VectorValue(x,z)
end

x_int = interpolate_everywhere(x->x[1],FESpace(Triangulation(model),ReferenceFE(lagrangian,Float64,2),conformity=:L2))
    


function iterate()
    F = FESpace(Triangulation(model),ReferenceFE(lagrangian,VectorValue{2,Float64},2),conformity=:L2)
    S = FESpace(Triangulation(model),ReferenceFE(lagrangian,Float64,2),conformity=:H1)

    s,b,ζ  = get_sb_fields(problem,model)
    h = s - b
    z = ζ*h + b
    z = interpolate_everywhere(z,S)


    # ϕx(x) = VectorValue(x[1],x[2]*(problem.s(x[1])-problem.b(x[1]))+problem.b(x[1]))
    # ϕ = interpolate_everywhere(ϕx,F)

    fh = fab.f0h
    μ = SachsVisc(fab.a2(fh),fab.a4(fh))

    sol, res = solve_up(zero(stk.X),μ,η,z,stk)
    uh, ph = sol
    dt = CFL*d/maximum(uh.free_values)

    # sol⁺ = FSSAsolve(zero(stk.X),res,dt,s,stk)
    # uh⁺, ph⁺ = sol⁺
    uh⁺ = uh

    writevtk(get_triangulation(model),"serial0", cellfields=["uh"=>uh,"ph"=>ph,"z"=>z,"s"=>s,"a2"=>fab.a2(fh),"h"=>h,"ϕ"=>ϕ,"detgrad"=>det(∇(ϕ))])


    for i = 1:nt
        # print("Solving for fabric at i = $i\n")
        # fh = fab.solve(fh,uh,εꜝ(ϕ,uh),ϕ,h,dt,fab)

        print("Solving for surface at i = $i\n")
        # Δs = solve_surface3d(model,s,dt,uh⁺,ϕ,h,LUSolver(),problem.❄️)

        # print("Imposing surface at i = $i\n")
        # Δs = impose(Δs,model,LUSolver())
        # s += Δs
        # # h = s - b
        # z = solve_mesh(b,s,model,1/ncell[2],LUSolver())

        z = solve_surface_combined(model,z,problem.b,dt,uh,LUSolver(),problem.❄️)
        # ϕ = ϕcalc∘(x_int,z)
        # # ϕ = interpolate_everywhere(ϕ,F)

        # hnew_free = 1.1*h.free_values
        # h = FEFunction(S,hnew_free)
        # ϕnew_free = ϕ.free_values
        # ϕnew_free[4609:end] = z.free_values
        # ϕ = FEFunction(F,ϕnew_free)
        

        μ = SachsVisc(fab.a2(fh),fab.a4(fh))
        print("Solving for velocity at i = $i\n")
        sol, res = solve_up(sol,μ,η,z,stk)
        uh, ph = sol
        dt = CFL*d/maximum(uh.free_values)

        # sol⁺ = FSSAsolve(sol⁺,res,dt,s,stk)
        # uh⁺, ph⁺ = sol⁺
        uh⁺ = uh

        

        writevtk(get_triangulation(model),"serial$i", cellfields=["uh"=>uh,"ph"=>ph,"z"=>z,"s"=>s,"a2"=>fab.a2(fh)])
    end
end


iterate()
