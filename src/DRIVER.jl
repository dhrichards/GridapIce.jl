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

model = CartesianDiscreteModel((0,problem.L,0,1),ncell,isperiodic=(true,false))
labels = get_labels(model,D)



solver = LUSolver()
stk = Stokes(model,problem,solver)

Ecc = 1.0; Eca = 25.0; n = 1.0; B = 100.0
function FlowLaw(A,A⁴)
    μ = SachsVisc(A,A⁴)
    τ(ε) = μ⊙ε
    return τ
end


fab = SpecFab(model,D,:H1implicit,1,0.3,2,LUSolver(),1)

CFL = 0.1
d = problem.L[1]/ncell[1]

nt = 500



function iterate()
    s,b,ζ,L = get_sb_fields(problem,model)
    M = Transform(ζ,b,s)
    h = s - b
    z = ζ*h + b

    fh = fab.f0h
    τ = FlowLaw(fab.a2∘(fh),fab.a4∘(fh))

    sol, res = solve_up_linear(zero(stk.X),τ,M,h,stk)
    uh, ph = sol
    dt = CFL*d/maximum(uh.free_values)
    writevtk(get_triangulation(model),"serial0", cellfields=["uh"=>uh,"ph"=>ph,"z"=>z,"s"=>s,"a2"=>fab.a2∘(fh)])


    for i = 1:nt

        fh = fab.solve(fh,uh,τ(ε(uh)),M,h,dt,fab)

        s = solve_surface3d(model,s,dt,uh,M,h,LUSolver(),problem.❄️)
        s = impose_surface(s,model,d,LUSolver())

        M = Transform(ζ,b,s)
        h = s - b
        z = ζ*h + b

        τ = FlowLaw(fab.a2∘(fh),fab.a4∘(fh))

        sol, res = solve_up_linear(zero(stk.X),τ,M,h,stk)
        uh, ph = sol
        dt = CFL*d/maximum(uh.free_values)

        

        writevtk(get_triangulation(model),"serial$i", cellfields=["uh"=>uh,"ph"=>ph,"z"=>z,"s"=>s,"a2"=>fab.a2∘(fh)])
    end
end
iterate()
