using Gridap
# using GridapIce
include("rawimports.jl")
using Gridap.TensorValues

problem = CustomB(1e3,0.5,0.5)
L = (5000)

D = problem.D
ncell = (30,20)
domain = (D == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
domain = (D == 2) ? (-0.5,0.5,0,1) : (-0.5,0.5,-0.5,0.5,0,1)


model = CartesianDiscreteModel(domain,ncell,isperiodic=problem.periodicity)
labels = get_labels(model,D)


using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton,iterations=50,xtol=1e-8,ftol=1e-8,linesearch=BackTracking())
solver = FESolver(nls)
# solver = LUSolver()

Ecc = 1.0; Eca = 25.0; n=3.0; B = 100.0
rheo = Rheology(Ecc,Eca,n,B,:Sachs,Triangulation(model),D)


# n=1.0; B = 2.1430373e-1
stk = Stokes(model,problem,solver,L)

fab = SpecFab(model,D,:H1implicit,1,0.3,4,L)

CFL = 0.1
d = minimum((L...,problem.H)./ncell)
dt = CFL*d/100.0
nt = 500




function iterate()
   
    surf = Surface(model,problem.b,problem.s,[problem.b],L,LUSolver(),["bottom"],1,1e-4)
    z = interpolate_everywhere(surf.z0,surf.Z)


    fh = fab.f0h
    sol, res = solve_up(zero(stk.X),fh,z,stk,rheo); uh, ph =sol
    
    # uh,ph = solve_up_linear(μ,z,stk)
    dt = CFL*d/maximum(uh.free_values)

    # sol⁺ = FSSAsolve(zero(stk.X),res,dt,s,stk)
    # uh⁺, ph⁺ = sol⁺
    uh⁺ = uh

    writevtk(get_triangulation(model),"results/custom0", cellfields=["uh"=>uh,"z"=>z,"a2"=>fab.a2(fh)])


    for i = 1:nt
        print("Solving for fabric at i = $i\n")
        ∇x = transform_gradient(z)
        εx(u) = symmetric_part(∇x(u))
        C = rheo.τhat(fh,εx(uh))
        fh = fab.solve(fh,uh,C,z,dt,fab)

        print("Solving for z at i = $i\n")
        solve_z!(z,surf,dt,uh,problem.❄️)
        

        print("Solving for velocity at i = $i\n")
        sol, res = solve_up(sol,fh,z,stk,rheo); uh, ph =sol
        # uh,ph = solve_up_linear(μ,z,stk)

        dt = CFL*d/maximum(uh.free_values)

        # sol⁺ = FSSAsolve(sol⁺,res,dt,s,stk)
        # uh⁺, ph⁺ = sol⁺
        uh⁺ = uh

        writevtk(get_triangulation(model),"results/custom$i", cellfields=["uh"=>uh,"z"=>z,"a2"=>fab.a2(fh)])
    end
end


iterate()
