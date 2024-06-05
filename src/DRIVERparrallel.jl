using Gridap
using Gridap.TensorValues
using Gridap.Geometry
using Gridap.Arrays
using Gridap.MultiField
using GridapDistributed
using GridapPETSc
using GridapPETSc: PETSC
using PartitionedArrays
using LineSearches: BackTracking
using Gridap.Arrays
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

#Run with below command from the root directory
# ./mpiexecjl --project=. -n 2 julia src/DRIVERparrallel.jl
# options = "-ksp_type cg -pc_type gamg -ksp_monitor"
# use mumps
options = "-ksp_type cg -pc_type gamg -pc_factor_mat_solver_type mumps -ksp_monitor"

# use complex numbers
# options = "-ksp_type fgmres -pc_type lu -ksp_monitor"
# function main(ranks)
function main(rank_partition,distribute)
    
    parts  = distribute(LinearIndices((prod(rank_partition),)))
    GridapPETSc.with(args=split(options)) do

        problem = ISMIPHOM(:B,1.0)
        L = (5e3,1e3)

        D = problem.D
        ncell = (32,16)

        model = CartesianDiscreteModel(parts,rank_partition,(0,1,0,1),ncell,isperiodic=(true,false))
        labels = get_labels(model,D)

        solver = PETScLinearSolver()
        
        stk = Stokes(model,problem,solver)
        Ecc = 1.0; Eca = 25.0; n = 1.0; B = 100.0

        fab = SpecFab(model,D,:H1implicit_real,1,0.3,4,solver)

        CFL = 1.0
        d = problem.L[1]/ncell[1]
        dt = CFL*d/1.0
        nt = 100

        
        z0(x) = x[2]*(problem.s(x[1])-problem.b(x[1]))+problem.b(x[1]) 
        

        S = FESpace(Triangulation(model),ReferenceFE(lagrangian,Float64,2),conformity=:H1)
        z = interpolate_everywhere(z0,S)

        fh = fab.f0h
        μ = SachsVisc(fab.a2(fh),fab.a4(fh))



        sol, res = solve_up_linear(zero(stk.X),μ,z,stk)
        uh, ph = sol
        # U = FESpace(Triangulation(model),ReferenceFE(lagrangian,VectorValue{2,Float64},2))
        # uh = interpolate_everywhere(x->VectorValue(x[2],0.0),U)
        writevtk(get_triangulation(model),"parrallel0", cellfields=["uh"=>uh,"z"=>z,"a2"=>fab.a2(fh)])


        for i = 1:nt
            ∇x = transform_gradient(z)
            εx(u) = symmetric_part(∇x(u))
            C = εx(uh)
            fh = fab.solve(fh,uh,C,z,dt,fab)

            # z = solve_surface_combined(model,z,problem.b,dt,uh,solver,problem.❄️)

            fh = fab.f0h
            μ = SachsVisc(fab.a2(fh),fab.a4(fh))

            # sol, res = solve_up_linear(zero(stk.X),μ,z,stk)
            # uh, ph = sol

            
            writevtk(get_triangulation(model),"parrallel$i", cellfields=["uh"=>uh,"z"=>z,"a2"=>fab.a2(fh)])
        end

    end

end




rank_partition = (2,1)
with_mpi() do distribute
    main(rank_partition,distribute)
end