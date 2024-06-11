using Gridap
using Gridap.TensorValues
using Gridap.Geometry
using Gridap.Arrays
using Gridap.MultiField
using GridapDistributed
using GridapPETSc
using GridapPETSc: PETSC
using PartitionedArrays
using FillArrays, BlockArrays
using LineSearches: BackTracking
using Gridap.Arrays
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers
using GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver, NonlinearSystemBlock, TriformBlock
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
include("fixes.jl")

#Run with below command from the root directory
# ./mpiexecjl --project=. -n 2 julia src/DRIVERparrallel.jl
# options = "-ksp_type cg -pc_type gamg -ksp_monitor"
# use mumps
options = "-ksp_type cg -pc_type gamg -pc_factor_mat_solver_type mumps -ksp_monitor"

L = (100e3,100e3)
# use complex numbers
# options = "-ksp_type fgmres -pc_type lu -ksp_monitor"
# function main(ranks)
function main(rank_partition,distribute)
    
    parts  = distribute(LinearIndices((prod(rank_partition),)))
    GridapPETSc.with(args=split(options)) do

        problem = ISMIPHOM(:F1)
        L = (100e3,100e3)
        ncell = (20,20,5)
        domain = (problem.D == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
        domain = (-0.5,0.5,-0.5,0.5,0,1)

        model = CartesianDiscreteModel(parts,rank_partition,domain,ncell,isperiodic=problem.periodicity)
        labels = get_labels(model,problem.D)

        solver = PETScLinearSolver()
        # solver = LUSolver()

        
        stk = Stokes(model,1.0,2.1430373e-1,problem,solver,L)
        Ecc = 1.0; Eca = 25.0

        fab = SpecFab(model,problem.D,:H1implicit_real,1,0.3,2,solver)

        CFL = 0.2
        d = minimum((L...,problem.H)./ncell)
        dt = CFL*d/100.0
        nt = 500
        

        
        z = init_z(model,problem.b,problem.s)

        fh = fab.f0h
        μ = SachsVisc(fab.a2(fh),fab.a4(fh))

        # sol, res = solve_up(zero(stk.X),μ,z,stk); uh, ph =sol
        uh = solve_up_linear(μ,z,stk)

        #gather
        # uh_max = 

        # U = FESpace(Triangulation(model),ReferenceFE(lagrangian,VectorValue{2,Float64},2))
        # uh = interpolate_everywhere(x->VectorValue(x[2],0.0),U)
        writevtk(get_triangulation(model),"results/parrallel0", cellfields=["uh"=>uh,"z"=>z,"a2"=>fab.a2(fh)])


        for i = 1:nt
            # ∇x = transform_gradient(z)
            # εx(u) = symmetric_part(∇x(u))
            # C = εx(uh)
            # fh = fab.solve(fh,uh,C,z,dt,fab)

            solve_z!(z,model,problem.b,dt,uh,problem.❄️,solver)

            # μ = SachsVisc(fab.a2(fh),fab.a4(fh))
            # sol, res = solve_up(sol,μ,z,stk); uh, ph =sol

            uh = solve_up_linear(μ,z,stk)

            writevtk(get_triangulation(model),"results/parrallel$i", cellfields=["uh"=>uh,"z"=>z,"a2"=>fab.a2(fh)])
        end

    end

end




rank_partition = (1,1,4)
with_mpi() do distribute
    main(rank_partition,distribute)
end