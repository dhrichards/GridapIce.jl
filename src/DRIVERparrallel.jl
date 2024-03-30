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

function testvalue(::Type{Tuple{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15}}) where {T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15}
    (testvalue(T1),testvalue(T2),testvalue(T3),testvalue(T4),testvalue(T5),testvalue(T6),testvalue(T7),testvalue(T8),testvalue(T9),testvalue(T10),testvalue(T11),testvalue(T12),testvalue(T13),testvalue(T14),testvalue(T15))
  end
  
function testvalue(::Type{Tuple{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15}}) where {T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15}
    (testvalue(T1),testvalue(T2),testvalue(T3),testvalue(T4),testvalue(T5),testvalue(T6),testvalue(T7),testvalue(T8),testvalue(T9),testvalue(T10),testvalue(T11),testvalue(T12),testvalue(T13),testvalue(T14),testvalue(T15))
  end



# options = "-ksp_type cg -pc_type gamg -ksp_monitor"
# use mumps
options = "-ksp_type fgmres -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor"
# function main(ranks)
function main(rank_partition,distribute)
    
    parts  = distribute(LinearIndices((prod(rank_partition),)))
    # GridapPETSc.with(args=split(options)) do

        problem = ISMIPHOM(:B,5e3)

        D = problem.D
        ncell = (32,16)

        model = CartesianDiscreteModel(parts,rank_partition,(0,problem.L,0,1),ncell,isperiodic=(true,false))
        labels = get_labels(model,D)
        
        stk = Stokes(model,problem,LUSolver())
        Ecc = 1.0; Eca = 25.0; n = 1.0; B = 100.0
        function FlowLaw(A,A⁴)
            μ = SachsVisc(A,A⁴)
            τ(ε) = μ⊙ε
            return τ
        end


        fab = SpecFab(model,D,:H1implicit,1,0.3,4)

        CFL = 10.0
        d = problem.L[1]/ncell[1]
        dt = 0.1
        nt = 500

        s,b,ζ,L = get_sb_fields(problem,model)
        M = Transform(ζ,b,s)
        h = s - b
        z = ζ*h + b

        fh = fab.f0h
        τ = FlowLaw(fab.a2∘(fh),fab.a4∘(fh))

        sol, res = solve_up_linear(zero(stk.X),τ,M,h,stk)
        uh, ph = sol
        writevtk(get_triangulation(model),"parrallel0", cellfields=["uh"=>uh,"ph"=>ph,"z"=>z,"s"=>s,"a2"=>fab.a2∘(fh)])


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

            
            writevtk(get_triangulation(model),"parrallel$i", cellfields=["uh"=>uh,"ph"=>ph,"z"=>z,"s"=>s,"a2"=>fab.a2∘(fh)])
        end

    # end

end




rank_partition = (2,1)
with_mpi() do distribute
    main(rank_partition,distribute)
end