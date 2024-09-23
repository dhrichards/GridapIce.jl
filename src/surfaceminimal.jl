using Gridap
using GridapDistributed
using PartitionedArrays
using GridapPETSc


function main_ex2(rank_partition,distribute)
    parts  = distribute(LinearIndices((prod(rank_partition),)))
    options = "-ksp_type cg -pc_type gamg -ksp_monitor"
    GridapPETSc.with(args=split(options)) do
      domain = (0,1,0,1)
      mesh_partition = (4,4)
      model = CartesianDiscreteModel(parts,rank_partition,domain,mesh_partition)
      order = 2
      u((x,y)) = (x+y)^order
      f(x) = -Δ(u,x)
      reffe = ReferenceFE(lagrangian,Float64,order)
      V = TestFESpace(model,reffe,dirichlet_tags="boundary")
      U = TrialFESpace(u,V)
      Ω = Triangulation(model)
      dΩ = Measure(Ω,2*order)
      a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
      l(v) = ∫( v*f )dΩ
      op = AffineFEOperator(a,l,U,V)
      solver = PETScLinearSolver()
      uh = solve(solver,op)
      writevtk(Ω,"results_ex2",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
    end
  end
  
  rank_partition = (1,1)
  with_mpi() do distribute
    main_ex2(rank_partition,distribute)
  end