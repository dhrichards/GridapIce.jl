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
using LinearAlgebra: eigvecs, Diagonal
using Gridap.Arrays
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers
using GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver, NonlinearSystemBlock, TriformBlock
# using PyCall

# const sf = PyNULL()

# function __init__()
#     copy!(sf, pyimport("specfabpy").specfab)
# end 


include("cases.jl")
include("tensorfunctions.jl")
include("gaunt_body.jl")
include("specfabinjulia.jl")
include("fabric_solve.jl")
include("surface_solvers.jl")
include("Rheology/sachs.jl")
include("meshes.jl")
include("coordinatetransform.jl")
include("stokessolvers.jl")
include("fixes.jl")
include("Rheology/Rathmann.jl")
include("Rheology/orthotropic.jl")
include("Rheology/flowlaws.jl")

