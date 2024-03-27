using Gridap
using GridapDistributed
using GridapDistributed.Geometry
using GridapGmsh
using GridapPETSc
using GridapPETSc: PETSC
using PartitionedArrays

H = 1e3
L = 5e3
b(x) = 0.5*H*sin(x[1]*2*pi/L)
s(x) = H
z_mwe(x) = 10.0*x[2]
function deform_point(x::VectorValue{2,Float64},b,s)
    z = b(x[1]) + (x[2]/H)*(s(x[1])-b(x[1]))
    return VectorValue([x[1],z])
end

function deform_point(x::VectorValue{2,Float64},z)
    znew = z(x)
    return VectorValue([x[1],znew])
end

function deform_mesh(model::DiscreteModelPortion,b,s)
    map!(x -> deform_point(x,b,s),model.parent_model.grid.node_coordinates,
                                model.parent_model.grid.node_coordinates)

    return model
end

function deform_mesh(model::DiscreteModelPortion,z)
    map!(x -> deform_point(x,z),model.parent_model.grid.node_coordinates,
                                model.parent_model.grid.node_coordinates)

    return model
end

function deform_mesh(model::GridapDistributed.DistributedDiscreteModel,b,s)
    map(local_views(model)) do m
        deform_mesh(m,b,s)
        # print(fieldnames(typeof(m.model.grid)))
        # print(fieldnames(typeof(m.parent_model.grid)))

    end
    return model
end

function deform_mesh(model::GridapDistributed.DistributedDiscreteModel,z)
    map(local_views(model)) do m
        deform_mesh(m,z)
        # print(fieldnames(typeof(m.model.grid)))
        # print(fieldnames(typeof(m.parent_model.grid)))

    end
    return model
end


function main(ranks)

    model = GmshDiscreteModel(ranks,"./meshes/rectangle.msh")


    model = deform_mesh(model,b,s) # works
    writevtk(get_triangulation(model),"model") 

    Z = FESpace(model,ReferenceFE(lagrangian,Float64,1))
    z = interpolate_everywhere(z_mwe,Z)
    model = deform_mesh(model,z) # does not work


end

with_mpi() do distribute 
    ranks = distribute_with_mpi(LinearIndices((2,)))
    main(ranks)
  end

