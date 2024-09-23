function removeZ(x::VectorValue{2,Float64})
    return VectorValue(x[1])
end

function removeZ(x::VectorValue{3,Float64})
    return VectorValue(x[1],x[2])
end

function Zonly(x::VectorValue)
    return Float64(x[end])
end


struct Surface
    order :: Int
    degree :: Int
    Lstretch :: Union{CellField,GridapDistributed.DistributedCellField}
    κ :: Float64
    b :: Function
    s :: Function
    dtags 
    bcs 
    Ω::Union{Triangulation,GridapDistributed.DistributedTriangulation}
    Γ::Union{BoundaryTriangulation,GridapDistributed.DistributedTriangulation}
    V :: Union{FESpace,GridapDistributed.DistributedFESpace}
    Z :: Union{FESpace,GridapDistributed.DistributedFESpace} 
    z0 :: Function
    solver

    function Surface(model,b,s,bcs,L,solver=LUSolver(),dtags=["bottom"],order=2,κ=1e-4)
        Lstretch = CellField(VectorValue(L),Triangulation(model))
        degree = 2*order
        reffe = ReferenceFE(lagrangian,Float64,order)
        Ω = Triangulation(model)
        Γ =  BoundaryTriangulation(model,tags="top")
        V = TestFESpace(Ω,reffe,dirichlet_tags=dtags)
        Z = TrialFESpace(V,bcs)
        z0 = x -> x[end]*(s(x)-b(x))+b(x)
        return new(order,degree,Lstretch,κ,b,s,dtags,bcs,Ω,Γ,V,Z,z0,solver)
    end
end

# function init_z(model,bcs,b,s,dtags=["bottom"],order=2)
#     reffeh = ReferenceFE(lagrangian,Float64,order)

#     Ω = Triangulation(model)

#     V = TestFESpace(Ω,reffeh,dirichlet_tags=dtags)
#     Z = TrialFESpace(V,bcs)

#     z0 = x -> x[end]*(s(x)-b(x))+b(x) 

#     zh = interpolate_everywhere(z0,Z)

#     return zh
# end


function solve_z!(zh,surf,dt,uh,acc=0.0)

    dΩ = Measure(surf.Ω,surf.degree)
    dΓ = Measure(surf.Γ,surf.degree)

    ∇x = transform_gradient(zh,surf.Lstretch)
    ∇s(u) = removeZ∘∇x(u)
    ∇z(u) = Zonly∘∇x(u)
    ∫_Ωx = transform_integral(zh,surf.Lstretch)

    us = removeZ∘uh
    w = Zonly∘uh
    
    # Solve for new surface
    aΓ(z,v) = ∫_Ωx( z*v + dt*us⋅∇s(z)*v )dΓ
    aΩ(z,v) = ∫_Ωx( surf.κ*∇x(z)⊙∇x(v) )dΩ

    lΓ(v) = ∫_Ωx( (zh + dt*(w + acc) )*v )dΓ  #

    a(z,v) = aΓ(z,v) + aΩ(z,v)
    l(v) = lΓ(v)
    

    op = AffineFEOperator(a,l,surf.Z,surf.V)
    
    solve!(zh,surf.solver,op)
    # zh = solve(solver,op)
end
