


function RathmannVisc(f,ε,Ecc,Eca,n)
    με = RathmannRheo_op∘(f,ε,Ecc,Eca,n)
    return με
end


function RathmannRheo_op(f::VectorValue,ε::SymTensorValue{3,Float64,6},Ecc::Float64,Eca::Float64,n::Float64)


    f = f.data
    # put C and W in arrays
    ε_array = [ε[1,1] ε[1,2] ε[1,3];
               ε[2,1] ε[2,2] ε[2,3];
               ε[3,1] ε[3,2] ε[3,3]]


    e1,e2,e3, eigvals = sf.frame(f,'e')

    α = 0.0125
    # Eij_grain = (Ecc^(2/(n_g+1)), Eca^(2/(n_g+1)))
    Eij_grain = (Ecc,Eca)

    Eij = sf.Eij_tranisotropic(f,e1,e2,e3,Eij_grain,α,1)

    τ = sf.rheo_rev_orthotropic_Martin(ε_array,1,n,e1,e2,e3,Eij)

    με = τ/(0.5*ε⊙ε+1e-9)^((1-n)/(2*n))

    return με

end


function RathmannRheo_op(f::VectorValue,ε::SymTensorValue{2,Float64,3},Ecc::Float64,Eca::Float64,n::Float64)
    ε = T₂toT₃_op(ε)
    return RathmannRheo_op(f,ε,Ecc,Eca,n)
end


