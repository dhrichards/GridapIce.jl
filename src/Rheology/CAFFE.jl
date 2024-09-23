
function CAFFE_E(fh,ε)
    A = a2calc(fh)
    A⁴ = a4calc(fh)

    D = Deformability(A,A⁴,ε)
    E = Enhancement_op∘(D)

    return E
end

function CAFFEVisc(fh,ε)
    E = CAFFE_E(fh,ε)
    return E^(-1/n)*ε
end


function Deformability(A,A⁴,ε)
    return (5/(ε⊙ε+1e-9))*((ε⋅A)⋅ε - (A⁴⊙ε)⊙ε)
end


function Enhancement_op(D::Float64,Emin=0.1,Emax=10.0)
    τ = (8/21)*((Emax-1)/(1-Emin))

    if D<1
        E = (1-Emin)*D^τ + Emin
    else
        E = (4D^2*(Emax-1) + 25 - 4*Emax)/21
    end
    return E
end
