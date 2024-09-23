

function M_matrices(m)
    j = (2,3,1)
    k = (3,1,2)
    Mi = zeros(SymTensorValue{3,Float64,6},3)
    Mi3 = zeros(SymTensorValue{3,Float64,6},3)

    for i in 1:3
        Mi[i] = symmetric_part((m[j[i]]⊗m[j[i]] - m[k[i]]⊗m[k[i]])/2)
        Mi3[i] = symmetric_part((m[j[i]]⊗m[k[i]] + m[k[i]]⊗m[j[i]])/2)
    end
    return Mi, Mi3
end

function λγ_Full(E,pow)
    j = (2,3,1)
    k = (3,1,2)
    λ = zeros(6)
    γ = 0
    p = 2/(pow+1)
    for i in 1:3
        λ[i] = (4/3)*(E[j[i],j[i]]^p + E[k[i],k[i]]^p - E[i,i]^p)
        λ[i+3] = 2*E[j[i],k[i]]^p
        γ += 2*(E[j[i],j[i]]*E[k[i],k[i]])^p - E[i,i]^(2*p)
    end
    return λ, γ
end

function λγ_Petit(E)
    j = (2,3,1)
    k = (3,1,2)
    λ = zeros(6)
    γ = 0
    for i in 1:3
        λ[i] = (4/3)*(E[j[i],j[i]] + E[k[i],k[i]] - E[i,i])
        λ[i+3] = 2*E[j[i],k[i]]
        γ += 2*(E[j[i],j[i]]*E[k[i],k[i]]) - E[i,i]^2
    end
    return λ, γ
end


function λγ_Martin(E,n)
    j = (2,3,1)
    k = (3,1,2)
    λ = zeros(6)

    function f!(F,λ)
        for i in 1:3
            F[i] = (3/16)^((n+1)/2)*2*(λ[j[i]]+ λ[k[i]])*(λ[j[i]]^2 + λ[k[i]]^2 + λ[j[i]]*λ[k[i]])^((n-1)/2) - E[i,i]
        end
    end

    function j!(J,λ)
        for i in 1:3
            J[i,i] = 0
            J[i,j[i]] = (3/16)^((n+1)/2)*2*((n-1)/2)*(λ[j[i]]+λ[k[i]])*(2*λ[j[i]]+λ[k[i]])*(λ[j[i]]^2 + λ[k[i]]^2 + λ[j[i]]*λ[k[i]])^((n-3)/2) +
                             (3/16)^((n+1)/2)*2*(λ[j[i]]^2 + λ[k[i]]^2 + λ[j[i]]*λ[k[i]])^((n-1)/2)

            J[i,k[i]] = (3/16)^((n+1)/2)*2*((n-1)/2)*(λ[j[i]]+λ[k[i]])*(2*λ[k[i]]+λ[j[i]])*(λ[j[i]]^2 + λ[k[i]]^2 + λ[j[i]]*λ[k[i]])^((n-3)/2) +
                             (3/16)^((n+1)/2)*2*(λ[j[i]]^2 + λ[k[i]]^2 + λ[j[i]]*λ[k[i]])^((n-1)/2)
        end
    end
    
    res = nlsolve(f!,j![1,0.5,0.2])

    λ[1:3] = res.zero
    for i in 1:3
        λ[i+3] = 2*E[j[i],k[i]]^(1/n)
    end

    γ = (9/16)*(λ[1]*λ[2] + λ[1]*λ[3] + λ[2]*λ[3])
    return λ, γ
end

function ortho_rhs(λ,γ,m)

    j = (2,3,1)
    k = (3,1,2)

    Mi, Mi3 = M_matrices(m)
    δ = one(SymTensorValue{3,Float64,6})
    μ = zero(SymFourthOrderTensorValue{3,Float64,36})

    for i in 1:3
        μ += (λ[i]/γ)*(Mi[j[i]]-Mi[k[i]])⊗symm_op((δ-3*m[i]⊗m[i])/2) +
            (4/λ[i+3])*Mi3[i]⊗Mi3[i]
    end

    return μ
end

function N_Petit(λ,γ,m)
    j = (2,3,1)
    k = (3,1,2)

    Mi, Mi3 = M_matrices(m)
    N = zero(SymFourthOrderTensorValue{3,Float64,36})
    for i in 1:3
        N += 1.5*(λ[i]/γ)^2*(Mi[j[i]]-Mi[k[i]])⊗(Mi[j[i]]-Mi[k[i]]) +
            (8/λ[i+3]^2)*(Mi3[i]⊗Mi3[i]) +
             1.5*(λ[j[i]]*λ[k[i]]/γ^2)*(Mi[i]-Mi[j[i]])⊗(Mi[i]-Mi[k[i]])
    end
    return N
end

function N_Full(λ,γ,m)
    j = (2,3,1)
    k = (3,1,2)

    Mi, Mi3 = M_matrices(m)
    N = zero(SymFourthOrderTensorValue{3,Float64,36})
    for i in 1:3
        N += (λ[i]/γ)*(Mi[j[i]]-M[k[i]])⊗(Mi[j[i]]-M[k[i]]) +
            (4/λ[i+3])*(Mi3[i]⊗Mi3[i])
    end
    return N
end


function Orthotropic(A::SymTensorValue{3},A⁴,Ecc,Eca)
    m = eigenvectors(A)

    E = Eij_tranisotropic(A,A⁴,Ecc,Eca,0.0125)
    λ, γ = λγ_Full(E,3)

    μ = ortho_rhs(λ,γ,m)

    return μ
end


function Orthotropic_N(A::SymTensorValue{3},A⁴,Ecc,Eca)
    m = eigenvectors(A)
    E = Eij_tranisotropic(A,A⁴,Ecc,Eca,0.0125)
    λ, γ = λγ_Full(E,3)
    return N_Full(λ,γ,m)
end



function Eij_tranisotropic(A,A⁴,Ecc,Eca,α)
    e = eigenvectors(A)

    E1 = Evw_tranisotropic(e[1],e[1],τ_vv(e[1]),A,A⁴,Ecc,Eca,α)
    E2 = Evw_tranisotropic(e[2],e[2],τ_vv(e[2]),A,A⁴,Ecc,Eca,α)
    E3 = Evw_tranisotropic(e[3],e[3],τ_vv(e[3]),A,A⁴,Ecc,Eca,α)
    E4 = Evw_tranisotropic(e[2],e[3],τ_vw(e[2],e[3]),A,A⁴,Ecc,Eca,α)
    E5 = Evw_tranisotropic(e[1],e[3],τ_vw(e[1],e[3]),A,A⁴,Ecc,Eca,α)
    E6 = Evw_tranisotropic(e[1],e[2],τ_vw(e[1],e[2]),A,A⁴,Ecc,Eca,α)

    E = SymTensorValue(E1,E6,E5,E2,E4,E3)
    return E
end

function Evw_tranisotropic(v,w,τ,A,A⁴,Ecc,Eca,α)
    δ = one(A)
    Ai = δ/3
    A⁴i = LinearClosure_op(Ai)


    vw = v⊗w
    Evw_sachs = sachshomo(τ,A,A⁴,Ecc,Eca)⊙vw/(sachshomo(τ,Ai,A⁴i,Ecc,Eca)⊙vw)
    Evw_taylor = taylorhomo(τ,A,A⁴,Ecc,Eca)⊙vw/(taylorhomo(τ,Ai,A⁴i,Ecc,Eca)⊙vw)

    Evw = (1-α)*Evw_sachs + α*Evw_taylor

    return Evw
end


function τ_vv(v::VectorValue{3,Float64})
    δ = one(SymTensorValue{3,Float64,6})
    return δ/3 - v⊗v
end

function τ_vw(v::VectorValue{3,Float64},w::VectorValue{3,Float64})
    return (v⊗w + w⊗v)
end

function sachshomo(τ,A,A⁴,Ecc,Eca)
    δ = one(A)
    p₁ = -(Ecc-1)/2
    p₂ = (3*(Ecc-1) - 4*(Eca-1))/2
    p₃ = Eca -1
    return τ + p₁*(A⊙τ)*δ + p₂*A⁴⊙τ + p₃*(τ⋅A + A⋅τ)
end

function taylorhomo(τ,A,A⁴,Ecc,Eca)
    μ = SachsFluidity_op(A,A⁴,1/Ecc,1/Eca)
    return inv4(μ)⊙τ
end

function eigenvectors(A::SymTensorValue{3,Float64,6})
    Am = [A[1,1] A[1,2] A[1,3];
          A[2,1] A[2,2] A[2,3];
          A[3,1] A[3,2] A[3,3]]
    M = eigvecs(Am)
    m = zeros(VectorValue{3,Float64},3)
    for i in 1:3
        m[i] = VectorValue(M[1,i],M[2,i],M[3,i])
    end
    return m
end


function LinearClosure_op(A::SymTensorValue{3,Float64,6})
    return SymFourthOrderTensorValue(6*A[1,1]/7 - 3/35,3*A[1,2]/7,3*A[1,3]/7,A[1,1]/7 + A[2,2]/7 - 1/35,
                                A[2,3]/7,A[1,1]/7 + A[3,3]/7 - 1/35,3*A[1,2]/7,A[1,1]/7 + A[2,2]/7 - 1/35,
                                A[2,3]/7,3*A[1,2]/7,A[1,3]/7,A[1,2]/7,3*A[1,3]/7,A[2,3]/7,A[1,1]/7 + A[3,3]/7 - 1/35,
                                A[1,3]/7,A[1,2]/7,3*A[1,3]/7,A[1,1]/7 + A[2,2]/7 - 1/35,3*A[1,2]/7,A[1,3]/7,
                                6*A[2,2]/7 - 3/35,3*A[2,3]/7,A[2,2]/7 + A[3,3]/7 - 1/35,A[2,3]/7,A[1,3]/7,
                                A[1,2]/7,3*A[2,3]/7,A[2,2]/7 + A[3,3]/7 - 1/35,3*A[2,3]/7,A[1,1]/7 + A[3,3]/7 - 1/35,
                                A[1,2]/7,3*A[1,3]/7,A[2,2]/7 + A[3,3]/7 - 1/35,3*A[2,3]/7,6*A[3,3]/7 - 3/35)
end