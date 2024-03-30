
function Sachsτhat(A,A⁴,uh,Ecc,Eca)
    μ = SachsVisc(A,A⁴,Ecc,Eca)
    τ = μ⊙ε(uh)
    ι = Eca/(0.4*Eca + 0.2*Ecc + 0.4)
    return ι*τ
end

function SachsVisc(A,A⁴,Ecc,Eca)
    μ = inv4∘SachsFluidity(A,A⁴,Ecc,Eca)
    return μ
end

function SachsVisc(A,A⁴)
    F = SachsFluidity_op∘(A,A⁴)
    μ = inv4∘F
    return μ
end



SachsFluidity(A,A⁴,Ecc,Eca) = Operation(SachsFluidity_op)(A,A⁴,Ecc,Eca)
function evaluate!(cache,::Broadcasting{typeof(SachsFluidity)},A,A⁴,Ecc,Eca)
    Broadcasting(Operation(SachsFluidity_op))(Broadcasting(A),Broadcasting(A⁴),Ecc,Eca)
end

function SachsFluidity_op(A::SymTensorValue,A⁴::SymFourthOrderTensorValue)
    return SachsFluidity_op(A,A⁴,1.0,25.0)
end

function SachsFluidity_op(A::SymTensorValue,A⁴::SymFourthOrderTensorValue,Ecc::Float64,Eca::Float64)
    p₁ = -(Ecc-1)/2
    p₂ = (3*(Ecc-1) - 4*(Eca-1))/2
    p₃ = Eca -1

    
    δ = one(A)
    δ⁴ = one(A⁴)
    F = δ⁴ + p₂*A⁴ + p₃*AntiCommuter(A) + p₁*δ⊗A 
    return F/(0.4*Eca + 0.2*Ecc + 0.4)
end


