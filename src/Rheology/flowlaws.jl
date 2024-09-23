struct Rheology
    Ecc::Float64
    Eca::Float64
    n :: Float64
    B :: Float64
    flowlaw::Symbol
    με::Function
    τhat::Function
    d_με::Function
    η::Function
    dη::Function


    function Rheology(Ecc,Eca,n,B,flowlaw,Ω,D)

        Ecc_cf = CellField(Ecc,Ω)
        Eca_cf = CellField(Eca,Ω)
        n_cf = CellField(n,Ω)
        
        η(ε) = B^(-1/n)*(0.5*ε⊙ε+1e-9)^((1-n)/(2*n))
        dη(dε,ε) = B^(-1/n)*(1-n)/(2*n)*(0.5*ε⊙ε+1e-9)^((1-n)/(2*n)-1)*0.5*(dε⊙ε+ε⊙dε)
    


        if flowlaw == :Sachs
            if D == 2
                με = (f,ε) -> SachsVisc(a2calc2d(f),a4calc2d(f),Ecc_cf,Eca_cf)⊙ε
                τhat = (f,ε) -> Sachsτhat(a2calc2d(f),a4calc2d(f),ε,Ecc_cf,Eca_cf)
                d_με = (f,dε,ε) -> SachsVisc(a2calc2d(f),a4calc2d(f),Ecc_cf,Eca_cf)⊙dε
            else
                με = (f,ε) -> SachsVisc(a2calc(f),a4calc(f),Ecc_cf,Eca_cf)⊙ε
                τhat = (f,ε) -> Sachsτhat(a2calc(f),a4calc(f),ε,Ecc_cf,Eca_cf)
                d_με = (f,dε,ε) -> SachsVisc(a2calc(f),a4calc(f),Ecc_cf,Eca_cf)⊙dε
            end
        elseif flowlaw == :Glen
            με = (f,ε) -> ε
            τhat = (f,ε) -> ε
            d_με = (f,dε,ε) -> dε
        elseif flowlaw == :RathmannMartin
            με = (f,ε) -> RathmannVisc(f,ε,Ecc_cf,Eca_cf,n_cf)
            τhat = με
            d_με = (f,dε,ε) -> RathmannVisc(f,dε,Ecc_cf,Eca_cf,n_cf)
        elseif flowlaw == :Rathmann
            if D ==2
                με = (f,ε) -> T₃toT₂(Orthotropic∘(a2calc(f),a4calc(f),Ecc_cf,Eca_cf))⊙ε
                τhat = με
                d_με = (f,dε,ε) -> T₃toT₂(Orthotropic∘(a2calc(f),a4calc(f),Ecc_cf,Eca_cf))⊙dε
            else
                με = (f,ε) -> Orthotropic(a2calc(f),a4calc(f),Ecc_cf,Eca_cf)⊙ε
                τhat = με
                d_με = (f,dε,ε) -> Orthotropic(a2calc(f),a4calc(f),Ecc_cf,Eca_cf)⊙dε
            end
        elseif flowlaw == :CAFFE
            με = (f,ε) -> CAFFEVisc(f,ε)*ε
            τhat = με
        else
            error("Unknown flowlaw")
        end


        return new(Ecc,Eca,n,B,flowlaw,με,τhat,d_με,η,dη)
    end
end

