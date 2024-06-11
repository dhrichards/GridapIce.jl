abstract type Case end

struct ISMIPHOM <: Case
    experiment::Symbol # B, C,
    D::Int # 2, 3
    H::Float64 # Depth
    s::Function # Surface
    b::Function # Bed
    β::Function # Basal friction coefficient
    α::Float64 # Slope
    ❄️::Float64 # Accumulation 
    periodicity::Tuple # (x,y,z)
    dtags # Dirichlet tags
    ubcs # Dirichlet values
    dmasks # Dirichlet masks

    function ISMIPHOM(experiment::Symbol)
        H = 1e3

        if experiment == :A
            s = x -> H
            D = 3
            b = x -> 0.5*H*sin(x[1]*2*π)*sin(x[2]*2*π)
            β = x -> 0.0
            α = 0.5 # degrees
            ❄️ = 0.0
            dtags = ["bottom"]
            ubcs = [VectorValue(0.0,0.0,0.0)]
            dmasks = [(true,true,true)]


        elseif experiment == :B
            H = 1e3
            s = x -> H
            D = 2
            b = x -> 0.5*H*sin(x[1]*2*π) 
            β = x -> 0.0
            α = 0.5 # degrees
            ❄️ = 0.0
            dtags = ["bottom"]
            ubcs = [VectorValue(0.0,0.0)]
            dmasks = [(true,true)]

        elseif experiment == :B_3D
            s = x -> H
            D = 3
            b = x -> 0.5*H*sin(x[1]*2*π)
            β = βBC
            α = 0.5
            ❄️ = 0.0
            dtags = ["bottom"]
            ubcs = [VectorValue(0.0,0.0,0.0)]
            dmasks = [(true,true,true)]

        
        elseif experiment == :C
            s = x -> H
            D = 2
            b = x -> 0.0
            β = x -> 1e-3*(1.0 + sin(2*π*x[1])*sin(2*π*x[2]))
            α = 0.5 # degrees
            ❄️ = 0.0
            dtags = []
            ubcs = []
            dmasks = []

        elseif experiment == :D
            s = x -> H
            D = 2
            b = x -> 0.0
            β = x -> 1e-3*(1.0 + sin(2*π*x[1]))
            α = 0.5 # degrees
            ❄️ = 0.0
            dtags = []
            ubcs = []
            dmasks = []

        elseif experiment == :F1
            s = x -> H
            D = 3
            σ = 0.1; a₀ = 100.0
            b = x -> a₀*(exp(-(x[1]^2 + x[2]^2)/σ^2))
            β = x -> 0.0
            α = 3.0 # degrees
            ❄️ = 0.0
            dtags = ["bottom"]
            ubcs = [VectorValue(0.0,0.0,0.0)]
            dmasks = [(true,true,true)]
        end

        

        
        if D == 2
            periodicity = (true,false)
        else
            periodicity = (true,true,false)
        end


        new(experiment,D,H,s,b,β,α,❄️,periodicity,dtags,ubcs,dmasks)
    end

end


struct Divide <: Case
    D::Int
    L::Tuple
    H::Float64
    s::Function
    b::Function
    β::Function
    α::Float64
    ❄️::Float64
    periodicity::Tuple
    dtags
    ubcs

    function Divide(L::Float64)
        H = 1e3
        L = (L)
        D = 2
        s(x) = H
        b(x) = 0.0
        β(x) = 0.0
        α = 0.0 # degrees
        ❄️ = 0.0
        periodicity = (false,false)
        dtags = ["bottom","right"]
        
         
        ubcs = [VectorValue(0.0,0.0)]

        new(D,L,H,s,b,β,α,❄️,periodicity,dtags,ubcs)
    end

end