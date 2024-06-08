abstract type Case end

struct ISMIPHOM <: Case
    experiment::Symbol # B, C,
    D::Int # 2, 3
    L::Union{Float64,Tuple} # Lx for 2D, (Lx,Ly) for 3D
    H::Float64 # Depth
    coords::Tuple
    s::Function # Surface
    b::Function # Bed
    β::Function # Basal friction coefficient
    α::Float64 # Slope
    ❄️::Float64 # Accumulation 
    periodicity::Tuple # (x,y,z)
    dtags # Dirichlet tags
    ubcs # Dirichlet values
    dmasks # Dirichlet masks

    function ISMIPHOM(experiment::Symbol,L::Float64)

        
        bB(x) = 0.5*H*sin(x[1]*2*pi/L)
        bC(x) = 0.5*H*sin(x[1]*2*pi/L)*sin(x[2]*2*pi/L)
        bD(x) = 0.0

        βBC(x) = 0.0
        βD(x) = 1e-3*(1.0 + sin(2*pi*x[1]/L))

        if experiment == :B
            s = x -> H
            H = 1e3
            L = (L)
            coords = (0,L,0,H)
            D = 2
            b = bB
            β = βBC
            α = 0.5 # degrees
            ❄️ = 0.0
            dtags = ["bottom"]
            ubcs = [VectorValue(0.0,0.0)]
            dmasks = [(true,true)]

        elseif experiment == :B_3D
            s = x -> H
            H = 1e3
            L = (L,1e3)
            coords = (0,L[1],0,L[2],0,H)
            D = 3
            b = x -> 0.5*H*sin(x[1]*2*pi/L[1])
            β = βBC
            α = 0.5
            ❄️ = 0.0
            dtags = ["bottom"]
            ubcs = [VectorValue(0.0,0.0,0.0)]
            dmasks = [(true,true,true)]

    
        elseif experiment == :A
            s = x -> H
            H = 1e3
            L = (L,L)
            coords = (0,L[1],0,L[2],0,H)
            D = 3
            b = x -> 0.5*H*sin(x[1]*2*pi/L[1])*sin(x[2]*2*pi/L[2])
            β = x -> 0.0
            α = 0.5 # degrees
            ❄️ = 0.0
            dtags = ["bottom"]
            ubcs = [VectorValue(0.0,0.0,0.0)]
            dmasks = [(true,true,true)]

        elseif experiment == :D
            s = x -> H
            H = 1e3
            L = (L)
            coords = (0,L,0,H)
            D = 2
            b = bD
            β = βD
            α = 0.5 # degrees
            ❄️ = 0.0
            dtags = []
            ubcs = []
            dmasks = []
        elseif experiment == :F1
            s = x -> H
            H = 1e3
            L = (L,L)
            coords = (0,L,0,L,0,H)
            D = 3
            σ = 10e3; a₀ = 100.0
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


        new(experiment,D,L,H,coords,s,b,β,α,❄️,periodicity,dtags,ubcs,dmasks)
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