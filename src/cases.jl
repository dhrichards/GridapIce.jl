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

        elseif experiment == :F1_2D
            s = x -> H
            D = 2
            σ = 0.1; a₀ = 100.0
            b = x -> a₀*(exp(-(x[1]^2)/σ^2))
            β = x -> 0.0
            α = 3.0 # degrees
            ❄️ = 0.0
            dtags = ["bottom"]
            ubcs = [VectorValue(0.0,0.0)]
            dmasks = [(true,true)]

        end

        

        
        if D == 2
            periodicity = (true,false)
        else
            periodicity = (true,true,false)
        end


        new(experiment,D,H,s,b,β,α,❄️,periodicity,dtags,ubcs,dmasks)
    end

end


struct CustomB <: Case
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

    function CustomB(H,bh,α)
        s = x -> H
        D = 2
        b = x -> bh*H*sin(x[1]*2*π) 
        β = x -> 0.0

        ❄️ = 0.0
        dtags = ["bottom"]
        ubcs = [VectorValue(0.0,0.0)]
        dmasks = [(true,true)]

        periodicity = (true,false)


        new(D,H,s,b,β,α,❄️,periodicity,dtags,ubcs,dmasks)

    end

end


#General
struct Problem <: Case
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
    dmasks

    function General(L,H,s = x-> 0.0, b = x-> 0.0, β = x-> 0.0, α = 0.0, ❄️ = 0.0, periodicity = (false,false), dtags = [], ubcs = [], dmasks = [])
        D = length(L)
        new(D,L,H,s,b,β,α,❄️,periodicity,dtags,ubcs,dmasks)
    end
end

struct StickSlip <: Case
    D::Int
    L
    H::Float64
    s::Function
    b::Function
    β::Function
    α::Float64
    ❄️::Float64
    periodicity::Tuple
    dtags
    ubcs
    dmasks

    function StickSlip(L,α=0.1,D=2)
        H = 1.0
        if D == 2
            L = (L)
            periodicity = (false,false)

            
        else
            L = (L,1.0)
            periodicity = (false,false,false)
        end
            
    
        s = x -> H
        b = x -> 0.0

        # β = 1e8 for x<0, 0 for x>0
        β = x -> 1e8*(x[1] < 0.0)
        ❄️ = 0.0
        
        if D == 2
            uin(x) = VectorValue(x[end] - 0.5*x[end]^2,0.0)
            ubcs = [uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0),VectorValue(1.0/3,0.0)]
            dmasks = [(true,true),(true,true),(false,true),(true,true)]
            dtags = ["left","bottom_left","bottom_right","right"]
        else
            uin = x-> VectorValue(x -> x[end] - 0.5*x[end]^2,0.0,0.0)
            ubcs = [uin,VectorValue(0.0,0.0,0.0),VectorValue(0.0,0.0,0.0),VectorValue(1.0/3,0.0,0.0),VectorValue(0.0,0.0,0.0),VectorValue(0.0,0.0,0.0)]
            dmasks = [(true,true,true),(true,true,true),(false,false,true)(true,true,true),(false,true,false),(false,true,false)]
            dtags = ["left","bottom_left","bottom_right","right","front","back"]
        end
         

        new(D,L,H,s,b,β,α,❄️,periodicity,dtags,ubcs,dmasks)
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