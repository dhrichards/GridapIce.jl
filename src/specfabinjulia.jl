

function nlm_len_from_L(Lmax::Int)
    return Int(((Lmax+1)*(Lmax+2))/2)
end


function makeM(W,C,λSR,L)
    return makeM_op∘(W,C,λSR,L)
end

function makeM_op(W::TensorValue,C::SymTensorValue{2,Float64,3},λSR::Float64,L::Int)
    C = T₂toT₃_op(C)
    W = T₂toT₃_op(W)
    return makeM_op(W,C,λSR,L)
end

function makeM_op(W::TensorValue,C::SymTensorValue{3,Float64,6},λSR::Float64,L::Int)
    # put C and W in arrays
    C = [C[1,1] C[1,2] C[1,3];
            C[2,1] C[2,2] C[2,3];
            C[3,1] C[3,2] C[3,3]]

    W = [W[1,1] W[1,2] W[1,3];
            W[2,1] W[2,2] W[2,3];
            W[3,1] W[3,2] W[3,3]]

    
    M_lrot = M_LROT(L,C,W,1.0)
    M_reg = M_REG(L,C)
    M_rrx = M_CDRX(L)

    M = TensorValue(M_lrot  + M_reg + λSR*M_rrx)
    return M
end


function M_LROT(Lmax::Int,ε::Array,ω::Array,ι::Float64)

    # Quadratic expansion coefficients
    qe = quad_rr(ε*ι)
    qo = quad_tp(ω)

    # Harmonic interaction weights
    g0_rot = [0,0,0.0,0,0,0]
    gz_rot = [-im*√(3)*qo[2],0,0,0,0,0]
    gn_rot = [-im*6/√(6)*qo[1],0,0,0,0,0]
    gp_rot = [im*6/√(6)*qo[3],0,0,0,0,0]

    g0_Tay = 3*[0,qe[1],qe[2],qe[3],qe[4],qe[5]]
    gz_Tay = [0,-qe[1],0,0,0,qe[5]]
    gn_Tay = [√(5/6)*qe[2],0,qe[1],√(2/3)*qe[2],√(3/2)*qe[3],2*qe[4]]
    gp_Tay = [√(5/6)*qe[4],2*qe[2],√(3/2)*qe[3],√(2/3)*qe[4],qe[5],0]

    g0 = g0_rot + g0_Tay
    gz = gz_rot + gz_Tay
    gn = gn_rot + gn_Tay
    gp = gp_rot + gp_Tay


    nlm_len = nlm_len_from_L(Lmax)
    M_LROT = zeros(ComplexF64,nlm_len,nlm_len)
    for i = 1:nlm_len
        M_LROT[i,:] = -1*(GC[i,1:nlm_len,1:6]*g0 
                        + GCm[i,1:nlm_len,1:6]*gz
                        + GC_m1[i,1:nlm_len,1:6]*gn
                        + GC_p1[i,1:nlm_len,1:6]*gp)
    end

    return M_LROT
end





function M_CDRX(Lmax::Int)
    # Continuous dynamic recrystallization (CDRX)
    # Rotation recrystallization (polygonization) as a Laplacian diffusion process (Godert, 2003).
    # Returns matrix M such that d/dt (nlm)_i = M_ij (nlm)_j
    
    # NOTICE: This gives the unscaled effect of CDRX. The caller must multiply by an appropriate CDRX 
    # rate factor (scale) that should depend on temperature, stress, etc.
    Ldiag = Float64[]
    for ll in 0:2:Lmax
        for mm in -ll:ll
            push!(Ldiag, -(ll * (ll + 1)))
        end
    end
    M_CDRX = Diagonal(Ldiag)  # Use Julia's Diagonal matrix constructor

    return M_CDRX
end

function M_REG(Lmax::Int,ε::Array)


    # For L=4 only
    expo = 1.700
    nu   = 1.9879322126397958

    Ldiag = Float64[]
    for ll in 0:2:Lmax
        for mm in -ll:ll
            push!(Ldiag, -(ll * (ll + 1)))
        end
    end

    Reg_diag = abs.(Ldiag./(Lmax*(Lmax+1))).^expo

    ratemag = nu*norm(ε)

    return -ratemag*Diagonal(Reg_diag)
end

function quad_rr(M::Array)
    fsq = √(2*π/15)
    return [fsq*    (M[1,1]-M[2,2] + 2*im*M[1,2]),
            +2*fsq* (M[1,3] + im*M[2,3]),
            -2/3*√(π/5)*(M[1,1] + M[2,2] - 2*M[3,3]),
            -2*fsq* (M[1,3] - im*M[2,3]),
            +fsq*    (M[1,1]-M[2,2] - 2*im*M[1,2])]
end

function quad_tp(M::Array)
    fsq = √(2*π/3)
    return fsq*[(M[2,3] - im*M[1,3]),
                √(2)*(M[1,2]),
                (-M[2,3] - im*M[1,3])]
end

function lm2idx(l::Int,m::Int)
    return Int((l*(l+1))/2 + m + 1)
end

function lm(Lmax)
    l = []
    m = []
    for ll in 0:2:Lmax
        for mm in -ll:ll
            push!(l,ll)
            push!(m,mm)
        end
    end
    return l,m
end



function a2calc(f::Union{MultiFieldFEFunction,GridapDistributed.DistributedMultiFieldFEFunction})
    fr,fi = f
    return a2calc_op∘(fr,fi)
end

function a4calc(f::Union{MultiFieldFEFunction,GridapDistributed.DistributedMultiFieldFEFunction})
    fr,fi = f
    return a4calc_op∘(fr,fi)
end


function a2calc(f0::CellField)
    return a2calc_op∘(f0)
end

function a4calc(f0::CellField)
    return a4calc_op∘(f0)
end

function a2calc_op(fr::VectorValue,fi::VectorValue)
    f = fr + im*fi
    return a2calc_op(f)
end

function a4calc_op(fr::VectorValue,fi::VectorValue)
    f = fr + im*fi
    return a4calc_op(f)
end



function a2calc2d(f::Union{MultiFieldFEFunction,GridapDistributed.DistributedMultiFieldFEFunction})
    fr,fi = f
    return a2calc2d_op∘(fr,fi)
end

function a4calc2d(f::Union{MultiFieldFEFunction,GridapDistributed.DistributedMultiFieldFEFunction})
    fr,fi = f
    return a4calc2d_op∘(fr,fi)
end

function a2calc2d(f::CellField)
    return a2calc2d_op∘(f)
end
function a4calc2d(f::CellField)
    return a4calc2d_op∘(f)
end

function a2calc2d_op(fr::VectorValue,fi::VectorValue)
    f = fr + im*fi
    return a2calc2d_op(f)
end

function a4calc2d_op(fr::VectorValue,fi::VectorValue)
    f = fr + im*fi
    return a4calc2d_op(f)
end

function a2calc2d_op(f::VectorValue)
    return T₃toT₂_op(a2calc_op(f))
end

function a4calc2d_op(f::VectorValue)
    return T₃toT₂_op(a4calc_op(f))
end

function a2calc_op(f::VectorValue)
    a11=((0.33333333333333333333)+(0)*im)*f[lm2idx(0,0)]+ ((-0.14907119849998597976)+(0)*im)*f[lm2idx(2,0)]+ 2*((0.18257418583505537115)+(0)*im)*f[lm2idx(2,2)]
    a12=2*((0)+(0.18257418583505537115)*im)*f[lm2idx(2,2)]
    a13=2*((-0.18257418583505537115)+(0)*im)*f[lm2idx(2,1)]
    a22=((0.33333333333333333333)+(0)*im)*f[lm2idx(0,0)]+ ((-0.14907119849998597976)+(0)*im)*f[lm2idx(2,0)]+ 2*((-0.18257418583505537115)+(0)*im)*f[lm2idx(2,2)]
    a23=2*((0)+(-0.18257418583505537115)*im)*f[lm2idx(2,1)]
    a33=((0.33333333333333333333)+(0)*im)*f[lm2idx(0,0)]+ ((0.29814239699997195952)+(0)*im)*f[lm2idx(2,0)]

    return SymTensorValue(real(a11),real(a12),real(a13),real(a22),real(a23),real(a33))*√(4*π)
end
    


    
function a4calc_op(f::VectorValue)

    a1111=((0.2)+(0)*im)*f[lm2idx(0,0)]+ ((-0.12777531299998798265)+(0)*im)*f[lm2idx(2,0)]+ 2*((0.15649215928719031813)+(0)*im)*f[lm2idx(2,2)] +
            ((0.028571428571428571429)+(0)*im)*f[lm2idx(4,0)]+ 2*((-0.030116930096841707924)+(0)*im)*f[lm2idx(4,2)]+ 2*((0.039840953644479787999)+(0)*im)*f[lm2idx(4,4)]

    a1112=2*((0)+(0.078246079643595159065)*im)*f[lm2idx(2,2)]+ 2*((0)+(-0.015058465048420853962)*im)*f[lm2idx(4,2)]+ 2*((0)+(0.039840953644479787999)*im)*f[lm2idx(4,4)]

    a1113=2*((-0.078246079643595159065)+(0)*im)*f[lm2idx(2,1)]+ 2*((0.031943828249996995663)+(0)*im)*f[lm2idx(4,1)]+ 2*((-0.028171808490950552584)+(0)*im)*f[lm2idx(4,3)]


    a1122=((0.066666666666666666667)+(0)*im)*f[lm2idx(0,0)]+ ((-0.042591770999995994217)+(0)*im)*f[lm2idx(2,0)]+ ((0.0095238095238095238095)+(0)*im)*f[lm2idx(4,0)]+ 2*((-0.039840953644479787999)+(0)*im)*f[lm2idx(4,4)]


    a1123=2*((0)+(-0.026082026547865053022)*im)*f[lm2idx(2,1)]+ 2*((0)+(0.010647942749998998554)*im)*f[lm2idx(4,1)]+ 2*((0)+(-0.028171808490950552584)*im)*f[lm2idx(4,3)]


    a1133=((0.066666666666666666667)+(0)*im)*f[lm2idx(0,0)]+ ((0.021295885499997997109)+(0)*im)*f[lm2idx(2,0)]+ 2*((0.026082026547865053022)+(0)*im)*f[lm2idx(2,2)] +
                ((-0.038095238095238095238)+(0)*im)*f[lm2idx(4,0)]+ 2*((0.030116930096841707924)+(0)*im)*f[lm2idx(4,2)]

    a1222=2*((0)+(0.078246079643595159065)*im)*f[lm2idx(2,2)]+ 2*((0)+(-0.015058465048420853962)*im)*f[lm2idx(4,2)]+ 2*((0)+(-0.039840953644479787999)*im)*f[lm2idx(4,4)]


    a1223=2*((-0.026082026547865053022)+(0)*im)*f[lm2idx(2,1)]+ 2*((0.010647942749998998554)+(0)*im)*f[lm2idx(4,1)]+ 2*((0.028171808490950552584)+(0)*im)*f[lm2idx(4,3)]


    a1233=2*((0)+(0.026082026547865053022)*im)*f[lm2idx(2,2)]+ 2*((0)+(0.030116930096841707924)*im)*f[lm2idx(4,2)]


    a1333=2*((-0.078246079643595159065)+(0)*im)*f[lm2idx(2,1)]+ 2*((-0.042591770999995994217)+(0)*im)*f[lm2idx(4,1)]


    a2222=((0.2)+(0)*im)*f[lm2idx(0,0)]+ ((-0.12777531299998798265)+(0)*im)*f[lm2idx(2,0)]+ 2*((-0.15649215928719031813)+(0)*im)*f[lm2idx(2,2)] +
         ((0.028571428571428571429)+(0)*im)*f[lm2idx(4,0)]+ 2*((0.030116930096841707924)+(0)*im)*f[lm2idx(4,2)]+ 2*((0.039840953644479787999)+(0)*im)*f[lm2idx(4,4)]

    a2223=2*((0)+(-0.078246079643595159065)*im)*f[lm2idx(2,1)]+ 2*((0)+(0.031943828249996995663)*im)*f[lm2idx(4,1)]+ 2*((0)+(0.028171808490950552584)*im)*f[lm2idx(4,3)]


    a2233=((0.066666666666666666667)+(0)*im)*f[lm2idx(0,0)]+ ((0.021295885499997997109)+(0)*im)*f[lm2idx(2,0)]+ 2*((-0.026082026547865053022)+(0)*im)*f[lm2idx(2,2)]+ ((-0.038095238095238095238)+(0)*im)*f[lm2idx(4,0)]+ 2*((-0.030116930096841707924)+(0)*im)*f[lm2idx(4,2)]

    a2333=2*((0)+(-0.078246079643595159065)*im)*f[lm2idx(2,1)]+ 2*((0)+(-0.042591770999995994217)*im)*f[lm2idx(4,1)]


    a3333=((0.2)+(0)*im)*f[lm2idx(0,0)]+ ((0.2555506259999759653)+(0)*im)*f[lm2idx(2,0)]+ ((0.076190476190476190476)+(0)*im)*f[lm2idx(4,0)]

    

    return SymFourthOrderTensorValue(   real(a1111),real(a1112),real(a1113),real(a1122),real(a1123),real(a1133),
                                        real(a1112),real(a1122),real(a1123),real(a1222),real(a1223),real(a1233),
                                        real(a1113),real(a1123),real(a1133),real(a1223),real(a1233),real(a1333),
                                        real(a1122),real(a1222),real(a1223),real(a2222),real(a2223),real(a2233),
                                        real(a1123),real(a1223),real(a1233),real(a2223),real(a2233),real(a2333),
                                        real(a1133),real(a1233),real(a1333),real(a2233),real(a2333),real(a3333))*√(4*π)
end
