function default_stokes_params()
  Dict(
    :n => 5,
    :d => 2,
    :d_bc => 2,
    :rel_exp => 1,
    :rel_disp => (0.0, 0.2, 0.0),
    :rot_angle => 0,
    :k_space => 2,
    :k_time => 1,
    :k_space_map => 1,
    :k_time_map => 1,
    :k_space_sol => 1,
    :k_time_sol => 1,
    :threshold => 1.0,
    :inlet => "inlet",
    :tini => 0,
    :tend => 1,
    :nsteps => 5,
    :nsubsteps => 1,
    :min_jacobian => 0.5,
    :umax => 1,
    :nu => 0.5,
    :Re => 10,
    :beta => 10,
    :beta_map => 10,
    :displacement => "translation",
    :expansion_type => "laplacian",
    :bodyfitted => false,
    :residuals => false,
    :verbose => true,
    :output => true,
    :outputname => "",
    :cond1 => true,
    :cond2 => false,
    :stlname => "cube",
    :problem_type => "stokes",
    :uniform_partition => false,
    :mesh_map_mask => (false,false,false),
    :mesh_map_order => 0.5,
    :solution_order => 1,
    :disp_smoother => 0,
    :aspect_ratio => (2,1,1),
    :box_exp_side => 0.0,
    :initial_value => "zero",
    :output_freq => 1,
    :tumax => 1/4,
    :force_term => (0,0,0),
    :sin_disp_A => (0,0,0),
    :sin_disp_w => (0,0,0),
    :sin_rot_A => (0,0,0),
    :sin_rot_w => (0,0,0),
  )
end

function setup_stokes_manufactured_solution(;d=2,m=2,δ=1,ν=1)
  @notimplementedif d != 2
  u = (xt) -> begin
    x,t = x_and_t(xt)
    VectorValue(
      x[1] - δ*t/2 + x[2]^m,
      -m*( x[1] - δ*t/2 )*x[2]^m,
      0
    )
  end
  p = (xt) -> begin
    x,t = x_and_t(xt)
    ( (x[1]+x[2]) - δ*t/2 )^(m-1)
  end
  f = (xt) -> ∂t_xt(u)(xt) - ν*(∇⋅(∇x(u)))(xt) + ∇x(p)(xt)
  g = (xt) -> tr(∇x(u)(xt))

  u,p,f,g
end

function inlet_velocity(xtmin,xtmax,d,Umax=1,dT=1/4)
  xmin,tmin = x_and_t(xtmin)
  xmax,tmax = x_and_t(xtmax)
  t0 = tmin
  T = tmax - tmin
  y0 = xmin[2]
  Ly = xmax[2] - xmin[2]
  z0 = xmin[3]
  Lz = xmax[3] - xmin[3]
  if dT > 0
    uin_max = Umax / dT
  else
    uin_max = Umax
  end
  tuin = T*dT

  function inlet_velocity_fun(xt)
    x,t = x_and_t(xt)
    y = (x[2]-y0)/Ly
    z = (x[3]-z0)/Lz
    if dT > 0
      t = min(t-t0,tuin)
    else
      t = 1.0
    end
    if d == 2
      ux = uin_max*t*(4*y-4*y^2)
    elseif d == 3
      ux = uin_max*t*(4*y-4*y^2)*(4*z-4*z^2)
    end
    VectorValue(ux,0.,0.)
  end
end

function time_smoother(tmin,tmax,s0=1/4;order=2)
  T = tmax - tmin
  t0::Float64 = s0*T
  f1::Float64 = 1 / t0^(order) * (t0/order)
  f2::Float64 = t0/order - t0
  function time_smoother_fun(t)
    Δt = t-tmin
    if Δt < t0
      Δt = (Δt)^order * f1
    else
      Δt = Δt + f2
    end
    tmin + Δt
  end
end

function stokes(;kwargs...)
  params = merge!(default_stokes_params(),Dict(kwargs...))
  stokes(params)
end

function stokes(params::Dict)
  all_params = merge!(default_stokes_params(),params)
  stokes(all_params,params)
end

function stokes(params::Dict,input_params::Dict)

  to = TimerOutput()

  nt::Int = params[:nsteps]
  nsubsteps::Int = params[:nsubsteps]
  tini::Float64 = params[:tini]
  tend::Float64 = params[:tend]
  stlname::String = params[:stlname]
  verbose::Bool = params[:verbose]
  # path_title::String = joinpath(dict[:path],dict[:title])
  threshold::Float64 = params[:threshold]
  ks::Int = params[:k_space]
  kt::Int = params[:k_time]
  rel_exp::Float64 = params[:rel_exp]
  rel_disp::NTuple{3,Float64} = params[:rel_disp]
  min_jacobian::Float64 = params[:min_jacobian]
  d::Int = params[:d]
  d_bc::Int = params[:d_bc]
  umax::Float64 = params[:umax]
  tumax::Float64 = params[:tumax]
  Re::Float64 = params[:Re]
  mesh_map_mask::NTuple{3,Bool} = params[:mesh_map_mask]
  mesh_map_order::Float64 = params[:mesh_map_order]
  tsmooth::Float64 = params[:disp_smoother]
  aratio::NTuple{3,Float64} = params[:aspect_ratio]
  δLmin::Float64 = params[:box_exp_side]
  initial_value::String = params[:initial_value]


  if isa(params[:n],Integer)
    n::Int = params[:n]
    @assert d == 2
    nx = ceil(Int,n*aratio[1])
    ny = ceil(Int,n*aratio[2])
    cells = (nx,ny,1)
  else
    cells::Tuple = params[:n]
  end

  γ = ks*(ks+1)
  γ0 = 1.0/10.0

  # μ = 1
  # ρ = 1
  # ν = μ/ρ
  β = params[:beta]
  # α = ρs

  D = 3
  Tv = VectorValue{D,Float64}
  Tp = Float64
  z_u = zero(Tv)
  z_p = zero(Tp)

  filename = joinpath(@__DIR__,"..","test","data","$(stlname).stl")
  if !isfile(filename)
    fileid = parse(Int,"$(stlname)")
    filename = download_thingi10k(fileid)
  end

  geo = STLGeometry(filename)

  # Scale to unit
  pmin,pmax = get_bounding_box(geo)
  Lmax = max( (pmax-pmin)... )
  geo = move_nodes(x->x/Lmax,geo)
  pmin,pmax = get_bounding_box(geo)

  expansion = (pmax-pmin)*rel_exp
  pmin -= expansion; pmax += expansion
  if d == 2
    L0 = pmax[2] - pmin[2]
  else
    L0 = pmax - pmin
  end
  L = VectorValue( Tuple(L0) .* (aratio) )
  ΔL = L - L0

  pmin -= ΔL*δLmin
  pmax += ΔL*(1-δLmin)

  hx = L[1] / cells[1] # average cell size
  Lx = L[1]
  xtmin = VectorValue(pmin...,tini)
  xtmax = VectorValue(pmax...,tend)

  Uin_max = params[:inlet] == "inlet" ? umax : 0.0
  Udisp_max = rel_disp[1]*hx / (tini-tend)
  Umax = max(Uin_max,Udisp_max)

  Lgeo = 1
  Umax = params[:inlet] == "manufactured" ? 1.0 : Umax
  ν = Umax * Lgeo / Re

  f = (x) -> zero(Tv)
  g = (x) -> zero(Tp)
  if params[:inlet] == "inlet"
    uin = inlet_velocity(xtmin,xtmax,d,umax,tumax)
  elseif params[:inlet] == "zero"
    uin = zero(Tv)
  elseif params[:inlet] == "manufactured"
    m::Int = params[:solution_order]
    u,p,f,g = setup_stokes_manufactured_solution(;d,ν,m)
    ut = t -> x -> u(VectorValue(x...,t))
    uin = u
  else
    @assert params[:inlet] == "free"
    f = VectorValue(params[:force_term]::Tuple)
  end


  mesh_map = get_mesh_map(mesh_map_mask,pmin,pmax,mesh_map_order)
  model = simplex_model(pmin,pmax,cells,mesh_map,d=d)
  tags = setup_tags!(model,d=d_bc)
  labels = get_face_labeling(model)
  wall_d_faces = get_tag_entities(labels,tags[:tags][:wall][1])


  # writevtk(model,"bgmodel")
  # writevtk(geo,"geo")
  inlet_faces = [5].+get_offset(HEX,2)
  wall_faces = [1,2,3,4].+get_offset(HEX,2)
  inlet_faces = sort(reduce(union,get_faces(HEX)[inlet_faces]))
  wall_faces = setdiff(wall_faces,wall_d_faces)
  wall_faces = sort(reduce(union,get_faces(HEX)[wall_faces]))
  wall_faces = setdiff(wall_faces,wall_d_faces)
  wall_faces = setdiff(wall_faces,inlet_faces)
  add_tag_from_tags!(labels,"inlet",inlet_faces)
  add_tag_from_tags!(labels,"wall",wall_faces)

  wall_d_tag = tags[:tags][:wall][1]
  wall_d_mask = tags[:masks][:wall][1]
  bnd_tag = tags[:tags][:boundary][1]
  bnd_mask = tags[:masks][:boundary][1]
  dirichlet_tags = ["wall",wall_d_tag]
  dirichlet_masks = [bnd_mask,wall_d_mask]
  dirichlet_funs = Any[z_u,z_u]
  if params[:inlet] != "free"
    pushfirst!(dirichlet_tags,"inlet")
    pushfirst!(dirichlet_masks,bnd_mask)
    pushfirst!(dirichlet_funs,uin)
  end

  p_tags = String[]
  p_funs = Any[]
  if params[:inlet] == "manufactured"
    dirichlet_tags = [bnd_tag,wall_d_tag]
    dirichlet_masks = [bnd_mask,wall_d_mask]
    dirichlet_funs = [u,z_u]
    p_tags = [bnd_tag,]
    p_funs = [p,]
  end

  disp_params = default_disp_params()
  disp_params[:beta] = params[:beta_map]
  disp_params[:verbose] = params[:verbose]
  disp_params[:order] = params[:k_space_map]
  disp_params[:type] = params[:expansion_type]
  disp_params[:dirichlet_tags] = tags[:tags][:dirichlet]
  disp_params[:dirichlet_masks] = tags[:masks][:dirichlet]


  τPSPG = β*(hx^2)/(4*ν)

  θt::Float64 = params[:rot_angle]
  pmin,pmax = get_bounding_box(geo)
  x0 = (pmin+pmax) / 2

  A_disp::Tuple = params[:sin_disp_A]
  A_disp = A_disp .* Tuple(pmax-pmin)
  ω_disp::Tuple = params[:sin_disp_w]

  A_rot::Tuple = params[:sin_rot_A]
  ω_rot::Tuple = params[:sin_rot_w]

  sin_disp(t) = VectorValue( (A_disp .* sin.( ω_disp .* t ))... )
  lindisp(t) = (t-tini)/(tend-tini)*hx*Point(rel_disp)
  disp(t) = sin_disp(t) + lindisp(t)

  sin_rot(t) = VectorValue( (A_rot .* sin.( ω_rot .* t ))... )
  lin_rot(t) = (t-tini)/(tend-tini)*θt*VectorValue(0,0,1)
  rot(t) = sin_rot(t) + lin_rot(t)
  ϕΓx(t) = translation(disp(t))
  ϕΓθ(t) = rotation(rot(t);x0=x0)
  invϕΓx(t) = inverse_translation(disp(t))
  invϕΓθ(t) = inverse_rotation(rot(t);x0=x0)

  _ϕΓ = (t) -> ϕΓx(t) ∘ ϕΓθ(t)
  _invϕΓ = (t) -> invϕΓθ(t) ∘ invϕΓx(t)

  ftime = time_smoother(tini,tend,tsmooth;order=2)
  ϕΓ = (t) -> _ϕΓ(ftime(t))
  invϕΓ = (t) -> _invϕΓ(ftime(t))

  strategy = AggregateCutCellsByThreshold(threshold)
  wΓ = (t::Tuple) -> x -> begin
    y = invϕΓ(t[1])(x)
    ϕΓ(t[2])(y) - ϕΓ(t[1])(y)
  end

  outputname::String = params[:outputname]
  output::Bool = params[:output]
  if outputname != ""
    output = true
  end
  if output
    if outputname == ""
      ignores=[:output,:verbose,:cond1,:cond2]
      ignores = [:output,:verbose,:cond1,:cond2,:savestatus,:outputpath,
      :outputname,:solver,:expansion_type,:nsubsteps,:tumax,:beta,:beta_map,
      :min_jacobian,:k_space,:k_time,:threshold,:rel_exp]
      outputname = savename(input_params;ignores)
    end
    mkpath("$(outputname)_vtu")
    pvd = createpvd("$(outputname).pvd")
    boundary_pvd = createpvd("$(outputname)_boundary.pvd")
    act_pvd = createpvd("$(outputname)_act.pvd")
    geo_pvd = createpvd("$(outputname)_geo.pvd")
    bg_pvd = createpvd("$(outputname)_bg.pvd")
  end

  @assert ks > 1
  reffe_t = ReferenceFE(lagrangian,Float64,kt)
  reffe_u = ReferenceFE(lagrangian,Tv,ks)
  reffe_p = ReferenceFE(lagrangian,Tp,ks-1)

  out = Dict{String,Any}()
  istep = 0
  Ωbg = Triangulation(model)
  @timeit to "cut" cutgeo,bgf_ioc = cut(STLCutter(),model,geo)
  remove_disconnected_parts!(cutgeo,OUT)
  aggregates = aggregate(strategy,cutgeo,cutgeo.geo,OUT,bgf_ioc)
  Ωact = Triangulation(cutgeo,ACTIVE_OUT)
  Ω0⁻ = Triangulation(cutgeo,PHYSICAL_OUT)
  Γ = EmbeddedBoundary(cutgeo)
  nΓ = -get_normal_vector(Γ)
  if params[:bodyfitted]
    Ω0⁻ = Ωact = Ωbg
    # Γ = BoundaryTriangulation(model;tags=bnd_tag)
    # wΓ = (T) -> x -> zero(Tv)
  end

  wh0 = zero
  if params[:inlet] == "manufactured"
    uh0⁻ = ut(tini)
    ph0 = z_p
  elseif initial_value == "zero"
    uh0⁻ = zero(Tv)
    ph0 = z_p
  elseif initial_value == "static"

    dirichlet_funs_0 = map(dirichlet_funs) do f
      if isa(f,Function)
        x -> f(Point(x...,tini))
      else
        f
      end
    end
    Ω = Ω0⁻

    Vstd = FESpace(Ωact,reffe_u;dirichlet_tags,dirichlet_masks,conformity=:H1)
    if params[:inlet] == "free"
      Qstd = FESpace(Ωact,reffe_p,constraint=:zeromean,conformity=:C0)
    else
      Qstd = FESpace(Ωact,reffe_p,conformity=:C0)
    end
    V = AgFEMSpace(Vstd,aggregates)
    Q = AgFEMSpace(Qstd,aggregates)
    U = TrialFESpace(V,dirichlet_funs_0)
    P = TrialFESpace(Q)
    X = MultiFieldFESpace([U,P])
    Y = MultiFieldFESpace([V,Q])

    ps = 2*ks
    pt = 2*kt
    dΩ = Measure(Ω,ps)
    dΓ = Measure(Γ,ps)

    uD0 = zero(Tv)
    ϕ = xt -> begin
     xt = Point(xt...)
     x,t = x_and_t(xt)
     ϕΓ(t)(x)
    end
    # ϕΓ( _extract_t(xt) )( _extract_x(xt) )
    uϕ = xt -> _extract_t(∇(ϕ)(xt))
    uD = x -> uϕ( Point(x...,tini) )
    # @show uD(Point(1,1,1))

    # writevtk(Ω,"trian",cellfields=["u"=>uD])
    # writevtk(Ωact,"trianact",cellfields=["u"=>uD])


    # f = Point(-1,0,0)
    τ = (β*ks^2)/hx
    cell_meas = get_cell_measure(Ωact)
    h = (cell_meas .* 6) .^(1/D)
    τ = CellField( (β*ks^2)./h, Ωact )


    function a0((u,p),(v,q))
      r = ∫(  ν*∇(v)⊙∇(u) - (∇⋅v)*p + (∇⋅u)*q )dΩ
      if !params[:bodyfitted]
        r = r +
          ∫( τ*v⋅u )dΓ +
          ∫( - v⋅(ν*nΓ⋅∇(u) - nΓ*p) )dΓ +
          ∫( - (ν*nΓ⋅∇(v) - nΓ*q)⋅u  )dΓ
      end
      r
    end

    function l0((v,q))
      r = ∫( v⋅(f) + q*(g) )dΩ
      if !params[:bodyfitted]
        r = r + ∫( τ*v⋅uD - (ν*nΓ⋅∇(v) - nΓ*q)⋅uD )dΓ
      end
      r
    end

    Tmat = SparseMatrixCSC{Float64,Int32}
    Tvec = Vector{Float64}
    assem = SparseMatrixAssembler(Tmat,Tvec,X,Y)


    if params[:problem_type] ∈  ("stokes","explicit-navier-stokes")

      @timeit to "assembly" op = AffineFEOperator(a0,l0,X,Y,assem)

      A = get_matrix(op)
      @show cond(A,1)
      if GridapPardiso.MKL_PARDISO_LOADED[]
        ls = PardisoSolver()
      else
        ls = LinearFESolver()
      end
      @timeit to "solve" uh,ph = solve(ls,op)

    elseif params[:problem_type] == "navier-stokes"
      c0(w,u,v) = ∫( v⋅((∇(u))'⋅w) )dΩ
      dc0(u,du,v) = c0(du,u,v) + c0(u,du,v)

      res0((u,p),(v,q)) = a0((u,p),(v,q)) + c0(u,u,v)- l0((v,q))
      jac0((u,p),(du,dp),(v,q)) = a0((du,dp),(v,q)) + dc0(u,du,v)
      ls = PardisoSolver()

      @time op = FEOperator(res0,jac0,X,Y,assem)
      # ls = LinearFESolver()
      nls = NLSolver(show_trace=true, method=:newton,iterations=10)
      # xh = interpolate([u0,p0],X)
      uin0 = (x) -> uin(Point(x...,tini))
      xh = interpolate([uin0,zero(Tp)],X)
     @time (uh,ph),cache = solve!(xh,nls,op)
    elseif params[:problem_type] == "none"
      uh = interpolate(zero(Tv),U)
      ph = interpolate(zero(Tp),P)
    else
      @unreachable
    end

    uh0⁻ = uh
    ph0 = ph
  else
    @unreachable "$(params[:problem_type]) not known"
  end

  n_Γ = -get_normal_vector(Γ)
  dΓ = Measure(Γ,ks*2)
  f_ν = sum(∫( n_Γ⋅∇( uh0⁻ ) )dΓ)
  f_p = sum(∫( n_Γ*ph0  )dΓ)
  out["pressure_force"] = [f_p]
  out["viscous_force"] = [f_ν]
  out["t"] = [tini]

  @timeit to "writevtk" if output
    cellfields = ["uh"=>uh0⁻,"wh"=>wh0,"w"=>wh0,"ph"=>ph0]
    pvd[tini] = createvtk(Ω0⁻,"$(outputname)_vtu/trian_$istep.vtu";cellfields)
    boundary_pvd[tini] = createvtk(Γ,"$(outputname)_vtu/boundary_$istep.vtu";cellfields)
    act_pvd[tini] = createvtk(Ωact,"$(outputname)_vtu/trian_act_$istep.vtu";cellfields)
    geo_pvd[tini] = createvtk(geo,"$(outputname)_vtu/geo_$istep.vtu")
    _aggregates = 1:num_cells(Ωbg)
    _colors = color_aggregates(_aggregates,model)
    bg_pvd[tini] = createvtk(Ωbg,"$(outputname)_vtu/bg_$istep.vtu",
      celldata = ["aggragate"=>_aggregates,"color" => _colors])
    # writevtk(Triangulation(get_stl(geo)),
    # "$(outputname)_vtu/_geo_0"*".vtu",
    # cellfields=[
    #   "w"=>zero(Tv)])
    #   writevtk(geo,
    #   "$(outputname)_vtu/geo_t_0"*".vtu")
  end

  nt > 0 || return out
  ts = LinRange(tini,tend,nt+1)

  tref = ts[1]
  cutgeo = aggregates = nothing
  isub = 0

  for istep in 1:nt

    t0 = ts[istep]
    t = ts[istep+1]
    isub = isub == nsubsteps ? 1 : isub+1
    tref = isub == 1 ? t0 : tref

    verbose = true
    if verbose
      println("Step $istep of $nt (substep $isub): t = [$t0,$t], t0 = $tref")
    end

    if t0 == tref
      geo_tref = move_nodes(ϕΓ(t0),geo)
      @timeit to "cut" cutgeo,bgf_ioc = cut(STLCutter(),model,geo_tref)
      remove_disconnected_parts!(cutgeo,OUT)
      aggregates = aggregate(strategy,cutgeo,cutgeo.geo,OUT,bgf_ioc)
      # add_cut_layer!(cutgeo,cutgeo.geo,OUT,aggregates)
    end
    Ωact = Triangulation(cutgeo,ACTIVE_OUT)
    Ω = Triangulation(cutgeo,PHYSICAL_OUT)
    Γ = EmbeddedBoundary(cutgeo)

    if params[:bodyfitted]
      Ω = Ωact = Ωbg
      aggregates = 1:num_cells(Ω)
    end

    dirichlet_funs_t0 = map(dirichlet_funs) do f
      if isa(f,Function)
        x -> f(Point(x...,t0))
      else
        f
      end
    end

    ΩI = time_extrusion(Ω,(t0,t))
    ΩIact = time_extrusion(Ωact,(t0,t))
    ΓI = time_extrusion(Γ,(t0,t))
    n_ΓI = -get_normal_vector(ΓI)
    n_Γ = -get_normal_vector(Γ)

    @timeit to "displacement" begin
      wh = setup_displacement(disp_params,Ω,Ωact,Γ,n_Γ,aggregates,wΓ((tref,t)))
      ϕ = space_time_geomap(wh0,wh,ΩIact,aggregates;
        k_space = params[:k_space_map], k_time = params[:k_time_map])
      verbose && println("Jt(ϕ) > $(minimum_jacobian(ϕ))")
    end

    Vstd = FESpace(Ωact,reffe_u;dirichlet_tags,dirichlet_masks,conformity=:H1)
    # Qstd = FESpace(Ωact,reffe_p;dirichlet_tags=p_tags)
    # Qstd = FESpace(Ωact,reffe_p;conformity=:L2,constraint=:zeromean)
    if params[:inlet] == "free"
      Qstd = FESpace(Ωact,reffe_p,constraint=:zeromean,conformity=:C0)
    else
      Qstd = FESpace(Ωact,reffe_p,conformity=:C0)
    end
    Vt = FESpace(ΩIact.time,reffe_t)

    VIstd = FESpaceST(ΩIact,Vstd,Vt)
    QIstd = FESpaceST(ΩIact,Qstd,Vt)
    V = AgFEMSpace(Vstd,aggregates)
    VI = AgFEMSpace(VIstd,aggregates)
    QI = AgFEMSpace(QIstd,aggregates)
    U = TrialFESpace(V,dirichlet_funs_t0)
    UI = TrialFESpace(VI,dirichlet_funs)
    PI = TrialFESpace(QI)
    # PI = QI
    XI = MultiFieldFESpace([UI,PI])
    YI = MultiFieldFESpace([VI,QI])

    ps = 2*ks
    pt = 2*kt
    dΩ0 = Measure(Ω,ps)
    dΓ = Measure(Γ,ps)
    dΩ0⁻ = Measure(Ω0⁻,ps)
    dΩI = Measure(ΩI,ps,pt)
    dΓI = Measure(ΓI,ps,pt)

    uD0 = zero(Tv)
    uϕ = _extract_x∘_extract_t∘∇(ϕ)
    uD = uD0 + uϕ

    if params[:inlet] == "manufactured"
      uD = u∘ϕ
    end

    cell_meas = get_cell_measure(Ωact)
    h = (cell_meas .* 6) .^(1/D)
    τ = CellField( (β*ks^2)./h, ΩIact )

    invJt = inv∘∇(ϕ)
    ∇_I(a) = invJt⋅∇(a)
    ∇x_I(a) = _extract_x∘∇_I(a)
    ∂t_I(a) = _extract_t∘∇_I(a)
    divx_I(a) = tr∘∇x_I(a)
    # Δx_I(a) = tr∘∇∇x_I(a)
    dJ = det∘(∇x(ϕ))

    dJΓn(j,c,n) = j*√(n⋅inv(c)⋅n)
    C = (j->j⋅j')∘(∇(ϕ))
    n_ΓI0 = push_normal∘(invJt,n_ΓI)
    nx_ΓI = _extract_x∘n_ΓI0
    dJΓ = dJΓn∘(dJ,C,n_ΓI)
    dJ0 = dJ(t0)

    ∫_ΩI(a) = ∫(a*dJ)
    ∫_ΓI(a) = ∫(a*dJΓ)
    ∫_Ω0(a) = ∫(a*dJ0)


    if params[:problem_type] ∈ ("explicit-navier-stokes","navier-stokes")

      ΩI⁻ = time_extrusion(Ω0⁻,(t0,t))
      dΩI⁻ = Measure(ΩI⁻,ps,pt)
      # @show typeof(Ω0⁻)
      # @show typeof(get_triangulation(uh0⁻))
      if get_triangulation(uh0⁻) !== Ω0⁻
        println("changing triangulation")
        _uh0⁻ = change_triangulation(uh0⁻,Ω0⁻)
      else
        _uh0⁻ = uh0⁻
      end
      uh⁻ = time_extrusion(_uh0⁻,ΩI⁻)

    end

    function a((u,p),(v,q))
      r =
        ∫_ΩI( ∂t_I(u)⋅v + ν*∇x_I(v)⊙∇x_I(u) - divx_I(v)*p + divx_I(u)*q )dΩI +
        ∫_Ω0( v(t0)⋅u(t0) )dΩ0
      if !params[:bodyfitted]
        r = r +
          ∫_ΓI( τ*v⋅u )dΓI +
          ∫_ΓI( - v⋅(ν*nx_ΓI⋅∇x_I(u) - nx_ΓI*p) )dΓI +
          ∫_ΓI( - (ν*nx_ΓI⋅∇x_I(v) - nx_ΓI*q)⋅u  )dΓI
      end
      # if params[:problem_type] == "explicit-navier-stokes"
      #   r = r + ∫_ΩI( v⋅((∇x_I(u))'⋅uΠ) )dΩI
      # end
      r
    end

    function l((v,q))
      r =
        ∫_ΩI( v⋅(f∘ϕ) + q*(g∘ϕ) )dΩI +
        ∫_Ω0( v(t0)⋅uh0⁻ )dΩ0⁻
      if !params[:bodyfitted]
        r = r + ∫_ΓI( τ*v⋅uD - (ν*nx_ΓI⋅∇x_I(v) - nx_ΓI*q)⋅uD )dΓI
      end
      if params[:problem_type] == "explicit-navier-stokes"
        r = r + ∫_ΩI( v⋅(∇x_I(uh⁻))'⋅uh⁻ )dΩI⁻
      end
      r
    end

    @show num_free_dofs(XI)

    vref = ∑( ∫(1)dΩ0 )
    vt0 = ∑( ∫(dJ(t0))dΩ0 )
    vt1 = ∑( ∫(dJ(t))dΩ0 )
    eΓ = √( l2(wh - wΓ((tref,t)),dΓ) )
    uΓ = √( l2(wh,dΓ) )

    @show (vt0-vref)/vref
    @show (vt1-vref)/vref
    @show eΓ
    @show uΓ


    GC.gc()

    Tmat = SparseMatrixCSC{Float64,Int32}
    Tvec = Vector{Float64}
    assem = SparseMatrixAssembler(Tmat,Tvec,XI,YI)

    if params[:problem_type] ∈ ("stokes","explicit-navier-stokes")
      @time @timeit to "assembly" op = AffineFEOperator(a,l,XI,YI,assem)
      A = get_matrix(op)
      if GridapPardiso.MKL_PARDISO_LOADED[]
        ls = PardisoSolver()
      else
        ls = LinearFESolver()
      end
      @time @timeit to "solve" uh,ph = solve(ls,op)

    elseif params[:problem_type] == "navier-stokes"
      c(w,u,v) = ∫_ΩI( v⋅((∇x_I(u))'⋅w) )dΩI
      dc(u,du,v) = c(du,u,v) + c(u,du,v)

      res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,u,v)- l((v,q))
      jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)
      ls = PardisoSolver()

      @time op = FEOperator(res,jac,XI,YI,assem)
      ls = LinearFESolver()
      nls = NLSolver(show_trace=true, method=:newton,iterations=10)
      # xh = interpolate([u0,p0],X)
      # TODO:
      #  - interpolate t0
      #  - interpolate tn-1
      # xh = interpolate([uh⁻,zero(Tp)],XI)
      xh = interpolate([uin,zero(Tp)],XI)
      # TODO:
      #  - fix nlsolver for FESpaceST
    @time @timeit to "solve"  xh,cache = solve!(xh,nls,op)
    uh,ph = xh
    b,A = Gridap.FESpaces.residual_and_jacobian(op,xh)
    elseif params[:problem_type] == "none"
      uh = interpolate(zero(Tv),UI)
      ph = interpolate(zero(Tp),PI)
    else
      error("Unknown problem type $(params[:problem_type])")
    end


    _freq = max(1,params[:output_freq])
    _trange = t0 .+ (1/_freq*(t-t0)).*(1:_freq)
    for t in _trange
      f_ν = sum(∫( n_Γ⋅∇( uh(t) ) * dJΓ(t) )dΓ)
      f_p = sum(∫( n_Γ*ph(t)  * dJΓ(t) )dΓ)
      push!(out["pressure_force"],f_p)
      push!(out["viscous_force"],f_ν)
      push!(out["t"],t)
    end

    if params[:cond1] && params[:problem_type] != "navier-stokes"
      A = get_matrix(op)
      condA = get(out,"cond1A",Float64[])
      @timeit to "cond" push!(condA,cond(A,1))
      out["cond1A"] = condA
      verbose && println("cond(A) = $(condA[end])")
    end

    if params[:inlet] == "manufactured"
      @timeit to "errors" begin
        l2err_u_t = l2(uh(t)-(u∘ϕ)(t),dΩ0)
        l2err_∇p_t = l2(∇x(ph)(t)-∇x(p∘ϕ)(t),dΩ0)
        l2err_p_t = l2((ph)(t)-(p∘ϕ)(t),dΩ0)
        l2_u_t = l2(uh(t),dΩ0)
        l2_∇p_t = l2(∇x(ph)(t),dΩ0)
      end
      l2err_u = get(out,"l2err_u",Float64[])
      l2err_∇p = get(out,"l2err_∇p",Float64[])
      l2err_p = get(out,"l2err_p",Float64[])
      l2_u = get(out,"l2_u",Float64[])
      l2_∇p = get(out,"l2_∇p",Float64[])
      push!(l2err_u,l2err_u_t)
      push!(l2err_∇p,l2err_∇p_t)
      push!(l2err_p,l2err_p_t)
      push!(l2_u,l2_u_t)
      push!(l2_∇p,l2_∇p_t)
      out["l2err_u"] = l2err_u
      out["l2err_∇p"] = l2err_∇p
      out["l2err_p"] = l2err_p
      out["l2_u"] = l2_u
      out["l2_∇p"] = l2_∇p

      verbose && (
        println("L2(err(u)) = $l2err_u_t");
        println("L2(err(∇p)) = $l2err_∇p_t");
        println("L2(err(p)) = $l2err_p_t") )

    end

    if params[:residuals]
      @assert params[:inlet] == "manufactured"
      @timeit to "residuals" begin
        ur = interpolate(u∘ϕ,UI)
        vr = interpolate(u,VI)
        pr = interpolate(p∘ϕ,PI)
        qr = interpolate(p,QI)
        f∂t = xt -> ∂t_xt(u)(xt)

        l2err_u = l2(ur-u∘ϕ,dΩI)
        l2err_p = l2(pr-p,dΩI)
        r0 = ((u,p),(v,q)) -> a((u,p),(v,q)) - l((v,q))
        r1 = ((u,p),(v,q)) -> ∫_ΩI( ∂t_I(u)⋅v - (f∂t∘ϕ)⋅v )dΩI
        r2 = ((u,p),(v,q)) -> ∫_Ω0( v(t0)⋅u(t0) )dΩ0 - ∫_Ω0( v(t0)⋅uh0⁻ )dΩ0⁻
        if !params[:bodyfitted]
          r3 = ((u,p),(v,q)) ->
            ∫_ΩI( ν*∇x_I(v)⊙∇x_I(u) - divx_I(v)*p + divx_I(u)*q )dΩI +
            ∫_ΓI( - v⋅(ν*nx_ΓI⋅∇x_I(u) - nx_ΓI*p) )dΓI -
            ∫_ΩI( v⋅(f∘ϕ) + q*(g∘ϕ) - v⋅(f∂t∘ϕ)  )dΩI
          r4 = ((u,p),(v,q)) ->
            ∫_ΓI( τ*v⋅u - (ν*nx_ΓI⋅∇x_I(v) - nx_ΓI*q)⋅u  )dΓI -
            ∫_ΓI( τ*v⋅uD - (ν*nx_ΓI⋅∇x_I(v) - nx_ΓI*q)⋅uD  )dΓI

          R = [r0,r1,r2,r3,r4]
        else
          r3 = ((u,p),(v,q)) ->
            ∫_ΩI( ν*∇x_I(v)⊙∇x_I(u) - divx_I(v)*p + divx_I(u)*q )dΩI -
            ∫_ΩI( v⋅(f∘ϕ) + q*(g∘ϕ) - v⋅(f∂t∘ϕ)  )dΩI

          R = [r0,r1,r2,r3]
        end

        res = map(r->∑(r((ur,pr),(vr,qr))),R)

        residuals = get(out,"residuals",Vector{Float64}[])
        l2err_ui = get(out,"l2err_ui",Float64[])
        l2err_pi = get(out,"l2err_pi",Float64[])
        push!(residuals,res)
        push!(l2err_ui,l2err_u)
        push!(l2err_pi,l2err_p)
        out["residuals"] = residuals
        out["l2err_ui"] = l2err_ui
        out["l2err_pi"] = l2err_pi

        verbose && ( println("Residuals = $res"))
        verbose && ( println("Interpolation error = $l2err_u"))
      end
    end

    if isub < nsubsteps && istep < nt
      @timeit to "displacement" begin
        t⁺ = ts[istep+2]
        wh⁺ = setup_displacement(disp_params,Ω,Ωact,Γ,n_Γ,aggregates,wΓ((tref,t⁺)))
        ϕ⁺ = space_time_geomap(wh,wh⁺,ΩIact,aggregates;
          k_space = params[:k_space_map], k_time = params[:k_time_map])
      end
      if minimum_jacobian(ϕ⁺) < min_jacobian
        isub = nsubsteps
      end
    end
    if isub == nsubsteps && istep < nt
      @timeit to "send_to_background" begin
        uh0⁻,Ω0⁻ = send_to_background(uh(t),Ωact,geo,ϕ,ϕΓ,t)
        wh0 = zero
      end
    else
      uh0⁻ = uh(t)
      Ω0⁻ = Ω
      wh0 = wh
    end

    @timeit to "writevtk" if output
      freq = params[:output_freq]
      if freq > 1
        trange = t0 .+ (1/freq*(t-t0)).*(1:freq)
      else
        period = floor(1/freq)
        if istep % period == 0 || istep == nt
          trange = [t]
        else
          trange = []
        end
      end


      for (i,t) in enumerate(trange)

        wht = _extract_x∘ϕ(t) - (x->x)

        k = (istep-1)*max(1,freq) + i

        pvd[t] = createvtk(Ω,"$(outputname)_vtu/trian_$k"*".vtu",
          cellfields=[
            "uh"=>uh(t),
            "ph"=>ph(t),
            "wh"=>wht ])
        boundary_pvd[t] = createvtk(Γ,"$(outputname)_vtu/boundary_$k"*".vtu",
          cellfields=[
            "uh"=>uh(t),
            "ph"=>ph(t),
            "wh"=>wht,
            "w"=>wΓ((tref,t))])
        act_pvd[t] = createvtk(Ωact,"$(outputname)_vtu/trian_act_$k"*".vtu",
            cellfields=[
              "uh"=>uh(t),
              "ph"=>ph(t),
              "wh"=>wht ])

        geo_pvd[t] = createvtk(move_nodes(ϕΓ(t),geo),
          "$(outputname)_vtu/geo_$k"*".vtu")

        colors = color_aggregates(aggregates,model)
        bg_pvd[t] = createvtk(Ωbg,"$(outputname)_vtu/bg_$istep.vtu",
            celldata = ["aggragate"=>aggregates,"color" => colors])

          # geo_tref = move_nodes(ϕΓ(tref),geo)
          # writevtk(Triangulation(get_stl(geo_tref)),
          # "$(outputname)_vtu/_geo_$k"*".vtu",
          # cellfields=[
          #   "w"=>wΓ((tref,t))])
          # geo_t = move_nodes(x->x+wΓ((tref,t))(x),geo_tref)
          # writevtk(geo_t,
          # "$(outputname)_vtu/geo_t_$k"*".vtu")

      end
      save("$(outputname).jld2",tostringdict(out))

    end

    GC.gc()
    if verbose
      show(to;sortby=:time)
      println("")
    end
  end


  @timeit to "writevtk" if output
    savepvd(pvd)
    savepvd(boundary_pvd)
    savepvd(act_pvd)
    savepvd(geo_pvd)
    savepvd(bg_pvd)
  end

  verbose = true
  if verbose
    show(to;sortby=:firstexec)
    println("")
  end

  out
end
