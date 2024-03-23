using GridapIce
using Documenter

DocMeta.setdocmeta!(GridapIce, :DocTestSetup, :(using GridapIce); recursive=true)

makedocs(;
    modules=[GridapIce],
    authors="Daniel Richards <daniel.richards@monash.edu>",
    sitename="GridapIce.jl",
    format=Documenter.HTML(;
        canonical="https://dhrichards.github.io/GridapIce.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dhrichards/GridapIce.jl",
    devbranch="main",
)
