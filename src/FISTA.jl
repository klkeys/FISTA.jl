module FISTA

using Distances
using PLINK
using Gadfly
using DataFrames
#using SparseRegression

typealias Float Union{Float32, Float64}

export lasso_reg
export cv_lasso
export mcp_reg

include("common.jl")
include("lasso.jl")
include("mcp.jl")

end
