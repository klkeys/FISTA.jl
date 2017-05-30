# ----------------------------------------- #
# thresholding functions

function softthreshold{T <: AbstractFloat}(x::T, λ::T) :: T
    @assert λ >= 0 "Argument λ must exceed 0"
    x >  λ && return (x - λ)
    x < -λ && return (x + λ)
    return zero(T)
end

function firmthreshold{T <: AbstractFloat}(x::T, λ::T, γ::T) :: T
    @assert γ > 1 "Argument γ must exceed 1"
    t = λ*γ
    abs(x) > t && return x
    d = (one(T) - one(T) / γ)
    return softthreshold(x, λ) / d
end

@inline function compute_lambdas{T <: AbstractFloat}(
    λmax :: T,
    ε    :: T,
    K    :: Int = 100
) :: Vector{T}
    λmin = λmax*ε
    Λ = collect(linspace(λmax, λmin, K))
    return Λ
end

function compute_lambdas{T <: AbstractFloat}(
    x :: AbstractMatrix{T},
    y :: AbstractVector{T},
    K :: Int = 100,
    ε :: T   = convert(T, 1e-3)
) :: Vector{T}

    # problem dimensions?
    n,p = size(x)
    @assert n == length(y) "Arguments x and y must have same number of rows"

    # use Jerome Friedman's approach to select reasonable estimates of λ
    # first compute scaled copies of x and y
    sy = std(y, corrected=false)
    my = mean(y)
    sx = std(x, corrected=false, 1)
    mx = mean(x, 1)
    x_scaled = (x .- mx) ./ sx
    y_scaled = (y .- my) ./ sy

    # λ_max is maximum absolute column sum of x'*y, divided by n
    λmax = maxabs(x_scaled' * y_scaled) / n

    # generate K λs inbetween the previous two λs
    return compute_lambdas(λmax, ε, K)
end

# ----------------------------------------- #
# functions to handle FISTA output

# an object that houses results returned from an FISTA run
immutable FISTAResults{T <: AbstractFloat, V <: DenseVector}
    time :: T
    loss :: T
    iter :: Int
    beta :: V

    FISTAResults(time::T, loss::T, iter::Int, beta::DenseVector{T}) = new(time, loss, iter, beta)
end

# strongly typed external constructor for FISTAResults
function FISTAResults{T <: AbstractFloat}(
    time :: T,
    loss :: T,
    iter :: Int,
    beta :: DenseVector{T}
)
    FISTAResults{T, typeof(beta)}(time, loss, iter, beta)
end


# ----------------------------------------- #
# functions for handling temporary arrays
type FISTAVariables{T <: AbstractFloat, V <: DenseVector}
    b     :: V
    b0    :: Vector{T}
    xb    :: V
    xb0   :: Vector{T}
    df    :: V
    r     :: V
    bdiff :: Vector{T}
    z     :: V
    lambdas :: Vector{T}
    gamma :: T
end


function FISTAVariables{T <: AbstractFloat}(
    b       :: DenseVector{T},
    b0      :: Vector{T},
    xb      :: DenseVector{T},
    xb0     :: Vector{T},
    df      :: DenseVector{T},
    r       :: DenseVector{T},
    bdiff   :: Vector{T},
    z       :: DenseVector{T},
    lambdas :: Vector{T},
    gamma   :: T
)
    FISTAVariables{T, typeof(b)}(b, b0, xb, xb0, df, r, bdiff, z, lambdas, gamma)
end

function FISTAVariables{T <: AbstractFloat}(
    x       :: Matrix{T},
    y       :: Vector{T},
    lambdas :: Vector{T},
    gamma   :: T
)
    n, p  = size(x)
    b     = zeros(T, p)
    b0    = zeros(T, p)
    xb0   = zeros(T, n)
    xb    = zeros(T, n)
    df    = zeros(T, p)
    r     = zeros(T, n)
    bdiff = zeros(T, p)
    z     = zeros(T, p)

    FISTAVariables{T, typeof(y)}(b, b0, xb, xb0, df, r, bdiff, z, lambdas, gamma)
end


function FISTAVariables{T <: AbstractFloat}(
    x       :: SharedMatrix{T},
    y       :: SharedVector{T},
    lambdas :: Vector{T},
    gamma   :: T
)
    n, p  = size(x)
    pids  = procs(x)
    V     = typeof(y)
    b     = SharedVector(T, (p,), pids=pids) :: V
    b0    = zeros(T, p)
    xb    = SharedVector(T, (n,), pids=pids) :: V
    xb0   = zeros(T, n)
    df    = SharedVector(T, (p,), pids=pids) :: V
    r     = SharedVector(T, (n,), pids=pids) :: V
    bdiff = zeros(T, p)
    z     = SharedVector(T, (p,), pids=pids) :: V

    FISTAVariables{T, typeof(y)}(b, b0, xb, xb0, df, r, bdiff, z, lambdas, gamma)
end

FISTAVariables{T <: AbstractFloat}(x::DenseMatrix{T},y::DenseVector{T},lambdas::Vector{T}) = FISTAVariables(x,y,lambdas,convert(T,3))


function FISTAVariables{T <: AbstractFloat}(
    x   :: DenseMatrix{T},
    y   :: DenseVector{T},
    K   :: Int = 100,
    tol :: T = convert(T, 1e-3)
)
    # compute lambdas
    lambdas = compute_lambdas(x, y, K, tol)

    # use Patrick Breheny's default of 3 for gamma
    gamma = convert(T, 3)

    # now initialize temporary arrays
    FISTAVariables(x, y, lambdas, gamma)
end

# ----------------------------------------- #
# crossvalidation routines

# subroutine to compute a default number of folds
@inline cv_get_num_folds(nmin::Int, nmax::Int) = max(nmin, min(Sys.CPU_CORES::Int, nmax))

"Can also be called with an `Int` argument `n` instead of the data vector `y`."
function cv_get_folds(n::Int, q::Int)
    m, r = divrem(n, q)
    shuffle!([repmat(1:q, m); 1:r])
end

"""
CREATE UNSTRATIFIED CROSSVALIDATION PARTITION

    cv_get_folds(y,q) -> Vector{Int}

This function will partition the `n` components of `y` into `q` disjoint sets for unstratified `q`-fold crossvalidation.

Arguments:

- `y` is the `n`-vector to partition.
- `q` is the number of disjoint sets in the partition.
"""
function cv_get_folds(y::DenseVector, q::Int)
    n, r = divrem(length(y), q)
    shuffle!([repmat(1:q, n); 1:r])
end


# return type for crossvalidation
immutable FISTACrossvalidationResults{T <: AbstractFloat}
    mses   :: Vector{T}
#    path   :: Vector{Int}
    #lambda :: Vector{T}
    lambda :: T
    b      :: Vector{T}
    bidx   :: Vector{Int}
    k      :: Int
    bids   :: Vector{String}

    #FISTACrossvalidationResults(mses::Vector{T}, path::Vector{Int}, lambda::Vector{T}, b::Vector{T}, bidx::Vector{Int}, k::Int, bids::Vector{String}) = new(mses, path, lambda, b, bidx, k, bids)
    #FISTACrossvalidationResults(mses::Vector{T}, lambda::Vector{T}, b::Vector{T}, bidx::Vector{Int}, k::Int, bids::Vector{String}) = new(mses, lambda, b, bidx, k, bids)
    FISTACrossvalidationResults(mses::Vector{T}, lambda::T, b::Vector{T}, bidx::Vector{Int}, k::Int, bids::Vector{String}) = new(mses, lambda, b, bidx, k, bids)
end

# constructor for when bids are not available
# simply makes vector of "V$i" where $i are drawn from bidx
function FISTACrossvalidationResults{T <: AbstractFloat}(
    mses   :: Vector{T},
#    path   :: Vector{Int},
    #lambda :: Vector{T},
    lambda :: T,
    b      :: Vector{T},
    bidx   :: Vector{Int},
    k      :: Int
)
    bids = convert(Vector{String}, ["V" * "$i" for i in bidx]) :: Vector{String}
    #FISTACrossvalidationResults{eltype(mses)}(mses, path, lambda, b, bidx, k, bids)
    FISTACrossvalidationResults{eltype(mses)}(mses, lambda, b, bidx, k, bids)
end

# function to view an FISTACrossvalidationResults object
function Base.show(io::IO, x::FISTACrossvalidationResults)
    println(io, "Crossvalidation results:")
    println(io, "Minimum MSE ", minimum(x.mses), " occurs at λ = $(x.lambda).")
    println(io, "Best model β has the following $(x.k) nonzero coefficients:")
    println(io, DataFrame(Predictor=x.bidx, Name=x.bids, Estimated_β=x.b))
    return nothing
end

function Gadfly.plot(x::FISTACrossvalidationResults)
    df = DataFrame(ModelSize=x.path, MSE=x.mses)
    plot(df, x="ModelSize", y="MSE", xintercept=[x.k], Geom.line, Geom.vline(color=colorant"red"), Guide.xlabel("Model size"), Guide.ylabel("MSE"), Guide.title("MSE versus model size"))
#    plot(df, x="ModelSize", y="MSE", xintercept=[x.k], Geom.line, Geom.vline(color="red"), Guide.xlabel("Model size"), Guide.ylabel("MSE"), Guide.title("MSE versus model size"))
end
