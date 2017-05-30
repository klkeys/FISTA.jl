function Base.SharedArray(y::SharedArray) :: typeof(y)
    sy = size(y)
    pids = procs(y)
    T = eltype(y)
    z = SharedArray(T, sy, pids=pids) :: typeof(y)
    copy!(z,y)
end

function one_fold{T <: Float}(
    x        :: BEDFile{T},
    y        :: SharedVector{T},
    lambdas  :: Vector{T},
    folds    :: DenseVector{Int},
    fold     :: Int;
    pids     :: Vector{Int} = procs(x),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true
)
    # dimensions of problem
    n,p = size(x)

    # make vector of indices for folds
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx
    mask_n    = convert(Vector{Int}, train_idx)
    mask_test = convert(Vector{Int}, test_idx)

    # first mask training index
    copy!(x.mask, mask_n)

    # compute the regularization path on the training set
    betas = iht_path(x, y, length(lambdas), mask_n=mask_n, max_iter=max_iter, quiet=quiet, max_step=max_step, pids=pids, tol=tol, lambdas=lambdas)

    # tidy up
    gc()

    # change mask to test
    copy!(x.mask, mask_test)

    # preallocate vector for output
    myerrors = zeros(T, length(path))

    # allocate an index vector for b
    #indices = falses(p)

    # allocate the arrays for the test set
    xb = SharedArray(T, (n,), pids=pids) :: SharedVector{T}
    b  = SharedArray(T, (p,), pids=pids) :: SharedVector{T}
    r  = SharedArray(T, (n,), pids=pids) :: SharedVector{T}

    # compute the mean out-of-sample error for the TEST set
    # do this for every computed model in regularization path
    for i = 1:size(betas,2)

        # pull ith model in dense vector format
        b2 = full(vec(betas[:,i]))

        # copy it into SharedArray b
        copy!(b,b2)

        # indices stores Boolean indexes of nonzeroes in b
        #update_indices!(indices, b)
        #indices .= b .!= 0

        # compute estimated response Xb with $(path[i]) nonzeroes
        #A_mul_B!(xb, x, b, indices, path[i], mask_test, pids=pids)
        A_mul_B!(xb, x, b)

        # compute residuals
        r .= y .- xb

        # mask data from training set
        # training set consists of data NOT in fold:
        # r[folds .!= fold] = zero(Float64)
        #mask!(r, mask_test, 0, zero(T))
        r[train_idx] = zero(T)

        # compute out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = sumabs2(r) / test_size / 2
    end

    return myerrors :: Vector{T}
end

function pfold(
    T          :: Type,
    xfile      :: String,
    xtfile     :: String,
    x2file     :: String,
    yfile      :: String,
    meanfile   :: String,
    precfile   :: String,
    K          :: Int,
    folds      :: DenseVector{Int},
    pids       :: Vector{Int},
    q          :: Int;
    max_iter   :: Int  = 100,
    max_step   :: Int  = 50,
    quiet      :: Bool = true,
    header     :: Bool = false
)

    # ensure correct type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

    # how many CPU processes can pfold use?
    np = length(pids)

    # report on CPU processes
    quiet || println("pfold: np = ", np)
    quiet || println("pids = ", pids)

    # set up function to share state (indices of folds)
    i = 1
    nextidx() = (idx=i; i+=1; idx)

    # preallocate cell array for results
    results = SharedArray(T, (length(path),q), pids=pids) :: SharedMatrix{T}

    # master process will distribute tasks to workers
    # master synchronizes results at end before returning
    @sync begin

        # loop over all workers
        for worker in pids

            # exclude process that launched pfold, unless only one process is available
            if worker != myid() || np == 1

                # asynchronously distribute tasks
                @async begin
                    while true

                        # grab next fold
                        current_fold = nextidx()

                        # if current fold exceeds total number of folds then exit loop
                        current_fold > q && break

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker.\n\n")
                        r = remotecall_fetch(worker) do
                            processes = [worker]
                            x = BEDFile(T, xfile, x2file, meanfile, precfile, pids=processes, header=header)
                            y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=processes) :: SharedVector{T}
                            lambdas = compute_lambdas(x,y,K)

                            one_fold(x, y, lambdas, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=processes)
                        end # end remotecall_fetch()
                        setindex!(results, r, :, current_fold)
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return (vec(sum(results, 2) ./ q)) :: Vector{T}
end

# default type for pfold is Float64
pfold(xfile::String, xtfile::String, x2file::String, yfile::String, meanfile::String, precfile::String, K::Int, folds::DenseVector{Int}, pids::Vector{Int}, q::Int; max_iter::Int=100, max_step::Int =50, quiet::Bool=true, header::Bool=false) = pfold(Float64, xfile, xtfile, x2file, yfile, meanfile, precfile, K, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

function pfold(
    T        :: Type,
    xfile    :: String,
    x2file   :: String,
    yfile    :: String,
    K        :: Int, 
    folds    :: DenseVector{Int},
    pids     :: Vector{Int},
    q        :: Int;
    max_iter :: Int  = 100,
    max_step :: Int  = 50,
    quiet    :: Bool = true,
    header   :: Bool = false
)

    # ensure correct type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

    # do not allow crossvalidation with fewer than 3 folds
    q > 2 || throw(ArgumentError("Number of folds q = $q must be at least 3."))

    # how many CPU processes can pfold use?
    np = length(pids)

    # report on CPU processes
    quiet || println("pfold: np = ", np)
    quiet || println("pids = ", pids)

    # set up function to share state (indices of folds)
    i = 1
    nextidx() = (idx=i; i+=1; idx)

    # preallocate array for results
    results = SharedArray(T, (length(path),q), pids=pids) :: SharedMatrix{T}

    # master process will distribute tasks to workers
    # master synchronizes results at end before returning
    @sync begin

        # loop over all workers
        for worker in pids

            # exclude process that launched pfold, unless only one process is available
            if worker != myid() || np == 1

                # asynchronously distribute tasks
                @async begin
                    while true

                        # grab next fold
                        current_fold = nextidx()

                        # if current fold exceeds total number of folds then exit loop
                        current_fold > q && break

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker.\n\n")

                        # launch job on worker
                        # worker loads data from file paths and then computes the errors in one fold
                        r = remotecall_fetch(worker) do
                            processes = [worker]
                            x = BEDFile(T, xfile, x2file, pids=processes, header=header)
                            y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=processes) :: SharedVector{T}
                            lambdas = compute_lambdas(x,y,K)

                            one_fold(x, y, lambdas, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, pids=processes)
                        end # end remotecall_fetch()
                        setindex!(results, r, :, current_fold)
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return (vec(sum(results, 2) ./ q)) :: Vector{T}
end

# default for previous function is Float64
pfold(xfile::String, x2file::String, yfile::String, K::Int, folds::DenseVector{Int}, pids::Vector{Int}, q::Int; max_iter::Int = 100, max_step::Int = 50, quiet::Bool = true, header::Bool = false) = pfold(Float64, xfile, x2file, yfile, K, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)



"""
    cv_iht(xfile, xtfile, x2file, yfile, meanfile, precfile, [q=max(3, min(CPU_CORES,5)), path=collect(1:min(p,20)), folds=cv_get_folds(n,q), pids=procs()])

This variant of `cv_iht()` performs `q`-fold crossvalidation with a `BEDFile` object loaded by `xfile`, `xtfile`, and `x2file`,
with column means stored in `meanfile` and column precisions stored in `precfile`.
The continuous response is stored in `yfile` with data particioned by the `Int` vector `folds`.
The folds are distributed across the processes given by `pids`.
The dimensions `n` and `p` are inferred from BIM and FAM files corresponding to the BED file path `xpath`.
"""
function cv_iht(
    T        :: Type,
    xfile    :: String,
    xtfile   :: String,
    x2file   :: String,
    yfile    :: String,
    meanfile :: String,
    precfile :: String;
    q        :: Int = cv_get_num_folds(3,5), 
    K        :: Int = 100, 
    folds    :: DenseVector{Int} = begin
           # find n from the corresponding FAM file, then make folds
            famfile = xfile[1:(endof(xfile)-3)] * "fam"
            n       = countlines(famfile)
            cv_get_folds(n, q)
            end,
    pids     :: Vector{Int} = procs(),
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    quiet    :: Bool  = true,
    header   :: Bool  = false
)

    # enforce type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

    # compute folds in parallel
    mses = pfold(T, xfile, xtfile, x2file, yfile, meanfile, precfile, path, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

    # what is the best model size?
    k = path[indmin(errors)] :: Int

    # print results
    !quiet && print_cv_results(mses, path, k)

    # recompute ideal model
    # first load data on *all* processes
    x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, header=header, pids=pids)
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids) :: SharedVector{T}

    # first use L0_reg to extract model
    output = L0_reg(x, y, k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, pids=pids)

    # which components of beta are nonzero?
    inferred_model = output.beta .!= zero(T)
    bidx = find(inferred_model)

    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model)

    # refit the best model
    b, bidx = refit_iht(x_inferred, y, k, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)

    bids = prednames(x)[bidx]
    return IHTCrossvalidationResults{T}(mses, sdata(path), b, bidx, k, bids)
end

# encodes default type FLoat64 for previous function
### 22 Sep 2016: Julia v0.5 warns that this conflicts with cv_iht for GPUs
### since this is no longer the default interface for cv_iht with CPUs,
### then it is commented out here
#cv_iht(xfile::String, xtfile::String, x2file::String, yfile::String, meanfile::String, precfile::String; q::Int = max(3, min(CPU_CORES, 5)), path::DenseVector{Int} = begin bimfile=xfile[1:(endof(xfile)-3)] * "bim"; p=countlines(bimfile); collect(1:min(20,p)) end, folds::DenseVector{Int} = begin famfile=xfile[1:(endof(xfile)-3)] * "fam"; n=countlines(famfile); cv_get_folds(n, q) end, pids::DenseVector{Int}=procs(), tol::Float64=1e-4, max_iter::Int=100, max_step::Int=50, quiet::Bool=true, header::Bool=false) = cv_iht(Float64, xfile, xtfile, x2file, yfile, meanfile, precfile, path=path, folds=folds, q=q, pids=pids, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

"""
    cv_iht(T::Type, xfile, x2file, yfile, [q=max(3, min(CPU_CORES,5)), path=collect(1:min(p,20)), folds=cv_get_folds(n,q), pids=procs()])

An abbreviated call to `cv_iht` that calculates means, precs, and transpose on the fly.
"""
function cv_iht(
    T        :: Type,
    xfile    :: String,
    x2file   :: String,
    yfile    :: String;
    q        :: Int = cv_get_num_folds(3,5), 
    path     :: DenseVector{Int} = begin
           # find p from the corresponding BIM file, then make path
            bimfile = xfile[1:(endof(xfile)-3)] * "bim"
            p       = countlines(bimfile)
            collect(1:min(20,p))
            end,
    folds    :: DenseVector{Int} = begin
           # find n from the corresponding FAM file, then make folds
            famfile = xfile[1:(endof(xfile)-3)] * "fam"
            n       = countlines(famfile)
            cv_get_folds(n, q)
            end,
    pids     :: Vector{Int} = procs(),
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    max_step :: Int   = 50,
    quiet    :: Bool  = true,
    header   :: Bool  = false
)

    # enforce type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

    # compute folds in parallel
    mses = pfold(T, xfile, x2file, yfile, K, folds, pids, q, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)

    # what is the best model size?
    k = path[indmin(errors)] :: Int

    # print results
    #!quiet && print_cv_results(mses, path, k)

    # recompute ideal model
    # first load data on *all* processes
    x = BEDFile(T, xfile, x2file, header=header, pids=pids)
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids)

    # first use L0_reg to extract model
    output = L0_reg(x, y, k, max_iter=max_iter, max_step=max_step, quiet=quiet, tol=tol, pids=pids)

    # which components of beta are nonzero?
    inferred_model = output.beta .!= zero(T)
    bidx = find(inferred_model)

    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model)

    # refit the best model
    b, bidx = refit_iht(x_inferred, y, k, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)

    bids = prednames(x)[bidx]
    return IHTCrossvalidationResults{T}(mses, sdata(path), b, bidx, k, bids)
end

"""
    cv_iht(xfile, x2file, yfile)

The default call to `cv_iht`. Here `xfile` points to the PLINK BED file stored on disk, `x2file` points to the nongenetic covariates stored in a delimited file, and `yfile` points to the response variable stored in a **binary** file.

Important optional arguments and defaults include:

- `q`, the number of crossvalidation folds. Defaults to `max(3, min(CPU_CORES,5))`
- `path`, an `Int` vector that contains the model sizes to test. Defaults to `collect(1:min(p,20))`, where `p` is the number of genetic predictors read from the PLINK BIM file.
- `folds`, an `Int` vector that specifies the fold structure. Defaults to `cv_get_folds(n,q)`, where `n` is the number of cases read from the PLINK FAM file.
- `pids`, an `Int` vector of process IDs. Defaults to `procs()`.
"""
cv_iht(xfile::String, x2file::String, yfile::String; q::Int = cv_get_num_folds(3,5), K::Int=100, folds::DenseVector{Int} = begin famfile=xfile[1:(endof(xfile)-3)] * "fam"; n=countlines(famfile); cv_get_folds(n, q) end, pids::Vector{Int}=procs(), tol::Float64=1e-4, max_iter::Int=100, max_step::Int=50, quiet::Bool=true, header::Bool=false) = cv_iht(Float64, xfile, x2file, yfile, K=K, folds=folds, q=q, pids=pids, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet, header=header)
