function lasso_reg{T <: AbstractFloat, V <: AbstractVector}(
    x :: AbstractMatrix{T},
    y :: V,
    λ :: T;
    K :: Int = 100,
    v :: FISTAVariables{T,V} = FISTAVariables(x, y, K),
    max_iter       :: Int  = 1000,
    tol            :: T    = convert(T, 1e-4),
    mu             :: T    = one(T),
    backtrack_size :: T    = convert(T, 0.5),
    quiet          :: Bool = true
) FISTAResults{T,V} 

    n,p = size(x)
    @assert n == length(y) "Arguments x and y must have same number of rows"
    loss  = convert(T, Inf)
    loss0 = loss
    iter  = 0

    # start timing
    tic()

    for iter = 1:max_iter
        quiet || println("Iter = ", iter)
        copy!(v.b0, v.b)
        copy!(v.xb0, v.xb)
        loss0 = loss
        steps = 0

        # apply Nesterov acceleration
        k = iter / (iter + 3)
        v.z .= v.b .+ k.*(v.b .- v.b0)

        while true

            steps += 1
            quiet || println("steps = ", steps)
  
            # compute estimated response xb0 .= x*b0
            A_mul_B!(v.xb0, x, v.z)

            # compute residuals
            v.r .= y .- v.xb0
            lossz = sumabs2(v.r) / 2

            # compute gradient
            #df  .= -x'*r
            At_mul_B!(v.df, x, v.r)

            # estimate coefficients
            v.b .= v.z .+ mu.*v.df
            v.b .= softthreshold.(v.b, λ)
          
            # recompute estimated response xb.= x*b
            A_mul_B!(v.xb, x, v.b)

            # update residuals
            v.r .= y .- v.xb

            # update sum of squares
            loss = sumabs2(v.r) / 2

            # difference of estimates
            v.bdiff .= v.b .- v.z

            # check backtracking condition and break if satisfied
            loss <= lossz - dot(v.df, v.bdiff) + one(T) / (2*mu) * sumabs2(v.bdiff) && break

            # at this point we must keep backtracking, so reduce step size and repeat loop
            mu *= backtrack_size
        end

        quiet || println("loss = ", loss)
        abs(loss - loss0) < tol && break
    end

    # get execution time
    exec_time = toq()

    #return v.b
    return FISTAResults(exec_time, loss, iter, v.b)
end



function lasso_path{T <: AbstractFloat, V <: AbstractVector}(
    x :: AbstractMatrix{T},
    y :: V;
    K :: Int = 100,
    lambdas        :: Vector{T} = compute_lambdas(x,y,K),
    max_iter       :: Int  = 1000,
    tol            :: T    = convert(T, 1e-4),
    mu             :: T    = one(T),
    backtrack_size :: T    = convert(T, 0.5),
    quiet          :: Bool = true
) :: SparseMatrixCSC{T,Int}

    # size of problem?
    n,p = size(x)

    # ensure conformable arguments
    @assert n == length(y) "Arguments x and y must have same number of rows"
   
    # prespecify a matrix to store calculated models
    betas = spzeros(T, p, K) 

    # allocate all intermediate arrays
    # insert dummy variable for MCP gamma
    v = FISTAVariables(x, y, lambdas, convert(T, 3))

    # compute the path
    for i = 1:K

        # current lambda?
        λ = v.lambdas[i]

        # track progress
        quiet || print_with_color(:blue, "Computing iteration $i with λ = $(λ).\n\n")

        # run FISTA on this lambda value
        output = lasso_reg(x, y, λ, v=v, max_iter=max_iter, tol=tol, mu=mu, backtrack_size=backtrack_size, quiet=quiet)

        # save current progress
        betas[:,i] = sparsevec(output.beta)

        # cut path short if b is completely full
        if countnz(output.beta) == p
            quiet || warn("Model at λ = ", λ, " is completely saturated.\nlasso_path will terminate early...\n")
            return betas[:,1:i]
        end
    end

    return betas
end


function one_fold{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
	K        :: Int,
    folds    :: DenseVector{Int},
    fold     :: Int;
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
	mu       :: T    = one(T),
	backtrack_size :: T = convert(T, 0.5),
    quiet    :: Bool = true,
    lambdas  :: Vector{T} = compute_lambdas(x,y,K)
) :: Vector{T}

    # make vector of indices for folds
    test_idx  = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # allocate the arrays for the training set
    x_train = x[train_idx,:]
    y_train = y[train_idx]

    # compute the regularization path on the training set
    betas = lasso_path(x_train, y_train, K=K, tol=tol, max_iter=max_iter, quiet=quiet, mu=mu, backtrack_size=backtrack_size, lambdas=lambdas)

    # compute the mean out-of-sample error for the TEST set
    Xb = view(x, test_idx, :) * betas
#    r  = broadcast(-, view(y, test_idx, 1), Xb)
#    r  = broadcast(-, y[test_idx], Xb)
    r  = y[test_idx] .- Xb
    er = vec(sumabs2(r, 1)) ./ (2*test_size)

    return er
end

function pfold{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
	K        :: Int,
    folds    :: DenseVector{Int},
    q        :: Int;
    pids     :: Vector{Int} = procs(),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
	mu       :: T    = one(T),
	backtrack_size :: T = convert(T, 0.5),
    quiet    :: Bool = true,
    lambdas  :: Vector{T} = compute_lambdas(x,y,K)
) :: Vector{T}

    # how many CPU processes can pfold use?
    np = length(pids)

    # report on CPU processes
    quiet || println("pfold: np = ", np)
    quiet || println("pids = ", pids)

    # set up function to share state (indices of folds)
    i = 1
    nextidx() = (idx=i; i+=1; idx)

    # preallocate cell array for results
    results = SharedArray(T, (K,q), pids=pids) :: SharedMatrix{T}

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
                                one_fold(x, y, K, folds, current_fold, tol=tol, max_iter=max_iter, backtrack_size=backtrack_size, mu=mu, quiet=quiet, lambdas=lambdas)
                        end # end remotecall_fetch()
                        setindex!(results, r, :, current_fold)
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return (vec(sum(results, 2) ./ q))
end

function cv_lasso{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T};
    q        :: Int  = cv_get_num_folds(3, 5),
	K        :: Int  = 20,
    lambdas  :: Vector{T} = compute_lambdas(x,y,K), 
    folds    :: DenseVector{Int} = cv_get_folds(sdata(y),q),
    pids     :: Vector{Int}      = procs(),
    tol      :: T    = convert(T, 1e-4),
    max_iter :: Int  = 100,
	mu       :: T    = one(T),
	backtrack_size :: T = convert(T, 0.5),
    quiet    :: Bool = true,
)
    # do not allow crossvalidation with fewer than 3 folds
    q > 2 || throw(ArgumentError("Number of folds q = $q must be at least 3."))

    # problem dimensions?
    n,p = size(x)

    # check for conformable arrays
    n == length(y) || throw(DimensionMismatch("Row dimension of x ($n) must match number of rows in y ($(length(y)))"))

    # want to compute a path for each fold
    # the folds are computed asynchronously over processes enumerated by pids
    # master process then reduces errors across folds and returns MSEs
    mses = pfold(x, y, K, folds, q, pids=pids, tol=tol, max_iter=max_iter, backtrack_size=backtrack=backtrack_size, mu=mu, quiet=quiet, lambdas=lambdas)

    # what is the best model size?
    # if there are multiple model sizes of EXACTLY the same MSE,
    # then this chooses the smaller of the two
    i = indmin(mses)
#    k = path[i] :: Int
    λ = lambdas[i] :: T
#    path = vec(mapslices(

    # print results
#    !quiet && print_cv_results(mses, path, k)

    # refit the best model
#    b, bidx = refit_iht(sdata(x), sdata(y), k, tol=tol, max_iter=max_iter, max_step=max_step, quiet=quiet)
    output = lasso_reg(sdata(x), sdata(y), λ, K=K, max_iter=max_iter, backtrack_size=backtrack_size, quiet=quiet, tol=tol, mu=mu)
    bidx  = find(output.beta)
    b     = output.beta[bidx]
    k     = countnz(output.beta)

    #return FISTACrossvalidationResults(mses, path, λ, b, bidx, k)
    return FISTACrossvalidationResults(mses, λ, b, bidx, k)
end
