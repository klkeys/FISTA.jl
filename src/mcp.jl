function mcp_reg{T <: AbstractFloat, V <: AbstractVector}(
    x :: AbstractMatrix{T},
    y :: V,
    λ :: T,
    γ :: T;
    v :: FISTAVariables{T,V} = FISTAVariables(x, y),
    max_iter       :: Int  = 1000,
    tol            :: T    = convert(T, 1e-4),
    mu             :: T    = one(T),
    backtrack_size :: T    = convert(T, 0.5),
    quiet          :: Bool = true
) :: typeof(y)

    n,p = size(x)
    @assert n == length(y) "Arguments x and y must have same number of rows"
    loss  = convert(T, Inf)
    loss0 = loss

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
            v.b .= firmthreshold.(v.b, λ, γ)
           
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

    return v.b
end
