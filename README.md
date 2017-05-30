# FISTA.jl
Accelerated soft- and firm-thresholding algorithms for sparse regression

This package features a Julia implementation of the *fast iterative shrinkage-thresholding algorithm* (FISTA) for sparse regression.
FISTA is a Nesterov-accelerated proximal gradient scheme that minimizes the sum of squares subject to a sparsity-inducing penalty.
Both soft- and firm-thresholding are included here.
Soft-thresholding corresponds to [LASSO regression](https://en.wikipedia.org/wiki/Lasso_(statistics)), implemented in the popular [`glmnet`](https://cran.r-project.org/web/packages/glmnet/index.html) package in R.
LASSO regression employs the convex L1-norm penalty.
Firm-thresholding corresponds to MCP regression, implemented in [`ncvreg`](https://cran.r-project.org/web/packages/ncvreg/index.html).
MCP regression employs the [*minimax concave penalty*](https://arxiv.org/abs/1002.4734).

Other Julia packages that implement LASSO regression include:

* [GLMNet.jl](https://github.com/simonster/GLMNet.jl), a Julia wrapper around the Fortran kernel in `glmnet`.
* [Lasso.jl](https://github.com/simonster/Lasso.jl), a pure Julia implementation of `glmnet`.
* [SparseRegression.jl](https://github.com/joshday/SparseRegression.jl), a package with another FISTA implementation of LASSO.
* [Regression.jl](https://github.com/lindahua/Regression.jl), a package that implements various techniques for solving regression with generalized linear models.

This package is the intellectual cousin of [IHT.jl](https://github.com/klkeys/IHT.jl), which implements *iterative hard thresholding* for sparse regression.

## Download

## Basic use

## GWAS

## References

* Amir Beck and Marc Teboulle. (2009) A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM J. Imaging Sciences **2**:1, 183--202. ([pdf](http://epubs.siam.org/doi/abs/10.1137/080716542?journalCode=sjisbi))
* Thomas Blumensath and Mike E. Davies. (2008) Iterative hard thresholding for compressed sensing. Applied and Computational Harmonic Analysis, **27**:3, 265--274. ([pdf](https://arxiv.org/pdf/0805.0510.pdf))
* Patrick Breheny and Jian Huang. (2011). Coordinate descent algorithms for nonconvex penalized regression, with applications to biological feature selection. *Annals of Applied Statistics* **5**:1, 232--253.
* Jerome Friedman, Trevor Hastie, and Robert Tibshirani. (2010) Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software* **33**:1, 1--22. ([pdf](https://www.jstatsoft.org/article/view/v033i01/v33i01.pdf))
* Cun-Hui Zhang. (2010) Nearly unbiased variable selection under minimax concave penalty. *The Annals of Statistics* **38**:2, 894--942. ([pdf](https://arxiv.org/abs/1002.4734))
