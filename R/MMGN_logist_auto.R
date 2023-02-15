#' MMGN for 1-bit matrix completion under the logistic noise model with a data-driven approach 
#' for selecting the parameeter r
#' 
#' \code{MMGN_logistic_auto} implements MMGN for 1-bit matrix completion under the logistic model 
#' and automatically select the parameter r using a data-driven approach.
#' 
#' @param Y the obseved data matrix with missing values 
#' @param omega the index set of observed entries (column major vectorization)
#' @param sigma the noise level (sigma) of the logistic model
#' @param rSeq a grid of positive integer values for the parameter r (default: 1 to 5)
#' @param rate the percentage of data working as the training set (default: 0.8)
#' @param seed set seed for reproducible results (default: 1234)
#' @param maxiters maximum number of iterations (default: 5e2)
#' @param tol tolerance for early stopping (default: 1e-4)
#' @param stopping which criterion for early stopping. 
#'                 objective: early stop if the relative change in the objective value is less than tol (default)
#'                 estimate: early stop if the relative change in the estimate is less than tol 
#' @return \code{U} the estimated factor matrix U at the selected r
#' @return \code{V} the estimated factor matrix V at the selected r. The estimated matrix M=UV'
#' @return \code{rhat} the selected r 
#' @return \code{loglik} the log-likelihood at each value of rSeq 
#' @return \code{iter} the number of iterations used for the MMGN algorithm to converege for estimating the underlying 
#'                     matrix with the selected r
#' @return \code{obj} the objective value at each iteration for estimating the underlying 
#'                     matrix with the selected r
#' @return \code{relchange} the relative change of the estimate/objective at each iteration for estimating the underlying 
#'                     matrix with the selected r
#' @return \code{nBacktracks} the number of backtracking steps at each iteration for estimating the underlying 
#'                     matrix with the selected r
#' @export
#' 
MMGN_logist_auto <- function(Y, omega, sigma, rSeq=1:5, rate=0.8, seed=1234, maxiters=5e2, tol=1e-4, stopping='objective'){
  
  numR <- length(rSeq)

  f <- function(x) {return(plogis(x, scale = sigma))}
  gradf <- function(x) {return(dlogis(x, scale = sigma))}
  
  # generate training/testing sets
  set.seed(seed)
  training <- sample(omega, size=ceiling(rate*length(omega)))
  testing <- setdiff(omega, training)
  
  
  loglik <- rep(NA, numR)
  #main loop for selecting r
  for (i in 1:numR) {
    # set r for this loop
    r <- rSeq[i]
    
    res <- MMGN_logist(Y, omega = training, R=r, sigma = sigma, maxiters = maxiters, tol=tol, stopping=stopping)
    Mhat <- res$U%*%t(res$V)
    
    # compute the log-likelihood on the testing set
    loglik[i] <- -loss_1bit(Y, Mhat, testing, f)
  }

  ind <- which.max(loglik)
  rhat <- rSeq[ind]
  
  res <- MMGN_logist(Y, omega = omega, R=rhat, sigma = sigma, maxiters = maxiters, tol=tol, stopping=stopping)
  iter <- res$iters
  
  return(list(U = res$U, V = res$V, rhat = rhat, loglik = loglik, iters=iter, obj = res$obj[1:iter],
               relchange = res$relchange[1:iter], nBacktracks = res$nBacktracks[1:iter]))
}
