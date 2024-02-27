#' MMGN for 1-bit matrix completion under the logistic noise model with a data-driven approach 
#' for selecting the parameter r
#' 
#' \code{logistPCA_auto} implements MMGN for 1-bit matrix completion under the logistic model 
#' and automatically select the parameter r using a data-driven approach.
#' 
#' @param Y the observed data matrix with missing values 
#' @param ind_omega the indicator vector of observations (column-major vectorization)
#' @param sigma the noise level (sigma) of the logistic model; logisticSVD only allows sigma=1
#' @param rSeq a grid of positive integer values for the parameter r (default: 1 to 5)
#' @param rate the percentage of data working as the training set (default: 0.8)
#' @param seed set seed for reproducible results (default: 1234)
#'                 objective: early stop if the relative change in the objective value is less than tol (default)
#'                 estimate: early stop if the relative change in the estimate is less than tol 
#' @return \code{U} the estimated factor matrix U at the selected r
#' @return \code{V} the estimated factor matrix V at the selected r. The estimated matrix M=UV'
#' @return \code{rhat} the selected r 
#' @export
#' 
logistPCA_auto <- function(Y, ind_omega, sigma=1, rSeq=1:5, rate=0.8, seed=1234){
  
  numR <- length(rSeq)
  
  f <- function(x) {return(plogis(x, scale = sigma))}
  gradf <- function(x) {return(dlogis(x, scale = sigma))}
  
  # generate training/testing sets
  set.seed(seed)
  omega <- which(ind_omega > 0)
  training <- sample(omega, size=ceiling(rate*length(omega)))
  testing <- setdiff(omega, training)
  m <- nrow(Y)
  n <- ncol(Y)
  ind_train <- ind_test <- rep(0, m*n)
  ind_train[training] <- 1
  ind_test[testing] <- 1
  
  # get the training matrix for logistPCA
  y_train <- as.vector(Y)
  y_train[-training] <- NA
  Y_train <- matrix(y_train, m, n)
  D_train <- (1+Y_train)/2
  
  
  # get the observed matrix for logistPCA
  y_obs <- as.vector(Y)
  y_obs[-omega] <- NA
  Y_obs <- matrix(y_obs, m, n)
  D_obs <- (1+Y_obs)/2
  
  loglik <- rep(NA, numR)
  #main loop for selecting r
  for (i in 1:numR) {
    # set r for this loop
    r <- rSeq[i]
    
    res <- logisticSVD(D_train, k = r, main_effects = FALSE, partial_decomp = FALSE)
    Mhat <- res$A%*%t(res$B)
    
    # compute the log-likelihood on the testing set
    #loglik[i] <- -loss_1bit(D_obs, Mhat, testing, f)
    loglik[i] <- -obj_1bit(D_obs[ind_test>0], Mhat[ind_test>0], f)
  }
  
  ind <- which.max(loglik)
  rhat <- rSeq[ind]
  
  res <- logisticSVD(D_obs, k = rhat, main_effects = FALSE, partial_decomp = FALSE)

  
  return(list(U = res$A, V = res$B, rhat = rhat, loglik = loglik))
}
