#' Loss function of the 1-bit matrix completion problem 
#' 
#' \code{loss_1bit} computes the loss (negative log-likelihood) in the 1Bit matrix completion problem 
#' 
#' @param Y the obseved data matrix with missing values 
#' @param M the current estimate of the underlying matrix
#' @param omega the index set of observed entries (column-major vectorization)
#' @param f the CDF of the noise (probit or logistic)
#' @return the loss at the current M
#' @export
#' 
loss_1bit <- function(Y, M, omega, f){
  D <- (1+Y)/2
  d <- as.vector(D)
  d <- d[omega]

  m <- as.vector(M)
  m <- m[omega]

  nOmega <- length(omega)
  ix1 <- which(d!=0)
  ix0 <- setdiff(1:nOmega, ix1)

  obj <- -sum(log(f(m[ix1]))) - sum(log(1-f(m[ix0])))
  
  return(obj)
}
