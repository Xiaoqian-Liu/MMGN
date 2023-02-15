#' MMGN for 1-bit matrix completion under the probit noise model
#' 
#' \code{MMGN_probit} implements MMGN for 1-bit matrix completion under the probit model
#' 
#' @param Y the obseved data matrix with missing values 
#' @param omega tthe index set of observed entries (column-major vectorization)
#' @param R the target rank in the rank constraint
#' @param sigma the noise level (sigma) of the probit model
#' @param maxiters maximum number of iterations
#' @param tol tolerance for the relative change in the estimate or objective (for early stopping)
#' @param stopping which criterion for early stopping. 
#'                 objective: early stop if the relative change in the objective value is less than tol (default)
#'                 estimate: early stop if the relative change in the estimate is less than tol 
#' @return \code{U} the estimated factor matrix U
#' @return \code{V} the estimated factor matrix V. The estimated matrix M=UV'
#' @return \code{iter} the number of iterations used for the MMGN algorithm to converege
#' @return \code{obj} the objective value at each iteration
#' @return \code{relchange} the relative change in the objective / estimate at each iteration
#' @return \code{nBacktracks} the number of backtracking steps at each iteration
#' @export
#' 
MMGN_probit <- function(Y, omega, R, sigma, maxiters=1e2, tol=1e-4, stopping = 'objective'){
  
  m <- nrow(Y)
  n <- ncol(Y)
  D <- (1+Y)/2

  f <- function(x) {return(pnorm(x, sd=sigma))}
  gradf <- function(x) {return(dnorm(x, sd=sigma))}

  # random initial values for MMGN
  set.seed(1234)
  U <- 0.1*matrix(data=rnorm(m*R), nrow = m)
  V <- 0.1*matrix(data=rnorm(n*R), nrow = n)
  Mhat_last <- U%*%t(V)
  obj_last <- loss_1bit(Y, Mhat_last, omega, f)


  #parameter for backtracking
  beta <- 0.5
  gamma <- 0.9

  obj <- rep(NA, maxiters)
  relchange <- rep(NA, maxiters)
  dec <- rep(NA, maxiters)
  alphas <- rep(NA, maxiters)
  nBacktracks <- rep(NA, maxiters)
  L <- 1/sigma^2

  for (i in 1:maxiters) {
    # construct X
    X <- (1/L)*Y*gradf(Mhat_last)/f(Y*Mhat_last)
    # construct x in the LS problem
    x <- as.vector(X)
    x <- x[omega]
    # construct J in the LS problem
    Phi <- cbind(kronecker(V, Diagonal(n=m, x=1)), kronecker(Diagonal(n=n, x=1), U))
    Phi <- Phi[omega, ]

    # solve the LS problem using LSQR
    eta <- glmnet(Phi, x, lambda = 0, intercept = FALSE)$beta
    # get du and dv
    du <- eta[1:(m*R)]
    dv <- eta[(m*R+1):(R*(m+n))]

    # recover dU and dV
    dU <- matrix(du,  nrow = m)
    dV <- t(matrix(dv, nrow = R))

    # update U and V with stepsize alpha chosen by backtracking
    alpha <- 1
    U1 <- U + alpha*dU
    V1 <- V + alpha*dV

    # compute dU V + U dV in the backtracking condition
    G <- dU%*%t(V)+ U%*%t(dV)
    g <- as.vector(G)
    g <- g[omega]

    dec[i] <- -L*x%*%g

    # start backtracking
    nBacktrack <- 0
    Mhat <- U1%*%t(V1)
    obj_new <- loss_1bit(Y, Mhat, omega, f)
    while (obj_new > obj_last + beta*alpha*dec[i] && nBacktrack<=100) {
      alpha <- gamma*alpha
      U1 <- U + alpha*dU
      V1 <- V + alpha*dV
      Mhat <- U1%*%t(V1)
      obj_new <- loss_1bit(Y, Mhat, omega, f)
      nBacktrack <- nBacktrack +1
    }

    # save some output
    obj[i] <- obj_new
    alphas[i] <- alpha
    nBacktracks[i] <- nBacktrack

    # relative change according to stopping 
    if(i >10){
       if (stopping == 'objective'){
         relchange[i] <- abs((obj_last - obj_new)/(obj_last + 1e-15))
       }else{
         relchange[i] <- norm(Mhat - Mhat_last, "F")^2/norm(Mhat_last, "F")^2      
       }
      
      # earlier stopping
      if(relchange[i]< tol){
        U <- U1
        V <- V1
        break
      }
    }
    
 

    # update for  the next iteration
    U <- U1
    V <- V1
    Mhat_last <- Mhat
    obj_last <- obj_new

  }

  return(list(U = U, V = V, iters=i, obj = obj[1:i],
              relchange = relchange[1:i], 
              nBacktracks = nBacktracks[1:i]))
}
