
binary_fit_auto <- function(Y, omega, sigma, alpha=1,  option=2, rSeq=1:5, rate=0.8, seed=1234){
  
  m <- nrow(Y)
  n <- ncol(Y)
  
  ###############  Logistic model ################
  ############### ############### ############### 
  f <- function(x) {return(plogis(x, scale = sigma))}
  gradf <- function(x) {return(dlogis(x, scale = sigma))}
  
  
  
  # generate training/testing sets
  set.seed(seed)
  training <- sample(omega, size=ceiling(rate*length(omega)))
  testing <- setdiff(omega, training)
  m <- nrow(Y)
  n <- ncol(Y)
  ind_train <- ind_test <- rep(0, m*n)
  ind_train[training] <- 1
  ind_test[testing] <- 1
  
  
  y_train <- as.vector(Y)
  y_train[-training] <- NA
  Y_train <- as.matrix(y_train, m, n)

  D <- (1+Y)/2
  d <- D[ind_test>0]
  
  
  numR <- length(rSeq)
  loglik <- rep(NA, numR)
  #main loop for selecting r
  for (i in 1:numR) {
    # set r for this loop
    r <- rSeq[i]
    
    YY <- as.tensor(array(1,dim = c(m,n,r)))@data
    for (k in 1:r) {
      YY[, , k] <- (1+Y_train)/2
    }
    

    fit <- binary_fit(YY,r,alpha,sigma,option,random.ini=FALSE,const=TRUE,nrand=3) ## estimate the low-rank tensor from the binary observations.
    Mhat <- fit$para[, , 1]
    
    # compute the log-likelihood on the testing set
    loglik[i] <- -obj_1bit(d, Mhat[ind_test>0], f)
  }
  
  ind <- which.max(loglik)
  rhat <- rSeq[ind]
  YY <- as.tensor(array(1,dim = c(m,n,rhat)))@data
  for (k in 1:rhat) {
    YY[, , k] <- (1+Y)/2 # now use the whole set of observation, omega
  }
  fit <- binary_fit(YY,rhat,alpha,sigma,option,random.ini=FALSE,const=TRUE,nrand=3)
  Mhat <-  fit$para[, , 1]
  
  
  return(list(Mhat = Mhat, rhat = rhat, loglik = loglik))
}
