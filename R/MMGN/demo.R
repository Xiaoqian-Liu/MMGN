library(MMGN)
library(lattice)
library(glmnet)
m <- 1000
n <- 1000
r <- 1

rho <- 0.5
sigma <- 0.5

# generate the underlying matrix, non-spiky
set.seed(2022)
U <- matrix(runif(m*r,min=-0.5,max=0.5),nrow = m)
V <- matrix(runif(n*r,min=-0.5,max=0.5),nrow = n)
M_star <- U%*%t(V)
M_star <- M_star/max(abs(M_star))

sratio <- sqrt(m*n)/norm(M_star, "F")




###############  Logistic model ################
############### ############### ############### 
f <- function(x) {return(plogis(x, scale = sigma))}
gradf <- function(x) {return(dlogis(x, scale = sigma))}


# generate the binary matrix Y
Y0 <- sign(f(M_star) - matrix(runif(m*n), nrow=m))

omega <- sample(1:(m*n), size = floor(rho*m*n))
ind_omega <- rep(0, m*n);
ind_omega[omega] <- 1

y <- as.vector(Y0)
y[-omega] <- 0
Y <- matrix(y, m, n)

## initialization
svd_out = svd(Y)
U0 <- svd_out$u[, 1:r]%*%sqrt(diag(x=svd_out$d[1:r], nrow=r))
V0 <- svd_out$v[, 1:r]%*%sqrt(diag(x=svd_out$d[1:r], nrow=r))

######## *** takes about 1 seconds ***
t <- proc.time()
res_mmgn <- MMGN_logist(Y, ind_omega, sigma=sigma, R=r, U0=U0, V0=V0, maxiters=500, tol=1e-4)
proc.time()- t


Mhat <- (res_mmgn$U)%*%t(res_mmgn$V)
# relative error
norm(M_star - Mhat, 'F')^2/norm(M_star, 'F')^2


######## *** takes about 10 seconds ***
t <- proc.time()
res_auto <- MMGN_logist_auto(Y, ind_omega, sigma, rSeq=1:5, rate=0.8)
proc.time()- t

Mhat <- (res_auto$U)%*%t(res_auto$V)
# relative error
norm(M_star - Mhat, 'F')^2/norm(M_star, 'F')^2









###############  Probit model ################
############### ############### ############### 
f <- function(x) {return(pnorm(x, sd=sigma))}
gradf <- function(x) {return(dnorm(x, sd=sigma))}


# generate the binary matrix Y
Y0 <- sign(f(M_star) - matrix(runif(m*n), nrow=m))

omega <- sample(1:(m*n), size = floor(rho*m*n))
ind_omega <- rep(0, m*n);
ind_omega[omega] <- 1

y <- as.vector(Y0)
y[-omega] <- 0
Y <- matrix(y, m, n)

## initialization
svd_out <- svd(Y)
U0 <- svd_out$u[, 1:r]%*%sqrt(diag(x=svd_out$d[1:r], nrow=r))
V0 <- svd_out$v[, 1:r]%*%sqrt(diag(x=svd_out$d[1:r], nrow=r))

########################################
######## *** takes about 1 seconds ***
t <- proc.time()
res_mmgn <- MMGN_probit(Y, ind_omega, sigma=sigma, R=r, U0=U0, V0=V0, maxiters=500, tol=1e-4)
proc.time()- t


Mhat <- (res_mmgn$U)%*%t(res_mmgn$V)
# relative error
norm(M_star - Mhat, 'F')^2/norm(M_star, 'F')^2


########################################
######## *** takes about 15 seconds ***
t <- proc.time()
res_auto <- MMGN_probit_auto(Y, ind_omega, sigma, rSeq=1:5, rate=0.8)
proc.time()- t

Mhat <- (res_auto$U)%*%t(res_auto$V)
# relative error
norm(M_star - Mhat, 'F')^2/norm(M_star, 'F')^2


